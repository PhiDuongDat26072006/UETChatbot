"""
tests/test_agent_eval.py - Kiểm thử đánh giá quyết định Agent (JEV Router, Answer Eval, Trajectory, Trace).
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.agent.graph import UETAgentGraph
from src.base import (
    BaseRAGPipeline,
    EvaluationSample,
    Response,
    RetrievedContext,
    UserQuery,
)
from src.decision import FakeDecisionClient, JevDecisionClient, Route, Verdict
from src.evaluation.agent_eval import (
    answer_eval_reference,
    compute_answer_eval_metrics,
    compute_loop_metrics,
    compute_roc_auc,
    compute_router_metrics,
    compute_sample_agent_flags,
    compute_trajectory_metrics,
    evaluate_agent_decisions,
    extract_rounds,
    score_sample_rounds,
)
from src.evaluation.diagnostics import (
    build_e2e_report,
    classify_sample_error,
    print_e2e_diagnostic_report,
)
from src.evaluation.evaluator import (
    generate_predictions,
    run_router_eval_cli,
    score_predictions,
)
from src.evaluation.llm_judge import UETLLMJudgeEvaluator
from src.evaluation.metrics import strip_agent_notes


class _StubRetriever:
    default_top_k = 4

    def __init__(self) -> None:
        self.calls = 0

    def retrieve(self, query: UserQuery, top_k: int = 4, filters: Optional[Dict[str, Any]] = None) -> List[RetrievedContext]:
        self.calls += 1
        return [
            RetrievedContext(
                chunk_id=f"c_{self.calls}_1",
                text="Điều kiện tốt nghiệp sớm là GPA >= 3.0 và tích lũy 90 tín chỉ.",
                similarity_score=0.92,
                rank=1,
                metadata={"document_id": "d_1", "title": "Quy chế đào tạo"},
            )
        ]


class _StubLLM:
    model_name = "stub-gen"

    def generate(self, augmented_prompt: Any, model: Optional[str] = None) -> Response:
        return Response(
            query_id=getattr(getattr(augmented_prompt, "user_query", None), "query_id", "q_stub"),
            answer="Điều kiện tốt nghiệp sớm là GPA >= 3.0 và tích lũy 90 tín chỉ.",
            sources=augmented_prompt.contexts,
            model_name=self.model_name,
            latency_seconds=0.05,
            metadata={"usage": {"prompt_tokens": 20, "completion_tokens": 15}},
        )


class _FakeAgenticPipeline(BaseRAGPipeline):
    def __init__(self, decision_client: FakeDecisionClient) -> None:
        self.agent_graph = UETAgentGraph(
            retriever=_StubRetriever(),
            llm_client=_StubLLM(),
            decision_client=decision_client,
            config={"agent": {"answer_eval": {"max_retries": 2}, "trace": {"verbose": False}}},
        )

    def query(self, query: UserQuery, top_k: Optional[int] = None) -> Response:
        state = self.agent_graph.run(query=query.query_text, user_query=query, top_k=top_k)
        agent_resp = state.get("response")
        route_val = (
            agent_resp.route.value
            if (agent_resp and agent_resp.route)
            else (state.get("route").value if state.get("route") else Route.QUALIFIED_QUESTION.value)
        )
        return Response(
            query_id=query.query_id,
            answer=state.get("answer", ""),
            sources=state.get("contexts") or [],
            model_name="stub-gen",
            latency_seconds=0.12,
            metadata={
                "status": agent_resp.status if agent_resp else "ok",
                "route": route_val,
                "retry": state.get("retry", 0),
                "fail_reason": state.get("fail_reason"),
                "missing_info": state.get("missing_info"),
                "used_queries": state.get("used_queries", []),
                "trace": state.get("trace", []),
            },
        )


class TestAgentTraceEnrichment(unittest.TestCase):
    def test_trace_verbose_toggle_and_chunk_ids(self):
        fake_client = FakeDecisionClient(
            routes=[Route.QUALIFIED_QUESTION],
            evaluate_answers=[Verdict.SUFFICIENT],
        )
        graph = UETAgentGraph(
            retriever=_StubRetriever(),
            llm_client=_StubLLM(),
            decision_client=fake_client,
            config={"agent": {"trace": {"verbose": False}}},
        )
        state_compact = graph.run("Điều kiện tốt nghiệp sớm?")
        trace_compact = state_compact["trace"]

        ret_entries = [t for t in trace_compact if t.get("node") == "retrieval_node"]
        gen_entries = [t for t in trace_compact if t.get("node") == "generation_node"]
        router_entries = [t for t in trace_compact if t.get("node") == "router_node"]
        eval_entries = [t for t in trace_compact if t.get("node") == "answer_eval_node"]

        self.assertEqual(len(ret_entries), 1)
        self.assertEqual(ret_entries[0]["round"], 0)
        self.assertEqual(ret_entries[0]["chunk_ids"], ["c_1_1"])
        self.assertEqual(ret_entries[0]["selected_chunk_ids"], ["c_1_1"])
        self.assertEqual(ret_entries[0]["queries"], ["Điều kiện tốt nghiệp sớm?"])

        self.assertEqual(len(gen_entries), 1)
        self.assertEqual(gen_entries[0]["round"], 0)
        self.assertNotIn("answer_text", gen_entries[0])

        self.assertEqual(router_entries[0]["raw_label"], "QUALIFIED_QUESTION")
        self.assertIs(router_entries[0]["parse_fallback"], False)
        self.assertEqual(eval_entries[0]["round"], 0)

        # Bật trace_verbose -> lưu answer_text trong generation_node
        graph.trace_verbose = True
        fake_client_v = FakeDecisionClient(
            routes=[Route.QUALIFIED_QUESTION],
            evaluate_answers=[Verdict.SUFFICIENT],
        )
        graph.decision_client = fake_client_v
        state_verbose = graph.run("Điều kiện tốt nghiệp sớm?")
        gen_v = [t for t in state_verbose["trace"] if t.get("node") == "generation_node"]
        self.assertIn("answer_text", gen_v[0])
        self.assertIn("GPA >= 3.0", gen_v[0]["answer_text"])

    def test_jev_client_route_parse_fallback_meta(self):
        client = JevDecisionClient(endpoint="http://localhost:9999/v1/systemone", api_key="test")
        client._call_decisions_api = lambda state, questions, session_id=None: (  # type: ignore[method-assign]
            {"answers": {"route": {"choice": "UNEXPECTED_LABEL"}}},
            {"source": "jev", "latency_s": 0.11, "server_latency_s": 0.08, "payload_bytes": 120},
        )
        route = client.route("Hỏi điểm chuẩn CNTT")
        self.assertEqual(route, Route.QUALIFIED_QUESTION)
        self.assertTrue(client.last_trace_meta["parse_fallback"])
        self.assertEqual(client.last_trace_meta["raw_label"], "UNEXPECTED_LABEL")

    def test_generate_predictions_enables_and_restores_trace_verbose(self):
        fake_client = FakeDecisionClient(
            routes=[Route.QUALIFIED_QUESTION, Route.OUT_OF_SCOPE],
            evaluate_answers=[Verdict.SUFFICIENT],
        )
        pipeline = _FakeAgenticPipeline(fake_client)
        self.assertFalse(pipeline.agent_graph.trace_verbose)

        dataset = [
            EvaluationSample(
                id="s1",
                query="Điều kiện tốt nghiệp sớm?",
                ground_truth="GPA >= 3.0 và tích lũy 90 tín chỉ.",
                metadata={"type": "factoid", "unit": "UET", "expected_route": "SINGLE_HOP"},
            ),
            EvaluationSample(
                id="s2",
                query="Thời tiết Hà Nội hôm nay?",
                ground_truth="Ngoài phạm vi UET.",
                metadata={"type": "out_of_domain", "unit": "out_of_domain", "expected_route": "OUT_OF_SCOPE"},
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            snap_path = Path(tmpdir) / "agent_preds.json"
            snapshot = generate_predictions(
                pipeline=pipeline,
                dataset=dataset,
                top_k=4,
                delay_between_queries=0.0,
                output_path=snap_path,
            )
            # Đảm bảo trace_verbose được khôi phục về False sau khi chạy xong
            self.assertFalse(pipeline.agent_graph.trace_verbose)
            self.assertEqual(snapshot["total_samples"], 2)

            rec0 = snapshot["predictions"][0]
            self.assertEqual(rec0["expected_route"], "SINGLE_HOP")
            self.assertIn("agent", rec0)
            self.assertEqual(rec0["agent"]["route"], "QUALIFIED_QUESTION")
            self.assertEqual(rec0["agent"]["status"], "ok")
            gen_trace = [t for t in rec0["agent"]["trace"] if t.get("node") == "generation_node"]
            self.assertEqual(len(gen_trace), 1)
            self.assertIn("answer_text", gen_trace[0])


class TestAgentDecisionMetrics(unittest.TestCase):
    def test_router_metrics_false_oos_and_confusion_matrix(self):
        records = [
            # In-domain đúng (explicit SINGLE_HOP -> QUALIFIED_QUESTION)
            {
                "sample_id": "r1",
                "is_out_of_domain": False,
                "expected_route": "SINGLE_HOP",
                "agent": {
                    "route": "QUALIFIED_QUESTION",
                    "status": "ok",
                    "trace": [{"node": "router_node", "result": "QUALIFIED_QUESTION", "parse_fallback": False}],
                },
            },
            # In-domain bị chặn nhầm (FALSE OOS)
            {
                "sample_id": "r2",
                "is_out_of_domain": False,
                "expected_route": "QUALIFIED_QUESTION",
                "agent": {
                    "route": "OUT_OF_SCOPE",
                    "status": "out_of_scope",
                    "trace": [{"node": "router_node", "result": "OUT_OF_SCOPE", "parse_fallback": False}],
                },
            },
            # Legacy AMBIGUOUS tự ánh xạ -> QUALIFIED_QUESTION + có parse_fallback
            {
                "sample_id": "r3",
                "is_out_of_domain": False,
                "expected_route": "AMBIGUOUS",
                "agent": {
                    "route": "AMBIGUOUS",
                    "status": "ok",
                    "trace": [{"node": "router_node", "result": "AMBIGUOUS", "parse_fallback": True}],
                },
            },
            # OOD tường minh bắt đúng
            {
                "sample_id": "r4",
                "is_out_of_domain": True,
                "expected_route": "OUT_OF_SCOPE",
                "agent": {
                    "route": "OUT_OF_SCOPE",
                    "status": "out_of_scope",
                    "trace": [{"node": "router_node", "result": "OUT_OF_SCOPE", "parse_fallback": False}],
                },
            },
            # OOD không có expected_route, đi qua QUALIFIED (Q1: không phạt false_oos)
            {
                "sample_id": "r5",
                "is_out_of_domain": True,
                "agent": {
                    "route": "QUALIFIED_QUESTION",
                    "status": "ok",
                    "trace": [{"node": "router_node", "result": "QUALIFIED_QUESTION", "parse_fallback": False}],
                },
            },
        ]

        m = compute_router_metrics(records)
        self.assertEqual(m["total_routed"], 5)
        self.assertEqual(m["in_domain_total"], 3)
        self.assertEqual(m["false_oos_count"], 1)
        self.assertAlmostEqual(m["false_oos_rate"], round(1 / 3, 4))
        self.assertEqual(m["ood_total"], 2)
        self.assertEqual(m["ood_caught_at_router_count"], 1)
        self.assertAlmostEqual(m["ood_caught_at_router_rate"], 0.5)
        self.assertEqual(m["parse_fallback_count"], 1)
        self.assertEqual(m["explicit_labeled_total"], 4)
        self.assertAlmostEqual(m["explicit_accuracy"], 0.75)
        self.assertEqual(m["confusion_matrix"]["QUALIFIED_QUESTION"]["OUT_OF_SCOPE"], 1)
        self.assertEqual(m["confusion_matrix"]["QUALIFIED_QUESTION"]["QUALIFIED_QUESTION"], 2)
        self.assertEqual(set(m["confusion_matrix"].keys()), {"OUT_OF_SCOPE", "QUALIFIED_QUESTION"})

    def test_answer_eval_metrics_auc_and_threshold_sweep(self):
        records = [
            # TP: Judge 5/5 (SUFFICIENT), JEV SUFFICIENT (0.85)
            {
                "sample_id": "a1",
                "is_out_of_domain": False,
                "error_label": "OK",
                "offline_metrics": {"answer_correctness": 0.9, "numeric_faithfulness": 1.0},
                "judge": {"status": "ok", "raw_scores": {"answer_correctness": 5, "faithfulness": 5}},
                "agent": {
                    "route": "QUALIFIED_QUESTION",
                    "status": "ok",
                    "retry": 0,
                    "trace": [{"node": "answer_eval_node", "result": "SUFFICIENT", "noul_prob": 0.85, "source": "jev"}],
                },
            },
            # FP (thiếu ý): Judge 2/5 (INSUFFICIENT), JEV SUFFICIENT (0.62), faithfulness=5
            {
                "sample_id": "a2",
                "is_out_of_domain": False,
                "error_label": "INCOMPLETE_OR_WRONG",
                "offline_metrics": {"answer_correctness": 0.4, "numeric_faithfulness": 1.0},
                "judge": {"status": "ok", "raw_scores": {"answer_correctness": 2, "faithfulness": 5}},
                "agent": {
                    "route": "QUALIFIED_QUESTION",
                    "status": "ok",
                    "retry": 0,
                    "trace": [{"node": "answer_eval_node", "result": "SUFFICIENT", "noul_prob": 0.62, "source": "jev"}],
                },
            },
            # FP (bịa đặt / hallucination): Judge corr 2/5, faith 2/5, JEV SUFFICIENT (0.55)
            {
                "sample_id": "a3",
                "is_out_of_domain": False,
                "error_label": "HALLUCINATION",
                "offline_metrics": {"answer_correctness": 0.4, "numeric_faithfulness": 0.5, "ungrounded_entities": ["2027"]},
                "judge": {"status": "ok", "raw_scores": {"answer_correctness": 2, "faithfulness": 2}},
                "agent": {
                    "route": "QUALIFIED_QUESTION",
                    "status": "ok",
                    "retry": 0,
                    "trace": [{"node": "answer_eval_node", "result": "SUFFICIENT", "noul_prob": 0.55, "source": "jev"}],
                },
            },
            # TN: Fallback offline (0.35 -> INSUFFICIENT), partial với 2 vòng (0.20, 0.35 -> best=0.35)
            {
                "sample_id": "a4",
                "is_out_of_domain": False,
                "error_label": "INCOMPLETE_OR_WRONG",
                "offline_metrics": {"answer_correctness": 0.35, "numeric_faithfulness": 1.0},
                "agent": {
                    "route": "QUALIFIED_QUESTION",
                    "status": "partial",
                    "retry": 2,
                    "trace": [
                        {"node": "answer_eval_node", "result": "INSUFFICIENT", "noul_prob": 0.20, "source": "jev"},
                        {"node": "answer_eval_node", "result": "INSUFFICIENT", "noul_prob": 0.35, "source": "jev"},
                    ],
                },
            },
            # FN (False Reject): Judge 4/5 (SUFFICIENT), nhưng JEV đánh partial với max noul_prob=0.45
            {
                "sample_id": "a5",
                "is_out_of_domain": False,
                "error_label": "OK",
                "offline_metrics": {"answer_correctness": 0.8, "numeric_faithfulness": 1.0},
                "judge": {"status": "ok", "raw_scores": {"answer_correctness": 4, "faithfulness": 5}},
                "agent": {
                    "route": "QUALIFIED_QUESTION",
                    "status": "partial",
                    "retry": 2,
                    "trace": [
                        {"node": "answer_eval_node", "result": "INSUFFICIENT", "noul_prob": 0.45, "source": "jev"},
                        {"node": "answer_eval_node", "result": "INSUFFICIENT", "noul_prob": 0.40, "source": "jev"},
                    ],
                },
            },
            # OOD -> phải bị bỏ qua khỏi answer_eval
            {
                "sample_id": "a6",
                "is_out_of_domain": True,
                "agent": {"route": "OUT_OF_SCOPE", "status": "out_of_scope", "trace": []},
            },
        ]

        ref_a4, src_a4 = answer_eval_reference(records[3])
        self.assertEqual((ref_a4, src_a4), ("INSUFFICIENT", "offline"))

        ae = compute_answer_eval_metrics(records)
        self.assertEqual(ae["evaluated_samples"], 5)
        self.assertEqual(ae["reference_source_counts"], {"judge": 4, "offline": 1})
        self.assertEqual(ae["confusion_matrix"]["SUFFICIENT"], {"SUFFICIENT": 1, "INSUFFICIENT": 1})
        self.assertEqual(ae["confusion_matrix"]["INSUFFICIENT"], {"SUFFICIENT": 2, "INSUFFICIENT": 1})
        self.assertEqual(ae["false_accept_count"], 2)
        self.assertEqual(ae["false_accept_incomplete_count"], 1)
        self.assertEqual(ae["false_accept_hallucination_count"], 1)
        self.assertEqual(ae["false_reject_count"], 1)
        self.assertAlmostEqual(ae["false_reject_rate"], 0.5)
        self.assertIsNotNone(ae["roc_auc"])
        self.assertEqual(len(ae["threshold_sweep"]), 6)

        # Ở ngưỡng 0.70: chỉ a1 (0.85) được chấp nhận -> FP=0, precision=1.0
        row_07 = [r for r in ae["threshold_sweep"] if r["threshold"] == 0.7][0]
        self.assertEqual(row_07["false_accept_count"], 0)
        self.assertEqual(row_07["precision"], 1.0)

        # Kiểm tra ROC-AUC helper (perfect separation, ties, single class)
        self.assertEqual(compute_roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(compute_roc_auc([0, 1], [0.5, 0.5]), 0.5)
        self.assertIsNone(compute_roc_auc([1, 1], [0.6, 0.8]))

    def test_trajectory_metrics(self):
        records = [
            {
                "sample_id": "t1",
                "total_latency_s": 2.0,
                "agent": {
                    "route": "QUALIFIED_QUESTION",
                    "status": "ok",
                    "retry": 1,
                    "trace": [
                        {"node": "router_node", "source": "jev", "latency_s": 0.2},
                        {"node": "retrieval_node", "source": "retriever", "latency_s": 0.3},
                        {"node": "generation_node", "source": "llm", "latency_s": 0.5},
                        {"node": "answer_eval_node", "source": "jev", "latency_s": 0.2},
                        {"node": "followup_query_node", "source": "llm", "latency_s": 0.3},
                        {"node": "retrieval_node", "source": "retriever", "latency_s": 0.2},
                        {"node": "generation_node", "source": "llm", "latency_s": 0.4},
                        {"node": "answer_eval_node", "source": "jev", "latency_s": 0.2},
                    ],
                },
            },
            {
                "sample_id": "t2",
                "total_latency_s": 0.3,
                "agent": {
                    "route": "OUT_OF_SCOPE",
                    "status": "out_of_scope",
                    "retry": 0,
                    "trace": [{"node": "router_node", "source": "jev", "latency_s": 0.2}],
                },
            },
            {
                "sample_id": "t3",
                "total_latency_s": 0.5,
                "agent": {
                    "route": None,
                    "status": "fail",
                    "retry": 0,
                    "fail_reason": "jev_unavailable",
                    "trace": [{"node": "router_node", "source": "jev", "error": "timeout", "latency_s": 0.5}],
                },
            },
        ]

        tr = compute_trajectory_metrics(records)
        self.assertEqual(tr["total_agent_samples"], 3)
        self.assertEqual(tr["status_distribution"]["ok"], 1)
        self.assertEqual(tr["status_distribution"]["out_of_scope"], 1)
        self.assertEqual(tr["status_distribution"]["fail"], 1)
        self.assertEqual(tr["fail_reason_counts"], {"jev_unavailable": 1})
        self.assertEqual(tr["avg_retries_rag"], 1.0)
        self.assertEqual(tr["retry_triggered_rate"], 1.0)
        self.assertAlmostEqual(tr["avg_jev_calls"], round((3 + 1 + 1) / 3, 4))
        self.assertAlmostEqual(tr["avg_llm_calls"], round((3 + 0 + 0) / 3, 4))
        self.assertAlmostEqual(tr["avg_retrieval_calls"], round((2 + 0 + 0) / 3, 4))
        self.assertEqual(tr["latency_by_route_s"]["QUALIFIED_QUESTION"], 2.0)
        self.assertEqual(tr["latency_by_route_s"]["OUT_OF_SCOPE"], 0.3)

    def test_diagnostics_router_false_oos_and_backward_compat(self):
        # 1. Câu in-domain bị router chặn -> ROUTER_FALSE_OOS (ưu tiên trước FALSE_REFUSAL)
        false_oos_rec = {
            "sample_id": "fo1",
            "query": "Điểm chuẩn ngành CNTT 2025?",
            "is_out_of_domain": False,
            "offline_metrics": {"is_refusal": True, "context_relevance": 0.0, "overall_score": 0.0},
            "retrieval_metrics": {"chunk_hit_at_k": 0.0},
            "agent": {"route": "OUT_OF_SCOPE", "status": "out_of_scope", "retry": 0, "trace": []},
        }
        lbl, _ = classify_sample_error(false_oos_rec)
        self.assertEqual(lbl, "ROUTER_FALSE_OOS")
        self.assertIn("ROUTER_FALSE_OOS", compute_sample_agent_flags(false_oos_rec))

        # 2. Snapshot cũ (modular, không có agent data) -> KHÔNG có `agent_metrics` trong report
        modular_samples = [
            {
                "sample_id": "m1",
                "query": "Q1",
                "type": "factoid",
                "unit": "UET",
                "category": "QUY_CHE",
                "is_out_of_domain": False,
                "retrieval_metrics": {"chunk_hit_at_k": 1.0},
                "offline_metrics": {"overall_score": 0.9, "answer_correctness": 0.9, "faithfulness": 0.9},
                "error_label": "OK",
                "error_reason": "OK",
            }
        ]
        self.assertIsNone(evaluate_agent_decisions(modular_samples))
        modular_report = build_e2e_report(modular_samples, top_k=5)
        self.assertNotIn("agent_metrics", modular_report)

        # 3. Score snapshot agentic end-to-end + in báo cáo không lỗi
        agentic_snapshot = {
            "top_k": 5,
            "predictions": [
                {
                    "sample_id": "ag1",
                    "query": "Điều kiện tốt nghiệp sớm?",
                    "type": "factoid",
                    "unit": "UET",
                    "category": "QUY_CHE",
                    "ground_truth": "GPA >= 3.0 và tích lũy 90 tín chỉ.",
                    "target_document_ids": ["d_1"],
                    "target_chunk_ids": {"c_1": 2},
                    "is_out_of_domain": False,
                    "retrieved_contexts": [
                        {"chunk_id": "c_1", "document_id": "d_1", "rank": 1, "score": 0.9, "text": "GPA >= 3.0 và tích lũy 90 tín chỉ."}
                    ],
                    "generated_answer": "Điều kiện tốt nghiệp sớm là GPA >= 3.0 và tích lũy 90 tín chỉ.",
                    "model_name": "stub-gen",
                    "total_latency_s": 1.5,
                    "agent": {
                        "route": "QUALIFIED_QUESTION",
                        "status": "ok",
                        "retry": 0,
                        "trace": [
                            {"node": "router_node", "source": "jev", "result": "QUALIFIED_QUESTION", "latency_s": 0.2, "parse_fallback": False},
                            {"node": "retrieval_node", "source": "retriever", "round": 0, "chunk_ids": ["c_1"], "latency_s": 0.3},
                            {"node": "generation_node", "source": "llm", "round": 0, "latency_s": 0.8},
                            {"node": "answer_eval_node", "source": "jev", "round": 0, "result": "SUFFICIENT", "noul_prob": 0.88, "latency_s": 0.2},
                        ],
                    },
                },
                {
                    "sample_id": "ag2",
                    "query": "Học phí ngành AI là bao nhiêu?",
                    "type": "factoid",
                    "unit": "UET",
                    "category": "HOC_PHI",
                    "ground_truth": "Học phí 35 triệu/năm.",
                    "is_out_of_domain": False,
                    "retrieved_contexts": [],
                    "generated_answer": "Nội dung bạn hỏi hiện nằm ngoài phạm vi hỗ trợ của tôi.",
                    "model_name": "stub-gen",
                    "total_latency_s": 0.2,
                    "agent": {
                        "route": "OUT_OF_SCOPE",
                        "status": "out_of_scope",
                        "retry": 0,
                        "trace": [
                            {"node": "router_node", "source": "jev", "result": "OUT_OF_SCOPE", "latency_s": 0.2, "parse_fallback": False},
                        ],
                    },
                },
            ],
        }
        report = score_predictions(agentic_snapshot, enable_semantic=False)
        self.assertIn("agent_metrics", report)
        self.assertEqual(report["error_taxonomy"]["ROUTER_FALSE_OOS"], 1)
        self.assertEqual(report["agent_metrics"]["router"]["false_oos_count"], 1)
        self.assertEqual(report["agent_metrics"]["answer_eval"]["evaluated_samples"], 1)
        self.assertIn("ROUTER_FALSE_OOS", report["samples"][1]["agent_flags"])

        buf = io.StringIO()
        with redirect_stdout(buf):
            print_e2e_diagnostic_report(report)
        printed = buf.getvalue()
        self.assertIn("ĐÁNH GIÁ QUYẾT ĐỊNH AGENT", printed)
        self.assertIn("Quét ngưỡng noul_prob", printed)

    def test_run_router_eval_cli_with_fake_client(self):
        golden_path = BASE_DIR / "data" / "golden_questions.json"
        if not golden_path.exists():
            self.skipTest("Không tìm thấy data/golden_questions.json")

        fake_client = FakeDecisionClient(
            routes=[
                Route.OUT_OF_SCOPE,
                Route.OUT_OF_SCOPE,
                Route.OUT_OF_SCOPE,
                Route.OUT_OF_SCOPE,
                Route.OUT_OF_SCOPE,
                Route.QUALIFIED_QUESTION,
            ],
            default_route=Route.QUALIFIED_QUESTION,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "router_report.json"
            buf = io.StringIO()
            with redirect_stdout(buf):
                report = run_router_eval_cli(
                    dataset_path=golden_path,
                    output_path=out_path,
                    decision_client=fake_client,
                    num_samples=6,
                )
            self.assertIsNotNone(report)
            assert report is not None
            self.assertTrue(out_path.exists())
            self.assertEqual(report["total_samples"], 6)
            self.assertIn("confusion_matrix", report["router_metrics"])


class _MultiRoundStubRetriever:
    default_top_k = 4

    def __init__(self) -> None:
        self.calls = 0

    def retrieve(self, query: UserQuery, top_k: int = 4, filters: Optional[Dict[str, Any]] = None) -> List[RetrievedContext]:
        self.calls += 1
        if self.calls == 1:
            return [
                RetrievedContext(
                    chunk_id="c_noise",
                    text="Thông tin chung về Trường Đại học Công nghệ.",
                    similarity_score=0.60,
                    rank=1,
                    metadata={"document_id": "d_0", "title": "Giới thiệu"},
                )
            ]
        return [
            RetrievedContext(
                chunk_id="c_gold",
                text="Điều kiện tốt nghiệp sớm là GPA >= 3.0 và tích lũy tối thiểu 90 tín chỉ.",
                similarity_score=0.95,
                rank=1,
                metadata={"document_id": "d_1", "title": "Quy chế đào tạo"},
            )
        ]


class _MultiRoundStubLLM:
    model_name = "stub-multi"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, augmented_prompt: Any, model: Optional[str] = None) -> Response:
        self.calls += 1
        qid = getattr(getattr(augmented_prompt, "user_query", None), "query_id", "q_mr")
        if self.calls == 1:
            ans = "Sinh viên cần đáp ứng một số điều kiện theo quy chế nhà trường."
        else:
            ans = "Điều kiện tốt nghiệp sớm là GPA >= 3.0 và tích lũy tối thiểu 90 tín chỉ."
        return Response(
            query_id=qid,
            answer=ans,
            sources=augmented_prompt.contexts,
            model_name=self.model_name,
            latency_seconds=0.05,
        )


class _MultiRoundAgenticPipeline(BaseRAGPipeline):
    def __init__(self) -> None:
        decision = FakeDecisionClient(
            routes=[Route.QUALIFIED_QUESTION],
            evaluate_answers=[Verdict.INSUFFICIENT, Verdict.SUFFICIENT],
        )
        self.agent_graph = UETAgentGraph(
            retriever=_MultiRoundStubRetriever(),
            llm_client=_MultiRoundStubLLM(),
            decision_client=decision,
            config={
                "agent": {
                    "answer_eval": {"max_retries": 2},
                    "followup": {"enable_llm": False},
                    "trace": {"verbose": False},
                }
            },
        )

    def query(self, query: UserQuery, top_k: Optional[int] = None) -> Response:
        state = self.agent_graph.run(query=query.query_text, user_query=query, top_k=top_k)
        agent_resp = state.get("response")
        route_val = agent_resp.route.value if (agent_resp and agent_resp.route) else "QUALIFIED_QUESTION"
        return Response(
            query_id=query.query_id,
            answer=state.get("answer", ""),
            sources=state.get("contexts") or [],
            model_name="stub-multi",
            latency_seconds=0.25,
            metadata={
                "status": agent_resp.status if agent_resp else "ok",
                "route": route_val,
                "retry": state.get("retry", 0),
                "fail_reason": state.get("fail_reason"),
                "missing_info": state.get("missing_info"),
                "used_queries": state.get("used_queries", []),
                "trace": state.get("trace", []),
            },
        )


class TestAgentLoopEvaluationPhaseB(unittest.TestCase):
    def test_strip_agent_notes(self):
        core = "Điều kiện tốt nghiệp sớm là GPA >= 3.0 và tích lũy 90 tín chỉ."
        incomplete_note = (
            "\n\n> **Lưu ý:** Câu trả lời trên có thể chưa đầy đủ hoàn toàn mọi ý trong câu hỏi của bạn"
            " (phần còn thiếu: học phí kỳ hè). "
            "Bạn có thể liên hệ Phòng Đào tạo (Phòng 105-E3, SĐT: 024.3754.7865) hoặc Cổng thông tin đào tạo UET để xác nhận thêm thông tin chi tiết nhé."
        )
        ambiguous_note = (
            "\n\n> **Gợi ý:** Nếu bạn muốn tra cứu chính xác theo trường hợp của mình, "
            "bạn có thể cho mình biết thêm **ngành học, khóa tuyển sinh hoặc hệ đào tạo (Chuẩn/CLC)** nhé."
        )
        voice_note = (
            " Nếu bạn muốn tra cứu chính xác hơn theo trường hợp của mình, "
            "bạn hãy cho mình biết thêm ngành học, khóa tuyển sinh hoặc hệ đào tạo cụ thể nhé."
        )

        self.assertEqual(strip_agent_notes(core + incomplete_note), core)
        self.assertEqual(strip_agent_notes(core + ambiguous_note), core)
        self.assertEqual(strip_agent_notes(core + incomplete_note + ambiguous_note), core)
        self.assertEqual(strip_agent_notes(core + voice_note), core)
        self.assertEqual(strip_agent_notes(core), core)

    def test_extract_rounds(self):
        trace = [
            {"node": "router_node", "result": "QUALIFIED_QUESTION"},
            {"node": "retrieval_node", "round": 0, "queries": ["Q0"], "chunk_ids": ["c1"], "selected_chunk_ids": ["c1"]},
            {"node": "generation_node", "round": 0, "answer_text": "A0"},
            {"node": "answer_eval_node", "round": 0, "result": "INSUFFICIENT", "noul_prob": 0.30},
            {"node": "followup_query_node", "round": 1, "missing_info": "thiếu tín chỉ", "followup_queries": ["Q1"]},
            {"node": "retrieval_node", "round": 1, "queries": ["Q1"], "chunk_ids": ["c2"], "selected_chunk_ids": ["c2", "c1"]},
            {"node": "generation_node", "round": 1, "answer_text": "A1"},
            {"node": "answer_eval_node", "round": 1, "result": "SUFFICIENT", "noul_prob": 0.85},
        ]
        rounds = extract_rounds(trace)
        self.assertEqual(len(rounds), 2)
        self.assertEqual(rounds[0]["round"], 0)
        self.assertEqual(rounds[0]["answer_text"], "A0")
        self.assertEqual(rounds[0]["noul_prob"], 0.30)
        self.assertEqual(rounds[1]["round"], 1)
        self.assertEqual(rounds[1]["followup_queries"], ["Q1"])
        self.assertEqual(rounds[1]["selected_chunk_ids"], ["c2", "c1"])
        self.assertEqual(rounds[1]["verdict"], "SUFFICIENT")

        # Snapshot cũ không có trường round -> suy luận tuần tự, không lỗi
        legacy_trace = [
            {"node": "retrieval_node", "chunks_retrieved": 2},
            {"node": "generation_node", "answer_chars": 50},
            {"node": "answer_eval_node", "result": "INSUFFICIENT"},
            {"node": "retrieval_node", "chunks_retrieved": 2},
            {"node": "generation_node", "answer_chars": 80},
            {"node": "answer_eval_node", "result": "SUFFICIENT"},
        ]
        legacy_rounds = extract_rounds(legacy_trace)
        self.assertEqual(len(legacy_rounds), 2)
        self.assertIsNone(
            score_sample_rounds({"is_out_of_domain": False, "agent": {"status": "ok", "trace": legacy_trace}})
        )

    def test_loop_metrics_retrieval_and_effect(self):
        gt = "Điều kiện tốt nghiệp sớm là GPA >= 3.0 và tích lũy 90 tín chỉ."
        # Mẫu 1 (ok, improved): vòng 0 thiếu chunk vàng, vòng 1 tìm thấy c_gold -> điểm tăng mạnh
        rec_improved = {
            "sample_id": "lp_1",
            "query": "Điều kiện tốt nghiệp sớm?",
            "ground_truth": gt,
            "target_chunk_ids": {"c_gold": 2},
            "is_out_of_domain": False,
            "agent": {
                "route": "QUALIFIED_QUESTION",
                "status": "ok",
                "retry": 1,
                "trace": [
                    {"node": "retrieval_node", "round": 0, "chunk_ids": ["c_x"], "selected_chunk_ids": ["c_x"]},
                    {"node": "generation_node", "round": 0, "answer_text": "Sinh viên cần đạt yêu cầu học vụ."},
                    {"node": "answer_eval_node", "round": 0, "result": "INSUFFICIENT", "noul_prob": 0.25},
                    {"node": "retrieval_node", "round": 1, "chunk_ids": ["c_gold"], "selected_chunk_ids": ["c_gold", "c_x"]},
                    {"node": "generation_node", "round": 1, "answer_text": gt},
                    {"node": "answer_eval_node", "round": 1, "result": "SUFFICIENT", "noul_prob": 0.90},
                ],
            },
        }
        # Mẫu 2 (partial, harmed + gold_evicted + chọn sai best_answer):
        # vòng 0 tìm được c_gold và trả đúng, vòng 1 bị đẩy c_gold khỏi selected_chunk_ids và trả tệ nhưng noul_prob cao hơn
        rec_harmed = {
            "sample_id": "lp_2",
            "query": "Điều kiện tốt nghiệp sớm?",
            "ground_truth": gt,
            "target_chunk_ids": {"c_gold": 2},
            "is_out_of_domain": False,
            "agent": {
                "route": "QUALIFIED_QUESTION",
                "status": "partial",
                "retry": 2,
                "trace": [
                    {"node": "retrieval_node", "round": 0, "chunk_ids": ["c_gold"], "selected_chunk_ids": ["c_gold"]},
                    {"node": "generation_node", "round": 0, "answer_text": gt},
                    {"node": "answer_eval_node", "round": 0, "result": "INSUFFICIENT", "noul_prob": 0.30},
                    {"node": "retrieval_node", "round": 1, "chunk_ids": ["c_noise"], "selected_chunk_ids": ["c_noise"]},
                    {"node": "generation_node", "round": 1, "answer_text": "Chưa rõ quy định chi tiết."},
                    {"node": "answer_eval_node", "round": 1, "result": "INSUFFICIENT", "noul_prob": 0.45},
                ],
            },
        }

        r1 = score_sample_rounds(rec_improved)
        r2 = score_sample_rounds(rec_harmed)
        assert r1 is not None and r2 is not None
        rec_improved["agent_rounds_eval"] = r1
        rec_harmed["agent_rounds_eval"] = r2

        self.assertEqual(r1["outcome"], "improved")
        self.assertEqual(r1["gold_recall_round0"], 0.0)
        self.assertEqual(r1["gold_recall_final"], 1.0)
        self.assertFalse(r1["gold_evicted"])

        self.assertEqual(r2["outcome"], "harmed")
        self.assertTrue(r2["gold_evicted"])
        self.assertFalse(r2["partial_best_selected"])

        flags2 = compute_sample_agent_flags(rec_harmed)
        self.assertIn("LOOP_HARMED", flags2)
        self.assertIn("GOLD_EVICTED", flags2)

        loop_m = compute_loop_metrics([rec_improved, rec_harmed])
        assert loop_m is not None
        self.assertEqual(loop_m["samples_with_retry"], 2)
        self.assertEqual(loop_m["samples_with_gold"], 2)
        self.assertEqual(loop_m["gold_recall_round0"], 0.5)
        self.assertEqual(loop_m["gold_recall_final"], 0.5)
        self.assertEqual(loop_m["gold_evicted_rate"], 0.5)
        self.assertEqual(loop_m["win_rate"], 0.5)
        self.assertEqual(loop_m["harm_rate"], 0.5)
        self.assertIn("r0", loop_m["correctness_by_round"])
        self.assertIn("r1", loop_m["correctness_by_round"])
        self.assertEqual(loop_m["partial_samples"], 1)
        self.assertEqual(loop_m["partial_best_answer_accuracy"], 0.0)

    def test_judge_rounds_uses_cache_suffix(self):
        seen_ids: List[str] = []

        def fake_judge_call(prompt: str) -> str:
            # Câu vòng 0 ngắn -> chấm 2/5, câu cuối đầy đủ -> chấm 5/5
            corr = 2 if "một số điều kiện" in prompt else 5
            return json.dumps({
                "faithfulness": {"score": 5, "unsupported_claims": [], "reason": "ok"},
                "answer_correctness": {"score": corr, "missing_facts": [], "wrong_facts": [], "reason": "ok"},
                "answer_relevance": {"score": 5, "reason": "ok"},
                "context_sufficiency": {"score": 5, "reason": "ok"},
            })

        with tempfile.TemporaryDirectory() as tmpdir:
            judge = UETLLMJudgeEvaluator(
                model_name="mock-judge",
                delay_seconds=0.0,
                cache_file=Path(tmpdir) / "judge_rounds_cache.json",
                call_fn=fake_judge_call,
            )
            orig_judge_record = judge.judge_record

            def spy_judge_record(*args: Any, **kwargs: Any) -> Dict[str, Any]:
                seen_ids.append(str(kwargs.get("sample_id") or ""))
                return orig_judge_record(*args, **kwargs)

            judge.judge_record = spy_judge_record  # type: ignore[method-assign]

            snapshot = {
                "top_k": 4,
                "predictions": [
                    {
                        "sample_id": "jr_01",
                        "query": "Điều kiện tốt nghiệp sớm?",
                        "ground_truth": "GPA >= 3.0 và tích lũy 90 tín chỉ.",
                        "is_out_of_domain": False,
                        "retrieved_contexts": [
                            {"chunk_id": "c_gold", "document_id": "d_1", "rank": 1, "score": 0.9, "text": "GPA >= 3.0 và 90 tín chỉ."}
                        ],
                        "generated_answer": "GPA >= 3.0 và tích lũy 90 tín chỉ.",
                        "agent": {
                            "route": "QUALIFIED_QUESTION",
                            "status": "ok",
                            "retry": 1,
                            "trace": [
                                {"node": "retrieval_node", "round": 0, "chunk_ids": ["c_x"], "selected_chunk_ids": ["c_x"]},
                                {"node": "generation_node", "round": 0, "answer_text": "Sinh viên cần một số điều kiện."},
                                {"node": "answer_eval_node", "round": 0, "result": "INSUFFICIENT", "noul_prob": 0.25},
                                {"node": "retrieval_node", "round": 1, "chunk_ids": ["c_gold"], "selected_chunk_ids": ["c_gold"]},
                                {"node": "generation_node", "round": 1, "answer_text": "GPA >= 3.0 và tích lũy 90 tín chỉ."},
                                {"node": "answer_eval_node", "round": 1, "result": "SUFFICIENT", "noul_prob": 0.92},
                            ],
                        },
                    }
                ],
            }

            report = score_predictions(
                snapshot,
                use_judge=True,
                judge=judge,
                enable_semantic=False,
                judge_rounds=True,
            )
            self.assertEqual(seen_ids, ["jr_01", "jr_01::r0"])
            reval = report["samples"][0]["agent_rounds_eval"]
            self.assertEqual(reval["score_source"], "judge")
            self.assertEqual(reval["delta"], 3.0)
            self.assertEqual(reval["outcome"], "improved")
            self.assertEqual(report["agent_metrics"]["loop"]["score_source"], "judge")

    def test_end_to_end_multi_round(self):
        pipeline = _MultiRoundAgenticPipeline()
        dataset = [
            EvaluationSample(
                id="mr_01",
                query="Điều kiện tốt nghiệp sớm?",
                ground_truth="Điều kiện tốt nghiệp sớm là GPA >= 3.0 và tích lũy tối thiểu 90 tín chỉ.",
                metadata={
                    "type": "factoid",
                    "unit": "UET",
                    "target_document_ids": ["d_1"],
                    "target_chunk_ids": {"c_gold": 2},
                },
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            pred_path = Path(tmpdir) / "mr_preds.json"
            snapshot = generate_predictions(
                pipeline=pipeline,
                dataset=dataset,
                top_k=4,
                delay_between_queries=0.0,
                output_path=pred_path,
            )
            report = score_predictions(snapshot, enable_semantic=False)
            self.assertIn("agent_metrics", report)
            self.assertIn("loop", report["agent_metrics"])
            lp = report["agent_metrics"]["loop"]
            self.assertEqual(lp["samples_with_retry"], 1)
            self.assertEqual(lp["gold_recall_round0"], 0.0)
            self.assertEqual(lp["gold_recall_final"], 1.0)
            self.assertEqual(lp["gold_evicted_rate"], 0.0)
            self.assertEqual(lp["win_rate"], 1.0)

            buf = io.StringIO()
            with redirect_stdout(buf):
                print_e2e_diagnostic_report(report)
            self.assertIn("Vòng lặp Follow-up", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
