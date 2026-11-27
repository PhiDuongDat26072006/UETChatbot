"""
src/agent/graph.py - Xây dựng đồ thị trạng thái điều phối Agentic RAG v3 ("Generate -> Evaluate -> Follow-up").

Helper functions and SSE update streaming live in ``src/agent/helpers.py`` and ``src/agent/streaming.py``.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Generator, List, Optional

from langgraph.graph import END, START, StateGraph

from src.agent.counter_helpers import route_after_answer_eval
from src.agent.followup import UETFollowupQueryGenerator
from src.agent.helpers import (
    ANCHOR_COUNT_PATTERNS,
    NODE_STATUS_MESSAGES as _NODE_STATUS_MESSAGES,
    build_augmented_prompt,
    chunk_answer_for_stream as _chunk_answer_for_stream,
    clean_query_for_retrieval,
    extract_anchor_count,
    extract_jev_trace_extras,
    merge_round_robin_results,
    resolve_agent_templates,
    select_accumulated_contexts,
)
from src.agent.rewriter import UETQueryRewriter
from src.agent.state import AgentState
from src.agent.streaming import stream_agent_graph
from src.base import AugmentedPrompt, RetrievedContext, UserQuery
from src.decision import (
    AgentResponse,
    DecisionClient,
    JevDecisionClient,
    Route,
    Verdict,
)
from src.llm.llm_client import LlmUnavailableError
from src.utils.helpers import get_logger, load_yaml_config

logger = get_logger("agent_graph")


class UETAgentGraph:
    """
    Bộ điều phối Đồ thị trạng thái Agentic RAG v3 (6 node chính + fail_node ẩn).
    """

    @staticmethod
    def extract_anchor_count(text: str, entity: str = "ngành") -> Optional[int]:
        """Trích xuất con số tổng cộng được nêu trực tiếp trong tài liệu."""
        return extract_anchor_count(text, entity=entity)

    def __init__(
        self,
        retriever=None,
        llm_client=None,
        prompt_augmenter=None,
        decision_client: Optional[DecisionClient] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.prompt_augmenter = prompt_augmenter
        self.raw_config = config or load_yaml_config()
        agent_cfg = self.raw_config.get("agent", self.raw_config)
        default_model = agent_cfg.get("default_model")

        jev_cfg = self.raw_config.get("jev", {})
        if decision_client is not None:
            self.decision_client = decision_client
        else:
            self.decision_client = JevDecisionClient(
                endpoint=jev_cfg.get("endpoint"),
                api_key=jev_cfg.get("api_key"),
                model=jev_cfg.get("model", "strands-decider-12B-gemma4-v1-2610"),
                timeout_s=float(jev_cfg.get("timeout_s", 30.0)),
                max_retries=int(jev_cfg.get("max_retries", 1)),
                max_concurrency=int(jev_cfg.get("max_concurrency", 1)),
            )

        self.max_contexts = int(agent_cfg.get("contexts", {}).get("max_chunks", 8))
        self.max_retries = int(agent_cfg.get("answer_eval", {}).get("max_retries", 3))
        self.recursion_limit = int(agent_cfg.get("recursion_limit", 100))

        followup_cfg = agent_cfg.get("followup", {})
        self.followup_generator = UETFollowupQueryGenerator(
            llm_client=llm_client,
            enable_llm=bool(followup_cfg.get("enable_llm", True)),
            model=followup_cfg.get("model") or default_model,
            max_queries=int(followup_cfg.get("max_queries", 2)),
        )

        self.gen_model = agent_cfg.get("generator", {}).get("model") or default_model
        self.trace_verbose = bool(agent_cfg.get("trace", {}).get("verbose", False))
        self.templates = resolve_agent_templates(self.raw_config)

        self.workflow = self._build_graph()
        self.app = self.workflow.compile()

    # -------------------------------------------------------------------------
    # BACKWARD-COMPATIBLE HELPERS
    # -------------------------------------------------------------------------

    def _select_accumulated_contexts(
        self,
        context_pool: Dict[str, RetrievedContext],
        new_chunk_ids: List[str],
    ) -> List[RetrievedContext]:
        return select_accumulated_contexts(context_pool, new_chunk_ids, max_contexts=self.max_contexts)

    @staticmethod
    def _clean_query_for_retrieval(q_text: str) -> str:
        return clean_query_for_retrieval(q_text)

    def _get_jev_trace_extras(self) -> Dict[str, Any]:
        return extract_jev_trace_extras(self.decision_client)

    def _build_augmented_prompt(
        self,
        query: str,
        contexts: List[RetrievedContext],
        session_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        extra_format_rules: Optional[str] = None,
    ) -> AugmentedPrompt:
        return build_augmented_prompt(
            self.prompt_augmenter,
            query=query,
            contexts=contexts,
            session_id=session_id,
            filters=filters,
            extra_format_rules=extra_format_rules,
        )

    # -------------------------------------------------------------------------
    # 6 NODE CHÍNH + fail_node
    # -------------------------------------------------------------------------

    def router_node(self, state: AgentState) -> Dict[str, Any]:
        """1. router_node (JEV): Phân loại câu hỏi thành 2 nhãn OUT_OF_SCOPE | QUALIFIED_QUESTION."""
        q = (state.get("query") or "").strip()
        session_id = state.get("session_id")
        start_t = time.time()
        try:
            try:
                route = self.decision_client.route(q, session_id=session_id)
            except TypeError:
                route = self.decision_client.route(q)
            latency = round(time.time() - start_t, 3)

            trace_entry = {
                "node": "router_node",
                "decision": "route",
                "result": route.value,
                "source": "jev",
                "latency_s": latency,
                **self._get_jev_trace_extras(),
            }
            logger.info(f"[RouterNode] Query: '{q[:50]}' -> Route: {route.value} ({latency}s)")
            return {
                "route": route,
                "trace": (state.get("trace") or []) + [trace_entry],
                "execution_steps": (state.get("execution_steps") or []) + [f"router: {route.value}"],
            }
        except Exception as exc:
            latency = round(time.time() - start_t, 3)
            logger.error(f"[RouterNode] JEV lỗi ({exc}). Chuyển sang fail_node.")
            trace_entry = {
                "node": "router_node",
                "decision": "route",
                "error": str(exc),
                "source": "jev",
                "fail_reason": "jev_unavailable",
                "latency_s": latency,
            }
            return {
                "fail_reason": "jev_unavailable",
                "trace": (state.get("trace") or []) + [trace_entry],
                "execution_steps": (state.get("execution_steps") or []) + ["router: ERROR -> fail_node"],
            }

    def out_of_scope_node(self, state: AgentState) -> Dict[str, Any]:
        """2. out_of_scope_node (Template): Trả lời mẫu thân thiện (state=0, 0 token LLM)."""
        ans = self.templates["out_of_scope"]
        resp = AgentResponse(
            status="out_of_scope",
            state=0,
            answer=ans,
            contexts=None,
            route=state.get("route", Route.OUT_OF_SCOPE),
            counters={"retry": state.get("retry", 0)},
            trace=state.get("trace", []),
            fail_reason=None,
        )
        return {
            "answer": ans,
            "response": resp,
            "contexts": None,
            "sources": [],
            "execution_steps": (state.get("execution_steps") or []) + ["out_of_scope"],
        }

    def retrieval_node(self, state: AgentState) -> Dict[str, Any]:
        """3. retrieval_node: Truy xuất hybrid và tích lũy ngữ cảnh vào context_pool."""
        queries = list(state["followup_queries"]) if state.get("followup_queries") else [state.get("query", "")]

        start_t = time.time()
        default_k = getattr(self.retriever, "default_top_k", 6) if self.retriever is not None else 6
        top_k = int(state.get("top_k") or default_k)
        per_query_results: List[List[RetrievedContext]] = []
        if self.retriever is not None:
            for q_text in queries:
                search_q = self._clean_query_for_retrieval(q_text)
                detected_facs = UETQueryRewriter.detect_faculties(search_q)
                sub_filters = {"faculty": detected_facs} if detected_facs else None
                contexts = self.retriever.retrieve(
                    query=UserQuery(query_text=search_q, filters=sub_filters),
                    top_k=top_k,
                    filters=sub_filters,
                )
                per_query_results.append(list(contexts))

        retrieved_this_round = merge_round_robin_results(per_query_results)

        context_pool: Dict[str, RetrievedContext] = dict(state.get("context_pool") or {})
        new_chunk_ids: List[str] = []
        for chunk in retrieved_this_round:
            new_chunk_ids.append(chunk.chunk_id)
            existing = context_pool.get(chunk.chunk_id)
            if existing is None or getattr(chunk, "similarity_score", 0.0) >= getattr(existing, "similarity_score", 0.0):
                context_pool[chunk.chunk_id] = chunk

        selected_contexts = self._select_accumulated_contexts(context_pool, new_chunk_ids)

        used_queries = list(state.get("used_queries") or [])
        for q_text in queries:
            if q_text not in used_queries:
                used_queries.append(q_text)

        latency = round(time.time() - start_t, 3)
        trace_entry = {
            "node": "retrieval_node",
            "source": "retriever",
            "round": int(state.get("retry", 0)),
            "queries": list(queries),
            "queries_count": len(queries),
            "chunks_retrieved": len(retrieved_this_round),
            "chunk_ids": [c.chunk_id for c in retrieved_this_round],
            "selected_count": len(selected_contexts),
            "selected_chunk_ids": [c.chunk_id for c in selected_contexts],
            "latency_s": latency,
        }

        logger.info(
            f"[RetrievalNode] Truy xuất {len(retrieved_this_round)} chunks mới | "
            f"Pool tích lũy: {len(context_pool)} | Chọn cho LLM: {len(selected_contexts)} | "
            f"used_queries: {len(used_queries)} ({latency}s)"
        )

        return {
            "retrieved_chunks": retrieved_this_round,
            "context_pool": context_pool,
            "contexts": selected_contexts,
            "used_queries": used_queries,
            "trace": (state.get("trace") or []) + [trace_entry],
            "execution_steps": (state.get("execution_steps") or []) + [f"retrieval: {len(retrieved_this_round)} chunks"],
        }

    def generation_node(self, state: AgentState) -> Dict[str, Any]:
        """4. generation_node (LLM): Sinh câu trả lời từ câu hỏi gốc + ngữ cảnh tích lũy."""
        q = state.get("query", "")
        contexts = state.get("contexts") or []
        user_query = state.get("user_query")
        filters = getattr(user_query, "filters", None) if user_query is not None else None

        start_t = time.time()
        try:
            usage_info = None
            if self.llm_client is None:
                context_summary = "\n\n".join(f"- {c.text}" for c in contexts[:3])
                answer = f"Dưới đây là thông tin trích xuất từ quy chế đào tạo UET:\n\n{context_summary}"
            else:
                augmented = self._build_augmented_prompt(
                    query=q,
                    contexts=contexts,
                    session_id=state.get("session_id"),
                    filters=filters,
                    extra_format_rules=state.get("extra_format_rules"),
                )
                try:
                    response = self.llm_client.generate(augmented, model=self.gen_model)
                except TypeError:
                    response = self.llm_client.generate(augmented)

                if getattr(response, "model_name", "") == "multi-provider-fallback":
                    raise LlmUnavailableError("Tất cả LLM providers đều từ chối dịch vụ.")
                answer = response.answer
                if isinstance(getattr(response, "metadata", None), dict):
                    usage_info = response.metadata.get("usage")

            latency = round(time.time() - start_t, 3)
            trace_entry: Dict[str, Any] = {
                "node": "generation_node",
                "source": "llm",
                "round": int(state.get("retry", 0)),
                "contexts_count": len(contexts),
                "answer_chars": len(answer),
                "latency_s": latency,
            }
            if self.trace_verbose:
                trace_entry["answer_text"] = answer
            if usage_info is not None:
                trace_entry["usage"] = usage_info

            logger.info(f"[GenerationNode] Đã sinh câu trả lời ({len(answer)} ký tự, {latency}s).")
            return {
                "answer": answer,
                "contexts": contexts,
                "sources": contexts,
                "trace": (state.get("trace") or []) + [trace_entry],
                "execution_steps": (state.get("execution_steps") or []) + ["generation_llm"],
            }
        except Exception as err:
            latency = round(time.time() - start_t, 3)
            logger.error(f"[GenerationNode] Lỗi sinh câu trả lời LLM: {err}. Chuyển sang fail_node (D12).")
            trace_entry = {
                "node": "generation_node",
                "decision": "generate",
                "error": str(err),
                "source": "llm",
                "fail_reason": "llm_unavailable",
                "latency_s": latency,
            }
            return {
                "fail_reason": "llm_unavailable",
                "trace": (state.get("trace") or []) + [trace_entry],
                "execution_steps": (state.get("execution_steps") or []) + ["generation: ERROR -> fail_node"],
            }

    def answer_eval_node(self, state: AgentState) -> Dict[str, Any]:
        """5. answer_eval_node (JEV): Đánh giá cặp (query, answer) — KHÔNG gửi chunk."""
        q = state.get("query", "")
        ans = state.get("answer", "")
        retry = state.get("retry", 0)
        route = state.get("route")

        start_t = time.time()
        session_id = state.get("session_id")
        try:
            try:
                verdict = self.decision_client.evaluate_answer(q, ans, session_id=session_id)
            except TypeError:
                verdict = self.decision_client.evaluate_answer(q, ans)
            latency = round(time.time() - start_t, 3)
        except Exception as exc:
            latency = round(time.time() - start_t, 3)
            logger.error(f"[AnswerEvalNode] JEV lỗi ({exc}). Chuyển sang fail_node.")
            trace_entry = {
                "node": "answer_eval_node",
                "decision": "evaluate_answer",
                "error": str(exc),
                "source": "jev",
                "fail_reason": "jev_unavailable",
                "latency_s": latency,
            }
            return {
                "fail_reason": "jev_unavailable",
                "trace": (state.get("trace") or []) + [trace_entry],
                "execution_steps": (state.get("execution_steps") or []) + ["answer_eval: ERROR -> fail_node"],
            }

        jev_extras = self._get_jev_trace_extras()
        trace_entry: Dict[str, Any] = {
            "node": "answer_eval_node",
            "decision": "evaluate_answer",
            "round": int(retry),
            "result": verdict.value,
            "retry_before": retry,
            "source": "jev",
            "latency_s": latency,
            **jev_extras,
        }

        if verdict == Verdict.SUFFICIENT:
            new_trace = (state.get("trace") or []) + [trace_entry]
            logger.info(f"[AnswerEvalNode] JEV phán quyết: ĐỦ ({latency}s) -> Hoàn tất RAG (state=1, status='ok').")
            resp = AgentResponse(
                status="ok",
                state=1,
                answer=ans,
                contexts=state.get("contexts"),
                route=route,
                counters={"retry": retry},
                trace=new_trace,
                fail_reason=None,
                missing_info=None,
            )
            return {
                "answer": ans,
                "answer_verdict": Verdict.SUFFICIENT,
                "response": resp,
                "trace": new_trace,
                "execution_steps": (state.get("execution_steps") or []) + ["answer_eval: SUFFICIENT"],
            }

        raw_prob = jev_extras.get("noul_prob")
        current_score = float(raw_prob) if raw_prob is not None else -1.0

        prev_best_answer = state.get("best_answer")
        prev_best_score = state.get("best_answer_score")
        prev_best_contexts = state.get("best_contexts") or []
        current_contexts = state.get("contexts") or []

        if prev_best_answer is None or prev_best_score is None or current_score >= float(prev_best_score):
            best_answer = ans
            best_answer_score = current_score
            best_contexts = list(current_contexts)
        else:
            best_answer = prev_best_answer
            best_answer_score = float(prev_best_score)
            best_contexts = list(prev_best_contexts)

        retry += 1
        trace_entry["retry_after"] = retry
        new_trace = (state.get("trace") or []) + [trace_entry]

        if retry <= self.max_retries:
            logger.info(
                f"[AnswerEvalNode] JEV phán quyết: THIẾU ({latency}s) -> retry={retry} <= {self.max_retries} -> followup_query_node"
            )
            return {
                "answer_verdict": Verdict.INSUFFICIENT,
                "retry": retry,
                "best_answer": best_answer,
                "best_answer_score": best_answer_score,
                "best_contexts": best_contexts,
                "trace": new_trace,
                "execution_steps": (state.get("execution_steps") or []) + [f"answer_eval: INSUFFICIENT ({retry})"],
            }

        chosen_answer = best_answer or ans
        chosen_contexts = best_contexts if best_contexts else list(current_contexts)
        missing_info = state.get("missing_info")
        missing_suffix = f" (phần còn thiếu: {missing_info})" if missing_info else ""
        note_template = self.templates["answer_incomplete_note"]
        try:
            note = note_template.format(missing_suffix=missing_suffix)
        except Exception:
            note = note_template

        final_answer = f"{chosen_answer}{note}" if note and note.strip() not in chosen_answer else chosen_answer
        if best_answer_score >= 0.0:
            trace_entry["selected_best_score"] = best_answer_score
        logger.info(
            f"[AnswerEvalNode] JEV phán quyết: THIẾU ({latency}s) -> retry={retry} > {self.max_retries} -> "
            f"trả RESPONSE partial (state=1, best_score={best_answer_score})."
        )
        resp = AgentResponse(
            status="partial",
            state=1,
            answer=final_answer,
            contexts=chosen_contexts,
            route=route,
            counters={"retry": retry},
            trace=new_trace,
            fail_reason=None,
            missing_info=missing_info,
        )
        return {
            "answer_verdict": Verdict.INSUFFICIENT,
            "retry": retry,
            "answer": final_answer,
            "best_answer": best_answer,
            "best_answer_score": best_answer_score,
            "best_contexts": best_contexts,
            "contexts": chosen_contexts,
            "sources": chosen_contexts,
            "response": resp,
            "trace": new_trace,
            "execution_steps": (state.get("execution_steps") or []) + [f"answer_eval: INSUFFICIENT ({retry}) -> partial"],
        }

    def followup_query_node(self, state: AgentState) -> Dict[str, Any]:
        """6. followup_query_node (LLM): Sinh 1-2 truy vấn tìm kiếm mới cho phần thông tin còn thiếu."""
        q = state.get("query", "")
        ans = state.get("answer", "")
        used_queries = list(state.get("used_queries") or [])

        start_t = time.time()
        missing_info, followup_queries, source = self.followup_generator.generate(
            query=q,
            answer=ans,
            used_queries=used_queries,
            session_id=state.get("session_id"),
        )
        latency = round(time.time() - start_t, 3)

        trace_entry = {
            "node": "followup_query_node",
            "round": int(state.get("retry", 0)),
            "source": source,
            "missing_info": missing_info,
            "followup_queries": followup_queries,
            "latency_s": latency,
        }
        logger.info(f"[FollowupQueryNode] source={source} | missing='{missing_info}' | queries={followup_queries}")

        return {
            "followup_queries": followup_queries,
            "missing_info": missing_info,
            "trace": (state.get("trace") or []) + [trace_entry],
            "execution_steps": (state.get("execution_steps") or []) + [f"followup_query: {len(followup_queries)} queries"],
        }

    def fail_node(self, state: AgentState) -> Dict[str, Any]:
        """7. fail_node (đường lỗi ẩn - Q5): Chỉ vào khi JEV hoặc LLM sập dịch vụ."""
        fail_reason = state.get("fail_reason") or "service_unavailable"
        ans = self.templates["fail_service"]

        resp = AgentResponse(
            status="fail",
            state=0,
            answer=ans,
            contexts=None,
            route=state.get("route"),
            counters={"retry": state.get("retry", 0)},
            trace=state.get("trace", []),
            fail_reason=fail_reason,
        )
        return {
            "answer": ans,
            "response": resp,
            "fail_reason": fail_reason,
            "contexts": None,
            "sources": [],
            "execution_steps": (state.get("execution_steps") or []) + ["fail_node"],
        }

    # -------------------------------------------------------------------------
    # CONDITIONAL EDGES & GRAPH COMPILATION
    # -------------------------------------------------------------------------

    def _route_decision(self, state: AgentState) -> str:
        if state.get("fail_reason"):
            return "fail_node"
        if state.get("route", Route.QUALIFIED_QUESTION) == Route.OUT_OF_SCOPE:
            return "out_of_scope_node"
        return "retrieval_node"

    def _generation_decision(self, state: AgentState) -> str:
        if state.get("fail_reason"):
            return "fail_node"
        return "answer_eval_node"

    def _answer_eval_decision(self, state: AgentState) -> str:
        if state.get("fail_reason"):
            return "fail_node"
        return route_after_answer_eval(state, max_retries=self.max_retries)

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("router_node", self.router_node)
        workflow.add_node("out_of_scope_node", self.out_of_scope_node)
        workflow.add_node("retrieval_node", self.retrieval_node)
        workflow.add_node("generation_node", self.generation_node)
        workflow.add_node("answer_eval_node", self.answer_eval_node)
        workflow.add_node("followup_query_node", self.followup_query_node)
        workflow.add_node("fail_node", self.fail_node)

        workflow.add_edge(START, "router_node")
        workflow.add_conditional_edges(
            "router_node",
            self._route_decision,
            {
                "out_of_scope_node": "out_of_scope_node",
                "retrieval_node": "retrieval_node",
                "fail_node": "fail_node",
            },
        )
        workflow.add_edge("out_of_scope_node", END)
        workflow.add_edge("retrieval_node", "generation_node")
        workflow.add_conditional_edges(
            "generation_node",
            self._generation_decision,
            {"answer_eval_node": "answer_eval_node", "fail_node": "fail_node"},
        )
        workflow.add_conditional_edges(
            "answer_eval_node",
            self._answer_eval_decision,
            {"END": END, "followup_query_node": "followup_query_node", "fail_node": "fail_node"},
        )
        workflow.add_edge("followup_query_node", "retrieval_node")
        workflow.add_edge("fail_node", END)
        return workflow

    # -------------------------------------------------------------------------
    # EXECUTION ENTRYPOINTS
    # -------------------------------------------------------------------------

    def _build_initial_state(
        self,
        query: str,
        user_query: Optional[UserQuery] = None,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        extra_format_rules: Optional[str] = None,
        voice_mode: bool = False,
    ) -> AgentState:
        effective_session_id = session_id or (user_query.session_id if user_query else None) or str(uuid.uuid4())
        if user_query and user_query.session_id is None:
            user_query.session_id = effective_session_id

        return {
            "query": query,
            "session_id": effective_session_id,
            "user_query": user_query,
            "top_k": top_k,
            "extra_format_rules": extra_format_rules,
            "voice_mode": bool(voice_mode),
            "route": None,
            "followup_queries": [],
            "used_queries": [],
            "retrieved_chunks": [],
            "context_pool": {},
            "contexts": [],
            "answer_verdict": None,
            "missing_info": None,
            "retry": 0,
            "answer": None,
            "best_answer": None,
            "best_answer_score": None,
            "best_contexts": [],
            "response": None,
            "trace": [],
            "execution_steps": ["start"],
            "sources": [],
            "fail_reason": None,
        }

    def run(
        self,
        query: str,
        user_query: Optional[UserQuery] = None,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        extra_format_rules: Optional[str] = None,
        voice_mode: bool = False,
    ) -> AgentState:
        """Thực thi toàn bộ luồng đồ thị v3 cho một câu hỏi."""
        initial_state = self._build_initial_state(
            query=query,
            user_query=user_query,
            session_id=session_id,
            top_k=top_k,
            extra_format_rules=extra_format_rules,
            voice_mode=voice_mode,
        )
        return self.app.invoke(initial_state, config={"recursion_limit": self.recursion_limit})

    def run_stream(
        self,
        query: str,
        user_query: Optional[UserQuery] = None,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        extra_format_rules: Optional[str] = None,
        voice_mode: bool = False,
    ) -> Generator[Dict[str, Any], None, None]:
        """Thực thi đồ thị bằng LangGraph stream(stream_mode='updates')."""
        initial_state = self._build_initial_state(
            query=query,
            user_query=user_query,
            session_id=session_id,
            top_k=top_k,
            extra_format_rules=extra_format_rules,
            voice_mode=voice_mode,
        )
        yield from stream_agent_graph(
            self.app,
            initial_state,
            recursion_limit=self.recursion_limit,
            llm_client=self.llm_client,
        )

    def close_session(self, session_id: Optional[str]) -> None:
        """Đóng kết nối HTTP Session của cả JEV và LLM cho session_id."""
        if not session_id:
            return
        if hasattr(self.decision_client, "close_session"):
            self.decision_client.close_session(session_id)
        if self.llm_client is not None and hasattr(self.llm_client, "close_session"):
            self.llm_client.close_session(session_id)
