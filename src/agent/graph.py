"""
src/agent/graph.py - Xây dựng đồ thị trạng thái điều phối Agentic RAG v3 ("Generate -> Evaluate -> Follow-up").
Gồm 9 node chính theo sơ đồ mục 2 của implementation_plan_v3.md + 1 node lỗi ẩn (fail_node cho lỗi dịch vụ JEV/LLM):
  1. router_node (JEV 3 nhãn: OUT_OF_SCOPE, AMBIGUOUS, QUALIFIED_QUESTION)
  2. out_of_scope_node (Template)
  3. clarify_ask_node (LLM có cấu trúc: known_info, missing_info, question)
  4. clarify_check_node (JEV: kiểm tra đủ từng mục trong missing_info)
  5. clarify_exhausted_node (Template)
  6. retrieval_node (BM25 + Vector + RRF + MMR + Rerank + Soft Faculty Filter + tích lũy context_pool)
  7. generation_node (LLM sinh câu trả lời từ ngữ cảnh tích lũy)
  8. answer_eval_node (JEV: đánh giá cặp query + answer, không gửi chunk)
  9. followup_query_node (LLM: xác định phần thiếu và sinh 1-2 truy vấn tìm kiếm mới)
 10. fail_node (đường lỗi ẩn khi JEV hoặc LLM gián đoạn dịch vụ)
"""

from __future__ import annotations
import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Set

from langgraph.graph import StateGraph, START, END

from src.base import AugmentedPrompt, RetrievedContext, UserQuery
from src.agent.state import AgentState, ClarifyData
from src.agent.clarify_store import ClarifySessionStore
from src.agent.counter_helpers import (
    route_after_clarify_check,
    route_after_answer_eval,
)
from src.agent.followup import UETFollowupQueryGenerator
from src.agent.rewriter import UETQueryRewriter
from src.decision import (
    Route,
    Verdict,
    AgentResponse,
    DecisionClient,
    JevDecisionClient,
)
from src.llm.llm_client import LlmUnavailableError
from src.prompts.prompts import (
    DEFAULT_CLARIFY_MISSING_INFO,
    DEFAULT_CLARIFY_QUESTION,
    LLM_CLARIFY_ASK_SYSTEM,
    LLM_CLARIFY_ASK_TEMPLATE,
    LLM_GENERATION_FALLBACK_SYSTEM,
    LLM_GENERATION_FALLBACK_TEMPLATE,
)
from src.utils.helpers import get_logger, load_yaml_config

logger = get_logger("agent_graph")

ANCHOR_COUNT_PATTERNS = [
    r"(?:gồm|có|đào tạo|tổng số|tổng cộng|hiện có|tất cả|quy mô)\s*(\d{1,2})\s*(?:ngành|chương trình đào tạo|chuyên ngành|khoa|viện|bộ môn|mức|tiêu chuẩn)",
    r"(\d{1,2})\s*(?:ngành|chương trình đào tạo|chuyên ngành)\s*(?:đào tạo|đại học|chính quy|chuẩn)?",
    r"(?:gồm có|bao gồm)\s*(\d{1,2})\s*(?:khoa|viện)",
]


class UETAgentGraph:
    """
    Bộ điều phối Đồ thị trạng thái Agentic RAG v3 (9 node chính + fail_node ẩn).
    Luồng chính:
      - Happy path: router (JEV) -> retrieval -> generation (LLM) -> answer_eval (JEV) -> END (2 JEV + 1 LLM).
      - Follow-up loop: answer_eval THIẾU (retry <= 3) -> followup_query (LLM) -> retrieval -> generation -> answer_eval.
      - Partial exit (Q1=A): answer_eval THIẾU (retry > 3) -> trả câu trả lời gần nhất (state=1, status='partial') kèm ghi chú phần thiếu.
      - Clarify loop: router AMBIGUOUS -> clarify_ask (LLM JSON) -> clarify_check (JEV + missing_info) -> retrieval | clarify_ask | clarify_exhausted.
    """

    @staticmethod
    def extract_anchor_count(text: str, entity: str = "ngành") -> Optional[int]:
        """Trích xuất con số tổng cộng được nêu trực tiếp trong tài liệu."""
        for pattern in ANCHOR_COUNT_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for m in matches:
                try:
                    num = int(m.group(1))
                    if 1 <= num <= 99:
                        return num
                except (ValueError, IndexError):
                    continue
        return None

    def __init__(
        self,
        retriever=None,
        llm_client=None,
        prompt_augmenter=None,
        decision_client: Optional[DecisionClient] = None,
        clarify_store: Optional[ClarifySessionStore] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.prompt_augmenter = prompt_augmenter
        self.raw_config = config or load_yaml_config()
        agent_cfg = self.raw_config.get("agent", self.raw_config)
        default_model = agent_cfg.get("default_model")

        # Cấu hình JEV Client
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

        # Cấu hình Clarify Session Store
        clarify_session_cfg = self.raw_config.get("clarify_session", {})
        self.clarify_store = clarify_store or ClarifySessionStore(
            ttl_s=float(clarify_session_cfg.get("ttl_s", 900.0))
        )

        # Cấu hình chọn ngữ cảnh tích lũy
        contexts_cfg = agent_cfg.get("contexts", {})
        self.max_contexts = int(contexts_cfg.get("max_chunks", 8))

        # Ngưỡng đếm retry từ config (không hard-code)
        self.max_retries = int(agent_cfg.get("answer_eval", {}).get("max_retries", 3))
        self.max_retried3 = int(agent_cfg.get("clarify", {}).get("max_retries", 3))
        self.recursion_limit = int(agent_cfg.get("recursion_limit", 100))

        # Follow-up Query Generator (LLM)
        followup_cfg = agent_cfg.get("followup", {})
        self.followup_generator = UETFollowupQueryGenerator(
            llm_client=llm_client,
            enable_llm=bool(followup_cfg.get("enable_llm", True)),
            model=followup_cfg.get("model") or default_model,
            max_queries=int(followup_cfg.get("max_queries", 2)),
        )

        # Generator & Clarify model
        gen_cfg = agent_cfg.get("generator", {})
        self.gen_model = gen_cfg.get("model") or default_model
        clarify_node_cfg = agent_cfg.get("clarify", {})
        self.clarify_model = clarify_node_cfg.get("model") or default_model

        # Mẫu template từ config
        templates_cfg = self.raw_config.get("templates", {})
        self.templates = {
            "out_of_scope": templates_cfg.get(
                "out_of_scope",
                "Chào bạn! Tôi là UET Chatbot - Trợ lý ảo tư vấn quy chế đào tạo và tuyển sinh của Trường Đại học Công nghệ (VNU-UET). "
                "Rất vui được phục vụ bạn! Nếu bạn cần tìm hiểu thông tin học vụ, quy chế, điểm chuẩn hay học phí, hãy chia sẻ để tôi hỗ trợ nhé!",
            ),
            "clarify_exhausted": templates_cfg.get(
                "clarify_exhausted",
                "Xin lỗi bạn, sau nhiều lần hỏi thêm thông tin, bạn vẫn chưa cung cấp đủ dữ liệu để tôi tra cứu chính xác. "
                "Bạn vui lòng đặt lại câu hỏi đầy đủ hơn (nêu rõ ngành, khóa, chương trình đào tạo) hoặc liên hệ trực tiếp Phòng Đào tạo (Phòng 105-E3) để được hướng dẫn chi tiết nhé.",
            ),
            "fail_service": templates_cfg.get(
                "fail_service",
                "Xin lỗi bạn, hệ thống kiểm tra và tra cứu thông tin đang gián đoạn dịch vụ, "
                "vui lòng thử lại sau ít phút hoặc liên hệ trực tiếp Phòng Đào tạo (Phòng 105-E3) để được hỗ trợ nhé.",
            ),
            "answer_incomplete_note": templates_cfg.get(
                "answer_incomplete_note",
                "\n\n> **Lưu ý:** Câu trả lời trên có thể chưa đầy đủ hoàn toàn mọi ý trong câu hỏi của bạn{missing_suffix}. "
                "Bạn có thể liên hệ Phòng Đào tạo (Phòng 105-E3, SĐT: 024.3754.7865) hoặc Cổng thông tin đào tạo UET để xác nhận thêm thông tin chi tiết nhé.",
            ),
        }

        # Xây dựng và biên dịch đồ thị LangGraph v3
        self.workflow = self._build_graph()
        self.app = self.workflow.compile()

    # =========================================================================
    # HÀM BỔ TRỢ CHỌN NGỮ CẢNH TÍCH LŨY (ƯU TIÊN VÒNG MỚI + ĐIỂM CAO VÒNG CŨ)
    # =========================================================================

    def _select_accumulated_contexts(
        self,
        context_pool: Dict[str, RetrievedContext],
        new_chunk_ids: List[str],
    ) -> List[RetrievedContext]:
        """
        Chọn tối đa max_contexts (mặc định 8) chunk đưa cho LLM:
        - Nếu tổng số chunk trong pool <= max_contexts: lấy toàn bộ, sắp xếp theo similarity_score giảm dần.
        - Nếu vượt quá max_contexts: ưu tiên chunk của vòng mới nhất theo thứ tự round-robin giữa các truy vấn
          (tránh trường hợp 1 truy vấn có điểm tuyệt đối cao chiếm hết slot và đẩy mất top-1 của truy vấn kia),
          đồng thời giữ các chunk điểm cao nhất của các vòng trước.
        """
        if not context_pool:
            return []

        if len(context_pool) <= self.max_contexts:
            return sorted(
                context_pool.values(),
                key=lambda c: getattr(c, "similarity_score", 0.0),
                reverse=True,
            )

        new_id_set: Set[str] = set(new_chunk_ids)
        new_chunks: List[RetrievedContext] = []
        seen_new: Set[str] = set()
        for cid in new_chunk_ids:
            if cid in context_pool and cid not in seen_new:
                seen_new.add(cid)
                new_chunks.append(context_pool[cid])

        older_chunks = sorted(
            [c for cid, c in context_pool.items() if cid not in new_id_set],
            key=lambda c: getattr(c, "similarity_score", 0.0),
            reverse=True,
        )

        # Dành tối thiểu nửa slot cho chunk cũ nếu có, phần còn lại ưu tiên tuyệt đối cho chunk vòng mới
        reserved_old = min(len(older_chunks), self.max_contexts // 2)
        take_new = min(len(new_chunks), self.max_contexts - reserved_old)
        take_old = min(len(older_chunks), self.max_contexts - take_new)

        return new_chunks[:take_new] + older_chunks[:take_old]

    @staticmethod
    def _clean_query_for_retrieval(q_text: str) -> str:
        """Gỡ khung '(Bổ sung: ...)' và mở rộng từ viết tắt/đồng nghĩa học vụ UET trước khi đưa vào BM25/Vector."""
        if not q_text:
            return q_text
        cleaned = q_text
        if "bổ sung:" in cleaned.lower():
            cleaned = re.sub(r"\s*\(\s*Bổ sung\s*:\s*([^)]*)\)", r" \1", cleaned, flags=re.IGNORECASE)
        return UETQueryRewriter.expand_for_retrieval(cleaned)

    @staticmethod
    def _normalize_str_list(val: Any) -> List[str]:
        if isinstance(val, list):
            return [str(item).strip() for item in val if str(item).strip()]
        if isinstance(val, str) and val.strip():
            return [val.strip()]
        return []

    def _get_jev_trace_extras(self) -> Dict[str, Any]:
        meta = getattr(self.decision_client, "last_trace_meta", None)
        if not isinstance(meta, dict):
            return {}
        extras = {}
        for key in ("server_latency_s", "payload_bytes", "noul_prob"):
            if key in meta and meta[key] is not None:
                extras[key] = meta[key]
        return extras

    # =========================================================================
    # 9 NODE CHÍNH + 1 NODE LỖI ẨN (fail_node)
    # =========================================================================

    def router_node(self, state: AgentState) -> Dict[str, Any]:
        """4.1 router_node (JEV): Phân loại câu hỏi thành 3 nhãn OUT_OF_SCOPE | AMBIGUOUS | QUALIFIED_QUESTION."""
        q = state.get("effective_query") or state.get("query", "").strip()
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
        except Exception as e:
            latency = round(time.time() - start_t, 3)
            logger.error(f"[RouterNode] JEV lỗi ({e}). Chuyển sang fail_node.")
            trace_entry = {
                "node": "router_node",
                "decision": "route",
                "error": str(e),
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
        """4.2 out_of_scope_node (Template): Trả lời mẫu thân thiện (state=0, 0 token LLM)."""
        ans = self.templates["out_of_scope"]
        resp = AgentResponse(
            status="out_of_scope",
            state=0,
            answer=ans,
            contexts=None,
            route=state.get("route", Route.OUT_OF_SCOPE),
            counters={
                "retry": state.get("retry", 0),
                "retried3": state.get("retried3", 0),
            },
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

    def clarify_ask_node(self, state: AgentState) -> Dict[str, Any]:
        """
        4.3 clarify_ask_node (LLM): Xác định đã có thông tin cụ thể gì (known_info),
        còn thiếu thông tin cụ thể gì (missing_info), và đặt một câu hỏi ngắn gọn cho phần thiếu.
        Đầu ra JSON: {"known_info": [...], "missing_info": [...], "question": "..."}.
        """
        q = state.get("query", "")
        clarify = dict(
            state.get("clarify")
            or {"asks": [], "supplements": [], "known_info": [], "missing_info": [], "pending": False}
        )
        asks = list(clarify.get("asks") or [])
        supplements = list(clarify.get("supplements") or [])
        prior_known = list(clarify.get("known_info") or [])
        prior_missing = list(clarify.get("missing_info") or [])

        known_info: List[str] = list(prior_known)
        missing_info: List[str] = []
        ask_question = ""
        parse_source = "fallback"

        if self.llm_client is not None:
            try:
                hist_lines = []
                if prior_known:
                    hist_lines.append("Thông tin đã biết trước đó: " + ", ".join(prior_known))
                if prior_missing:
                    hist_lines.append("Thông tin trước đó còn thiếu: " + ", ".join(prior_missing))
                if asks:
                    hist_lines.append("Các câu hỏi đã hỏi: " + " | ".join(asks))
                if supplements:
                    hist_lines.append("Sinh viên đã bổ sung: " + " | ".join(supplements))
                hist_str = ("\n" + "\n".join(hist_lines)) if hist_lines else ""

                prompt_text = LLM_CLARIFY_ASK_TEMPLATE.format(query=q, hist_str=hist_str)
                uq = UserQuery(query_text=q, session_id=state.get("session_id"))
                ap = AugmentedPrompt(
                    user_query=uq,
                    contexts=[],
                    system_instruction=LLM_CLARIFY_ASK_SYSTEM,
                    formatted_prompt=prompt_text,
                )

                data = None
                if hasattr(self.llm_client, "generate_json"):
                    data = self.llm_client.generate_json(ap, model=self.clarify_model)
                else:
                    try:
                        resp = self.llm_client.generate(ap, model=self.clarify_model)
                    except TypeError:
                        resp = self.llm_client.generate(ap)
                    text = (resp.answer or "").strip()
                    first_b = text.find("{")
                    last_b = text.rfind("}")
                    if first_b != -1 and last_b > first_b:
                        data = json.loads(text[first_b : last_b + 1])

                if isinstance(data, dict) and str(data.get("question", "")).strip():
                    ask_question = str(data["question"]).strip().strip('"')
                    known_info = self._normalize_str_list(data.get("known_info"))
                    missing_info = self._normalize_str_list(data.get("missing_info"))
                    parse_source = "llm_json"
            except Exception as e:
                logger.warning(f"[ClarifyAskNode] LLM/JSON lỗi ({e}). Dùng câu hỏi và missing_info mặc định.")

        if not ask_question:
            ask_question = DEFAULT_CLARIFY_QUESTION
            missing_info = prior_missing or list(DEFAULT_CLARIFY_MISSING_INFO)
        elif not missing_info:
            missing_info = prior_missing or list(DEFAULT_CLARIFY_MISSING_INFO)

        clarify["asks"] = asks + [ask_question]
        clarify["known_info"] = known_info
        clarify["missing_info"] = missing_info
        clarify["pending"] = True

        # Lưu trạng thái chờ vào ClarifySessionStore (gồm known_info và missing_info)
        session_id = state.get("session_id")
        if session_id:
            self.clarify_store.save_session(
                session_id=session_id,
                query=q,
                clarify_asks=clarify["asks"],
                clarify_supplements=supplements,
                retried3=state.get("retried3", 0),
                known_info=known_info,
                missing_info=missing_info,
            )

        trace_entry = {
            "node": "clarify_ask_node",
            "source": parse_source,
            "known_info": known_info,
            "missing_info": missing_info,
        }
        new_trace = (state.get("trace") or []) + [trace_entry]

        resp = AgentResponse(
            status="clarify",
            state=0,
            answer=ask_question,
            contexts=None,
            route=state.get("route", Route.AMBIGUOUS),
            counters={
                "retry": state.get("retry", 0),
                "retried3": state.get("retried3", 0),
            },
            trace=new_trace,
            fail_reason=None,
        )

        logger.info(
            f"[ClarifyAskNode] known={known_info} | missing={missing_info} | ask='{ask_question[:60]}...'"
        )

        return {
            "answer": ask_question,
            "clarify": clarify,
            "response": resp,
            "contexts": None,
            "trace": new_trace,
            "execution_steps": (state.get("execution_steps") or []) + ["clarify_ask"],
        }

    def clarify_check_node(self, state: AgentState) -> Dict[str, Any]:
        """4.4 clarify_check_node (JEV): Kiểm tra người dùng đã cung cấp đủ từng mục trong missing_info chưa."""
        q = state.get("query", "")
        clarify = dict(
            state.get("clarify")
            or {"asks": [], "supplements": [], "known_info": [], "missing_info": [], "pending": False}
        )
        last_ask = clarify["asks"][-1] if clarify.get("asks") else ""
        supplements = list(clarify.get("supplements") or [])
        missing_info = list(clarify.get("missing_info") or [])
        known_info = list(clarify.get("known_info") or [])

        start_t = time.time()
        session_id = state.get("session_id")
        try:
            try:
                verdict = self.decision_client.check_supplement(
                    q,
                    last_ask,
                    supplements,
                    missing_info=missing_info,
                    session_id=session_id,
                )
            except TypeError:
                verdict = self.decision_client.check_supplement(
                    q,
                    last_ask,
                    supplements,
                    missing_info=missing_info,
                )
            latency = round(time.time() - start_t, 3)

            retried3 = state.get("retried3", 0)
            trace_entry = {
                "node": "clarify_check_node",
                "decision": "check_supplement",
                "result": verdict.value,
                "missing_info": missing_info,
                "retried3_before": retried3,
                "source": "jev",
                "latency_s": latency,
                **self._get_jev_trace_extras(),
            }

            if verdict == Verdict.INSUFFICIENT:
                retried3 += 1
                trace_entry["retried3_after"] = retried3
                logger.info(
                    f"[ClarifyCheckNode] JEV phán quyết: THIẾU ({latency}s) -> retried3 tăng lên {retried3}"
                )

                session_id = state.get("session_id")
                if session_id:
                    self.clarify_store.save_session(
                        session_id=session_id,
                        query=q,
                        clarify_asks=clarify.get("asks", []),
                        clarify_supplements=supplements,
                        retried3=retried3,
                        known_info=known_info,
                        missing_info=missing_info,
                    )

                return {
                    "clarify_verdict": Verdict.INSUFFICIENT,
                    "retried3": retried3,
                    "clarify": clarify,
                    "trace": (state.get("trace") or []) + [trace_entry],
                    "execution_steps": (state.get("execution_steps") or [])
                    + [f"clarify_check: INSUFFICIENT ({retried3})"],
                }
            else:
                # ĐỦ: Nối chuỗi tất định tạo effective_query, xóa phiên chờ
                clean_supps = []
                for s in supplements:
                    s_clean = re.sub(r"[:;=8xX][-~]?[)(/\\dDpP]+|[^\w\s\d,._\-]", " ", s).strip()
                    s_clean = re.sub(r"\s+", " ", s_clean)
                    if s_clean:
                        clean_supps.append(s_clean)
                supp_text = "; ".join(clean_supps) if clean_supps else "; ".join(supplements)
                effective_query = f"{q}\n(Bổ sung: {supp_text})" if supp_text else q
                clarify["pending"] = False

                session_id = state.get("session_id")
                if session_id:
                    self.clarify_store.clear_session(session_id)

                logger.info(
                    f"[ClarifyCheckNode] JEV phán quyết: ĐỦ ({latency}s) -> effective_query='{effective_query[:60]}...'"
                )

                return {
                    "clarify_verdict": Verdict.SUFFICIENT,
                    "effective_query": effective_query,
                    "clarify": clarify,
                    "trace": (state.get("trace") or []) + [trace_entry],
                    "execution_steps": (state.get("execution_steps") or []) + ["clarify_check: SUFFICIENT"],
                }
        except Exception as e:
            latency = round(time.time() - start_t, 3)
            logger.error(f"[ClarifyCheckNode] JEV lỗi ({e}). Chuyển sang fail_node.")
            trace_entry = {
                "node": "clarify_check_node",
                "decision": "check_supplement",
                "error": str(e),
                "source": "jev",
                "fail_reason": "jev_unavailable",
                "latency_s": latency,
            }
            return {
                "fail_reason": "jev_unavailable",
                "trace": (state.get("trace") or []) + [trace_entry],
                "execution_steps": (state.get("execution_steps") or []) + ["clarify_check: ERROR -> fail_node"],
            }

    def clarify_exhausted_node(self, state: AgentState) -> Dict[str, Any]:
        """4.5 clarify_exhausted_node (Template): Thông báo quá số lần bổ sung (retried3 > 3)."""
        ans = self.templates["clarify_exhausted"]
        session_id = state.get("session_id")
        if session_id:
            self.clarify_store.clear_session(session_id)

        resp = AgentResponse(
            status="clarify_exhausted",
            state=0,
            answer=ans,
            contexts=None,
            route=state.get("route"),
            counters={
                "retry": state.get("retry", 0),
                "retried3": state.get("retried3", 0),
            },
            trace=state.get("trace", []),
            fail_reason=None,
        )
        return {
            "answer": ans,
            "response": resp,
            "contexts": None,
            "sources": [],
            "execution_steps": (state.get("execution_steps") or []) + ["clarify_exhausted"],
        }

    def retrieval_node(self, state: AgentState) -> Dict[str, Any]:
        """
        4.5 retrieval_node: Thực thi hybrid search (BM25 + Vector + RRF + MMR + Rerank + Soft Faculty Filter).
        - Vòng đầu: truy vấn [effective_query or query].
        - Vòng bổ sung: truy vấn followup_queries.
        - Tích lũy vào context_pool theo chunk_id (khử trùng lặp) và chọn tối đa max_contexts chunk ưu tiên vòng mới.
        """
        if state.get("followup_queries"):
            queries = list(state["followup_queries"])
        else:
            main_q = state.get("effective_query") or state.get("query", "")
            queries = [main_q]

        per_query_results: List[List[RetrievedContext]] = []
        if self.retriever is not None:
            for q_text in queries:
                search_q = self._clean_query_for_retrieval(q_text)
                detected_facs = UETQueryRewriter.detect_faculties(search_q)
                sub_filters = {"faculty": detected_facs} if detected_facs else None

                uq = UserQuery(query_text=search_q, filters=sub_filters)
                contexts = self.retriever.retrieve(query=uq, top_k=5, filters=sub_filters)
                per_query_results.append(list(contexts))

        # Gộp luân phiên (round-robin: #1 của mỗi query trước, rồi #2...) để mỗi truy vấn đều có đại diện trong top ngữ cảnh
        retrieved_this_round: List[RetrievedContext] = []
        seen_in_round: Set[str] = set()
        max_len = max((len(lst) for lst in per_query_results), default=0)
        for idx in range(max_len):
            for lst in per_query_results:
                if idx < len(lst):
                    c = lst[idx]
                    if c.chunk_id not in seen_in_round:
                        seen_in_round.add(c.chunk_id)
                        retrieved_this_round.append(c)

        # Tích lũy vào context_pool (dict theo chunk_id, giữ điểm cao hơn nếu trùng)
        context_pool: Dict[str, RetrievedContext] = dict(state.get("context_pool") or {})
        new_chunk_ids: List[str] = []
        for c in retrieved_this_round:
            new_chunk_ids.append(c.chunk_id)
            existing = context_pool.get(c.chunk_id)
            if existing is None or getattr(c, "similarity_score", 0.0) >= getattr(existing, "similarity_score", 0.0):
                context_pool[c.chunk_id] = c

        selected_contexts = self._select_accumulated_contexts(context_pool, new_chunk_ids)

        # Cập nhật used_queries theo thứ tự
        used_queries = list(state.get("used_queries") or [])
        for q_text in queries:
            if q_text not in used_queries:
                used_queries.append(q_text)

        logger.info(
            f"[RetrievalNode] Truy xuất {len(retrieved_this_round)} chunks mới | "
            f"Pool tích lũy: {len(context_pool)} | Chọn cho LLM: {len(selected_contexts)} | "
            f"used_queries: {len(used_queries)}"
        )

        return {
            "retrieved_chunks": retrieved_this_round,
            "context_pool": context_pool,
            "contexts": selected_contexts,
            "used_queries": used_queries,
            "execution_steps": (state.get("execution_steps") or []) + [f"retrieval: {len(retrieved_this_round)} chunks"],
        }

    def generation_node(self, state: AgentState) -> Dict[str, Any]:
        """
        4.6 generation_node (LLM): Sinh toàn bộ câu trả lời từ câu hỏi gốc + ngữ cảnh tích lũy (Q2).
        Lỗi LLM -> fail_node với fail_reason='llm_unavailable' (D12).
        """
        q = state.get("effective_query") or state.get("query", "")
        contexts = state.get("contexts") or []
        uq = UserQuery(
            query_text=q,
            session_id=state.get("session_id"),
            filters=state.get("filters"),
        )

        start_t = time.time()
        try:
            usage_info = None
            if self.llm_client is None:
                context_summary = "\n\n".join(f"- {c.text}" for c in contexts[:3])
                answer = f"Dưới đây là thông tin trích xuất từ quy chế đào tạo UET:\n\n{context_summary}"
            else:
                if self.prompt_augmenter is not None:
                    augmented = self.prompt_augmenter.augment(query=uq, contexts=contexts)
                else:
                    context_block = "\n\n".join(f"[Đoạn {i+1}]: {c.text}" for i, c in enumerate(contexts))
                    prompt_str = LLM_GENERATION_FALLBACK_TEMPLATE.format(
                        context_block=context_block,
                        query=q,
                    )
                    augmented = AugmentedPrompt(
                        user_query=uq,
                        contexts=contexts,
                        system_instruction=LLM_GENERATION_FALLBACK_SYSTEM,
                        formatted_prompt=prompt_str,
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
                "contexts_count": len(contexts),
                "answer_chars": len(answer),
                "latency_s": latency,
            }
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
        """
        4.7 answer_eval_node (JEV): Đánh giá cặp (query, answer) — KHÔNG gửi chunk.
        - ĐỦ -> AgentResponse(status='ok', state=1, contexts) -> END.
        - THIẾU -> retry += 1:
            + retry <= max_retries (3) -> followup_query_node.
            + retry > max_retries (3) -> Q1=A: trả câu trả lời gần nhất (status='partial', state=1)
              kèm ghi chú phần còn thiếu + gợi ý liên hệ Phòng Đào tạo -> END.
        - Lỗi JEV -> fail_node (Q5).
        """
        q = state.get("effective_query") or state.get("query", "")
        ans = state.get("answer", "")
        retry = state.get("retry", 0)
        retried3 = state.get("retried3", 0)

        start_t = time.time()
        session_id = state.get("session_id")
        try:
            try:
                verdict = self.decision_client.evaluate_answer(q, ans, session_id=session_id)
            except TypeError:
                verdict = self.decision_client.evaluate_answer(q, ans)
            latency = round(time.time() - start_t, 3)
        except Exception as e:
            latency = round(time.time() - start_t, 3)
            logger.error(f"[AnswerEvalNode] JEV lỗi ({e}). Chuyển sang fail_node.")
            trace_entry = {
                "node": "answer_eval_node",
                "decision": "evaluate_answer",
                "error": str(e),
                "source": "jev",
                "fail_reason": "jev_unavailable",
                "latency_s": latency,
            }
            return {
                "fail_reason": "jev_unavailable",
                "trace": (state.get("trace") or []) + [trace_entry],
                "execution_steps": (state.get("execution_steps") or []) + ["answer_eval: ERROR -> fail_node"],
            }

        trace_entry = {
            "node": "answer_eval_node",
            "decision": "evaluate_answer",
            "result": verdict.value,
            "retry_before": retry,
            "source": "jev",
            "latency_s": latency,
            **self._get_jev_trace_extras(),
        }

        if verdict == Verdict.SUFFICIENT:
            new_trace = (state.get("trace") or []) + [trace_entry]
            logger.info(f"[AnswerEvalNode] JEV phán quyết: ĐỦ ({latency}s) -> Hoàn tất RAG (state=1, status='ok').")
            resp = AgentResponse(
                status="ok",
                state=1,
                answer=ans,
                contexts=state.get("contexts"),
                route=state.get("route"),
                counters={"retry": retry, "retried3": retried3},
                trace=new_trace,
                fail_reason=None,
                missing_info=None,
            )
            return {
                "answer_verdict": Verdict.SUFFICIENT,
                "response": resp,
                "trace": new_trace,
                "execution_steps": (state.get("execution_steps") or []) + ["answer_eval: SUFFICIENT"],
            }

        # THIẾU -> Cập nhật best_answer (ưu tiên noul_prob cao nhất; nếu bằng nhau hoặc None thì giữ bản mới nhất)
        jev_extras = self._get_jev_trace_extras()
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

        # Tăng retry trước rồi so sánh
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

        # retry > max_retries -> Q1=A: trả câu trả lời tốt nhất qua các vòng kèm ghi chú phần còn thiếu (status='partial', state=1)
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
            route=state.get("route"),
            counters={"retry": retry, "retried3": retried3},
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
        """
        4.8 followup_query_node (LLM): Xác định phần thông tin còn thiếu trong câu trả lời
        và sinh 1-2 truy vấn tìm kiếm mới cho đúng phần đó, không trùng used_queries.
        """
        q = state.get("effective_query") or state.get("query", "")
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
            "source": source,
            "missing_info": missing_info,
            "followup_queries": followup_queries,
            "latency_s": latency,
        }
        logger.info(
            f"[FollowupQueryNode] source={source} | missing='{missing_info}' | queries={followup_queries}"
        )

        return {
            "followup_queries": followup_queries,
            "missing_info": missing_info,
            "trace": (state.get("trace") or []) + [trace_entry],
            "execution_steps": (state.get("execution_steps") or [])
            + [f"followup_query: {len(followup_queries)} queries"],
        }

    def fail_node(self, state: AgentState) -> Dict[str, Any]:
        """4.9 fail_node (đường lỗi ẩn - Q5): Chỉ vào khi JEV hoặc LLM sập dịch vụ."""
        fail_reason = state.get("fail_reason") or "service_unavailable"
        ans = self.templates["fail_service"]

        session_id = state.get("session_id")
        if session_id:
            self.clarify_store.clear_session(session_id)

        resp = AgentResponse(
            status="fail",
            state=0,
            answer=ans,
            contexts=None,
            route=state.get("route"),
            counters={
                "retry": state.get("retry", 0),
                "retried3": state.get("retried3", 0),
            },
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

    # =========================================================================
    # ĐIỀU KIỆN RẼ NHÁNH (CONDITIONAL EDGES)
    # =========================================================================

    def _start_decision(self, state: AgentState) -> str:
        """Rẽ nhánh từ START (D3): Nếu có phiên làm rõ đang chờ -> clarify_check_node, ngược lại -> router_node."""
        clarify = state.get("clarify") or {}
        if clarify.get("pending") and clarify.get("supplements"):
            return "clarify_check_node"
        return "router_node"

    def _route_decision(self, state: AgentState) -> str:
        """Rẽ nhánh từ router_node (3 nhãn + fail_node khi JEV lỗi)."""
        if state.get("fail_reason"):
            return "fail_node"
        route = state.get("route", Route.QUALIFIED_QUESTION)
        if route == Route.OUT_OF_SCOPE:
            return "out_of_scope_node"
        if route == Route.AMBIGUOUS:
            return "clarify_ask_node"
        return "retrieval_node"

    def _clarify_ask_decision(self, state: AgentState) -> str:
        """
        Sau clarify_ask_node:
        Nếu kịch bản test truyền sẵn supplements, chuyển thẳng sang clarify_check_node.
        Ngược lại tạm dừng trả câu hỏi làm rõ cho người dùng (END).
        """
        clarify = state.get("clarify") or {}
        supplements = clarify.get("supplements", [])
        asks = clarify.get("asks", [])
        if len(supplements) >= len(asks) and len(supplements) > 0:
            return "clarify_check_node"
        return END

    def _clarify_check_decision(self, state: AgentState) -> str:
        """Rẽ nhánh sau clarify_check_node."""
        if state.get("fail_reason"):
            return "fail_node"
        return route_after_clarify_check(state, max_retried3=self.max_retried3)

    def _generation_decision(self, state: AgentState) -> str:
        """Rẽ nhánh sau generation_node."""
        if state.get("fail_reason"):
            return "fail_node"
        return "answer_eval_node"

    def _answer_eval_decision(self, state: AgentState) -> str:
        """Rẽ nhánh sau answer_eval_node."""
        if state.get("fail_reason"):
            return "fail_node"
        return route_after_answer_eval(state, max_retries=self.max_retries)

    # =========================================================================
    # LẮP RÁP GRAPH (9 NODE CHÍNH + fail_node ẨN)
    # =========================================================================

    def _build_graph(self) -> StateGraph:
        """Lắp ráp đồ thị v3 theo đúng sơ đồ mục 2 của implementation_plan_v3.md."""
        workflow = StateGraph(AgentState)

        # 9 node chính theo sơ đồ
        workflow.add_node("router_node", self.router_node)
        workflow.add_node("out_of_scope_node", self.out_of_scope_node)
        workflow.add_node("clarify_ask_node", self.clarify_ask_node)
        workflow.add_node("clarify_check_node", self.clarify_check_node)
        workflow.add_node("clarify_exhausted_node", self.clarify_exhausted_node)
        workflow.add_node("retrieval_node", self.retrieval_node)
        workflow.add_node("generation_node", self.generation_node)
        workflow.add_node("answer_eval_node", self.answer_eval_node)
        workflow.add_node("followup_query_node", self.followup_query_node)
        # Đường lỗi ẩn cho JEV / LLM gián đoạn dịch vụ (Q5)
        workflow.add_node("fail_node", self.fail_node)

        # START -> router_node | clarify_check_node (resume phiên AMBIGUOUS)
        workflow.add_conditional_edges(
            START,
            self._start_decision,
            {
                "router_node": "router_node",
                "clarify_check_node": "clarify_check_node",
            },
        )

        # router_node -> out_of_scope_node | clarify_ask_node | retrieval_node | fail_node
        workflow.add_conditional_edges(
            "router_node",
            self._route_decision,
            {
                "out_of_scope_node": "out_of_scope_node",
                "clarify_ask_node": "clarify_ask_node",
                "retrieval_node": "retrieval_node",
                "fail_node": "fail_node",
            },
        )

        workflow.add_edge("out_of_scope_node", END)

        # clarify_ask_node -> clarify_check_node | END (chờ người dùng bổ sung)
        workflow.add_conditional_edges(
            "clarify_ask_node",
            self._clarify_ask_decision,
            {
                "clarify_check_node": "clarify_check_node",
                END: END,
            },
        )

        # clarify_check_node -> clarify_ask_node | clarify_exhausted_node | retrieval_node | fail_node
        workflow.add_conditional_edges(
            "clarify_check_node",
            self._clarify_check_decision,
            {
                "clarify_ask_node": "clarify_ask_node",
                "clarify_exhausted_node": "clarify_exhausted_node",
                "retrieval_node": "retrieval_node",
                "fail_node": "fail_node",
            },
        )

        workflow.add_edge("clarify_exhausted_node", END)

        # retrieval_node -> generation_node
        workflow.add_edge("retrieval_node", "generation_node")

        # generation_node -> answer_eval_node | fail_node
        workflow.add_conditional_edges(
            "generation_node",
            self._generation_decision,
            {
                "answer_eval_node": "answer_eval_node",
                "fail_node": "fail_node",
            },
        )

        # answer_eval_node -> END (ĐỦ hoặc THIẾU retry>3) | followup_query_node (THIẾU retry<=3) | fail_node
        workflow.add_conditional_edges(
            "answer_eval_node",
            self._answer_eval_decision,
            {
                "END": END,
                "followup_query_node": "followup_query_node",
                "fail_node": "fail_node",
            },
        )

        # followup_query_node -> retrieval_node
        workflow.add_edge("followup_query_node", "retrieval_node")

        workflow.add_edge("fail_node", END)

        return workflow

    # =========================================================================
    # ĐIỂM THỰC THI (RUN / EXECUTE)
    # =========================================================================

    def run(
        self,
        query: str,
        user_query: Optional[UserQuery] = None,
        session_id: Optional[str] = None,
        supplements: Optional[List[str]] = None,
    ) -> AgentState:
        """
        Thực thi toàn bộ luồng đồ thị v3 cho một câu hỏi.
        Tự động resume phiên chờ AMBIGUOUS qua cạnh có điều kiện từ START.
        """
        effective_session_id = session_id or (user_query.session_id if user_query else None) or str(uuid.uuid4())
        if user_query and user_query.session_id is None:
            user_query.session_id = effective_session_id

        # Kiểm tra phiên chờ bổ sung của session_id
        if effective_session_id and self.clarify_store.has_pending_session(effective_session_id):
            session_data = self.clarify_store.get_session(effective_session_id)
            if session_data:
                logger.info(
                    f"[UETAgentGraph.run] Tiếp tục phiên làm rõ cho session_id='{effective_session_id}' qua Graph"
                )
                original_query = session_data["query"]
                prior_asks = session_data.get("asks", [])
                prior_supplements = session_data.get("supplements", [])
                prior_known = session_data.get("known_info", [])
                prior_missing = session_data.get("missing_info", [])
                new_supplements = prior_supplements + [query]

                initial_state: AgentState = {
                    "query": original_query,
                    "session_id": effective_session_id,
                    "user_query": user_query,
                    "route": Route.AMBIGUOUS,
                    "effective_query": None,
                    "followup_queries": [],
                    "used_queries": [],
                    "retrieved_chunks": [],
                    "context_pool": {},
                    "contexts": [],
                    "clarify_verdict": None,
                    "answer_verdict": None,
                    "missing_info": None,
                    "retry": 0,
                    "retried3": session_data.get("retried3", 0),
                    "clarify": {
                        "asks": prior_asks,
                        "supplements": new_supplements,
                        "known_info": prior_known,
                        "missing_info": prior_missing,
                        "pending": True,
                    },
                    "answer": None,
                    "best_answer": None,
                    "best_answer_score": None,
                    "best_contexts": [],
                    "response": None,
                    "trace": [],
                    "execution_steps": [f"resume_clarify_for: {effective_session_id}"],
                    "sources": [],
                    "fail_reason": None,
                }
                return self.app.invoke(
                    initial_state,
                    config={"recursion_limit": self.recursion_limit},
                )

        # Luồng chuẩn khởi tạo từ START -> router_node
        clarify_init: ClarifyData = {
            "asks": [],
            "supplements": supplements or [],
            "known_info": [],
            "missing_info": [],
            "pending": False,
        }

        initial_state: AgentState = {
            "query": query,
            "session_id": effective_session_id,
            "user_query": user_query,
            "route": None,
            "effective_query": None,
            "followup_queries": [],
            "used_queries": [],
            "retrieved_chunks": [],
            "context_pool": {},
            "contexts": [],
            "clarify_verdict": None,
            "answer_verdict": None,
            "missing_info": None,
            "retry": 0,
            "retried3": 0,
            "clarify": clarify_init,
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

        return self.app.invoke(
            initial_state,
            config={"recursion_limit": self.recursion_limit},
        )

    def close_session(self, session_id: Optional[str]) -> None:
        """Đóng kết nối HTTP Session của cả JEV và LLM và dọn phiên clarify cho session_id."""
        if not session_id:
            return
        self.clarify_store.clear_session(session_id)
        if hasattr(self.decision_client, "close_session"):
            self.decision_client.close_session(session_id)
        if self.llm_client is not None and hasattr(self.llm_client, "close_session"):
            self.llm_client.close_session(session_id)
