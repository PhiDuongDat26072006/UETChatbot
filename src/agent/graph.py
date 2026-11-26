"""
src/agent/graph.py - Xây dựng đồ thị trạng thái điều phối Agentic RAG v3 ("Generate -> Evaluate -> Follow-up").
Gồm 6 node chính + 1 node lỗi ẩn (fail_node cho lỗi dịch vụ JEV/LLM):
  1. router_node (JEV 3 nhãn: OUT_OF_SCOPE, AMBIGUOUS, QUALIFIED_QUESTION)
     - OUT_OF_SCOPE -> out_of_scope_node
     - QUALIFIED_QUESTION & AMBIGUOUS -> đi thẳng vào retrieval_node (không chặn lượt hỏi ngược);
       riêng AMBIGUOUS tự động gắn câu gợi ý mềm ở cuối phản hồi cuối cùng đã qua answer_eval.
  2. out_of_scope_node (Template)
  3. retrieval_node (BM25 + Vector + GraphRAG + RRF + Rerank + Soft Faculty Filter + tích lũy context_pool)
  4. generation_node (LLM sinh câu trả lời theo chiến lược Answer-first + Soft Follow-up)
  5. answer_eval_node (JEV: đánh giá cặp query + answer, không gửi chunk)
  6. followup_query_node (LLM: xác định phần thiếu và sinh 1-2 truy vấn tìm kiếm mới)
  7. fail_node (đường lỗi ẩn khi JEV hoặc LLM gián đoạn dịch vụ)
"""

from __future__ import annotations
import re
import time
import uuid
from typing import Any, Dict, Generator, List, Optional, Set

from langgraph.graph import StateGraph, START, END

from src.base import AugmentedPrompt, RetrievedContext, UserQuery
from src.agent.state import AgentState
from src.agent.counter_helpers import route_after_answer_eval
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

_NODE_STATUS_MESSAGES: Dict[str, str] = {
    "router_node": "Đang phân loại câu hỏi...",
    "retrieval_node": "Đang tra cứu quy chế và tài liệu liên quan...",
    "generation_node": "Đang tổng hợp câu trả lời...",
    "answer_eval_node": "Đang kiểm tra độ đầy đủ câu trả lời...",
    "followup_query_node": "Đang tìm kiếm bổ sung thông tin còn thiếu...",
}


def _chunk_answer_for_stream(text: str, pieces_per_chunk: int = 4) -> Generator[str, None, None]:
    """Tách câu trả lời cuối thành từng cụm nhỏ cho SSE mà vẫn bảo toàn nguyên vẹn xuống dòng & bảng Markdown."""
    if not text:
        return
    pieces = re.findall(r"\S+\s*|\s+", text)
    if not pieces:
        yield text
        return
    buf: List[str] = []
    for piece in pieces:
        buf.append(piece)
        if len(buf) >= pieces_per_chunk:
            yield "".join(buf)
            buf = []
    if buf:
        yield "".join(buf)


class UETAgentGraph:
    """
    Bộ điều phối Đồ thị trạng thái Agentic RAG v3 (6 node chính + fail_node ẩn).
    Luồng chính:
      - Happy path: router (JEV) -> retrieval -> generation (LLM) -> answer_eval (JEV) -> END (2 JEV + 1 LLM).
      - AMBIGUOUS path: router=AMBIGUOUS -> vẫn đi thẳng retrieval -> generation -> answer_eval -> gắn câu gợi ý cuối câu -> END.
      - Follow-up loop: answer_eval THIẾU (retry <= max_retries) -> followup_query (LLM) -> retrieval -> generation -> answer_eval.
      - Partial exit (Q1=A): answer_eval THIẾU (retry > max_retries) -> trả câu trả lời tốt nhất (state=1, status='partial') kèm ghi chú.
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

        # Cấu hình chọn ngữ cảnh tích lũy
        contexts_cfg = agent_cfg.get("contexts", {})
        self.max_contexts = int(contexts_cfg.get("max_chunks", 8))

        # Ngưỡng đếm retry từ config (không hard-code)
        self.max_retries = int(agent_cfg.get("answer_eval", {}).get("max_retries", 3))
        self.recursion_limit = int(agent_cfg.get("recursion_limit", 100))

        # Follow-up Query Generator (LLM)
        followup_cfg = agent_cfg.get("followup", {})
        self.followup_generator = UETFollowupQueryGenerator(
            llm_client=llm_client,
            enable_llm=bool(followup_cfg.get("enable_llm", True)),
            model=followup_cfg.get("model") or default_model,
            max_queries=int(followup_cfg.get("max_queries", 2)),
        )

        # Generator model
        gen_cfg = agent_cfg.get("generator", {})
        self.gen_model = gen_cfg.get("model") or default_model

        # Mẫu template từ config
        templates_cfg = self.raw_config.get("templates", {})
        self.templates = {
            "out_of_scope": templates_cfg.get(
                "out_of_scope",
                "Chào bạn! Tôi là UET Chatbot - Trợ lý ảo tư vấn quy chế đào tạo và tuyển sinh của Trường Đại học Công nghệ (VNU-UET). "
                "Nội dung bạn hỏi hiện nằm ngoài phạm vi hỗ trợ của tôi. Rất vui được phục vụ bạn nếu bạn cần tìm hiểu thông tin học vụ, quy chế, điểm chuẩn hay học phí của UET nhé!",
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
            "ambiguous_followup_note": templates_cfg.get(
                "ambiguous_followup_note",
                "\n\n> **Gợi ý:** Nếu bạn muốn tra cứu chính xác theo trường hợp của mình, "
                "bạn có thể cho mình biết thêm **ngành học, khóa tuyển sinh hoặc hệ đào tạo (Chuẩn/CLC)** nhé.",
            ),
        }

        # Xây dựng và biên dịch đồ thị LangGraph v3 (6 node chính + fail_node)
        self.workflow = self._build_graph()
        self.app = self.workflow.compile()

    # =========================================================================
    # HÀM BỔ TRỢ CHỌN NGỮ CẢNH & ĐỊNH DẠNG PHẢN HỒI
    # =========================================================================

    def _select_accumulated_contexts(
        self,
        context_pool: Dict[str, RetrievedContext],
        new_chunk_ids: List[str],
    ) -> List[RetrievedContext]:
        """
        Chọn tối đa max_contexts (mặc định 8) chunk đưa cho LLM:
        - Nếu tổng số chunk trong pool <= max_contexts: lấy toàn bộ, sắp xếp theo similarity_score giảm dần.
        - Nếu vượt quá max_contexts: ưu tiên chunk của vòng mới nhất theo thứ tự round-robin giữa các truy vấn,
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

        reserved_old = min(len(older_chunks), self.max_contexts // 2)
        take_new = min(len(new_chunks), self.max_contexts - reserved_old)
        take_old = min(len(older_chunks), self.max_contexts - take_new)

        return new_chunks[:take_new] + older_chunks[:take_old]

    @staticmethod
    def _clean_query_for_retrieval(q_text: str) -> str:
        """Gỡ khung '(Bổ sung: ...)' (nếu có) và mở rộng từ viết tắt/đồng nghĩa học vụ UET trước khi đưa vào BM25/Vector."""
        if not q_text:
            return q_text
        cleaned = q_text
        if "bổ sung:" in cleaned.lower():
            cleaned = re.sub(r"\s*\(\s*Bổ sung\s*:\s*([^)]*)\)", r" \1", cleaned, flags=re.IGNORECASE)
        return UETQueryRewriter.expand_for_retrieval(cleaned)

    def _apply_ambiguous_followup_note(
        self,
        answer: str,
        route: Optional[Route],
        voice_mode: bool = False,
    ) -> str:
        """
        Khi JEV Router phân loại câu hỏi là AMBIGUOUS, gắn thêm 1 câu gợi ý ngắn ở cuối phản hồi cuối cùng
        đã qua answer_eval_node (chỉ gắn đúng 1 lần, không chèn Markdown khi ở voice_mode).
        """
        if route != Route.AMBIGUOUS or not answer:
            return answer

        if voice_mode:
            note = (
                " Nếu bạn muốn tra cứu chính xác hơn theo trường hợp của mình, "
                "bạn hãy cho mình biết thêm ngành học, khóa tuyển sinh hoặc hệ đào tạo cụ thể nhé."
            )
        else:
            note = self.templates.get("ambiguous_followup_note") or ""

        if note and note.strip() not in answer:
            return f"{answer}{note}"
        return answer

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
    # 6 NODE CHÍNH + 1 NODE LỖI ẨN (fail_node)
    # =========================================================================

    def router_node(self, state: AgentState) -> Dict[str, Any]:
        """1. router_node (JEV): Phân loại câu hỏi thành 3 nhãn OUT_OF_SCOPE | AMBIGUOUS | QUALIFIED_QUESTION."""
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
        """
        3. retrieval_node: Thực thi hybrid search (BM25 + Vector + GraphRAG + RRF + Rerank + Soft Faculty Filter).
        - Vòng đầu: truy vấn [query].
        - Vòng bổ sung: truy vấn followup_queries.
        - Tích lũy vào context_pool theo chunk_id (khử trùng lặp) và chọn tối đa max_contexts chunk ưu tiên vòng mới.
        """
        if state.get("followup_queries"):
            queries = list(state["followup_queries"])
        else:
            queries = [state.get("query", "")]

        start_t = time.time()
        default_k = getattr(self.retriever, "default_top_k", 6) if self.retriever is not None else 6
        top_k = int(state.get("top_k") or default_k)
        per_query_results: List[List[RetrievedContext]] = []
        if self.retriever is not None:
            for q_text in queries:
                search_q = self._clean_query_for_retrieval(q_text)
                detected_facs = UETQueryRewriter.detect_faculties(search_q)
                sub_filters = {"faculty": detected_facs} if detected_facs else None

                uq = UserQuery(query_text=search_q, filters=sub_filters)
                contexts = self.retriever.retrieve(query=uq, top_k=top_k, filters=sub_filters)
                per_query_results.append(list(contexts))

        # Gộp luân phiên (round-robin: #1 của mỗi query trước, rồi #2...) để tránh query starvation
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

        used_queries = list(state.get("used_queries") or [])
        for q_text in queries:
            if q_text not in used_queries:
                used_queries.append(q_text)

        latency = round(time.time() - start_t, 3)
        trace_entry = {
            "node": "retrieval_node",
            "source": "retriever",
            "queries_count": len(queries),
            "chunks_retrieved": len(retrieved_this_round),
            "selected_count": len(selected_contexts),
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

    def _build_augmented_prompt(
        self,
        query: str,
        contexts: List[RetrievedContext],
        session_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        extra_format_rules: Optional[str] = None,
    ) -> AugmentedPrompt:
        """Dựng AugmentedPrompt chuẩn, kèm quy tắc định dạng Web/Voice (nếu có)."""
        uq = UserQuery(
            query_text=query,
            session_id=session_id,
            filters=filters,
        )
        if self.prompt_augmenter is not None:
            augmented = self.prompt_augmenter.augment(query=uq, contexts=contexts)
        else:
            context_block = "\n\n".join(f"[Đoạn {i + 1}]: {c.text}" for i, c in enumerate(contexts))
            prompt_str = LLM_GENERATION_FALLBACK_TEMPLATE.format(
                context_block=context_block,
                query=query,
            )
            augmented = AugmentedPrompt(
                user_query=uq,
                contexts=contexts,
                system_instruction=LLM_GENERATION_FALLBACK_SYSTEM,
                formatted_prompt=prompt_str,
            )

        if extra_format_rules and isinstance(getattr(augmented, "formatted_prompt", None), str):
            if hasattr(augmented, "model_copy"):
                augmented = augmented.model_copy(
                    update={"formatted_prompt": augmented.formatted_prompt + extra_format_rules}
                )
            else:
                augmented.formatted_prompt = augmented.formatted_prompt + extra_format_rules
        return augmented

    def generation_node(self, state: AgentState) -> Dict[str, Any]:
        """
        4. generation_node (LLM): Sinh toàn bộ câu trả lời từ câu hỏi gốc + ngữ cảnh tích lũy (Answer-first).
        Lỗi LLM -> fail_node với fail_reason='llm_unavailable' (D12).
        """
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
        5. answer_eval_node (JEV): Đánh giá cặp (query, answer) — KHÔNG gửi chunk.
        - ĐỦ -> gắn ambiguous_followup_note (nếu route=AMBIGUOUS) -> AgentResponse(status='ok', state=1) -> END.
        - THIẾU -> retry += 1:
            + retry <= max_retries -> followup_query_node.
            + retry > max_retries -> trả câu trả lời tốt nhất (status='partial', state=1) kèm ghi chú + gợi ý nếu AMBIGUOUS -> END.
        - Lỗi JEV -> fail_node (Q5).
        """
        q = state.get("query", "")
        ans = state.get("answer", "")
        retry = state.get("retry", 0)
        route = state.get("route")
        voice_mode = bool(state.get("voice_mode", False))

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
            final_answer = self._apply_ambiguous_followup_note(ans, route, voice_mode=voice_mode)
            logger.info(f"[AnswerEvalNode] JEV phán quyết: ĐỦ ({latency}s) -> Hoàn tất RAG (state=1, status='ok').")
            resp = AgentResponse(
                status="ok",
                state=1,
                answer=final_answer,
                contexts=state.get("contexts"),
                route=route,
                counters={"retry": retry},
                trace=new_trace,
                fail_reason=None,
                missing_info=None,
            )
            return {
                "answer": final_answer,
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

        # retry > max_retries -> trả câu trả lời tốt nhất kèm ghi chú phần thiếu (+ gợi ý nếu AMBIGUOUS)
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
        final_answer = self._apply_ambiguous_followup_note(final_answer, route, voice_mode=voice_mode)

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
        """
        6. followup_query_node (LLM): Xác định phần thông tin còn thiếu trong câu trả lời
        và sinh 1-2 truy vấn tìm kiếm mới cho đúng phần đó, không trùng used_queries.
        """
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

    # =========================================================================
    # ĐIỀU KIỆN RẼ NHÁNH (CONDITIONAL EDGES)
    # =========================================================================

    def _route_decision(self, state: AgentState) -> str:
        """
        Rẽ nhánh từ router_node:
        - Lỗi JEV -> fail_node
        - OUT_OF_SCOPE -> out_of_scope_node
        - QUALIFIED_QUESTION hoặc AMBIGUOUS -> đi thẳng vào retrieval_node
        """
        if state.get("fail_reason"):
            return "fail_node"
        route = state.get("route", Route.QUALIFIED_QUESTION)
        if route == Route.OUT_OF_SCOPE:
            return "out_of_scope_node"
        return "retrieval_node"

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
    # LẮP RÁP GRAPH (6 NODE CHÍNH + fail_node ẨN)
    # =========================================================================

    def _build_graph(self) -> StateGraph:
        """Lắp ráp đồ thị v3 gọn nhẹ (6 node chính + fail_node ẩn, không còn nhánh Clarify chặn lượt)."""
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
            {
                "answer_eval_node": "answer_eval_node",
                "fail_node": "fail_node",
            },
        )

        workflow.add_conditional_edges(
            "answer_eval_node",
            self._answer_eval_decision,
            {
                "END": END,
                "followup_query_node": "followup_query_node",
                "fail_node": "fail_node",
            },
        )

        workflow.add_edge("followup_query_node", "retrieval_node")
        workflow.add_edge("fail_node", END)

        return workflow

    # =========================================================================
    # ĐIỂM THỰC THI (RUN / RUN_STREAM)
    # =========================================================================

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
        return self.app.invoke(
            initial_state,
            config={"recursion_limit": self.recursion_limit},
        )

    def run_stream(
        self,
        query: str,
        user_query: Optional[UserQuery] = None,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        extra_format_rules: Optional[str] = None,
        voice_mode: bool = False,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Thực thi đồ thị bằng LangGraph `stream(stream_mode='updates')`:
        - Phát sự kiện `status` theo từng node để cập nhật tiến độ tức thì và giữ kết nối SSE sống.
        - Bắt buộc đi qua `answer_eval_node` (+ `followup_query_node` nếu thiếu).
        - CHỈ xuất `sources` + `token` của câu trả lời cuối cùng đã được duyệt (không rò rỉ bản nháp vòng trước).
        """
        start_t = time.time()
        initial_state = self._build_initial_state(
            query=query,
            user_query=user_query,
            session_id=session_id,
            top_k=top_k,
            extra_format_rules=extra_format_rules,
            voice_mode=voice_mode,
        )

        final_response: Optional[AgentResponse] = None
        final_contexts: List[RetrievedContext] = []
        final_route: Optional[Route] = None
        final_retry: int = 0
        final_answer: str = ""

        for update in self.app.stream(
            initial_state,
            config={"recursion_limit": self.recursion_limit},
            stream_mode="updates",
        ):
            if not isinstance(update, dict):
                continue
            for node_name, node_output in update.items():
                if not isinstance(node_output, dict):
                    continue

                if node_output.get("route") is not None:
                    final_route = node_output["route"]
                if "retry" in node_output and node_output["retry"] is not None:
                    final_retry = int(node_output["retry"])
                if node_output.get("contexts") is not None:
                    final_contexts = list(node_output["contexts"] or [])
                if node_output.get("answer") is not None:
                    final_answer = str(node_output["answer"])
                if node_output.get("response") is not None:
                    final_response = node_output["response"]
                    final_answer = final_response.answer or ""
                    final_contexts = list(final_response.contexts or [])

                if node_name in _NODE_STATUS_MESSAGES:
                    msg = _NODE_STATUS_MESSAGES[node_name]
                    if node_name == "followup_query_node" and final_retry > 0:
                        msg = f"Đang tìm kiếm bổ sung thông tin (lần {final_retry})..."
                    yield {
                        "type": "status",
                        "node": node_name,
                        "message": msg,
                    }

        status_val = final_response.status if final_response else "ok"
        state_val = final_response.state if final_response else (1 if final_contexts else 0)
        route_val = (
            final_response.route.value
            if (final_response and final_response.route)
            else (final_route.value if final_route else Route.QUALIFIED_QUESTION.value)
        )
        model_name = getattr(self.llm_client, "model_name", None) or "jev-agentic-rag-v3"

        yield {
            "type": "sources",
            "sources": final_contexts,
            "model_name": model_name,
            "route": route_val,
            "status": status_val,
        }

        for chunk in _chunk_answer_for_stream(final_answer):
            yield {
                "type": "token",
                "content": chunk,
            }

        yield {
            "type": "done",
            "status": status_val,
            "state": state_val,
            "route": route_val,
            "retry": final_retry,
            "answer": final_answer,
            "sources": final_contexts,
            "model_name": model_name,
            "latency_seconds": round(time.time() - start_t, 3),
        }

    def close_session(self, session_id: Optional[str]) -> None:
        """Đóng kết nối HTTP Session của cả JEV và LLM cho session_id."""
        if not session_id:
            return
        if hasattr(self.decision_client, "close_session"):
            self.decision_client.close_session(session_id)
        if self.llm_client is not None and hasattr(self.llm_client, "close_session"):
            self.llm_client.close_session(session_id)
