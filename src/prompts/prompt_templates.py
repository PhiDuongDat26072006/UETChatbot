"""
src/prompts/prompt_templates.py - Quản lý các mẫu prompt và ráp bối cảnh.
Thành viên phụ trách: Prompt Engineer / RAG Designer.
Nhiệm vụ: Thiết kế chỉ dẫn hệ thống (System Prompt) và kết hợp câu hỏi + ngữ cảnh thành AugmentedPrompt hoàn chỉnh.
"""

from __future__ import annotations
from typing import List, Optional
from src.base import BasePromptAugmenter, UserQuery, RetrievedContext, AugmentedPrompt
from src.prompts.prompts import (
    DEFAULT_UET_SYSTEM_PROMPT,
    FEW_SHOT_EXAMPLES,
    LLM_GENERATION_USER_TEMPLATE,
)
from src.utils.helpers import get_logger, load_yaml_config

logger = get_logger("prompts")


class UETPromptAugmenter(BasePromptAugmenter):
    """
    Bộ ghép nối Prompt tăng cường cho UET Chatbot.
    Tự động đọc mẫu prompt từ config.yaml nếu không truyền trực tiếp.
    Tích hợp Few-Shot Examples và RAG Guardrails để tối ưu hóa chỉ số Ragas.
    """

    def __init__(self, system_prompt: Optional[str] = None, include_few_shot: Optional[bool] = None):
        config = load_yaml_config()
        prompt_cfg = config.get("prompt", {}) if isinstance(config, dict) else {}
        if include_few_shot is not None:
            self.include_few_shot = bool(include_few_shot)
        else:
            self.include_few_shot = bool(prompt_cfg.get("include_few_shot", True))
        if system_prompt:
            self.system_prompt = system_prompt.strip()
        else:
            cfg_prompt = prompt_cfg.get("system_prompt")
            self.system_prompt = (cfg_prompt or DEFAULT_UET_SYSTEM_PROMPT).strip()

    def format_contexts(
        self,
        contexts: List[RetrievedContext],
        lightrag_overview: Optional[str] = None,
    ) -> str:
        """
        Định dạng danh sách các đoạn trích dẫn văn bản thành khối ngữ cảnh có cấu trúc.
        Nếu có Bản đồ Liên kết Quy chế 2 tầng từ LightRAGRetriever (lightrag_overview),
        chèn khối bản đồ này lên đầu trước các DataChunk nguyên văn đã qua Reranker.
        """
        resolved_overview = (lightrag_overview or "").strip()
        if not resolved_overview:
            for ctx in (contexts or []):
                ov = (ctx.metadata or {}).get("lightrag_overview")
                if isinstance(ov, str) and ov.strip():
                    resolved_overview = ov.strip()
                    break

        if not contexts:
            if resolved_overview:
                return f"[BẢN ĐỒ LIÊN KẾT QUY CHẾ (LIGHTRAG 2-TẦNG)]\n{resolved_overview}"
            return "*(Không tìm thấy tài liệu quy chế nào liên quan trực tiếp từ cơ sở dữ liệu)*"

        formatted_parts: List[str] = []
        if resolved_overview:
            formatted_parts.append(
                f"[BẢN ĐỒ LIÊN KẾT QUY CHẾ (LIGHTRAG 2-TẦNG)]\n{resolved_overview}"
            )

        for i, ctx in enumerate(contexts, 1):
            meta = ctx.metadata or {}
            title = meta.get("title") or meta.get("source") or f"Tài liệu căn cứ {i}"
            page_info = f" - Trang {meta['page']}" if meta.get("page") is not None else ""
            score_info = f" [Điểm Cross-Encoder: {ctx.similarity_score:.3f} ({int(round(ctx.similarity_score * 100))}堆)]".replace("堆", "%") if ctx.similarity_score is not None else ""
            
            search_method = meta.get("search_source")
            if not search_method:
                parts: List[str] = []
                if meta.get("bm25_rank") is not None:
                    parts.append("BM25")
                if meta.get("vector_rank") is not None:
                    parts.append("Vector Search")
                if meta.get("graph_rank") is not None:
                    parts.append("GraphRAG")
                search_method = " & ".join(parts)
            method_info = f" [Nguồn tìm kiếm: {search_method}]" if search_method else ""

            header = f"[Tài liệu {i}: {title}{page_info}{score_info}{method_info}]"
            graph_triples = meta.get("graph_triples") or []
            if graph_triples:
                triples_str = "; ".join(str(t) for t in graph_triples[:5])
                formatted_parts.append(
                    f"{header}\n[Quan hệ Đồ thị Tri thức: {triples_str}]\n{ctx.text.strip()}"
                )
            else:
                formatted_parts.append(f"{header}\n{ctx.text.strip()}")

        return "\n\n".join(formatted_parts)

    def augment(
        self,
        query: UserQuery,
        contexts: List[RetrievedContext],
        lightrag_overview: Optional[str] = None,
    ) -> AugmentedPrompt:
        """
        Ghép câu hỏi và danh sách ngữ cảnh vào mẫu prompt hoàn chỉnh cho LLM.
        :param query: Đối tượng câu hỏi của sinh viên
        :param contexts: Danh sách các đoạn văn bản trích xuất sau Reranker
        :param lightrag_overview: Tùy chọn khối tóm tắt liên kết 2 tầng của LightRAGRetriever
        :return: Đối tượng AugmentedPrompt
        """
        logger.info(f"Đang ghép prompt với {len(contexts)} đoạn ngữ cảnh cho query_id: {query.query_id}")

        context_block = self.format_contexts(contexts, lightrag_overview=lightrag_overview)
        few_shot_section = f"{FEW_SHOT_EXAMPLES}\n\n" if self.include_few_shot else ""

        formatted_prompt = LLM_GENERATION_USER_TEMPLATE.format(
            system_prompt=self.system_prompt,
            few_shot_section=few_shot_section,
            context_block=context_block,
            query=query.query_text,
        )

        return AugmentedPrompt(
            user_query=query,
            contexts=contexts,
            system_instruction=self.system_prompt,
            formatted_prompt=formatted_prompt,
        )


if __name__ == "__main__":
    import sys
    from pathlib import Path
    _BASE_DIR = Path(__file__).resolve().parent.parent.parent
    if str(_BASE_DIR) not in sys.path:
        sys.path.insert(0, str(_BASE_DIR))

    # Kiểm thử độc lập module Prompt Augmenter
    print("=" * 60)
    print("KIỂM THỬ ĐỘC LẬP: UETPromptAugmenter (src/prompts)")
    print("=" * 60)

    augmenter = UETPromptAugmenter()
    test_query = UserQuery(query_text="Sinh viên UET cần bao nhiêu tín chỉ để được xét tốt nghiệp?")
    test_contexts = [
        RetrievedContext(
            chunk_id="chunk_01",
            text="Sinh viên phải tích lũy tối thiểu 130 tín chỉ đối với hệ cử nhân, đạt GPA từ 2.00 trở lên.",
            similarity_score=0.92,
            rank=1,
            metadata={"title": "Quy chế Đào tạo Đại học Chính quy UET", "page": 42},
        ),
        RetrievedContext(
            chunk_id="chunk_02",
            text="Chuẩn đầu ra ngoại ngữ yêu cầu chứng chỉ tương đương VSTEP bậc 4 (B2) hoặc IELTS 5.5.",
            similarity_score=0.85,
            rank=2,
            metadata={"title": "Quy định Chuẩn đầu ra Ngoại ngữ ĐHQGHN", "page": 5},
        ),
    ]

    augmented = augmenter.augment(query=test_query, contexts=test_contexts)
    print(f"\n[+] Query ID: {augmented.user_query.query_id}")
    print(f"[+] Số đoạn ngữ cảnh: {len(augmented.contexts)}")
    print("\n--- NỘI DUNG PROMPT HOÀN CHỈNH ---")
    print(augmented.formatted_prompt)
    print("=" * 60)
    print(" KIỂM THỬ PROMPTS THÀNH CÔNG!")
