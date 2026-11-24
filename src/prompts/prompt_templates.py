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

    def format_contexts(self, contexts: List[RetrievedContext]) -> str:
        """
        Định dạng danh sách các đoạn trích dẫn văn bản thành khối ngữ cảnh có cấu trúc.
        Bao gồm tên tài liệu, số hiệu/trang và điểm tương đồng (nếu có).
        """
        if not contexts:
            return "*(Không tìm thấy tài liệu quy chế nào liên quan trực tiếp từ cơ sở dữ liệu)*"

        formatted_parts: List[str] = []
        for i, ctx in enumerate(contexts, 1):
            meta = ctx.metadata or {}
            title = meta.get("title") or meta.get("source") or f"Tài liệu căn cứ {i}"
            page_info = f" - Trang {meta['page']}" if meta.get("page") is not None else ""
            score_info = f" [Điểm Cross-Encoder: {ctx.similarity_score:.3f} ({int(round(ctx.similarity_score * 100))}堆)]".replace("堆", "%") if ctx.similarity_score is not None else ""
            
            search_method = meta.get("search_source")
            if not search_method:
                if meta.get("bm25_rank") is not None and meta.get("vector_rank") is not None:
                    search_method = "BM25 & Vector Search"
                elif meta.get("bm25_rank") is not None:
                    search_method = "BM25"
                elif meta.get("vector_rank") is not None:
                    search_method = "Vector Search"
            method_info = f" [Nguồn tìm kiếm: {search_method}]" if search_method else ""

            header = f"[Tài liệu {i}: {title}{page_info}{score_info}{method_info}]"
            formatted_parts.append(f"{header}\n{ctx.text.strip()}")

        return "\n\n".join(formatted_parts)

    def augment(
        self,
        query: UserQuery,
        contexts: List[RetrievedContext],
    ) -> AugmentedPrompt:
        """
        Ghép câu hỏi và danh sách ngữ cảnh vào mẫu prompt hoàn chỉnh cho LLM.
        :param query: Đối tượng câu hỏi của sinh viên
        :param contexts: Danh sách các đoạn văn bản trích xuất từ Vector DB
        :return: Đối tượng AugmentedPrompt
        """
        logger.info(f"Đang ghép prompt với {len(contexts)} đoạn ngữ cảnh cho query_id: {query.query_id}")

        # =========================================================================
        # TODO: THÀNH VIÊN PHỤ TRÁCH TỰ CÀI ĐẶT PHẦN NÀY:
        # Gợi ý:
        # 1. Định dạng các đoạn trích từ contexts:
        #    `context_str = "\n\n".join([f"[Nguồn: {c.metadata.get('title', 'Tài liệu')}] {c.text}" for c in contexts])`
        # 2. Xây dựng formatted_prompt kết hợp:
        #    - System instruction
        #    - Ngữ cảnh trích xuất
        #    - Câu hỏi của người dùng
        # 3. Trả về đối tượng `AugmentedPrompt(...)` theo định nghĩa trong base.py
        # =========================================================================

        context_block = self.format_contexts(contexts)
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
