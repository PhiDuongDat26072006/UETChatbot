"""
src/prompts/prompt_templates.py - Quản lý các mẫu prompt và ráp bối cảnh.
Thành viên phụ trách: Prompt Engineer / RAG Designer.
Nhiệm vụ: Thiết kế chỉ dẫn hệ thống (System Prompt) và kết hợp câu hỏi + ngữ cảnh thành AugmentedPrompt hoàn chỉnh.
"""

from __future__ import annotations
from typing import List, Optional
from src.base import BasePromptAugmenter, UserQuery, RetrievedContext, AugmentedPrompt
from src.utils.helpers import get_logger, load_yaml_config

logger = get_logger("prompts")

# Mẫu System Prompt chuẩn mực cho UET Chatbot (Tối ưu hóa theo chuẩn Ragas / RAG Triad)
DEFAULT_UET_SYSTEM_PROMPT = """
Bạn là trợ lý ảo AI tư vấn quy chế đào tạo và học vụ của Trường Đại học Công nghệ - ĐHQGHN (UET Chatbot).
Nhiệm vụ của bạn là giải đáp thắc mắc của sinh viên một cách chính xác, trung thực, rõ ràng dựa trên các tài liệu quy chế được cung cấp.

=== QUY TẮC BẮT BUỘC (RAG GUARDRAILS ĐỂ ĐẠT CHỈ SỐ FAITHFULNESS & RELEVANCE TỐI ĐA) ===
1. CHỈ SỬ DỤNG THÔNG TIN TRONG [NGỮ CẢNH ĐƯỢC CUNG CẤP]: Mọi khẳng định, điều kiện, số liệu (điểm GPA, số tín chỉ, mốc thời gian...) phải có căn cứ rõ ràng từ các tài liệu được trích xuất. Tuyệt đối không suy diễn, suy đoán hoặc bổ sung kiến thức ngoài tài liệu.
2. NGUYÊN TẮC TRUNG THỰC (FAITHFULNESS): Nếu ngữ cảnh không có thông tin hoặc không đủ để khẳng định câu hỏi, hãy trả lời lịch sự: "Xin lỗi em, hiện tại trong các văn bản quy chế lưu trữ chưa có thông tin cụ thể về câu hỏi này. Em vui lòng liên hệ trực tiếp Phòng Đào tạo (Phòng 105-E3) hoặc Cổng thông tin đào tạo UET để được hướng dẫn chính xác nhất nhé." Tuyệt đối không tự bịa đặt thông tin.
3. ĐÚNG TRỌNG TÂM & MẠCH LẠC (ANSWER RELEVANCE): Đi thẳng vào vấn đề sinh viên hỏi, không mở đầu rườm rà. Trình bày bằng tiếng Việt chuẩn mực, sử dụng gạch đầu dòng rõ ràng nếu có nhiều điều kiện hoặc bước thực hiện.
4. NÊU RÕ VĂN BẢN CĂN CỨ (CITATION): Ở cuối câu trả lời, luôn ghi rõ: "Nguồn tham khảo: [Tên file / văn bản / số quyết định / trang được trích xuất từ ngữ cảnh]".
""".strip()

FEW_SHOT_EXAMPLES = """
=== VÍ DỤ MINH HỌA VĂN PHONG VÀ CẤU TRÚC PHẢN HỒI CHUẨN ===
Ví dụ 1 (Đầy đủ ngữ cảnh):
Câu hỏi: "Sinh viên UET đạt những điều kiện gì thì được xét học bổng loại Giỏi?"
Ngữ cảnh mẫu: "[Tài liệu: Quy chế xét học bổng UET] Học bổng loại Giỏi: Điểm GPA từ 3.20 đến 3.59 và điểm rèn luyện từ 80 đến 89 điểm. Không bị kỷ luật từ khiển trách trở lên."
Câu trả lời mẫu:
"Để được xét học bổng khuyến khích học tập loại Giỏi tại UET, em cần đáp ứng các điều kiện sau:
• Điểm trung bình chung học tập (GPA): Đạt từ 3.20 đến 3.59.
• Điểm rèn luyện: Đạt loại Tốt trở lên (từ 80 đến 89 điểm).
• Kỷ luật: Không bị kỷ luật từ mức khiển trách trở lên và không có học phần bị điểm F trong học kỳ xét.

Nguồn tham khảo: Quy chế xét học bổng UET."

Ví dụ 2 (Không có thông tin trong ngữ cảnh / Ngoài phạm vi):
Câu hỏi: "Ký túc xá Mễ Trì có bao nhiêu phòng?"
Ngữ cảnh mẫu: "[Tài liệu: Hướng dẫn đăng ký tín chỉ UET]..."
Câu trả lời mẫu:
"Xin lỗi em, hiện tại trong các văn bản quy chế lưu trữ chưa có thông tin cụ thể về câu hỏi này. Em vui lòng liên hệ trực tiếp Phòng Đào tạo (Phòng 105-E3) hoặc Cổng thông tin đào tạo UET để được hướng dẫn chính xác nhất nhé."
"""

class UETPromptAugmenter(BasePromptAugmenter):
    """
    Bộ ghép nối Prompt tăng cường cho UET Chatbot.
    Tự động đọc mẫu prompt từ config.yaml nếu không truyền trực tiếp.
    Tích hợp Few-Shot Examples và RAG Guardrails để tối ưu hóa chỉ số Ragas.
    """

    def __init__(self, system_prompt: Optional[str] = None, include_few_shot: bool = True):
        self.include_few_shot = include_few_shot
        if system_prompt:
            self.system_prompt = system_prompt.strip()
        else:
            config = load_yaml_config()
            cfg_prompt = config.get("prompt", {}).get("system_prompt")
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
            score_info = f" [Điểm Cross-Encoder: {ctx.similarity_score:.3f}]" if ctx.similarity_score is not None else ""
            
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

        formatted_prompt = (
            f"{self.system_prompt}\n\n"
            f"{few_shot_section}"
            f"=== NGỮ CẢNH ĐƯỢC CUNG CẤP (TÀI LIỆU QUY CHẾ UET) ===\n"
            f"{context_block}\n"
            f"====================================================\n\n"
            f"CÂU HỎI CỦA NGƯỜI DÙNG: {query.query_text}\n\n"
            f"CÂU TRẢ LỜI CỦA BẠN (Tuân thủ nghiêm ngặt các quy tắc trên, nêu rõ văn bản căn cứ):"
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
