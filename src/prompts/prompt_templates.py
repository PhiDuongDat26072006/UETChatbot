"""
src/prompts/prompt_templates.py - Quản lý các mẫu prompt và ráp bối cảnh.
Thành viên phụ trách: Prompt Engineer / RAG Designer.
Nhiệm vụ: Thiết kế chỉ dẫn hệ thống (System Prompt) và kết hợp câu hỏi + ngữ cảnh thành AugmentedPrompt hoàn chỉnh.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Optional

_BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from src.base import BasePromptAugmenter, UserQuery, RetrievedContext, AugmentedPrompt
from src.utils.helpers import get_logger, load_yaml_config

logger = get_logger("prompts")

# Mẫu System Prompt chuẩn mực cho UET Chatbot (Đồng bộ với config.yaml)
DEFAULT_UET_SYSTEM_PROMPT = """
Bạn là trợ lý ảo tư vấn quy chế đào tạo và học vụ của Trường Đại học Công nghệ - ĐHQGHN (UET Chatbot).
Nhiệm vụ của bạn là giải đáp thắc mắc của sinh viên một cách chính xác, thân thiện, rõ ràng dựa trên các tài liệu quy chế được cung cấp.

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin có trong phần [NGỮ CẢNH ĐƯỢC CUNG CẤP] được trích xuất để trả lời.
2. Nếu ngữ cảnh không có thông tin hoặc không đủ để trả lời câu hỏi, hãy trả lời lịch sự: "Xin lỗi em, hiện tại trong các văn bản quy chế lưu trữ chưa có thông tin cụ thể về câu hỏi này. Em vui lòng liên hệ trực tiếp Phòng Đào tạo (Phòng 105-E3) hoặc Cổng thông tin đào tạo UET để được hướng dẫn chính xác nhất nhé." Tuyệt đối không tự bịa đặt thông tin.
3. Trả lời bằng tiếng Việt chuẩn mực, mạch lạc, có thể dùng gạch đầu dòng để làm rõ các điều kiện, tiêu chí.
4. Cuối câu trả lời, hãy nêu rõ nguồn tài liệu tham khảo (Tên file, văn bản hoặc số quyết định) được trích xuất từ ngữ cảnh.
"""


class UETPromptAugmenter(BasePromptAugmenter):
    """
    Bộ ghép nối Prompt tăng cường cho UET Chatbot.
    Tự động đọc mẫu prompt từ config.yaml nếu không truyền trực tiếp.
    """

    def __init__(self, system_prompt: Optional[str] = None):
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
            score_info = f" [Độ phù hợp: {int(ctx.similarity_score * 100)}%]" if ctx.similarity_score is not None else ""

            header = f"[Tài liệu {i}: {title}{page_info}{score_info}]"
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

        context_block = self.format_contexts(contexts)

        formatted_prompt = (
            f"{self.system_prompt}\n\n"
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
