"""
src/prompts/prompt_templates.py - Quản lý các mẫu prompt và ráp bối cảnh.
Thành viên phụ trách: Prompt Engineer / RAG Designer.
Nhiệm vụ: Thiết kế chỉ dẫn hệ thống (System Prompt) và kết hợp câu hỏi + ngữ cảnh thành AugmentedPrompt hoàn chỉnh.
"""

from __future__ import annotations
from typing import List
from src.base import BasePromptAugmenter, UserQuery, RetrievedContext, AugmentedPrompt
from src.utils.helpers import get_logger

logger = get_logger("prompts")

# Mẫu System Prompt chuẩn mực cho UET Chatbot
DEFAULT_UET_SYSTEM_PROMPT = """
Bạn là Trợ lý AI Thông minh hỗ trợ tư vấn Quy chế Đào tạo và Học vụ của Trường Đại học Công nghệ - Đại học Quốc gia Hà Nội (VNU-UET).
Nhiệm vụ của bạn là giải đáp chính xác, lịch sự và rõ ràng các thắc mắc của sinh viên, giảng viên dựa trên các văn bản quy chế được cung cấp.

Quy tắc ứng xử và phản hồi:
1. Chỉ sử dụng thông tin trong phần 'NGỮ CẢNH ĐƯỢC CUNG CẤP' dưới đây để trả lời.
2. Nếu ngữ cảnh không có thông tin hoặc không đủ để khẳng định, hãy thẳng thắn thông báo: "Xin lỗi, hiện tôi chưa tìm thấy quy định cụ thể về vấn đề này trong tài liệu hiện có. Bạn vui lòng liên hệ trực tiếp Phòng Đào tạo (uet.vnu.edu.vn) để được hỗ trợ chính xác nhất." Tuyệt đối không tự suy diễn hoặc bịa đặt quy chế.
3. Luôn chỉ rõ số hiệu văn bản, điều khoản hoặc tên tài liệu làm căn cứ nếu có trong ngữ cảnh.
4. Trình bày câu trả lời mạch lạc, gạch đầu dòng rõ ràng nếu có nhiều điều kiện hoặc bước thực hiện.
"""


class UETPromptAugmenter(BasePromptAugmenter):
    """
    Bộ ghép nối Prompt tăng cường cho UET Chatbot.
    Thành viên phụ trách cần kế thừa BasePromptAugmenter từ base.py và tự cài đặt hàm augment().
    """

    def __init__(self, system_prompt: str = DEFAULT_UET_SYSTEM_PROMPT):
        self.system_prompt = system_prompt.strip()

    def augment(
        self,
        query: UserQuery,
        contexts: List[RetrievedContext],
    ) -> AugmentedPrompt:
        """
        Ghép câu hỏi và danh sách ngữ cảnh vào mẫu prompt hoàn chỉnh.
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

        raise NotImplementedError(
            "TODO: Thành viên phụ trách Prompts cần tự cài đặt logic hàm augment() trong src/prompts/prompt_templates.py!"
        )
