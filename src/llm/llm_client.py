"""
src/llm/llm_client.py - Phụ trách kết nối và gọi mô hình ngôn ngữ lớn (LLM).
Thành viên phụ trách: Người tích hợp mô hình LLM.
Nhiệm vụ: Gọi API của Google Gemini, OpenAI, Claude hoặc LLM cục bộ (Ollama) để sinh câu trả lời cuối cùng kèm trích dẫn nguồn.
"""

from __future__ import annotations
import os
from typing import Optional, Generator
from base import BaseLLM, AugmentedPrompt, Response
from src.utils.helpers import get_logger, Timer

logger = get_logger("llm")


class UETLLMClient(BaseLLM):
    """
    Client tương tác với LLM cho UET Chatbot.
    Thành viên phụ trách cần kế thừa BaseLLM từ base.py và tự cài đặt:
    - property model_name
    - generate(prompt: AugmentedPrompt) -> Response
    """

    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        self._model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    @property
    def model_name(self) -> str:
        """Tên mô hình LLM đang sử dụng."""
        return self._model_name

    def generate(self, prompt: AugmentedPrompt) -> Response:
        """
        Gửi prompt đến LLM để nhận câu trả lời cho người dùng.
        :param prompt: Đối tượng AugmentedPrompt chứa câu hỏi và ngữ cảnh
        :return: Đối tượng Response
        """
        logger.info(f"Đang gửi yêu cầu sinh câu trả lời đến mô hình '{self.model_name}'...")

        # =========================================================================
        # TODO: THÀNH VIÊN PHỤ TRÁCH TỰ CÀI ĐẶT PHẦN NÀY:
        # Gợi ý:
        # 1. Khởi tạo client của Google GenAI SDK (hoặc OpenAI client):
        #    `from google import genai; client = genai.Client(api_key=self.api_key)`
        # 2. Gọi API sinh phản hồi:
        #    `res = client.models.generate_content(model=self.model_name, contents=prompt.formatted_prompt)`
        # 3. Đo độ trễ (latency_seconds) bằng Timer trong src.utils.helpers.
        # 4. Đóng gói kết quả thành đối tượng `Response(...)` theo base.py:
        #    - query_id=prompt.user_query.query_id
        #    - answer=res.text
        #    - sources=prompt.contexts
        #    - model_name=self.model_name
        #    - latency_seconds=...
        # =========================================================================

        raise NotImplementedError(
            "TODO: Thành viên phụ trách LLM cần tự cài đặt logic hàm generate() trong src/llm/llm_client.py!"
        )
