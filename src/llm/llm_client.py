"""
src/llm/llm_client.py - Phụ trách kết nối và gọi mô hình ngôn ngữ lớn (LLM).
Thành viên phụ trách: Người tích hợp mô hình LLM.
Nhiệm vụ: Gọi API của Google Gemini, OpenAI, Claude hoặc LLM cục bộ (Ollama) để sinh câu trả lời cuối cùng kèm trích dẫn nguồn.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Optional, Generator

_BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from src.base import BaseLLM, AugmentedPrompt, Response
from src.utils.helpers import get_logger, Timer, load_yaml_config

logger = get_logger("llm")


class UETLLMClient(BaseLLM):
    """
    Client tương tác với LLM cho UET Chatbot.
    Kế thừa BaseLLM từ base.py và cài đặt các hàm generate() và generate_stream().
    Tự động đồng bộ các tham số (model_name, candidate_models, temperature) từ config.yaml.
    """

    DEFAULT_CANDIDATE_MODELS = [
        "gemini-3.8-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-flash-latest",
    ]

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        config = load_yaml_config()
        llm_cfg = config.get("llm", {})

        self._model_name = model_name or llm_cfg.get("model_name", "gemini-3.8-flash")
        self.candidate_models = list(llm_cfg.get("candidate_models", self.DEFAULT_CANDIDATE_MODELS))
        # Thuộc tính tương thích ngược
        self.CANDIDATE_MODELS = self.candidate_models
        self.temperature = float(llm_cfg.get("temperature", 0.3))
        self.api_key = (api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")).strip()
        self._client = None

        if self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
                logger.info(f"Đã khởi tạo thành công Google GenAI Client cho mô hình '{self._model_name}'")
            except Exception as e:
                logger.warning(f"Chưa thể kết nối Google GenAI Client ({e}). Hệ thống sẽ dùng chế độ fallback.")
        else:
            logger.info("Chưa tìm thấy GEMINI_API_KEY. Sẽ sử dụng phản hồi trích xuất trực tiếp khi sinh câu trả lời.")

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

        with Timer() as timer:
            # 1. Gọi Google GenAI SDK nếu có client
            if self._client:
                # Danh sách model thử nghiệm ưu tiên model cấu hình, sau đó thử các fallback model
                models_to_try = [self.model_name] + [m for m in self.candidate_models if m != self.model_name]
                generate_config = {"temperature": self.temperature}

                for m in models_to_try:
                    try:
                        res = self._client.models.generate_content(
                            model=m,
                            contents=prompt.formatted_prompt,
                            config=generate_config,
                        )
                        if res and res.text:
                            meta = {}
                            if hasattr(res, "usage_metadata") and res.usage_metadata:
                                meta["prompt_tokens"] = getattr(res.usage_metadata, "prompt_token_count", None)
                                meta["candidates_tokens"] = getattr(res.usage_metadata, "candidates_token_count", None)

                            return Response(
                                query_id=prompt.user_query.query_id,
                                answer=res.text.strip(),
                                sources=prompt.contexts,
                                model_name=m,
                                latency_seconds=round(timer.elapsed, 3),
                                metadata=meta,
                            )
                    except Exception as e:
                        logger.warning(f"Mô hình {m} gặp lỗi ({e}), đang chuyển sang mô hình dự phòng tiếp theo...")

            # 2. Chế độ Fallback thông minh khi chưa có API key hoặc mất mạng
            logger.info("Sử dụng chế độ fallback trích đoạn tài liệu quy chế.")
            if prompt.contexts:
                context_preview = "\n\n".join([
                    f"- **{c.metadata.get('title', 'Tài liệu quy chế')}**:\n  {c.text.strip()}"
                    for c in prompt.contexts[:3]
                ])
                fallback_answer = (
                    f"Dựa trên các văn bản quy chế đào tạo UET được tra cứu:\n\n"
                    f"{context_preview}\n\n"
                    f"*(Lưu ý: Để kích hoạt phản hồi tổng hợp thông minh từ AI, vui lòng kiểm tra GEMINI_API_KEY trong file .env)*"
                )
            else:
                fallback_answer = (
                    "Xin lỗi, hiện tôi chưa tìm thấy quy định cụ thể về vấn đề này trong tài liệu hiện có. "
                    "Bạn vui lòng liên hệ trực tiếp Phòng Đào tạo (uet.vnu.edu.vn) để được hỗ trợ chính xác nhất."
                )

        return Response(
            query_id=prompt.user_query.query_id,
            answer=fallback_answer,
            sources=prompt.contexts,
            model_name=f"{self.model_name}-fallback",
            latency_seconds=round(timer.elapsed, 3),
        )

    def generate_stream(self, prompt: AugmentedPrompt) -> Generator[str, None, None]:
        """
        Sinh phản hồi dạng luồng (streaming) cho giao diện người dùng.
        """
        if self._client:
            models_to_try = [self.model_name] + [m for m in self.candidate_models if m != self.model_name]
            generate_config = {"temperature": self.temperature}
            for m in models_to_try:
                try:
                    stream = self._client.models.generate_content_stream(
                        model=m,
                        contents=prompt.formatted_prompt,
                        config=generate_config,
                    )
                    has_content = False
                    for chunk in stream:
                        if chunk.text:
                            has_content = True
                            yield chunk.text
                    if has_content:
                        return
                except Exception as e:
                    logger.warning(f"Lỗi streaming từ mô hình {m}: {e}. Đang thử mô hình khác...")

        # Fallback stream nếu không có client hoặc streaming thất bại
        yield self.generate(prompt).answer


if __name__ == "__main__":
    import sys
    from pathlib import Path
    _BASE_DIR = Path(__file__).resolve().parent.parent.parent
    if str(_BASE_DIR) not in sys.path:
        sys.path.insert(0, str(_BASE_DIR))

    # Kiểm thử độc lập module LLM Client
    from src.prompts.prompt_templates import UETPromptAugmenter
    from src.base import UserQuery, RetrievedContext

    print("=" * 60)
    print("KIỂM THỬ ĐỘC LẬP: UETLLMClient (src/llm)")
    print("=" * 60)

    llm_client = UETLLMClient()
    print(f"[+] Model đang cấu hình: {llm_client.model_name}")

    sample_query = UserQuery(query_text="Điều kiện xét học bổng khuyến khích học tập loại Giỏi tại UET là gì?")
    sample_contexts = [
        RetrievedContext(
            chunk_id="ctx_01",
            text="Học bổng loại Giỏi: Sinh viên có điểm GPA từ 3.20 đến 3.59 và điểm rèn luyện đạt loại Tốt trở lên (từ 80 đến 89 điểm). Mức học bổng bằng 100% mức trần học phí.",
            similarity_score=0.95,
            rank=1,
            metadata={"title": "Quy chế xét học bổng khuyến khích học tập UET", "page": 6},
        )
    ]

    augmenter = UETPromptAugmenter()
    prompt = augmenter.augment(query=sample_query, contexts=sample_contexts)

    print("\n[+] Đang gọi hàm generate()...")
    res = llm_client.generate(prompt)

    print(f"[+] Phản hồi từ mô hình: {res.model_name}")
    print(f"[+] Thời gian xử lý: {res.latency_seconds} giây")
    print(f"[+] Số nguồn dẫn: {len(res.sources)}")
    print("\n--- NỘI DUNG CÂU TRẢ LỜI ---")
    print(res.answer)
    print("=" * 60)
    print(" KIỂM THỬ LLM CLIENT THÀNH CÔNG!")
