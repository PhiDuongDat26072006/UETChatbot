"""
src/llm/llm_client.py - Phụ trách kết nối và gọi mô hình ngôn ngữ lớn (LLM).
Thành viên phụ trách: Người tích hợp mô hình LLM.
Nhiệm vụ: Gọi API của Google Gemini, OpenAI, Claude hoặc LLM cục bộ (Ollama) để sinh câu trả lời cuối cùng kèm trích dẫn nguồn.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Optional, Generator, List, Any

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

    # Danh sách API Keys tĩnh (nếu bạn muốn khai báo danh sách trực tiếp trong mã nguồn):
    # Khuyến nghị: Ưu tiên khai báo trong file 'api_keys.txt' hoặc file '.env' để tránh lộ bí mật khi push Git.
    DEFAULT_API_KEYS: List[str] = [
        # "AIzaSy...key_du_phong_1",
        # "AIzaSy...key_du_phong_2",
    ]

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_keys: Optional[List[str]] = None,
    ):
        config = load_yaml_config()
        llm_cfg = config.get("llm", {})

        self._model_name = model_name or llm_cfg.get("model_name", "gemini-3.8-flash")
        self.candidate_models = list(llm_cfg.get("candidate_models", self.DEFAULT_CANDIDATE_MODELS))
        # Thuộc tính tương thích ngược
        self.CANDIDATE_MODELS = self.candidate_models
        self.temperature = float(llm_cfg.get("temperature", 0.3))

        # Tải danh sách tất cả các API Keys khả dụng (hỗ trợ chuyển đổi tự động khi hết hạn mức)
        self.api_keys = self._load_api_keys(custom_key=api_key, custom_keys=api_keys)
        self.current_key_index = 0
        self._clients: dict[str, Any] = {}

        if self.api_keys:
            masked = [self._mask_key(k) for k in self.api_keys]
            logger.info(f"Đã phát hiện {len(self.api_keys)} API Key Google Gemini: {', '.join(masked)} (Hỗ trợ Auto-Failover).")
            # Khởi tạo trước client cho key đầu tiên
            self._get_client_for_key(self.api_keys[0])
        else:
            logger.info("Chưa tìm thấy GEMINI_API_KEY. Sẽ sử dụng phản hồi trích xuất trực tiếp khi sinh câu trả lời.")

    @staticmethod
    def _mask_key(key: str) -> str:
        """Che bớt ký tự của API Key để bảo mật thông tin khi log."""
        if not key:
            return ""
        if len(key) <= 8:
            return "***"
        return f"{key[:4]}...{key[-4:]}"

    @classmethod
    def _load_api_keys(
        cls,
        custom_key: Optional[str] = None,
        custom_keys: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Nạp danh sách tất cả API Key từ nhiều nguồn:
        1. custom_keys / custom_key (nếu truyền vào hàm khởi tạo)
        2. File 'api_keys.txt' ở thư mục gốc (mỗi dòng 1 key, rất tiện để bổ sung nhanh)
        3. cls.DEFAULT_API_KEYS (danh sách khai báo trực tiếp trong class UETLLMClient)
        4. config.yaml -> llm.api_keys
        5. GEMINI_API_KEYS (danh sách phân tách bởi dấu phẩy, chấm phẩy hoặc xuống dòng)
        6. GEMINI_API_KEY (khóa chính)
        7. GEMINI_API_KEY_1..20 (các khóa dự phòng đánh số)
        8. GEMINI_BACKUP_API_KEY
        """
        keys: List[str] = []

        if custom_key == "":
            # Truyền rỗng cố ý -> chạy offline không dùng key
            return []

        if custom_keys:
            for k in custom_keys:
                k_str = str(k).strip()
                if k_str and k_str not in keys:
                    keys.append(k_str)

        if custom_key:
            c = custom_key.strip()
            if c and c not in keys:
                keys.append(c)

        # 1. Đọc từ file api_keys.txt ở thư mục gốc dự án (nếu có)
        keys_file = _BASE_DIR / "api_keys.txt"
        if keys_file.exists():
            try:
                with open(keys_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and line not in keys:
                            keys.append(line)
            except Exception as e:
                logger.warning(f"Không thể đọc file api_keys.txt: {e}")

        # 2. Danh sách tĩnh DEFAULT_API_KEYS khai báo trong class
        if hasattr(cls, "DEFAULT_API_KEYS") and isinstance(cls.DEFAULT_API_KEYS, list):
            for k in cls.DEFAULT_API_KEYS:
                k_str = str(k).strip()
                if k_str and not k_str.startswith("#") and k_str not in keys:
                    keys.append(k_str)

        # 3. Đọc từ config.yaml (llm.api_keys)
        try:
            cfg = load_yaml_config()
            cfg_keys = cfg.get("llm", {}).get("api_keys", [])
            if isinstance(cfg_keys, list):
                for k in cfg_keys:
                    k_str = str(k).strip()
                    if k_str and k_str not in keys:
                        keys.append(k_str)
            elif isinstance(cfg_keys, str) and cfg_keys.strip():
                for item in cfg_keys.replace(";", ",").replace("\n", ",").split(","):
                    k = item.strip()
                    if k and k not in keys:
                        keys.append(k)
        except Exception:
            pass

        # 4. Danh sách nhiều key qua GEMINI_API_KEYS (phân tách bởi dấu phẩy, chấm phẩy, hoặc xuống dòng)
        env_keys_list = os.getenv("GEMINI_API_KEYS", "")
        if env_keys_list:
            for item in env_keys_list.replace(";", ",").replace("\n", ",").split(","):
                k = item.strip()
                if k and k not in keys:
                    keys.append(k)

        # 5. Khóa chính GEMINI_API_KEY
        main_key = os.getenv("GEMINI_API_KEY", "").strip()
        if main_key and main_key not in keys:
            keys.append(main_key)

        # 6. Các khóa dự phòng đánh số: GEMINI_API_KEY_1, GEMINI_API_KEY_2, ..., GEMINI_API_KEY_20
        for i in range(1, 21):
            k = os.getenv(f"GEMINI_API_KEY_{i}", "").strip()
            if k and k not in keys:
                keys.append(k)

        # 7. Khóa dự phòng GEMINI_BACKUP_API_KEY
        backup_key = os.getenv("GEMINI_BACKUP_API_KEY", "").strip()
        if backup_key and backup_key not in keys:
            keys.append(backup_key)

        return keys

    def _get_client_for_key(self, key: str):
        """Khởi tạo hoặc tái sử dụng Google GenAI Client cho một API key cụ thể."""
        if not key:
            return None
        if key not in self._clients:
            try:
                from google import genai
                self._clients[key] = genai.Client(api_key=key)
            except Exception as e:
                logger.warning(f"Lỗi khởi tạo GenAI Client cho key [{self._mask_key(key)}]: {e}")
                return None
        return self._clients.get(key)

    @property
    def api_key(self) -> str:
        """Trả về API Key đang được sử dụng hiện tại."""
        if self.api_keys and 0 <= self.current_key_index < len(self.api_keys):
            return self.api_keys[self.current_key_index]
        return ""

    @api_key.setter
    def api_key(self, value: str):
        if value:
            self.api_keys = [value]
            self.current_key_index = 0
            self._clients = {}
            self._get_client_for_key(value)
        else:
            self.api_keys = []
            self.current_key_index = 0

    @property
    def _client(self):
        """Trả về GenAI Client của API Key đang hoạt động."""
        current_k = self.api_key
        return self._get_client_for_key(current_k) if current_k else None

    @property
    def model_name(self) -> str:
        """Tên mô hình LLM đang sử dụng."""
        return self._model_name

    def generate(self, prompt: AugmentedPrompt) -> Response:
        """
        Gửi prompt đến LLM để nhận câu trả lời cho người dùng.
        Hỗ trợ tự động chuyển đổi mô hình (Model Failover) và tự động đổi API Key (Key Failover) khi hết hạn mức.
        """
        logger.info(f"Đang gửi yêu cầu sinh câu trả lời đến mô hình '{self.model_name}'...")

        errors_encountered: List[str] = []

        with Timer() as timer:
            # 1. Duyệt qua các API Keys khả dụng (bắt đầu từ key đang hoạt động tốt)
            if self.api_keys:
                ordered_keys = (
                    self.api_keys[self.current_key_index:] +
                    self.api_keys[:self.current_key_index]
                )
                models_to_try = [self.model_name] + [m for m in self.candidate_models if m != self.model_name]
                generate_config = {"temperature": self.temperature}

                for key_idx, key in enumerate(ordered_keys):
                    client = self._get_client_for_key(key)
                    if not client:
                        continue

                    key_exhausted = False
                    for m in models_to_try:
                        try:
                            res = client.models.generate_content(
                                model=m,
                                contents=prompt.formatted_prompt,
                                config=generate_config,
                            )
                            if res and res.text:
                                meta = {}
                                if hasattr(res, "usage_metadata") and res.usage_metadata:
                                    meta["prompt_tokens"] = getattr(res.usage_metadata, "prompt_token_count", None)
                                    meta["candidates_tokens"] = getattr(res.usage_metadata, "candidates_token_count", None)

                                # Lưu lại index của key hoạt động tốt để các lượt truy vấn sau tiếp tục dùng
                                self.current_key_index = self.api_keys.index(key)

                                return Response(
                                    query_id=prompt.user_query.query_id,
                                    answer=res.text.strip(),
                                    sources=prompt.contexts,
                                    model_name=m,
                                    latency_seconds=round(timer.elapsed, 3),
                                    metadata=meta,
                                )
                        except Exception as e:
                            err_str = str(e)
                            errors_encountered.append(f"Key[{self._mask_key(key)}]-Model[{m}]: {err_str}")
                            logger.warning(f"Key [{self._mask_key(key)}] - Model {m} gặp lỗi ({err_str}).")

                            # Nếu gặp lỗi quá hạn mức (429) hoặc lỗi quyền (403), chuyển ngay sang API Key tiếp theo
                            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "403" in err_str:
                                if len(ordered_keys) > 1 and key_idx < len(ordered_keys) - 1:
                                    next_k = ordered_keys[key_idx + 1]
                                    logger.warning(
                                        f"🔄 API Key [{self._mask_key(key)}] hết hạn mức / từ chối. "
                                        f"Tự động chuyển sang API Key dự phòng: [{self._mask_key(next_k)}]..."
                                    )
                                key_exhausted = True
                                break

                    if key_exhausted:
                        continue

            # 2. Xử lý trường hợp tất cả mô hình và tất cả API Key đều từ chối dịch vụ hoặc chưa có API Key
            logger.info("Sử dụng chế độ thông báo lỗi / fallback trích đoạn tài liệu quy chế.")

            if errors_encountered:
                combined_err = " | ".join(errors_encountered)
                if "503" in combined_err or "UNAVAILABLE" in combined_err:
                    error_title = "⚠️ **MÔ HÌNH AI TỪ CHỐI DỊCH VỤ (MÃ LỖI 503 - SERVER QUÁ TẢI)**"
                    error_detail = (
                        "Máy chủ Google Gemini hiện đang chịu tải quá lớn (High Demand) trên toàn cầu "
                        "và tạm thời từ chối xử lý yêu cầu lúc này."
                    )
                elif "429" in combined_err or "RESOURCE_EXHAUSTED" in combined_err:
                    error_title = "⚠️ **MÔ HÌNH AI TỪ CHỐI DỊCH VỤ (MÃ LỖI 429 - TẤT CẢ API KEY ĐỀU HẾT HẠN MỨC)**"
                    error_detail = (
                        f"Tất cả {len(self.api_keys)} API Key Google Gemini đã sử dụng hết hạn mức yêu cầu miễn phí (Quota Exceeded) "
                        "hoặc gửi quá số lượng request cho phép trong ngày."
                    )
                elif "403" in combined_err or "PERMISSION_DENIED" in combined_err:
                    error_title = "⚠️ **MÔ HÌNH AI TỪ CHỐI DỊCH VỤ (MÃ LỖI 403 - LỖI QUYỀN TRUY CẬP)**"
                    error_detail = "Các khóa API không có quyền truy cập mô hình này hoặc đã bị Google vô hiệu hóa."
                elif "SAFETY" in combined_err or "BLOCK" in combined_err:
                    error_title = "⚠️ **MÔ HÌNH AI TỪ CHỐI DỊCH VỤ (BỘ LỌC AN TOÀN - SAFETY FILTER)**"
                    error_detail = "Nội dung câu hỏi hoặc phản hồi đã bị bộ lọc an toàn của Google Gemini từ chối."
                else:
                    error_title = "⚠️ **MÔ HÌNH AI TỪ CHỐI DỊCH VỤ / LỖI KẾT NỐI API**"
                    short_err = errors_encountered[-1].split("\n")[0][:120]
                    error_detail = f"Không thể nhận phản hồi từ dịch vụ Google Gemini ({short_err})."

                if prompt.contexts:
                    context_preview = "\n\n".join([
                        f"- **{c.metadata.get('title', 'Tài liệu quy chế')}**:\n  {c.text.strip()}"
                        for c in prompt.contexts[:3]
                    ])
                    fallback_answer = (
                        f"{error_title}\n\n"
                        f"> {error_detail}\n\n"
                        f"---\n"
                        f"📂 **HỆ THỐNG TỰ ĐỘNG CHUYỂN SANG CHẾ ĐỘ TRÍCH XUẤT TÀI LIỆU CĂN CỨ:**\n\n"
                        f"{context_preview}\n\n"
                        f"*(Bạn có thể cấu hình thêm API Key dự phòng trong file .env hoặc đối chiếu với các trích đoạn văn bản trên).*"
                    )
                else:
                    fallback_answer = (
                        f"{error_title}\n\n"
                        f"> {error_detail}\n\n"
                        f"Đồng thời, hệ thống chưa tìm thấy văn bản quy chế nào phù hợp với câu hỏi này trong cơ sở dữ liệu. Vui lòng thử lại sau."
                    )
                resp_model_name = "gemini-service-denied-fallback"

            elif not self.api_keys:
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
                resp_model_name = "offline-fallback"

            else:
                fallback_answer = (
                    "Xin lỗi em, hiện tại hệ thống chưa nhận được phản hồi phù hợp từ mô hình AI. Vui lòng thử lại sau."
                )
                resp_model_name = f"{self.model_name}-fallback"

        return Response(
            query_id=prompt.user_query.query_id,
            answer=fallback_answer,
            sources=prompt.contexts,
            model_name=resp_model_name,
            latency_seconds=round(timer.elapsed, 3),
        )

    def generate_stream(self, prompt: AugmentedPrompt) -> Generator[str, None, None]:
        """
        Sinh phản hồi dạng luồng (streaming) cho giao diện người dùng.
        Hỗ trợ tự động chuyển sang API Key khác nếu key hiện tại hết hạn mức.
        """
        if self.api_keys:
            ordered_keys = (
                self.api_keys[self.current_key_index:] +
                self.api_keys[:self.current_key_index]
            )
            models_to_try = [self.model_name] + [m for m in self.candidate_models if m != self.model_name]
            generate_config = {"temperature": self.temperature}

            for key in ordered_keys:
                client = self._get_client_for_key(key)
                if not client:
                    continue

                for m in models_to_try:
                    try:
                        stream = client.models.generate_content_stream(
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
                            self.current_key_index = self.api_keys.index(key)
                            return
                    except Exception as e:
                        err_str = str(e)
                        logger.warning(f"Streaming thất bại với Key [{self._mask_key(key)}] - Model {m}: {err_str}")
                        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "403" in err_str:
                            break

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
