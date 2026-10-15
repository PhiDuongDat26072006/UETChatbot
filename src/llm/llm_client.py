"""
src/llm/llm_client.py - Phụ trách kết nối và gọi mô hình ngôn ngữ lớn (LLM).
Thành viên phụ trách: Người tích hợp mô hình LLM.
Nhiệm vụ: Gọi API của Google Gemini, OpenAI, Claude hoặc LLM cục bộ (Ollama) để sinh câu trả lời cuối cùng kèm trích dẫn nguồn.
"""

from __future__ import annotations
import os
from typing import Optional, Generator
from src.base import BaseLLM, AugmentedPrompt, Response
from src.utils.helpers import get_logger, Timer

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
        # Nhiệt độ mặc định 0.2 (thấp để hạn chế tối đa ảo giác, tăng tính trung thực Faithfulness)
        self.temperature = float(llm_cfg.get("temperature", 0.2))

        # Tải danh sách tất cả các API Keys khả dụng từ .env (hỗ trợ tự động chuyển đổi khi hết hạn mức)
        self.api_keys = self._load_api_keys(custom_key=api_key, custom_keys=api_keys)
        self.current_key_index = 0
        self._clients: dict[str, Any] = {}

        if self.api_keys:
            logger.info(f"Đã phát hiện {len(self.api_keys)} API Key Google Gemini (Hỗ trợ Auto-Failover).")
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
        Nạp danh sách tất cả API Key từ file biến môi trường .env:
        1. custom_keys / custom_key (nếu truyền trực tiếp vào hàm khởi tạo)
        2. GEMINI_API_KEYS (danh sách nhiều key phân tách bởi dấu phẩy, chấm phẩy hoặc xuống dòng)
        3. GEMINI_API_KEY (khóa chính)
        4. GEMINI_API_KEY_1, GEMINI_API_KEY_2, ..., GEMINI_API_KEY_50 (các khóa dự phòng đánh số)
        5. GEMINI_BACKUP_API_KEY
        """
        keys: List[str] = []

        if custom_key == "":
            # Truyền rỗng cố ý -> chạy offline không dùng key
            return []

        if custom_keys:
            for k in custom_keys:
                k_str = str(k).strip().strip("'\"")
                if k_str and k_str not in keys:
                    keys.append(k_str)

        if custom_key:
            c = custom_key.strip().strip("'\"")
            if c and c not in keys:
                keys.append(c)

        # 1. Danh sách nhiều key qua biến GEMINI_API_KEYS (phân tách bởi dấu phẩy, chấm phẩy, hoặc xuống dòng)
        env_keys_list = os.getenv("GEMINI_API_KEYS", "")
        if env_keys_list:
            for item in env_keys_list.replace(";", "\n").replace(",", "\n").split("\n"):
                k = item.strip().strip("'\"")
                if k and not k.startswith("#") and k not in keys:
                    keys.append(k)

        # 2. Khóa chính GEMINI_API_KEY
        main_key = os.getenv("GEMINI_API_KEY", "").strip().strip("'\"")
        if main_key and main_key not in keys:
            keys.append(main_key)

        # 3. Các khóa dự phòng đánh số: GEMINI_API_KEY_1, GEMINI_API_KEY_2, ..., GEMINI_API_KEY_50
        for i in range(1, 51):
            k = os.getenv(f"GEMINI_API_KEY_{i}", "").strip().strip("'\"")
            if k and k not in keys:
                keys.append(k)

        # 4. Khóa dự phòng GEMINI_BACKUP_API_KEY
        backup_key = os.getenv("GEMINI_BACKUP_API_KEY", "").strip().strip("'\"")
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
