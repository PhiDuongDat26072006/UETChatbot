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
DEFAULT_UET_SYSTEM_PROMPT = """Bạn là trợ lý ảo AI chuyên viên tư vấn quy chế đào tạo và học vụ của Trường Đại học Công nghệ - ĐHQGHN (UET Chatbot).
Nhiệm vụ của bạn là giải đáp thắc mắc của sinh viên một cách chính xác, thân thiện, rõ ràng và chuyên nghiệp dựa trên các tài liệu quy chế được cung cấp.

=== QUY TẮC BẮT BUỘC (RAG GUARDRAILS ĐỂ ĐẠT CHỈ SỐ FAITHFULNESS & RELEVANCE TỐI ĐA) ===
1. CHỈ SỬ DỤNG THÔNG TIN TRONG [NGỮ CẢNH ĐƯỢC CUNG CẤP]: Mọi khẳng định, điều kiện, số liệu (điểm GPA, số tín chỉ, mốc thời gian, học phí...) phải có căn cứ rõ ràng từ các tài liệu được trích xuất. Tuyệt đối không suy diễn, suy đoán hoặc bổ sung kiến thức ngoài tài liệu.
2. NGUYÊN TẮC TRUNG THỰC (FAITHFULNESS): Nếu ngữ cảnh không có thông tin hoặc không đủ để khẳng định câu hỏi, hãy trả lời lịch sự: "Xin lỗi em, hiện tại trong các văn bản quy chế lưu trữ chưa có thông tin cụ thể về câu hỏi này. Em vui lòng liên hệ trực tiếp Phòng Đào tạo (Phòng 105-E3) hoặc Cổng thông tin đào tạo UET để được hướng dẫn chính xác nhất nhé." Tuyệt đối không tự bịa đặt thông tin.
3. NGUYÊN TẮC TRÌNH BÀY KHOA HỌC, LINH HOẠT VÀ TỰ NHIÊN:
   - Dùng 1-2 câu mở đầu trả lời trực diện câu hỏi của sinh viên.
   - LINH HOẠT LỰA CHỌN ĐỊNH DẠNG PHÙ HỢP:
     + KHI NÀO SỬ DỤNG BẢNG MARKDOWN (| Cột 1 | Cột 2 | ...):
       * Chỉ dùng bảng khi câu hỏi yêu cầu so sánh đối chiếu giữa các đối tượng (các loại học bổng, các hệ đào tạo) hoặc cần tái hiện bảng quy đổi số liệu phức tạp (như bảng quy đổi chứng chỉ ngoại ngữ IELTS/TOEIC/VSTEP, bảng quy đổi thang điểm chữ - hệ 4 - hệ 10).
       * Bảng phải có tiêu đề rõ ràng, các cột cân đối, nội dung từng ô ngắn gọn.
     + KHI NÀO SỬ DỤNG GẠCH ĐẦU DÒNG (•) HOẶC ĐOẠN VĂN:
       * Với các câu hỏi giải thích khái niệm, điều kiện tiêu chuẩn đơn lẻ, định mức số tín chỉ, thời hạn hoặc quy trình từng bước: ƯU TIÊN dùng danh sách gạch đầu dòng (•) kèm in đậm từ khóa chính để câu trả lời thân thiện, mạch lạc và dễ đọc.
     + Dùng khối trích dẫn (> **Lưu ý quan trọng:** ...) để bổ sung các điều kiện kèm theo hoặc trường hợp đặc biệt.""".strip()

FEW_SHOT_EXAMPLES = """=== VÍ DỤ MINH HỌA VĂN PHONG VÀ CẤU TRÚC PHẢN HỒI CHUẨN ===
Ví dụ 1 (Hỏi về điều kiện / quy định học vụ -> Trả lời tự nhiên bằng gạch đầu dòng):
Câu hỏi: "Sinh viên UET có được đăng ký học cải thiện điểm không? Điều kiện và quy tắc tính điểm thế nào?"
Ngữ cảnh mẫu: "[Tài liệu: Quy chế đào tạo đại học UET] Sinh viên được quyền đăng ký học lại học phần đã đạt điểm D, D+ để cải thiện điểm trung bình chung tích lũy. Điểm học phần mới sẽ thay thế cho điểm học phần cũ. Học phí tính theo số tín chỉ đăng ký học lại. Không áp dụng học cải thiện cho điểm C trở lên."
Câu trả lời mẫu:
Chào em, theo quy chế đào tạo của Trường Đại học Công nghệ (UET), sinh viên được phép đăng ký học cải thiện điểm với các quy định cụ thể như sau:

• **Điều kiện điểm số:** Chỉ áp dụng đối với các học phần đã đạt điểm **D** hoặc **D+**. Các học phần đã đạt từ điểm C trở lên không được phép đăng ký học cải thiện.
• **Quy tắc tính điểm GPA:** Điểm của lần học cải thiện sau sẽ thay thế hoàn toàn cho điểm của lần học trước và được dùng để tính điểm trung bình chung tích lũy.
• **Học phí:** Sinh viên nộp học phí theo số tín chỉ của học phần đăng ký học lại theo định mức hiện hành.

> **Lưu ý quan trọng:** Em nên cân nhắc kỹ lịch học và khối lượng tín chỉ trong kỳ để việc học cải thiện đạt kết quả cao nhất mà không ảnh hưởng tới tiến độ các học phần chính khóa nhé.

Ví dụ 2 (Bảng quy đổi chứng chỉ ngoại ngữ phức tạp -> Bắt buộc dùng Bảng Markdown):
Câu hỏi: "Chuẩn đầu ra ngoại ngữ tốt nghiệp tại UET quy định thế nào? Cần IELTS hay TOEIC bao nhiêu điểm?"
Ngữ cảnh mẫu: "[Tài liệu: Minh chứng chuẩn đầu ra trình độ ngoại ngữ được sử dụng để xét tốt nghiệp bậc đại học] Đối với CTĐT ngoại ngữ tiếng Anh: Bậc 3 (CTĐT chuẩn): IELTS 4.5; TOEFL 460 ITP / 42 iBT; TOEIC Reading 275 Listening 275 Speaking 120 Writing 120; Aptis B1 (General); VSTEP.3-5 (4.0). Bậc 4 (CTĐT CLC): IELTS 5.5; TOEFL 543 ITP / 72 iBT; TOEIC Reading 385 Listening 400 Speaking 160 Writing 150; Aptis B2 (General); VSTEP.3-5 (6.0)..."
Câu trả lời mẫu:
Chuẩn đầu ra trình độ tiếng Anh để xét tốt nghiệp đại học tại Trường Đại học Công nghệ (UET) được áp dụng theo Bảng quy đổi chứng chỉ cụ thể như sau:

### Bảng quy đổi chứng chỉ chuẩn đầu ra tiếng Anh UET
| Khung năng lực VN | CTĐT áp dụng | IELTS | TOEFL | TOEIC (4 kỹ năng) | VSTEP | Aptis |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Bậc 3** | CTĐT Chuẩn | **4.5** | 460 ITP / 42 iBT | R: 275, L: 275, S: 120, W: 120 | 4.0 | B1 |
| **Bậc 4** | CTĐT Chất lượng cao (CLC) | **5.5** | 543 ITP / 72 iBT | R: 385, L: 400, S: 160, W: 150 | 6.0 | B2 |

> **Lưu ý quan trọng:** Chứng chỉ ngoại ngữ nộp về Phòng Đào tạo phải còn hiệu lực tính đến thời điểm nộp. Sinh viên chưa có chứng chỉ quốc tế có thể đăng ký thi đánh giá năng lực ngoại ngữ VNU-TEST do Trường ĐH Ngoại ngữ - ĐHQGHN tổ chức.

Ví dụ 3 (Hỏi về định mức & thời hạn -> Trả lời súc tích bằng gạch đầu dòng):
Câu hỏi: "Quy định về số tín chỉ tối thiểu, tối đa và thời gian rút bớt học phần tại UET thế nào?"
Ngữ cảnh mẫu: "[Tài liệu: Quy chế đào tạo đại học UET] Trong mỗi học kỳ chính, sinh viên phải đăng ký tối thiểu 14 tín chỉ (đối với học lực bình thường) hoặc tối thiểu 10 tín chỉ (đối với sinh viên bị cảnh báo học tập). Số tín chỉ tối đa được đăng ký là 24 tín chỉ. Sinh viên được phép rút bớt học phần trong vòng 2 tuần đầu của học kỳ chính."
Câu trả lời mẫu:
Chào em, quy định về định mức tín chỉ và thời hạn rút bớt học phần trong học kỳ chính tại UET như sau:

• **Thời hạn rút bớt học phần:** Được thực hiện trong **2 tuần đầu** của học kỳ chính. Học phần được chấp nhận rút sẽ không ghi điểm vào bảng điểm và không phải đóng học phí.
• **Định mức số tín chỉ tối thiểu:**
  - Sinh viên học lực bình thường: Đăng ký tối thiểu **14 tín chỉ** / học kỳ chính.
  - Sinh viên đang bị cảnh báo học tập: Đăng ký tối thiểu **10 tín chỉ** / học kỳ chính.
• **Định mức số tín chỉ tối đa:** Được đăng ký tối đa **24 tín chỉ** / học kỳ chính.

Ví dụ 4 (Không có thông tin trong ngữ cảnh / Ngoài phạm vi):
Câu hỏi: "Ký túc xá Mễ Trì có bao nhiêu phòng?"
Ngữ cảnh mẫu: "[Tài liệu: Hướng dẫn đăng ký tín chỉ UET]..."
Câu trả lời mẫu:
Xin lỗi em, hiện tại trong các văn bản quy chế lưu trữ chưa có thông tin cụ thể về câu hỏi này. Em vui lòng liên hệ trực tiếp Phòng Đào tạo (Phòng 105-E3) hoặc Cổng thông tin đào tạo UET để được hướng dẫn chính xác nhất nhé."""

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
            score_info = f" [Điểm Cross-Encoder: {ctx.similarity_score:.3f}]" if ctx.similarity_score is not None else ""
            
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

        formatted_prompt = (
            f"{self.system_prompt}\n\n"
            f"{few_shot_section}"
            f"=== NGỮ CẢNH ĐƯỢC CUNG CẤP (TÀI LIỆU QUY CHẾ UET) ===\n"
            f"{context_block}\n"
            f"====================================================\n\n"
            f"CÂU HỎI CỦA NGƯỜI DÙNG: {query.query_text}\n\n"
            f"CÂU TRẢ LỜI CỦA BẠN (Tuân thủ nghiêm ngặt các quy tắc trên):"
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
