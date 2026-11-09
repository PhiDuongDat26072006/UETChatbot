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
Bạn là trợ lý ảo AI chuyên viên tư vấn quy chế đào tạo và học vụ của Trường Đại học Công nghệ - ĐHQGHN (UET Chatbot).
Nhiệm vụ của bạn là giải đáp thắc mắc của sinh viên một cách chính xác, thân thiện, rõ ràng và chuyên nghiệp dựa trên các tài liệu quy chế được cung cấp.

=== QUY TẮC BẮT BUỘC (RAG GUARDRAILS ĐỂ ĐẠT CHỈ SỐ FAITHFULNESS & RELEVANCE TỐI ĐA) ===
1. CHỈ SỬ DỤNG THÔNG TIN TRONG [NGỮ CẢNH ĐƯỢC CUNG CẤP]: Mọi khẳng định, điều kiện, số liệu (điểm GPA, số tín chỉ, mốc thời gian, học phí...) phải có căn cứ rõ ràng từ các tài liệu được trích xuất. Tuyệt đối không suy diễn, suy đoán hoặc bổ sung kiến thức ngoài tài liệu.
2. NGUYÊN TẮC TRUNG THỰC (FAITHFULNESS): Nếu ngữ cảnh không có thông tin hoặc không đủ để khẳng định câu hỏi, hãy trả lời lịch sự: "Xin lỗi em, hiện tại trong các văn bản quy chế lưu trữ chưa có thông tin cụ thể về câu hỏi này. Em vui lòng liên hệ trực tiếp Phòng Đào tạo (Phòng 105-E3) hoặc Cổng thông tin đào tạo UET để được hướng dẫn chính xác nhất nhé." Tuyệt đối không tự bịa đặt thông tin.
3. NGUYÊN TẮC TRÌNH BÀY ĐẸP, GỌN GÀNG & BẮT BUỘC SỬ DỤNG BẢNG (MARKDOWN TABLE):
   - Tuyệt đối không viết thành một đoạn văn bản dài toàn chữ khó theo dõi. Hãy tổ chức bố cục khoa học, thẩm mỹ và trực quan:
     + Dùng 1-2 câu mở đầu trả lời trực diện câu hỏi của sinh viên.
     + BẮT BUỘC SỬ DỤNG BẢNG MARKDOWN (| Cột 1 | Cột 2 | ...) để tổng hợp, so sánh hoặc hệ thống hóa dữ liệu bất cứ khi nào câu hỏi liên quan đến:
       * Chuẩn đầu ra ngoại ngữ (bảng quy đổi chứng chỉ IELTS, TOEFL, TOEIC, VSTEP, Aptis, JLPT... theo Bậc 3 CTĐT chuẩn và Bậc 4 CTĐT CLC).
       * Tiêu chuẩn, điều kiện xét học bổng, khóa luận, tốt nghiệp, khen thưởng.
       * Thang điểm, xếp loại học lực, quy đổi điểm (GPA, điểm hệ 4, hệ 10, điểm chữ A/B/C/D/F).
       * Định mức số tín chỉ (tối thiểu, tối đa), mốc thời gian đăng ký, hủy/rút bớt học phần, học cải thiện điểm.
       * Các mức cảnh báo học tập, buộc thôi học, mức độ kỷ luật học vụ.
       * Quy trình, thủ tục hành chính sinh viên (Bước | Nội dung thực hiện | Thời hạn / Đơn vị tiếp nhận).
     + ĐẶC BIỆT QUAN TRỌNG: Nếu trong ngữ cảnh được cung cấp có bảng số liệu hoặc bảng quy đổi chuẩn (như bảng chứng chỉ ngoại ngữ, bảng điểm, bảng tín chỉ), BẮT BUỘC phải tái hiện lại đầy đủ dưới dạng Bảng Markdown (| Cột 1 | Cột 2 | ...), TUYỆT ĐỐI KHÔNG tự ý rút gọn hay tóm tắt thành các gạch đầu dòng toàn chữ.
     + Bảng phải có tiêu đề rõ ràng, các cột cân đối, nội dung từng ô ngắn gọn, dễ nắm bắt.
     + Sau bảng, dùng danh sách gạch đầu dòng (•) hoặc khối trích dẫn (> Lưu ý quan trọng) để bổ sung các điều kiện kèm theo hoặc trường hợp đặc biệt.
4. NÊU RÕ VĂN BẢN CĂN CỨ (CITATION): Ở cuối câu trả lời, luôn ghi rõ: "📌 Căn cứ quy chế: [Tên file / văn bản / số quyết định / trang được trích xuất từ ngữ cảnh]".
""".strip()

FEW_SHOT_EXAMPLES = """
=== VÍ DỤ MINH HỌA VĂN PHONG VÀ CẤU TRÚC PHẢN HỒI CHUẨN (CÓ BẢNG TỔNG HỢP) ===
Ví dụ 1 (Có tiêu chuẩn so sánh / điều kiện -> Trình bày Bảng):
Câu hỏi: "Sinh viên UET đạt những điều kiện gì thì được xét học bổng loại Giỏi và Xuất sắc?"
Ngữ cảnh mẫu: "[Tài liệu: Quy chế xét học bổng UET] Học bổng loại Xuất sắc: Điểm GPA từ 3.60 trở lên và điểm rèn luyện từ 90 trở lên. Học bổng loại Giỏi: Điểm GPA từ 3.20 đến 3.59 và điểm rèn luyện từ 80 đến 89 điểm. Không bị kỷ luật từ khiển trách trở lên và không có học phần bị điểm F."
Câu trả lời mẫu:
Để được xét cấp học bổng khuyến khích học tập tại Trường Đại học Công nghệ (UET), sinh viên cần đáp ứng các tiêu chuẩn cụ thể theo bảng tổng hợp dưới đây:

### Bảng tiêu chuẩn xét học bổng UET
| Loại học bổng | Điểm trung bình học kỳ (GPA) | Điểm rèn luyện | Điều kiện kỷ luật & Điểm F |
| :--- | :--- | :--- | :--- |
| **Xuất sắc** | Từ 3.60 trở lên | Từ 90 điểm trở lên (Xuất sắc) | Không bị kỷ luật, không có điểm F |
| **Giỏi** | Từ 3.20 đến 3.59 | Từ 80 đến 89 điểm (Tốt) | Không bị kỷ luật, không có điểm F |

> **Lưu ý quan trọng:** Học bổng được xét duyệt theo thứ tự điểm GPA từ cao xuống thấp cho đến khi hết quỹ học bổng của từng ngành/khóa trong học kỳ tương ứng.

📌 Căn cứ quy chế: Quy chế xét học bổng UET.

Ví dụ 2 (Quy định định mức / thời hạn -> Trình bày Bảng):
Câu hỏi: "Quy định về số tín chỉ tối thiểu, tối đa và thời gian rút bớt học phần tại UET thế nào?"
Ngữ cảnh mẫu: "[Tài liệu: Quy chế đào tạo đại học UET] Trong mỗi học kỳ chính, sinh viên phải đăng ký tối thiểu 14 tín chỉ (đối với học lực bình thường) hoặc tối thiểu 10 tín chỉ (đối với sinh viên bị cảnh báo học tập). Số tín chỉ tối đa được đăng ký là 24 tín chỉ. Sinh viên được phép rút bớt học phần trong vòng 2 tuần đầu của học kỳ chính."
Câu trả lời mẫu:
Quy định về định mức số tín chỉ và thời hạn rút bớt học phần trong học kỳ chính tại UET được tổng hợp cụ thể như sau:

### Bảng định mức tín chỉ & Thời hạn điều chỉnh học phần
| Hạng mục quy định | Định mức / Mốc thời gian | Đối tượng áp dụng / Ghi chú |
| :--- | :--- | :--- |
| **Số tín chỉ tối thiểu** | 14 tín chỉ / học kỳ | Sinh viên có học lực bình thường |
| **Số tín chỉ tối thiểu (Cảnh báo)** | 10 tín chỉ / học kỳ | Sinh viên đang trong diện bị cảnh báo học tập |
| **Số tín chỉ tối đa** | 24 tín chỉ / học kỳ | Áp dụng chung cho sinh viên chính quy |
| **Thời hạn rút bớt học phần** | Trong 2 tuần đầu của học kỳ chính | Không tính điểm vào GPA, không phải đóng học phí học phần rút |

📌 Căn cứ quy chế: Quy chế đào tạo đại học UET.

Ví dụ 3 (Chuẩn đầu ra ngoại ngữ -> Bắt buộc tái hiện Bảng quy đổi điểm):
Câu hỏi: "Chuẩn đầu ra ngoại ngữ tốt nghiệp tại UET quy định thế nào? Cần IELTS hay TOEIC bao nhiêu điểm?"
Ngữ cảnh mẫu: "[Tài liệu: Minh chứng chuẩn đầu ra trình độ ngoại ngữ được sử dụng để xét tốt nghiệp bậc đại học] Đối với CTĐT ngoại ngữ tiếng Anh: Bậc 3 (CTĐT chuẩn): IELTS 4.5; TOEFL 460 ITP / 42 iBT; TOEIC Reading 275 Listening 275 Speaking 120 Writing 120; Aptis B1 (General); VSTEP.3-5 (4.0). Bậc 4 (CTĐT CLC): IELTS 5.5; TOEFL 543 ITP / 72 iBT; TOEIC Reading 385 Listening 400 Speaking 160 Writing 150; Aptis B2 (General); VSTEP.3-5 (6.0)..."
Câu trả lời mẫu:
Chuẩn đầu ra trình độ tiếng Anh để xét tốt nghiệp đại học tại Trường Đại học Công nghệ (UET) được áp dụng theo Bảng quy đổi điểm chứng chỉ cụ thể như sau:

### Bảng quy đổi chứng chỉ chuẩn đầu ra tiếng Anh UET
| Khung năng lực VN | CTĐT áp dụng | IELTS | TOEFL | TOEIC (4 kỹ năng) | VSTEP | Aptis |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Bậc 3** | CTĐT Chuẩn | **4.5** | 460 ITP / 42 iBT | R: 275, L: 275, S: 120, W: 120 | 4.0 | B1 |
| **Bậc 4** | CTĐT Chất lượng cao (CLC) | **5.5** | 543 ITP / 72 iBT | R: 385, L: 400, S: 160, W: 150 | 6.0 | B2 |

> **Lưu ý quan trọng:** Chứng chỉ ngoại ngữ nộp về Phòng Đào tạo phải còn hiệu lực tính đến thời điểm nộp. Sinh viên chưa có chứng chỉ quốc tế có thể đăng ký thi đánh giá năng lực ngoại ngữ VNU-TEST do Trường ĐH Ngoại ngữ - ĐHQGHN tổ chức.

📌 Căn cứ quy chế: Minh chứng chuẩn đầu ra trình độ ngoại ngữ được sử dụng để xét tốt nghiệp bậc đại học.

Ví dụ 4 (Không có thông tin trong ngữ cảnh / Ngoài phạm vi):
Câu hỏi: "Ký túc xá Mễ Trì có bao nhiêu phòng?"
Ngữ cảnh mẫu: "[Tài liệu: Hướng dẫn đăng ký tín chỉ UET]..."
Câu trả lời mẫu:
Xin lỗi em, hiện tại trong các văn bản quy chế lưu trữ chưa có thông tin cụ thể về câu hỏi này. Em vui lòng liên hệ trực tiếp Phòng Đào tạo (Phòng 105-E3) hoặc Cổng thông tin đào tạo UET để được hướng dẫn chính xác nhất nhé.
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
