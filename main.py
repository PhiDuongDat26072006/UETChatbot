"""
main.py - Điểm khởi chạy (Entry Point & Command Router) của hệ thống UET Chatbot.
Điều phối thực thi: Web UI, CLI Chat, Đánh giá chất lượng và Nạp dữ liệu đa chặng.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Cấu hình UTF-8 cho console Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def print_banner():
    banner = """
========================================================================
     🎓 UET CHATBOT - HỆ THỐNG TRA CỨU QUY CHẾ ĐÀO TẠO UET (VNU)
                 Kiến trúc RAG Modular cho Nhóm Phát Triển
========================================================================
    """
    print(banner)


def main():
    parser = argparse.ArgumentParser(description="UET Chatbot RAG System")

    # Nhóm tham số Khởi chạy Chatbot & Trải nghiệm
    parser.add_argument("--run", action="store_true", help="Kích hoạt khởi chạy giao diện Chatbot")
    parser.add_argument("--cli", action="store_true", help="Giao diện Chatbot dòng lệnh trên Terminal")
    parser.add_argument("--web", action="store_true", help="Giao diện Chatbot Web trên trình duyệt local")
    parser.add_argument("--host", default="127.0.0.1", help="Địa chỉ host cho Web UI")
    parser.add_argument("--port", type=int, default=8000, help="Cổng port cho Web UI")

    # Nhóm tham số Đánh giá chất lượng (Evaluation)
    parser.add_argument("--eval", "--evaluate", action="store_true", dest="eval", help="Kích hoạt quy trình đánh giá chất lượng hệ thống")
    parser.add_argument("--e2e", action="store_true", help="Đánh giá toàn trình End-to-End RAG (RAG Triad metrics)")
    parser.add_argument("--retrieval", action="store_true", help="Đánh giá hiệu quả phân hệ Retrieval (Precision, Recall, MRR, NDCG)")

    # Nhóm tham số điều phối nạp dữ liệu đa chặng (Multi-Stage Ingestion)
    parser.add_argument("--ingest", action="store_true", help="Kích hoạt quy trình điều phối và nạp dữ liệu qua các chặng")
    parser.add_argument("--status", action="store_true", help="Xem báo cáo thống kê số lượng dữ liệu ở từng chặng")
    parser.add_argument("--from", dest="from_stage", default="raw", choices=["raw", "processed", "chunked"], help="Chặng dữ liệu xuất phát (mặc định: raw)")
    parser.add_argument("--to", dest="to_stage", default="vectordb", choices=["processed", "chunked", "vectordb", "indexed_data"], help="Chặng dữ liệu đích đến (mặc định: vectordb)")

    args = parser.parse_args()
    print_banner()

    if args.run:
        if args.web:
            from src.interfaces.routes import run_web
            run_web(host=args.host, port=args.port, open_browser=True)
        elif args.cli:
            from src.interfaces.cli import run_cli
            run_cli()
        else:
            print("⚠️  Vui lòng chỉ định một trong hai hình thức trải nghiệm:")
            print("  1. Khởi chạy Giao diện WEB:      python main.py --run --web")
            print("  2. Khởi chạy Chatbot trên CLI:   python main.py --run --cli\n")

    elif args.eval:
        if args.e2e:
            from src.evaluation.evaluator import run_eval_e2e_cli
            run_eval_e2e_cli()
        elif args.retrieval:
            from src.evaluation.evaluate_retrieval import run_eval_retrieval_cli
            run_eval_retrieval_cli()
        else:
            print("⚠️  Vui lòng chỉ định một trong hai hình thức đánh giá:")
            print("  1. Đánh giá toàn trình RAG:      python main.py --eval --e2e")
            print("  2. Đánh giá phân hệ truy xuất:   python main.py --eval --retrieval\n")

    elif args.ingest:
        if args.status:
            from src.pipeline import run_ingest_status_cli
            run_ingest_status_cli()
        else:
            from src.pipeline import run_ingest_cli
            run_ingest_cli(from_stage=args.from_stage, to_stage=args.to_stage)

    else:
        print("📌 CÁC LỆNH CHÍNH ĐỂ SỬ DỤNG HỆ THỐNG:")
        print("  1. Khởi chạy Giao diện WEB:          python main.py --run --web")
        print("  2. Khởi chạy Chatbot trên CLI:       python main.py --run --cli")
        print("  3. Đánh giá toàn trình RAG:          python main.py --eval --e2e")
        print("  4. Đánh giá phân hệ truy xuất:       python main.py --eval --retrieval")
        print("  5. Thống kê trạng thái dữ liệu:      python main.py --ingest --status")
        print("  6. Nạp dữ liệu đa chặng (Ingest):    python main.py --ingest --from <stage> --to <stage>")
        print("\n💡 Ví dụ nạp dữ liệu:")
        print("  • Từ raw -> processed:               python main.py --ingest --from raw --to processed")
        print("  • Từ processed -> chunked:           python main.py --ingest --from processed --to chunked")
        print("  • Từ chunked -> vectordb:            python main.py --ingest --from chunked --to vectordb")
        print("  • Từ chunked -> indexed_data:        python main.py --ingest --from chunked --to indexed_data")
        print("  • Chạy toàn trình raw -> vectordb:   python main.py --ingest --from raw --to vectordb\n")


if __name__ == "__main__":
    main()