"""
main.py - Điểm khởi chạy (Entry Point) của toàn bộ hệ thống UET Chatbot.
Hỗ trợ chạy CLI, kiểm tra kiến trúc nhóm hoặc chạy API Server.
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

from src.utils.helpers import get_logger

logger = get_logger("main")


def print_banner():
    banner = """
========================================================================
     🎓 UET CHATBOT - HỆ THỐNG TRA CỨU QUY CHẾ ĐÀO TẠO UET (VNU)
                 Kiến trúc RAG Modular cho Nhóm Phát Triển
========================================================================
    """
    print(banner)


def run_api(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True):
    """Khởi chạy API Server và Giao diện Web với FastAPI và Uvicorn."""
    try:
        import uvicorn
        import webbrowser
        import threading
        from src.interfaces.routes import create_app

        app = create_app()
        if app is None:
            print("Vui lòng cài đặt: pip install fastapi uvicorn")
            return

        url = f"http://{host}:{port}"
        print(f"\n🌐 GIAO DIỆN WEB LOCAL ĐANG CHẠY TẠI: {url}")
        print(f"📚 Tài liệu REST API Swagger:        {url}/docs")
        print("💡 Nhấn Ctrl+C để dừng server.\n")

        if open_browser:
            threading.Timer(1.2, lambda: webbrowser.open(url)).start()

        uvicorn.run(app, host=host, port=port)
    except ImportError:
        print("Lỗi: Cần cài đặt uvicorn để chạy server (pip install uvicorn)")


def run_evaluation():
    """Chạy đánh giá chất lượng RAG trên tập Benchmark QA."""
    print("\n📊 ĐANG CHẠY QUY TRÌNH ĐÁNH GIÁ CHẤT LƯỢNG RAG (EVALUATION)...")
    from src.evaluation.evaluator import UETEvaluator
    from src.pipeline import get_rag_pipeline

    dataset_path = BASE_DIR / "data" / "eval_data" / "test_qa_dataset.json"
    dataset = UETEvaluator.load_dataset(dataset_path)

    if not dataset:
        print(f"Không tìm thấy tập dữ liệu đánh giá tại: {dataset_path}")
        return

    # Lấy RAG Pipeline chính thức từ src/pipeline.py
    rag_pipeline = get_rag_pipeline()
    evaluator = UETEvaluator()
    report = evaluator.evaluate_pipeline(rag_pipeline, dataset)

    print("\n" + "=" * 65)
    print("📈 BÁO CÁO KẾT QUẢ ĐÁNH GIÁ (RAG EVALUATION REPORT)")
    print("=" * 65)
    print(f"Tổng số mẫu kiểm thử: {report.total_samples}")
    print("\nĐiểm số trung bình (Thang điểm 0.0 - 1.0):")
    for metric, score in report.average_metrics.items():
        print(f"  • {metric.replace('_', ' ').capitalize():<22}: {score:.3f}")

    print("\nChi tiết từng mẫu câu hỏi:")
    for idx, res in enumerate(report.results, 1):
        print(f"  [{idx}] Câu hỏi: {res.query}")
        print(f"      Trả lời: {res.generated_answer}")
        print(f"      Điểm số: {res.metrics}")
    print("=" * 65)


def run_cli():
    """Khởi chạy giao diện Chatbot trực tiếp trên Terminal với giao diện Rich."""
    from src.interfaces.cli import run_cli as _run_cli
    _run_cli()


def main():
    parser = argparse.ArgumentParser(description="UET Chatbot RAG System")
    parser.add_argument("--web", action="store_true", help="Khởi chạy giao diện Chatbot Web trên trình duyệt local")
    parser.add_argument("--cli", action="store_true", help="Khởi chạy giao diện Chatbot dòng lệnh trên Terminal")
    parser.add_argument("--eval", "--evaluate", action="store_true", dest="eval", help="Chạy đánh giá chất lượng RAG trên tập Benchmark QA")
    parser.add_argument("--host", default="127.0.0.1", help="Địa chỉ host cho Web UI")
    parser.add_argument("--port", type=int, default=8000, help="Cổng port cho Web UI")

    args = parser.parse_args()
    print_banner()

    if args.web:
        run_api(host=args.host, port=args.port, open_browser=True)
    elif args.cli:
        run_cli()
    elif args.eval:
        run_evaluation()
    else:
        print("📌 3 LỆNH CHÍNH ĐỂ SỬ DỤNG HỆ THỐNG:")
        print("  1. Khởi chạy Giao diện WEB:       python main.py --web    <-- (KHUYÊN DÙNG)")
        print("  2. Khởi chạy Chatbot trên CLI:    python main.py --cli    <-- (TERMINAL CHAT)")
        print("  3. Đánh giá chất lượng RAG:       python main.py --eval   <-- (BENCHMARK QA)")
        print("\n💡 Mẹo:")
        print("  • Chạy kiểm thử tự động: pytest")
        print("  • Xem kịch bản code mẫu: python examples/mock_pipeline_demo.py\n")


if __name__ == "__main__":
    main()