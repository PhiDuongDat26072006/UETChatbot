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


def run_ingest(from_stage: str = "raw", to_stage: str = "vectordb"):
    """Khởi chạy quy trình điều phối và nạp dữ liệu đa chặng."""
    from src.pipeline import get_ingestion_pipeline

    print("\n" + "=" * 65)
    print(f"📦 BẮT ĐẦU ĐIỀU PHỐI DỮ LIỆU ĐA CHẶNG: [{from_stage.upper()}] ──> [{to_stage.upper()}]")
    print("=" * 65)

    pipeline = get_ingestion_pipeline()
    try:
        res = pipeline.run_stages(from_stage=from_stage, to_stage=to_stage)
    except Exception as e:
        print(f"\n❌ Lỗi trong quá trình điều phối dữ liệu: {e}")
        return

    print("\n" + "=" * 65)
    print("📊 BÁO CÁO KẾT QUẢ ĐIỀU PHỐI DỮ LIỆU (INGESTION REPORT)")
    print("=" * 65)
    transitions = res.get("transitions", {})
    if "raw_to_processed" in transitions:
        t = transitions["raw_to_processed"]
        print(f"  • Chặng [raw ──> processed]    : {t.get('processed', 0)} đã xử lý, {t.get('skipped', 0)} bỏ qua (trùng lặp)")
    if "processed_to_chunked" in transitions:
        t = transitions["processed_to_chunked"]
        print(f"  • Chặng [processed ──> chunked]: {t.get('processed_docs', 0)} tài liệu bóc tách -> {t.get('chunks_created', 0)} chunks mới, {t.get('skipped_docs', 0)} bỏ qua")
    if "chunked_to_vectordb" in transitions:
        t = transitions["chunked_to_vectordb"]
        print(f"  • Chặng [chunked ──> vectordb]  : {t.get('chunks_indexed', 0)} chunks nhúng mới, {t.get('skipped_chunks', 0)} bỏ qua (đã có trong DB)")
        print(f"    Tổng số vectors hiện có trong ChromaDB: {t.get('total_in_db', 0)}")

    print(f"\n⏱️  Tổng thời gian thực thi: {res.get('latency_seconds', 0)}s")
    print("=" * 65 + "\n")


def run_ingest_status():
    """Xem báo cáo thống kê số lượng dữ liệu ở từng chặng."""
    from src.pipeline import get_ingestion_pipeline

    pipeline = get_ingestion_pipeline()
    stats = pipeline.get_stage_stats()
    stages = stats.get("stages", {})
    manifest = stats.get("manifest_tracking", {})

    print("\n" + "=" * 65)
    print("📊 THỐNG KÊ TRẠNG THÁI DỮ LIỆU CÁC CHẶNG (INGESTION STAGES)")
    print("=" * 65)
    print(f"  1. Giai đoạn [RAW]      : {stages.get('raw', {}).get('file_count', 0)} tệp/thư mục ({stages.get('raw', {}).get('path')})")
    print(f"  2. Giai đoạn [PROCESSED]: {stages.get('processed', {}).get('document_count', 0)} tài liệu sạch ({stages.get('processed', {}).get('path')})")
    print(f"  3. Giai đoạn [CHUNKED]  : {stages.get('chunked', {}).get('chunk_count', 0)} chunks ({stages.get('chunked', {}).get('path')})")
    print(f"  4. Giai đoạn [VECTORDB] : {stages.get('vectordb', {}).get('vector_count', 0)} vectors ({stages.get('vectordb', {}).get('path')})")

    print("\n📋 Sổ cái theo dõi chống trùng lặp (Manifest Tracker):")
    print(f"  • Tổng tài liệu đang theo dõi : {manifest.get('total_tracked_documents', 0)}")
    comp = manifest.get("stages_completed", {})
    print(f"  • Hoàn thành chặng processed  : {comp.get('processed', 0)}")
    print(f"  • Hoàn thành chặng chunked    : {comp.get('chunked', 0)}")
    print(f"  • Hoàn thành chặng vectordb   : {comp.get('vectordb', 0)}")
    print(f"  • Lần cập nhật gần nhất       : {manifest.get('last_updated') or 'Chưa có'}")
    print("=" * 65 + "\n")


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

    # Nhóm tham số điều phối nạp dữ liệu đa chặng (Multi-Stage Ingestion)
    parser.add_argument("--ingest", action="store_true", help="Kích hoạt quy trình điều phối và nạp dữ liệu qua các chặng")
    parser.add_argument("--from", dest="from_stage", default="raw", choices=["raw", "processed", "chunked"], help="Chặng dữ liệu xuất phát (mặc định: raw)")
    parser.add_argument("--to", dest="to_stage", default="vectordb", choices=["processed", "chunked", "vectordb"], help="Chặng dữ liệu đích đến (mặc định: vectordb)")
    parser.add_argument("--ingest-status", action="store_true", help="Xem báo cáo thống kê số lượng dữ liệu ở từng chặng")

    args = parser.parse_args()
    print_banner()

    if args.web:
        run_api(host=args.host, port=args.port, open_browser=True)
    elif args.cli:
        run_cli()
    elif args.eval:
        run_evaluation()
    elif args.ingest_status:
        run_ingest_status()
    elif args.ingest:
        run_ingest(
            from_stage=args.from_stage,
            to_stage=args.to_stage,
        )
    else:
        print("📌 CÁC LỆNH CHÍNH ĐỂ SỬ DỤNG HỆ THỐNG:")
        print("  1. Khởi chạy Giao diện WEB:          python main.py --web")
        print("  2. Khởi chạy Chatbot trên CLI:       python main.py --cli")
        print("  3. Đánh giá chất lượng RAG:          python main.py --eval")
        print("  4. Thống kê trạng thái dữ liệu:      python main.py --ingest-status")
        print("  5. Nạp dữ liệu đa chặng (Ingest):    python main.py --ingest --from <stage> --to <stage>")
        print("\n💡 Ví dụ nạp dữ liệu:")
        print("  • Từ raw -> processed:               python main.py --ingest --from raw --to processed")
        print("  • Từ processed -> chunked:           python main.py --ingest --from processed --to chunked")
        print("  • Từ chunked -> vectordb:            python main.py --ingest --from chunked --to vectordb")
        print("  • Chạy toàn trình raw -> vectordb:   python main.py --ingest --from raw --to vectordb\n")


if __name__ == "__main__":
    main()