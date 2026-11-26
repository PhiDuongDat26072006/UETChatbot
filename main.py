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

    # 1. Nhóm Khởi chạy Chatbot (mặc định Hybrid; bật tùy chọn --graph / --agentic)
    run_group = parser.add_argument_group("1. Chạy Chatbot (--run)")
    run_group.add_argument("--run", action="store_true", help="Kích hoạt khởi chạy giao diện Chatbot")
    run_group.add_argument("--cli", action="store_true", help="Giao diện Chatbot dòng lệnh trên Terminal")
    run_group.add_argument("--web", action="store_true", help="Giao diện Chatbot Web trên trình duyệt local")
    run_group.add_argument("--graph", action="store_true", help="Bật nhánh GraphRAG kết hợp với Hybrid Search (mặc định: tắt)")
    run_group.add_argument("--agentic", action="store_true", help="Bật chế độ Agentic RAG LangGraph + JEV (mặc định: tắt, chạy Hybrid)")
    run_group.add_argument(
        "--llm",
        type=str,
        choices=["openrouter", "custom_api", "groq", "sambanova", "gemini"],
        default=None,
        help="Chọn nhanh LLM provider không cần sửa config.yaml (vd: --llm openrouter hoặc --llm custom_api)",
    )
    run_group.add_argument("--host", default="127.0.0.1", help="Địa chỉ host cho Web UI")
    run_group.add_argument("--port", type=int, default=8000, help="Cổng port cho Web UI")

    # 2. Nhóm Đánh giá cơ bản
    eval_group = parser.add_argument_group("2. Đánh giá cơ bản (--eval)")
    eval_group.add_argument("--eval", "--evaluate", action="store_true", dest="eval", help="Kích hoạt quy trình đánh giá chất lượng hệ thống")
    eval_group.add_argument("--e2e", action="store_true", help="Đánh giá toàn trình End-to-End RAG (hỗ trợ --graph / --agentic)")
    eval_group.add_argument("--retrieval", action="store_true", help="Đánh giá phân hệ Retrieval (Precision, Recall, MRR, NDCG)")
    eval_group.add_argument("--all", action="store_true", help="Chạy đánh giá toàn diện cả Retrieval và End-to-End")
    eval_group.add_argument("--use-judge", action="store_true", help="Bật LLM Judge (Structured JSON tiếng Việt)")
    eval_group.add_argument("--samples", "--num-samples", type=int, default=None, dest="samples", help="Số lượng mẫu kiểm thử (mặc định: toàn bộ)")
    eval_group.add_argument("--domain", type=str, default=None, help="Lọc theo khoa/lĩnh vực: fit, fet, fema, sae, handbook, tuyensinh, overview, all")
    eval_group.add_argument("--top_k", "--top-k", type=int, default=None, dest="top_k", help="Số lượng đoạn ngữ cảnh k (mặc định: lấy từ config.yaml)")
    eval_group.add_argument("--engine", type=str, choices=["vector", "lightrag", "hybrid", "auto"], default=None, help="Động cơ truy xuất: vector, lightrag, hybrid, auto (mặc định: lấy từ config.yaml)")
    eval_group.add_argument("--output", type=str, default=None, help="Đường dẫn file JSON để lưu kết quả đánh giá")

    # 3. Nhóm Đánh giá nâng cao (Benchmark 2 pha & tùy chọn chuyên sâu)
    eval_adv_group = parser.add_argument_group("3. Đánh giá nâng cao (Benchmark 2 pha & tùy chọn chuyên sâu)")
    eval_adv_group.add_argument("--level", type=str, choices=["doc", "chunk", "both"], default="both", help="Cấp độ đánh giá retrieval: doc, chunk, hoặc both")
    eval_adv_group.add_argument("--mode", type=str, choices=["live", "mock"], default="live", help="Chế độ đánh giá retrieval: 'live' hoặc 'mock'")
    eval_adv_group.add_argument("--dataset", type=str, default=None, help="Đường dẫn tới file dataset QA tùy chỉnh")
    eval_adv_group.add_argument("--generate-only", action="store_true", help="Chỉ chạy Pha 1 sinh snapshot predictions.json")
    eval_adv_group.add_argument("--score-from", type=str, default=None, help="Chạy Pha 2 chấm điểm từ snapshot predictions.json có sẵn")
    eval_adv_group.add_argument("--save-predictions", type=str, default=None, help="Đường dẫn lưu snapshot predictions.json")
    eval_adv_group.add_argument("--judge-provider", type=str, default=None, help="Ghi đè provider cho LLM Judge (groq, openrouter, sambanova, gemini)")
    eval_adv_group.add_argument("--judge-model", type=str, default=None, help="Ghi đè model cho LLM Judge")
    eval_adv_group.add_argument("--no-semantic", action="store_true", help="Tắt semantic cosine (bge-m3) khi chấm E2E offline")
    eval_adv_group.add_argument("--no-resume", action="store_true", help="Không tái sử dụng snapshot cũ khi chạy Pha 1")
    eval_adv_group.add_argument("--use-cache", action="store_true", help="Bật cache retrieval theo hash cấu hình (mặc định: tắt)")
    eval_adv_group.add_argument("--no-cache", action="store_true", help="Tắt cache retrieval (mặc định)")
    eval_adv_group.add_argument("--use-ragas", action="store_true", help="Sử dụng Ragas thay vì bộ đánh giá mặc định")

    # 4. Nhóm Nạp & lập chỉ mục dữ liệu đa chặng
    ingest_group = parser.add_argument_group("4. Nạp & lập chỉ mục dữ liệu (--ingest)")
    ingest_group.add_argument("--ingest", action="store_true", help="Kích hoạt quy trình điều phối và nạp dữ liệu qua các chặng")
    ingest_group.add_argument("--status", action="store_true", help="Xem báo cáo thống kê số lượng dữ liệu ở từng chặng")
    ingest_group.add_argument("--from", dest="from_stage", default="raw", choices=["raw", "processed", "chunked"], help="Chặng dữ liệu xuất phát (mặc định: raw)")
    ingest_group.add_argument("--to", dest="to_stage", default="vectordb", choices=["processed", "chunked", "vectordb", "indexed_data", "graph_data", "lightrag"], help="Chặng dữ liệu đích đến (mặc định: vectordb)")
    ingest_group.add_argument("--clear", "--reset", action="store_true", dest="clear", help="Xóa dữ liệu cũ ở chặng đích trước khi nạp lại")

    args = parser.parse_args()
    print_banner()

    import os

    def _apply_mode_flags() -> None:
        """Mặc định luôn chạy Hybrid (modular, tắt Graph); chỉ bật Agentic / Graph khi có cờ --agentic / --graph."""
        os.environ["RAG_MODE"] = "agentic" if args.agentic else "modular"
        os.environ["ENABLE_GRAPH"] = "1" if args.graph else "0"
        if args.llm:
            os.environ["LLM_PROVIDER"] = args.llm

    if args.run:
        _apply_mode_flags()
        if args.web:
            from src.interfaces.routes import run_web
            run_web(host=args.host, port=args.port, open_browser=True)
        elif args.cli:
            from src.interfaces.cli import run_cli
            run_cli(engine=args.engine)
        else:
            print("⚠️  Vui lòng chỉ định một trong hai hình thức trải nghiệm:")
            print("  1. Khởi chạy Giao diện WEB:      python main.py --run --web [--graph] [--agentic]")
            print("  2. Khởi chạy Chatbot trên CLI:   python main.py --run --cli [--graph] [--agentic] [--engine hybrid|lightrag|vector|auto]\n")

    elif args.eval:
        from src.utils.helpers import load_yaml_config
        cfg = load_yaml_config()
        configured_k = cfg.get("retrieval", {}).get("final_top_k", 10)
        eval_top_k = args.top_k if args.top_k is not None else configured_k
        out_p = Path(args.output) if args.output else None
        eval_dataset_target = args.dataset or cfg.get("paths", {}).get("eval_dataset_file")
        retrieval_use_cache = bool(args.use_cache)

        if getattr(args, "all", False):
            print("\n" + "=" * 75)
            print("🌟 KHỞI CHẠY ĐÁNH GIÁ TOÀN DIỆN CẢ RETRIEVAL VÀ END-TO-END RAG")
            print("=" * 75)
            from src.evaluation.evaluate_retrieval import run_eval_retrieval_cli
            from src.evaluation.evaluator import run_eval_e2e_cli

            ret_out = Path(args.output).with_name(f"{Path(args.output).stem}_retrieval.json") if args.output else None
            e2e_out = Path(args.output).with_name(f"{Path(args.output).stem}_e2e.json") if args.output else None

            print("\n[CHẶNG 1/2] ĐÁNH GIÁ PHÂN HỆ RETRIEVAL (DOC & CHUNK DUAL-LEVEL)")
            run_eval_retrieval_cli(
                eval_dir=eval_dataset_target,
                top_k=eval_top_k,
                output=ret_out,
                level=args.level,
                mode=args.mode,
                domain=args.domain,
                use_cache=retrieval_use_cache,
            )

            print("\n[CHẶNG 2/2] ĐÁNH GIÁ PHÂN HỆ END-TO-END GENERATION")
            _apply_mode_flags()
            run_eval_e2e_cli(
                dataset_path=eval_dataset_target,
                domain=args.domain,
                use_ragas=args.use_ragas,
                use_judge=args.use_judge,
                judge_provider=args.judge_provider,
                judge_model=args.judge_model,
                num_samples=args.samples,
                top_k=eval_top_k,
                output_path=e2e_out,
                generate_only=args.generate_only,
                score_from=args.score_from,
                save_predictions=args.save_predictions,
                enable_semantic=not args.no_semantic,
                resume=not args.no_resume,
            )
            print("\n" + "=" * 75)
            print("✅ ĐÃ HOÀN TẤT ĐÁNH GIÁ TOÀN DIỆN CẢ 2 PHÂN HỆ!")
            print("=" * 75 + "\n")

        elif args.e2e or args.score_from or args.generate_only or args.use_judge or args.use_ragas:
            _apply_mode_flags()
            from src.evaluation.evaluator import run_eval_e2e_cli
            run_eval_e2e_cli(
                dataset_path=eval_dataset_target,
                domain=args.domain,
                use_ragas=args.use_ragas,
                use_judge=args.use_judge,
                judge_provider=args.judge_provider,
                judge_model=args.judge_model,
                num_samples=args.samples,
                top_k=eval_top_k,
                output_path=out_p,
                generate_only=args.generate_only,
                score_from=args.score_from,
                save_predictions=args.save_predictions,
                enable_semantic=not args.no_semantic,
                resume=not args.no_resume,
            )
        else:
            # Mặc định khi chạy python main.py --eval hoặc --eval --retrieval: Giữ nguyên hành vi đọc config.yaml
            from src.evaluation.evaluate_retrieval import run_eval_retrieval_cli
            run_eval_retrieval_cli(
                eval_dir=eval_dataset_target,
                top_k=eval_top_k,
                output=out_p,
                level=args.level,
                mode=args.mode,
                domain=args.domain,
                use_cache=retrieval_use_cache,
            )

    elif args.ingest:
        if args.status:
            from src.pipeline import run_ingest_status_cli
            run_ingest_status_cli()
        else:
            from src.pipeline import run_ingest_cli
            run_ingest_cli(
                from_stage=args.from_stage,
                to_stage=args.to_stage,
                clear=args.clear,
                clear_existing=args.clear,
                domain=args.domain,
                max_chunks=args.samples,
            )

    else:
        print("📌 CÁC LỆNH CHÍNH ĐỂ SỬ DỤNG HỆ THỐNG:")
        print("  1. Khởi chạy Giao diện WEB (mặc định Hybrid): python main.py --run --web [--graph] [--agentic]")
        print("  2. Khởi chạy Chatbot CLI (mặc định Hybrid):")
        print("     • Hybrid thuần (BM25 + Vector):   python main.py --run --cli")
        print("     • Graph + Hybrid:                 python main.py --run --cli --graph")
        print("     • Agentic + Hybrid:               python main.py --run --cli --agentic")
        print("     • Graph + Hybrid + Agentic:       python main.py --run --cli --graph --agentic")
        print("  3. Đánh giá toàn diện cả 2 chặng:    python main.py --eval --all [--graph] [--agentic]")
        print("     • Đánh giá toàn diện mock nhanh:  python main.py --eval --all --mode mock")
        print("     • Đánh giá toàn diện theo domain: python main.py --eval --all --domain FET")
        print("  4. Đánh giá toàn trình RAG (E2E - mặc định Hybrid):")
        print("     • Hybrid thuần:                   python main.py --eval --e2e")
        print("     • Graph + Hybrid:                 python main.py --eval --e2e --graph")
        print("     • Agentic + Hybrid:               python main.py --eval --e2e --agentic")
        print("     • Graph + Hybrid + Agentic:       python main.py --eval --e2e --graph --agentic")
        print("     • Đánh giá theo domain/khoa:      python main.py --eval --e2e --domain FET")
        print("     • Thử nghiệm với số mẫu nhỏ:      python main.py --eval --e2e --samples 5")
        print("     • Đánh giá với top_k tùy ý:       python main.py --eval --e2e --top_k 10")
        print("     • Lưu kết quả đánh giá:           python main.py --eval --e2e --output eval_e2e.json")
        print("  5. Đánh giá phân hệ truy xuất:       python main.py --eval --retrieval")
        print("     • Đánh giá song song (mặc định):  python main.py --eval --retrieval --level both")
        print("     • Đánh giá mức Đoạn cắt (Chunk):  python main.py --eval --retrieval --level chunk")
        print("     • Đánh giá mức Tài liệu (Doc):    python main.py --eval --retrieval --level doc")
        print("     • Đánh giá nhanh mock mode:       python main.py --eval --retrieval --mode mock")
        print("     • Lọc theo domain/khoa:           python main.py --eval --retrieval --domain FET")
        print("     • Lưu kết quả retrieval:          python main.py --eval --retrieval --output eval_retrieval.json")
        print("  6. Thống kê trạng thái dữ liệu:      python main.py --ingest --status")
        print("  7. Nạp dữ liệu đa chặng (Ingest):    python main.py --ingest --from <stage> --to <stage>")
        print("\n💡 Quy trình nạp mới dữ liệu toàn diện (4 bước tuần tự):")
        print("  • Bước 1 (Chunking & ChromaDB):      python main.py --ingest --from processed --to vectordb --clear")
        print("  • Bước 2 (BM25 Sparse Index):        python main.py --ingest --from chunked --to indexed_data --clear")
        print("  • Bước 3 (LightRAG Knowledge Graph): python main.py --ingest --from processed --to lightrag")
        print("  • Bước 4 (Unified ChunkBridge):      python main.py --ingest --from chunked --to graph_data --clear")
        print("\n📌 Các thao tác đơn chặng:")
        print("  • Từ raw -> processed:               python main.py --ingest --from raw --to processed")
        print("  • Từ processed -> chunked:           python main.py --ingest --from processed --to chunked")
        print("  • Tái tạo nhanh BM25 từ chunked:     python main.py --ingest --from chunked --to indexed_data")
        print("  • Đồng bộ cầu nối GraphIndex:        python main.py --ingest --from chunked --to graph_data")
        print("  • Cập nhật đồ thị LightRAG (resume): python main.py --ingest --from processed --to lightrag\n")
        print("  9. Chatbot dòng lệnh với các động cơ (Engines):")
        print("     • Tự động chọn (mặc định):        python main.py --run --cli --engine auto")
        print("     • Động cơ lai (Graph + Vector):   python main.py --run --cli --engine hybrid")
        print("     • Động cơ đồ thị LightRAG:        python main.py --run --cli --engine lightrag")
        print("     • Động cơ Vector truyền thống:    python main.py --run --cli --engine vector\n")
        print(" 10. Lệnh chạy kiểm tra/đánh nhãn lại evaluate dataset:  python tests/eval_reviewer_app.py")

if __name__ == "__main__":
    main()
