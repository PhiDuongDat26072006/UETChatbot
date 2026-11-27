"""Terminal CLI runners and status reporters for multi-stage data ingestion."""

from __future__ import annotations

from typing import Optional

from src.pipelines.factory import get_ingestion_pipeline


def run_ingest_cli(
    from_stage: str = "raw",
    to_stage: str = "vectordb",
    clear: bool = False,
    clear_existing: Optional[bool] = None,
    domain: Optional[str] = None,
    max_chunks: Optional[int] = None,
) -> None:
    """Khởi chạy quy trình điều phối và nạp dữ liệu đa chặng với báo cáo kết quả trên terminal."""
    if clear_existing is not None:
        clear = clear or clear_existing

    print("\n" + "=" * 65)
    if clear:
        print("🗑️  [--clear] Đã bật cờ làm mới: Dữ liệu cũ sẽ bị xóa và nạp lại toàn bộ!")
    print(f"📦 BẮT ĐẦU ĐIỀU PHỐI DỮ LIỆU ĐA CHẶNG: [{from_stage.upper()}] ──> [{to_stage.upper()}]")
    if domain or max_chunks:
        print(f"   • Phạm vi thử nghiệm: domain={domain or 'ALL'}, max_chunks={max_chunks or 'ALL'}")
    print("=" * 65)

    pipeline = get_ingestion_pipeline()
    try:
        res = pipeline.run_stages(
            from_stage=from_stage,
            to_stage=to_stage,
            clear=clear,
            clear_existing=clear,
            domain=domain,
            max_chunks=max_chunks,
        )
    except Exception as err:
        print(f"\n❌ Lỗi trong quá trình điều phối dữ liệu: {err}")
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
        faculties = t.get("faculties", {})
        faculties_str = f" ({len(faculties)} file khoa: {', '.join(faculties.keys())})" if faculties else ""
        print(
            f"  • Chặng [processed ──> chunked]: {t.get('processed_docs', 0)} tài liệu bóc tách -> "
            f"{t.get('chunks_created', 0)} chunks mới{faculties_str}, {t.get('skipped_docs', 0)} bỏ qua"
        )
    if "chunked_to_vectordb" in transitions:
        t = transitions["chunked_to_vectordb"]
        print(
            f"  • Chặng [chunked ──> vectordb]     : {t.get('chunks_indexed', 0)} chunks nhúng mới, "
            f"{t.get('skipped_chunks', 0)} bỏ qua (đã có trong DB)"
        )
        print(f"    Tổng số vectors hiện có trong ChromaDB: {t.get('total_in_db', 0)}")
    if "chunked_to_indexed_data" in transitions:
        t = transitions["chunked_to_indexed_data"]
        print(f"  • Chặng [chunked ──> indexed_data] : {t.get('chunks_indexed', 0)} chunks đã lập chỉ mục BM25")
        print(f"    File lưu trữ BM25 index: {t.get('sparse_file')}")
    if "chunked_to_graph_data" in transitions:
        t = transitions["chunked_to_graph_data"]
        lr_info_str = (
            f", LightRAG merged: +{t.get('lightrag_nodes_merged', 0)} nodes / +{t.get('lightrag_edges_merged', 0)} edges"
            if t.get("lightrag_nodes_merged") or t.get("lightrag_edges_merged")
            else ""
        )
        print(
            f"  • Chặng [chunked ──> graph_data]   : {t.get('chunks_indexed', 0)} chunks -> "
            f"{t.get('entities_count', 0)} entities, {t.get('triples_count', 0)} triples "
            f"(OpenRouter LLM calls: {t.get('llm_calls', 0)}{lr_info_str})"
        )
        print(f"    File lưu trữ Graph index: {t.get('graph_file')}")
    t_lr = transitions.get("processed_to_lightrag") or transitions.get("chunked_to_lightrag")
    if t_lr:
        lr_nodes = t_lr.get("final_nodes", t_lr.get("total_entities", 0))
        lr_edges = t_lr.get("final_edges", t_lr.get("total_relations", 0))
        lr_file = t_lr.get("graph_path") or t_lr.get("graph_file")
        status_note = " (tạm dừng sớm do lỗi LLM liên tiếp)" if t_lr.get("aborted_due_to_llm_errors") else ""
        print(f"  • Chặng [processed ──> lightrag]   : Đã xây dựng Knowledge Graph{status_note}")
        print(f"    Thực thể: {lr_nodes}, Quan hệ: {lr_edges}, File: {lr_file}")

    print(f"\n⏱️  Tổng thời gian thực thi: {res.get('latency_seconds', 0)}s")
    print("=" * 65 + "\n")


def run_ingest_status_cli() -> None:
    """Xem báo cáo thống kê số lượng dữ liệu ở từng chặng hiển thị trên terminal."""
    pipeline = get_ingestion_pipeline()
    stats = pipeline.get_stage_stats()
    stages = stats.get("stages", {})
    manifest = stats.get("manifest_tracking", {})

    print("\n" + "=" * 65)
    print("📊 THỐNG KÊ TRẠNG THÁI DỮ LIỆU CÁC CHẶNG (INGESTION STAGES)")
    print("=" * 65)
    print(f"  1. Giai đoạn [RAW]         : {stages.get('raw', {}).get('file_count', 0)} tệp/thư mục ({stages.get('raw', {}).get('path')})")
    print(f"  2. Giai đoạn [PROCESSED]   : {stages.get('processed', {}).get('document_count', 0)} tài liệu sạch ({stages.get('processed', {}).get('path')})")
    print(f"  3. Giai đoạn [CHUNKED]     : {stages.get('chunked', {}).get('chunk_count', 0)} chunks ({stages.get('chunked', {}).get('path')})")
    print(f"  4. Giai đoạn [VECTORDB]    : {stages.get('vectordb', {}).get('vector_count', 0)} vectors ({stages.get('vectordb', {}).get('path')})")
    idx_info = stages.get("indexed_data", {})
    status_str = f"Sẵn sàng ({idx_info.get('size_mb')} MB)" if idx_info.get("exists") else "Chưa tạo"
    print(f"  5. Giai đoạn [INDEXED_DATA]: {status_str} ({idx_info.get('path')})")
    graph_info = stages.get("graph_data", {})
    g_status_str = f"Sẵn sàng ({graph_info.get('size_mb')} MB)" if graph_info.get("exists") else "Chưa tạo"
    print(f"  6. Giai đoạn [GRAPH_DATA]  : {g_status_str} ({graph_info.get('path')})")

    lr_info = stages.get("lightrag", {})
    if lr_info.get("exists"):
        lr_status = f"Sẵn sàng ({lr_info.get('entity_count')} entities, {lr_info.get('relation_count')} relations, {lr_info.get('size_mb')} MB)"
    else:
        lr_status = "Chưa tạo (Sẵn sàng khởi chạy với: python main.py --ingest --to lightrag)"
    print(f"  7. Giai đoạn [LIGHTRAG]    : {lr_status} ({lr_info.get('path')})")

    print("\n📋 Sổ cái theo dõi chống trùng lặp (Manifest Tracker):")
    print(f"  • Tổng tài liệu đang theo dõi : {manifest.get('total_tracked_documents', 0)}")
    comp = manifest.get("stages_completed", {})
    print(f"  • Hoàn thành chặng processed  : {comp.get('processed', 0)}")
    print(f"  • Hoàn thành chặng chunked    : {comp.get('chunked', 0)}")
    print(f"  • Hoàn thành chặng vectordb   : {comp.get('vectordb', 0)}")
    print(f"  • Hoàn thành chặng indexed_data: {comp.get('indexed_data', 0)}")
    print(f"  • Hoàn thành chặng graph_data  : {comp.get('graph_data', 0)}")
    print(f"  • Lần cập nhật gần nhất       : {manifest.get('last_updated') or 'Chưa có'}")
    print("=" * 65 + "\n")
