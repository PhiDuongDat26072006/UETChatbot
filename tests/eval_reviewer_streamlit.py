"""
tests/eval_reviewer_streamlit.py - Phiên bản Streamlit để kiểm tra và gán nhãn Ground Truth.

Cách chạy:
    streamlit run tests/eval_reviewer_streamlit.py
"""

import sys
import glob
import json
import datetime
from pathlib import Path
import streamlit as st
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill

# Thiết lập UTF-8 trên Windows console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
CHUNK_DIR = BASE_DIR / "data" / "chunked_data"
EVAL_FILE = BASE_DIR / "data" / "eval_data" / "e2e_eval_100.json"
LABELS_FILE = BASE_DIR / "data" / "eval_data" / "e2e_eval_100_labels.json"

st.set_page_config(
    page_title="Ground Truth Reviewer - UET Chatbot",
    page_icon="🔍",
    layout="wide",
)


@st.cache_resource(show_spinner="Đang nạp 13,500+ chunks dữ liệu...")
def load_all_chunks():
    chunks = {}
    files = glob.glob(str(CHUNK_DIR / "*.jsonl"))
    for p in files:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    c = json.loads(line)
                    cid = c.get("chunk_id")
                    if cid:
                        chunks[cid] = c
    return chunks


def load_eval_data():
    if not EVAL_FILE.exists():
        st.error(f"Không tìm thấy file {EVAL_FILE}")
        return []
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_labels():
    labels = {}
    if LABELS_FILE.exists():
        try:
            with open(LABELS_FILE, "r", encoding="utf-8") as f:
                labels = json.load(f)
        except Exception:
            labels = {}
    return labels


def save_label(qid, label, note):
    labels = load_labels()
    labels[qid] = {
        "label": label,
        "note": note,
        "updated_at": datetime.datetime.now().isoformat(),
    }
    LABELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)


def main():
    chunks_cache = load_all_chunks()
    eval_items = load_eval_data()
    labels_dict = load_labels()

    st.title("🔍 Ground Truth Reviewer - UET Chatbot")
    st.caption("Công cụ kiểm tra & gán nhãn ground truth (data/eval_data)")

    # Tính toán thống kê
    total = len(eval_items)
    dung = sum(1 for it in eval_items if labels_dict.get(it["id"], {}).get("label") == "đúng")
    sai = sum(1 for it in eval_items if labels_dict.get(it["id"], {}).get("label") == "sai")
    thieu = sum(1 for it in eval_items if labels_dict.get(it["id"], {}).get("label") == "thiếu")
    none_cnt = total - dung - sai - thieu
    progress = round(((total - none_cnt) / total * 100), 1) if total > 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Tổng số câu hỏi", total)
    col2.metric("🟢 Đúng", dung)
    col3.metric("🔴 Sai", sai)
    col4.metric("🟠 Thiếu", thieu)
    col5.metric("⚪ Chưa gán (None)", none_cnt, f"{progress}% hoàn thành")

    st.progress(progress / 100.0)

    # Sidebar Filter
    st.sidebar.header("Bộ lọc & Danh sách")
    filter_choice = st.sidebar.selectbox(
        "Lọc theo trạng thái:",
        ["Tất cả", "Chưa gán (None)", "Đúng", "Sai", "Thiếu"]
    )
    search_keyword = st.sidebar.text_input("Tìm kiếm (ID hoặc từ khóa):", "").lower().strip()

    # Lọc danh sách câu hỏi
    filtered_items = []
    for it in eval_items:
        qid = it["id"]
        lbl = labels_dict.get(qid, {}).get("label", "none")
        if filter_choice == "Chưa gán (None)" and lbl != "none":
            continue
        if filter_choice == "Đúng" and lbl != "đúng":
            continue
        if filter_choice == "Sai" and lbl != "sai":
            continue
        if filter_choice == "Thiếu" and lbl != "thiếu":
            continue

        if search_keyword:
            text_match = (
                search_keyword in qid.lower()
                or search_keyword in it.get("query", "").lower()
                or search_keyword in it.get("ground_truth", "").lower()
            )
            if not text_match:
                continue
        filtered_items.append(it)

    if not filtered_items:
        st.warning("Không có câu hỏi nào khớp với bộ lọc!")
        return

    # Question selector
    item_labels = [
        f"{it['id']} [{labels_dict.get(it['id'], {}).get('label', 'none').upper()}]"
        for it in filtered_items
    ]
    selected_idx = st.sidebar.selectbox(
        f"Chọn câu hỏi ({len(filtered_items)} câu):",
        range(len(filtered_items)),
        format_func=lambda i: item_labels[i],
    )

    current_item = filtered_items[selected_idx]
    qid = current_item["id"]
    current_rec = labels_dict.get(qid, {})
    current_label = current_rec.get("label", "none")
    current_note = current_rec.get("note", "")

    # Layout chính: Review & Label
    st.divider()
    top_col1, top_col2 = st.columns([3, 1])
    with top_col1:
        st.subheader(f"📌 Câu hỏi: {qid}")
        st.markdown(f"**Chuyên mục:** `{current_item.get('category')}` | **Đơn vị:** `{current_item.get('unit')}` | **Loại:** `{current_item.get('type')}`")
    with top_col2:
        status_colors = {
            "đúng": "🟢 ĐÚNG",
            "sai": "🔴 SAI",
            "thiếu": "🟠 THIẾU",
            "none": "⚪ CHƯA GÁN",
        }
        st.markdown(f"### Trạng thái: {status_colors.get(current_label, '⚪ CHƯA GÁN')}")

    # Box câu hỏi & ground truth
    st.info(f"**Nội dung câu hỏi:**\n\n{current_item.get('query')}")

    st.success(f"**Ground Truth hiện tại:**\n\n{current_item.get('ground_truth')}")

    # Form gán nhãn
    with st.container():
        st.write("#### ✍️ Gán nhãn cho Ground Truth:")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("🟢 Đúng", use_container_width=True, type="primary" if current_label == "đúng" else "secondary"):
                save_label(qid, "đúng", current_note)
                st.rerun()
        with c2:
            if st.button("🔴 Sai", use_container_width=True, type="primary" if current_label == "sai" else "secondary"):
                save_label(qid, "sai", current_note)
                st.rerun()
        with c3:
            if st.button("🟠 Thiếu", use_container_width=True, type="primary" if current_label == "thiếu" else "secondary"):
                save_label(qid, "thiếu", current_note)
                st.rerun()
        with c4:
            if st.button("⚪ Bỏ chọn (None)", use_container_width=True, type="primary" if current_label == "none" else "secondary"):
                save_label(qid, "none", current_note)
                st.rerun()

        note_val = st.text_input("Ghi chú (lý do sai/thiếu thông tin gì...):", value=current_note, key=f"note_{qid}")
        if note_val != current_note:
            if st.button("Lưu ghi chú"):
                save_label(qid, current_label, note_val)
                st.success("Đã lưu ghi chú!")
                st.rerun()

    st.divider()

    # Hiển thị Chunks
    target_chunks = current_item.get("target_chunk_ids", {})
    lvl2_ids = [cid for cid, lvl in target_chunks.items() if lvl == 2]
    lvl1_ids = [cid for cid, lvl in target_chunks.items() if lvl == 1]

    tab1, tab2 = st.tabs([f"📦 Chunk Mức độ 2 ({len(lvl2_ids)})", f"📄 Chunk Mức độ 1 ({len(lvl1_ids)})"])

    with tab1:
        if not lvl2_ids:
            st.write("*(Không có chunk mức độ 2 cho câu hỏi này)*")
        for i, cid in enumerate(lvl2_ids):
            cdata = chunks_cache.get(cid, {})
            meta = cdata.get("metadata", {})
            title = meta.get("title") or meta.get("heading") or "Tài liệu"
            breadcrumb = meta.get("breadcrumb") or ""
            with st.expander(f"Chunk #{i+1}: {cid} ({breadcrumb} › {title})", expanded=True):
                st.code(cdata.get("text", "[Không tìm thấy nội dung chunk]"), language="markdown")

    with tab2:
        if not lvl1_ids:
            st.write("*(Không có chunk mức độ 1 cho câu hỏi này)*")
        for i, cid in enumerate(lvl1_ids):
            cdata = chunks_cache.get(cid, {})
            meta = cdata.get("metadata", {})
            title = meta.get("title") or meta.get("heading") or "Tài liệu"
            breadcrumb = meta.get("breadcrumb") or ""
            with st.expander(f"Chunk #{i+1}: {cid} ({breadcrumb} › {title})", expanded=True):
                st.code(cdata.get("text", "[Không tìm thấy nội dung chunk]"), language="markdown")


if __name__ == "__main__":
    main()
