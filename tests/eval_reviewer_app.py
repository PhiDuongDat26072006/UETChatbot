"""
tests/eval_reviewer_app.py - Web App kiểm tra và gán nhãn Ground Truth cho UET Chatbot.

Sử dụng 100% THƯ VIỆN CHUẨN CỦA PYTHON (http.server) - KHÔNG CẦN CÀI FLASK HAY BẤT KỲ THƯ VIỆN NGOÀI NÀO.
Chạy được ngay trên mọi môi trường Python / Conda.

Cách chạy:
    python tests/eval_reviewer_app.py

Các tính năng:
- Hiển thị trực quan: ID câu hỏi, Query, Ground truth, Chunks mức độ 2 & 1 (nội dung lấy từ data/chunked_data).
- Gán nhãn: Đúng / Sai / Thiếu / None (mặc định là none).
- Phím tắt siêu tốc: Phím 1 (Đúng), Phím 2 (Sai), Phím 3 (Thiếu), Phím 0 (None), Mũi tên Trái/Phải (Trước/Sau).
- Tự động lưu trạng thái gán nhãn vào file data/eval_data/e2e_eval_100_labels.json (tự load lại ở các lần sau).
- Tìm kiếm, lọc theo trạng thái, ghi chú lý do sai/thiếu.
- Xuất dữ liệu ra Excel (.xlsx) và JSON kèm nhãn bất cứ lúc nào.
"""

import sys
import os
import glob
import json
import socket
import datetime
import threading
import webbrowser
import urllib.parse
import io
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import openpyxl
    from openpyxl.styles import Font, Alignment, PatternFill
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

# Thiết lập UTF-8 trên Windows console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "eval_reviewer.html"

# Đường dẫn dữ liệu
CHUNK_DIR = BASE_DIR / "data" / "chunked_data"
EVAL_FILE = BASE_DIR / "data" / "eval_data" / "e2e_eval_100_annotated.json"
LABELS_FILE = BASE_DIR / "data" / "eval_data" / "e2e_eval_100_labels.json"

# In-memory caches
CHUNKS_CACHE = {}
EVAL_ITEMS = []
LABELS_CACHE = {}
DATA_LOCK = threading.Lock()


def load_chunks():
    """Nạp tất cả chunks từ data/chunked_data/*.jsonl vào bộ nhớ."""
    global CHUNKS_CACHE
    CHUNKS_CACHE.clear()
    pattern = str(CHUNK_DIR / "*.jsonl")
    files = glob.glob(pattern)
    print(f"[INFO] Đang nạp chunks từ {len(files)} files trong {CHUNK_DIR}...")
    for p in files:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    c = json.loads(line)
                    cid = c.get("chunk_id")
                    if cid:
                        CHUNKS_CACHE[cid] = {
                            "chunk_id": cid,
                            "document_id": c.get("document_id"),
                            "text": c.get("text", ""),
                            "metadata": c.get("metadata", {}),
                        }
    print(f"[INFO] Đã nạp thành công {len(CHUNKS_CACHE):,} chunks.")


def load_eval_data():
    """Nạp danh sách câu hỏi kiểm thử từ file eval."""
    global EVAL_ITEMS
    if not EVAL_FILE.exists():
        print(f"[CẢNH BÁO] Không tìm thấy file {EVAL_FILE}")
        EVAL_ITEMS = []
        return

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        EVAL_ITEMS = json.load(f)
    print(f"[INFO] Đã nạp {len(EVAL_ITEMS)} câu hỏi kiểm thử từ {EVAL_FILE.name}")


def load_labels():
    """Nạp trạng thái gán nhãn đã lưu từ LABELS_FILE."""
    global LABELS_CACHE
    LABELS_CACHE.clear()
    if LABELS_FILE.exists():
        try:
            with open(LABELS_FILE, "r", encoding="utf-8") as f:
                LABELS_CACHE = json.load(f)
            print(f"[INFO] Đã nạp trạng thái {len(LABELS_CACHE)} nhãn từ {LABELS_FILE.name}")
        except Exception as e:
            print(f"[LỖI] Không thể đọc {LABELS_FILE}: {e}")
            LABELS_CACHE = {}
    else:
        print(f"[INFO] Chưa có file nhãn lưu trước đó. Mặc định tất cả là 'none'.")

    # Đảm bảo tất cả câu hỏi đều có mục nhãn mặc định là 'none'
    for it in EVAL_ITEMS:
        qid = it.get("id")
        if qid and qid not in LABELS_CACHE:
            LABELS_CACHE[qid] = {
                "label": "none",
                "note": "",
                "updated_at": None,
            }


def save_labels_to_file():
    """Lưu LABELS_CACHE ra file JSON."""
    try:
        LABELS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LABELS_FILE, "w", encoding="utf-8") as f:
            json.dump(LABELS_CACHE, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[LỖI] Không thể lưu nhãn vào {LABELS_FILE}: {e}")


def calculate_stats():
    """Tính toán thống kê số lượng nhãn hiện tại."""
    total = len(EVAL_ITEMS)
    dung = 0
    sai = 0
    thieu = 0
    none_count = 0

    for item in EVAL_ITEMS:
        qid = item.get("id")
        rec = LABELS_CACHE.get(qid, {})
        label = rec.get("label", "none") if isinstance(rec, dict) else (rec or "none")
        if label == "đúng":
            dung += 1
        elif label == "sai":
            sai += 1
        elif label == "thiếu":
            thieu += 1
        else:
            none_count += 1

    labeled = total - none_count
    progress = round((labeled / total * 100), 1) if total > 0 else 0

    return {
        "total": total,
        "dung": dung,
        "sai": sai,
        "thieu": thieu,
        "none": none_count,
        "labeled": labeled,
        "progress": progress,
    }


def find_free_port(start_port=5000):
    """Tìm một cổng mạng còn trống."""
    port = start_port
    while port < 6000:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
            port += 1
    return start_port


def generate_excel_bytes():
    """Tạo file Excel trả về dạng bytes."""
    if not HAS_OPENPYXL:
        raise RuntimeError("Thư viện openpyxl chưa được cài đặt để xuất Excel")

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Ground_Truth_Review"

    headers = [
        "ID câu hỏi",
        "Nội dung câu hỏi",
        "Ground truth",
        "Trạng thái gán nhãn",
        "Ghi chú kiểm tra",
        "ID chunk mức độ 2",
        "Nội dung chunk mức độ 2",
        "ID chunk mức độ 1",
        "Nội dung chunk mức độ 1",
    ]
    ws.append(headers)

    header_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    header_font = Font(bold=True, size=11)
    header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)

    for col_idx in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = header_align

    green_fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
    red_fill = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")
    yellow_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")

    for item in EVAL_ITEMS:
        qid = item.get("id")
        rec = LABELS_CACHE.get(qid, {})
        label = rec.get("label", "none") if isinstance(rec, dict) else (rec or "none")
        note = rec.get("note", "") if isinstance(rec, dict) else ""

        target_chunks = item.get("target_chunk_ids", {})
        lvl2_ids = []
        lvl1_ids = []
        if isinstance(target_chunks, dict):
            lvl2_ids = [cid for cid, lvl in target_chunks.items() if lvl == 2]
            lvl1_ids = [cid for cid, lvl in target_chunks.items() if lvl == 1]
        elif isinstance(target_chunks, list):
            lvl2_ids = target_chunks

        def format_chunk_cells(cids):
            if not cids:
                return "", ""
            ids_str = "\n".join(cids)
            parts = []
            for i, cid in enumerate(cids):
                c_data = CHUNKS_CACHE.get(cid, {})
                c_text = c_data.get("text", "[Không tìm thấy]")
                if len(cids) > 1:
                    parts.append(f"[{i+1}] ({cid})\n{c_text}")
                else:
                    parts.append(c_text)
            return ids_str, "\n\n---\n\n".join(parts)

        lvl2_id_str, lvl2_text_str = format_chunk_cells(lvl2_ids)
        lvl1_id_str, lvl1_text_str = format_chunk_cells(lvl1_ids)

        row = [
            qid,
            item.get("query", ""),
            item.get("ground_truth", ""),
            label,
            note,
            lvl2_id_str,
            lvl2_text_str,
            lvl1_id_str,
            lvl1_text_str,
        ]
        ws.append(row)

        row_idx = ws.max_row
        lbl_cell = ws.cell(row=row_idx, column=4)
        if label == "đúng":
            lbl_cell.fill = green_fill
        elif label == "sai":
            lbl_cell.fill = red_fill
        elif label == "thiếu":
            lbl_cell.fill = yellow_fill

    data_align = Alignment(vertical="top", wrap_text=True)
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=1, max_col=len(headers)):
        for cell in row:
            cell.alignment = data_align

    widths = {
        "A": 14, "B": 38, "C": 45, "D": 16,
        "E": 28, "F": 38, "G": 65, "H": 38, "I": 65,
    }
    for col_letter, w in widths.items():
        ws.column_dimensions[col_letter].width = w

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:I{ws.max_row}"

    out_stream = io.BytesIO()
    wb.save(out_stream)
    return out_stream.getvalue()


class EvalReviewerHandler(BaseHTTPRequestHandler):
    """Xử lý các request HTTP chuẩn của web app (chuẩn Python http.server)."""

    def log_message(self, format, *args):
        # Ẩn log truy cập tĩnh để console luôn sạch đẹp
        pass

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            if not TEMPLATE_PATH.exists():
                self.send_error(404, "Không tìm thấy file giao diện HTML")
                return
            with open(TEMPLATE_PATH, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        elif path == "/api/data":
            items = []
            with DATA_LOCK:
                load_eval_data()
                load_labels()
                for item in EVAL_ITEMS:
                    qid = item.get("id")
                    rec = LABELS_CACHE.get(qid, {})
                    if isinstance(rec, dict):
                        label = rec.get("label", "none")
                        note = rec.get("note", "")
                    else:
                        label = rec or "none"
                        note = ""

                    target_chunks = item.get("target_chunk_ids", {})
                    lvl2_chunks = []
                    lvl1_chunks = []

                    if isinstance(target_chunks, dict):
                        for cid, lvl in target_chunks.items():
                            chunk_obj = CHUNKS_CACHE.get(
                                cid,
                                {
                                    "chunk_id": cid,
                                    "text": "[Không tìm thấy nội dung chunk trong data/chunked_data]",
                                    "metadata": {},
                                },
                            )
                            if lvl == 2:
                                lvl2_chunks.append(chunk_obj)
                            elif lvl == 1:
                                lvl1_chunks.append(chunk_obj)
                    elif isinstance(target_chunks, list):
                        for cid in target_chunks:
                            chunk_obj = CHUNKS_CACHE.get(
                                cid,
                                {
                                    "chunk_id": cid,
                                    "text": "[Không tìm thấy nội dung chunk trong data/chunked_data]",
                                    "metadata": {},
                                },
                            )
                            lvl2_chunks.append(chunk_obj)

                    items.append({
                        "id": qid,
                        "query": item.get("query", ""),
                        "ground_truth": item.get("ground_truth", ""),
                        "type": item.get("type", ""),
                        "unit": item.get("unit", ""),
                        "category": item.get("category", ""),
                        "label": label,
                        "note": note,
                        "level_2_chunks": lvl2_chunks,
                        "level_1_chunks": lvl1_chunks,
                    })

                stats = calculate_stats()

            self.send_json({"items": items, "stats": stats})

        elif path == "/api/export_excel":
            try:
                excel_bytes = generate_excel_bytes()
                out_path = BASE_DIR / "data" / "eval_data" / "e2e_eval_100_reviewed.xlsx"
                with open(out_path, "wb") as f:
                    f.write(excel_bytes)

                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
                self.send_header(
                    "Content-Disposition",
                    'attachment; filename="e2e_eval_100_reviewed.xlsx"',
                )
                self.send_header("Content-Length", str(len(excel_bytes)))
                self.end_headers()
                self.wfile.write(excel_bytes)
            except Exception as e:
                self.send_json({"error": str(e)}, status=500)

        else:
            self.send_error(404, "Không tìm thấy đường dẫn")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            req_data = json.loads(body.decode("utf-8"))
        except Exception:
            req_data = {}

        if path == "/api/save_label":
            qid = req_data.get("id")
            label = req_data.get("label", "none")
            note = req_data.get("note", "")

            if not qid:
                self.send_json({"success": False, "error": "Thiếu question id"}, status=400)
                return

            with DATA_LOCK:
                LABELS_CACHE[qid] = {
                    "label": label,
                    "note": note,
                    "updated_at": datetime.datetime.now().isoformat(),
                }
                save_labels_to_file()
                stats = calculate_stats()

            self.send_json({"success": True, "stats": stats})

        elif path == "/api/export_json":
            out_items = []
            with DATA_LOCK:
                for item in EVAL_ITEMS:
                    item_copy = dict(item)
                    qid = item.get("id")
                    rec = LABELS_CACHE.get(qid, {})
                    if isinstance(rec, dict):
                        item_copy["review_label"] = rec.get("label", "none")
                        item_copy["review_note"] = rec.get("note", "")
                        item_copy["review_updated_at"] = rec.get("updated_at", "")
                    else:
                        item_copy["review_label"] = rec or "none"
                        item_copy["review_note"] = ""
                    out_items.append(item_copy)

            out_file = BASE_DIR / "data" / "eval_data" / "e2e_eval_100_annotated.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(out_items, f, ensure_ascii=False, indent=2)

            self.send_json({
                "success": True,
                "message": f"Đã lưu thành công ra file data/eval_data/{out_file.name}",
            })

        else:
            self.send_error(404, "Không tìm thấy đường dẫn")


def open_browser(port):
    """Mở trình duyệt sau khi server khởi động."""
    import time
    time.sleep(1.0)
    url = f"http://127.0.0.1:{port}"
    print(f"\n[INFO] Đang mở trình duyệt tại: {url}")
    webbrowser.open_new_tab(url)


def main():
    print("=" * 60)
    print(" UET RAG EVALUATION - GROUND TRUTH REVIEWER WEB APP")
    print("=" * 60)

    load_chunks()
    load_eval_data()
    load_labels()

    port = find_free_port(5000)
    server_address = ("127.0.0.1", port)
    httpd = ThreadingHTTPServer(server_address, EvalReviewerHandler)

    print(f"\n[INFO] Web App đang chạy tại: http://127.0.0.1:{port}")
    print("[INFO] Bấm Ctrl+C để dừng server khi hoàn tất.\n")

    threading.Thread(target=open_browser, args=(port,), daemon=True).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Đã dừng máy chủ thành công.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
