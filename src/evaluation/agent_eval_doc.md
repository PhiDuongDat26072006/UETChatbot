# Tài Liệu Kỹ Thuật: Đánh Giá Quyết Định Agentic RAG (JEV / LLM)

Tài liệu này đặc tả quy trình và các chỉ số đánh giá các quyết định trung gian của Agentic RAG v3 (`router_node`, `answer_eval_node`, và vòng lặp `followup_query_node`) trong `src/evaluation/agent_eval.py`.

---

## 1. Phạm Vi & Kiến Trúc 2 Pha

Bộ đánh giá giữ nguyên kiến trúc 2 pha của `src/evaluation/evaluator.py`:
- **Pha 1 (`generate_predictions`):** Khi chạy với `UETAgenticRAGPipeline`, evaluator tự bật tạm `agent_graph.trace_verbose = True` (và khôi phục lại `False` sau khi xong) để lưu đầy đủ khối `agent` vào snapshot (`route`, `status`, `retry`, `fail_reason`, `missing_info`, `used_queries`, `trace` gồm `round`, `chunk_ids`, `selected_chunk_ids`, `answer_text`, `noul_prob`, `parse_fallback`).
- **Pha 2 (`score_predictions`):** Chấm 100% cục bộ (0 gọi JEV, 0 chạy lại RAG) từ snapshot, đồng thời loại bỏ các ghi chú template (`answer_incomplete_note`, `ambiguous_followup_note`) trước khi chấm `offline_metrics` qua `strip_agent_notes()`.

Snapshot modular cũ (không có khối `agent`) không sinh mục `agent_metrics` để giữ tương thích ngược.

---

## 2. Danh Mục Chỉ Số (`report["agent_metrics"]`)

### 2.1. Router JEV (`agent_metrics.router`)
- **`route_distribution`:** Phân bố 3 nhãn `OUT_OF_SCOPE`, `AMBIGUOUS`, `QUALIFIED_QUESTION`.
- **`false_oos_rate` (`false_oos_count / in_domain_total`):** Chỉ số quan trọng nhất — tỉ lệ câu **in-domain** bị router chặn nhầm thành `OUT_OF_SCOPE` (gắn nhãn lỗi `ROUTER_FALSE_OOS` trong `error_taxonomy`).
- **`ood_caught_at_router_rate`:** Tỉ lệ câu OOD bị bắt ngay tại router (chỉ mang tính tham khảo; với câu OOD không có `expected_route` tường minh, cho phép đi qua RAG miễn câu trả lời cuối vẫn từ chối đúng).
- **`parse_fallback_count`:** Số lần JEV trả về nhãn lạ ngoài enum khiến client âm thầm fallback sang `QUALIFIED_QUESTION`.
- **`explicit_accuracy`, `confusion_matrix`, `per_class_metrics`:** Tính trên các mẫu có `expected_route` tường minh (ví dụ `data/golden_questions.json`, tự map nhãn cũ `SINGLE_HOP`/`MULTI_HOP` → `QUALIFIED_QUESTION`).

### 2.2. Answer Eval JEV (`agent_metrics.answer_eval`)
- **Nhãn tham chiếu:** Ưu tiên LLM Judge (`raw_scores.answer_correctness >= 4` → `SUFFICIENT`), fallback sang offline (`offline_metrics.answer_correctness >= 0.70`) khi không bật `--use-judge`. Bỏ qua câu OOD, lỗi sinh, hoặc nhánh `out_of_scope`/`fail`.
- **Chỉ số phân loại:** `confusion_matrix`, `accuracy`, `precision`, `recall`, `f1`.
- **`false_accept_rate`:** Tỉ lệ câu sai/thiếu nhưng JEV vẫn cho qua (`SUFFICIENT`). Tách riêng 2 nhóm:
  - `false_accept_incomplete_count`: câu thiếu ý — lỗi thực sự của JEV.
  - `false_accept_hallucination_count`: câu bịa đặt — **giới hạn thiết kế** do `answer_eval_node` không nhìn thấy chunk.
- **`false_reject_rate`:** Tỉ lệ câu đã đủ nhưng JEV vẫn đánh `INSUFFICIENT` (gây tốn vòng lặp).
- **`roc_auc` & `threshold_sweep` (`0.3 -> 0.8`):** Dùng `noul_prob` đã lưu để chọn ngưỡng tối ưu mà không cần chạy lại pipeline. Cột `EstRetries` ước lượng cận dưới số vòng retry trên chuỗi `noul_prob` quan sát được.

### 2.3. Thống Kê Luồng (`agent_metrics.trajectory`)
- `status_distribution` (`ok`, `partial`, `out_of_scope`, `fail`) và `fail_reason_counts` (`jev_unavailable`, `llm_unavailable`).
- `retry_distribution`, `avg_retries_rag`, `retry_triggered_rate`.
- `avg_jev_calls`, `avg_llm_calls`, `avg_retrieval_calls`.
- `node_latency_avg_s` và `latency_by_route_s`.

### 2.4. Vòng Lặp Follow-up (`agent_metrics.loop`) — 4 Metric Cốt Lõi
Chỉ xuất hiện khi snapshot có ít nhất 1 mẫu in-domain có `retry >= 1` và dữ liệu theo vòng:

| # | Metric | Ý nghĩa | Tham số cần cân nhắc |
|---|---|---|---|
| 1 | `gold_recall_round0` → `gold_recall_final` | Tỉ lệ chunk vàng có trong `selected_chunk_ids` ở vòng 0 so với vòng cuối | Follow-up có giúp retrieval không? |
| 2 | `gold_evicted_rate` | % mẫu từng tìm thấy chunk vàng nhưng bị loại khỏi ngữ cảnh cuối do giới hạn `max_chunks = 8` | Tăng `max_chunks` hoặc sửa `_select_accumulated_contexts` |
| 3 | `win_rate`, `harm_rate`, `avg_delta` | % mẫu vòng lặp cải thiện / làm tệ đi câu trả lời (ngưỡng `±0.05` offline hoặc `±1` Likert nếu bật `--judge-rounds`) | Có nên giữ vòng lặp không? |
| 4 | `correctness_by_round` & `partial_best_answer_accuracy` | Điểm trung bình theo vòng (`r0, r1, r2, r3`) và tỉ lệ chọn đúng vòng tốt nhất ở nhóm `partial` theo `noul_prob` | Chỉnh `max_retries`; kiểm chứng cơ chế chọn `best_answer` |

**Cờ theo mẫu (`samples[i].agent_flags`):**
- Giai đoạn A: `ROUTER_FALSE_OOS`, `ROUTER_PARSE_FALLBACK`, `EVAL_FALSE_ACCEPT`, `EVAL_FALSE_REJECT`, `PARTIAL_EXHAUSTED`, `AGENT_SERVICE_FAIL`.
- Giai đoạn B: `LOOP_HARMED`, `GOLD_EVICTED`.

---

## 3. Hướng Dẫn Chạy

Chạy từ thư mục gốc `UET_chatbot`:

```powershell
# 1. Pha 1: sinh snapshot agentic (tự lưu trace đầy đủ theo vòng)
.venv\Scripts\python.exe src/evaluation/evaluator.py --agentic --generate-only `
    --save-predictions data/eval_data/agentic_predictions.json

# 2. Pha 2 (offline, 0 API): chấm toàn bộ metrics E2E + Agent (A + B)
.venv\Scripts\python.exe src/evaluation/evaluator.py `
    --score-from data/eval_data/agentic_predictions.json `
    --output data/eval_data/agentic_eval_offline.json

# 3. Pha 2 (có LLM Judge + tùy chọn chấm thêm câu vòng 0 bằng --judge-rounds)
.venv\Scripts\python.exe src/evaluation/evaluator.py `
    --score-from data/eval_data/agentic_predictions.json `
    --use-judge --judge-rounds `
    --output data/eval_data/agentic_eval_judge.json

# 4. Đánh giá riêng JEV Router trên 35 câu golden (không chạy RAG)
.venv\Scripts\python.exe src/evaluation/evaluator.py --router-eval data/golden_questions.json `
    --output data/eval_data/router_eval_results.json
```

---

## 4. Hạn Chế Cần Lưu Ý Khi Đọc Kết Quả

1. **`answer_eval_node` không nhìn thấy ngữ cảnh (chunk):** JEV chỉ đọc `(query, answer)` nên không thể phát hiện bịa đặt có hình thức trôi chảy. Luôn xem `false_accept_incomplete_count` tách biệt với `false_accept_hallucination_count`.
2. **Số lượng mẫu OOD trong bộ 100 câu rất nhỏ (4 câu):** Không đủ độ tin cậy thống kê cho router OOD; dùng `--router-eval data/golden_questions.json` để đánh giá router đầy đủ 3 nhãn (`OUT_OF_SCOPE`, `AMBIGUOUS`, `QUALIFIED_QUESTION`).
3. **Ngữ cảnh khi bật `--judge-rounds`:** Snapshot chỉ lưu toàn văn text của `retrieved_contexts` cuối. Nếu chunk của vòng 0 bị đẩy ra khỏi ngữ cảnh cuối, judge chấm vòng 0 sẽ fallback sang ngữ cảnh cuối.
4. **Điểm offline là ước lượng gần đúng:** Khi không bật `--use-judge` / `--judge-rounds`, nhãn tham chiếu `answer_eval` và đường cong `correctness_by_round` dựa trên offline `answer_correctness` (từ vựng + số liệu + semantic `bge-m3`).
5. **Tách template ghi chú chỉ áp dụng cho offline:** Giữ nguyên đầu vào cho LLM Judge (rubric judge đã dặn không trừ điểm khối lưu ý) để tái sử dụng cache judge cũ.
