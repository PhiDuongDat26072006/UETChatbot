"""
scripts/jev_smoke.py - Smoke test kiểm thử 3 quyết định của JEV Decision Model v3.
  1. route (OUT_OF_SCOPE | QUALIFIED_QUESTION)
  2. check_supplement (với missing_info)
  3. evaluate_answer (chỉ query + answer)
Chạy trực tiếp: python scripts/jev_smoke.py
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.decision import JevDecisionClient
from src.utils.helpers import load_yaml_config


def main():
    print("=" * 70)
    print("🚀 SMOKE TEST: JEV DECISION MODEL v3 (3 MICRO-DECISIONS)")
    print("=" * 70)

    cfg = load_yaml_config()
    jev_cfg = cfg.get("jev", {})
    endpoint = os.getenv("JEV_ENDPOINT") or jev_cfg.get("endpoint")
    api_key = os.getenv("JEV_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    model = jev_cfg.get("model", "strands-decider-12B-gemma4-v1-2610")
    timeout_s = float(jev_cfg.get("timeout_s", 30))

    if "openrouter.ai" in (endpoint or "") and not api_key:
        print("⚠️  Chưa phát hiện biến môi trường OPENROUTER_API_KEY / JEV_API_KEY.")
        print("✅ Bỏ qua kiểm thử trực tiếp qua mạng (Skip gracefully).")
        return

    print(f"[*] Endpoint: {endpoint}")
    print(f"[*] Model: {model}")
    print(f"[*] Timeout: {timeout_s}s")

    client = JevDecisionClient(
        endpoint=endpoint,
        api_key=api_key,
        model=model,
        timeout_s=timeout_s,
    )

    # 1. Test Decision 1: route (3 nhãn)
    print("\n[1/3] Kiểm thử Quyết định 'route' (3 nhãn):")
    for q in [
        "Xin chào bạn, bạn là ai?",
        "Thời gian rút bớt học phần trong học kỳ chính là bao lâu?",
        "So sánh chuẩn đầu ra ngoại ngữ giữa hệ Chuẩn và CLC?",
        "Học bổng kỳ này cần điều kiện gì ạ?",
    ]:
        r = client.route(q)
        print(f"  - '{q}' -> Route: {r.value}")

    # 2. Test Decision 2: check_supplement (với missing_info)
    print("\n[2/3] Kiểm thử Quyết định 'check_supplement':")
    q_ambig = "Học bổng kỳ này cần điều kiện gì ạ?"
    last_ask = "Bạn đang học ngành nào và thuộc khóa tuyển sinh năm bao nhiêu?"
    missing_info = ["ngành", "khóa tuyển sinh"]
    supp_ok = ["Dạ em học ngành Công nghệ thông tin khóa QH-2023 ạ"]
    print(
        f"  - Bổ sung đủ: {supp_ok} -> Verdict: {client.check_supplement(q_ambig, last_ask, supp_ok, missing_info=missing_info).value}"
    )

    supp_bad = ["Em cảm ơn"]
    print(
        f"  - Bổ sung thiếu: {supp_bad} -> Verdict: {client.check_supplement(q_ambig, last_ask, supp_bad, missing_info=missing_info).value}"
    )

    # 3. Test Decision 3: evaluate_answer (chỉ query + answer)
    print("\n[3/3] Kiểm thử Quyết định 'evaluate_answer':")
    q_cmp = "So sánh chuẩn đầu ra ngoại ngữ giữa hệ Chuẩn và CLC UET?"
    good_ans = "Theo quy chế UET, chương trình Chuẩn yêu cầu ngoại ngữ Bậc 3, còn chương trình CLC yêu cầu Bậc 4 theo KNLNN 6 bậc."
    print(f"  - Câu trả lời đầy đủ -> Verdict: {client.evaluate_answer(q_cmp, good_ans).value}")

    partial_ans = "Theo quy chế UET, chương trình Chuẩn yêu cầu ngoại ngữ Bậc 3. Hiện chưa có thông tin về chương trình CLC."
    print(f"  - Câu trả lời thiếu 1 vế -> Verdict: {client.evaluate_answer(q_cmp, partial_ans).value}")

    print("\n" + "=" * 70)
    print("✅ HOÀN TẤT SMOKE TEST JEV v3!")
    print("=" * 70)


if __name__ == "__main__":
    main()
