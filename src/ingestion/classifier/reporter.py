"""Báo cáo và hiển thị bảng tổng hợp phân loại endpoints toàn trường."""
from __future__ import annotations

from .taxonomy import Category


def print_classification_summary(stats: list[dict], elapsed_time: float = None):
    """In bảng tổng hợp phân loại chuyên nghiệp theo chuẩn dự án."""
    print("\n" + "=" * 90)
    print("[✓] BẢNG TỔNG HỢP PHÂN LOẠI ENDPOINTS TOÀN TRƯỜNG ĐẠI HỌC CÔNG NGHỆ (UET)")
    if elapsed_time:
        print(f"    Thời gian hoàn thành: {elapsed_time} giây")
    print("=" * 90)

    header = f"{'Domain':<22} | {'Tổng':<5} | {'DAO_TAO':<7} | {'TUYEN_SINH':<10} | {'QUY_CHE':<7} | {'NC_HOPTAC':<9} | {'CAN_BO':<6} | {'TIN_TUC':<7} | {'KHAC':<5}"
    print(header)
    print("-" * 90)

    totals = {cat.value: 0 for cat in Category}
    grand_total = 0

    for s in stats:
        folder = s["folder"]
        tot = s["total"]
        grand_total += tot
        c = s["categories"]
        for cat in Category:
            totals[cat.value] += c.get(cat.value, 0)

        line = (
            f"{folder:<22} | {tot:<5} | "
            f"{c.get(Category.DAO_TAO.value, 0):<7} | "
            f"{c.get(Category.TUYEN_SINH.value, 0):<10} | "
            f"{c.get(Category.QUY_CHE_BIEU_MAU.value, 0):<7} | "
            f"{c.get(Category.NGHIEN_CUU_HOP_TAC.value, 0):<9} | "
            f"{c.get(Category.CAN_BO_GIANG_VIEN.value, 0):<6} | "
            f"{c.get(Category.TIN_TUC_SU_KIEN.value, 0):<7} | "
            f"{c.get(Category.KHAC.value, 0):<5}"
        )
        print(line)

    print("-" * 90)
    tot_line = (
        f"{'TỔNG CỘNG':<22} | {grand_total:<5} | "
        f"{totals[Category.DAO_TAO.value]:<7} | "
        f"{totals[Category.TUYEN_SINH.value]:<10} | "
        f"{totals[Category.QUY_CHE_BIEU_MAU.value]:<7} | "
        f"{totals[Category.NGHIEN_CUU_HOP_TAC.value]:<9} | "
        f"{totals[Category.CAN_BO_GIANG_VIEN.value]:<6} | "
        f"{totals[Category.TIN_TUC_SU_KIEN.value]:<7} | "
        f"{totals[Category.KHAC.value]:<5}"
    )
    print(tot_line)
    print("=" * 90)
