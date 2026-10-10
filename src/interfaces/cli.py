"""
src/interfaces/cli.py - Giao diện dòng lệnh Terminal (CLI Mode) của UET Chatbot.
Sử dụng thư viện Rich để mang lại trải nghiệm chat terminal trực quan, màu sắc và chuyên nghiệp.
"""

from __future__ import annotations
import sys
from pathlib import Path

# Cấu hình UTF-8 cho console Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.base import UserQuery
from src.pipeline import get_rag_pipeline


def run_cli():
    """Khởi chạy giao diện Chatbot trực tiếp trên Terminal với giao diện Rich."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.markdown import Markdown
        from rich.table import Table
    except ImportError:
        print("Lỗi: Cần cài đặt thư viện 'rich' (pip install rich)")
        return

    console = Console()

    banner = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                 🎓 UET RAG CHATBOT (CLI MODE)                     ║
    ║        Trợ lý thông minh tra cứu học vụ & quy chế VNU-UET         ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    console.print(f"[bold cyan]{banner}[/bold cyan]")
    console.print(
        "[dim]Nhập câu hỏi của bạn. Gõ '[bold yellow]stats[/bold yellow]' để xem thống kê, '[bold yellow]clear[/bold yellow]' để xóa màn hình, '[bold yellow]exit[/bold yellow]' để thoát.[/dim]\n"
    )

    with console.status("[bold green]Đang khởi động cỗ máy RAG và tải cơ sở tri thức...[/bold green]"):
        rag_pipeline = get_rag_pipeline()

    db_count = rag_pipeline.vector_store.count()
    console.print(f"[green]✔ Hệ thống đã sẵn sàng với [bold]{db_count}[/bold] chunks vector tri thức.[/green]\n")

    while True:
        try:
            user_input = console.input("[bold green]Sinh viên UET:[/bold green] ").strip()

            if not user_input:
                continue

            cmd = user_input.lower()
            if cmd in ["exit", "quit", "q"]:
                console.print("[cyan]Tạm biệt! Chúc bạn học tập tốt tại UET.[/cyan]")
                break

            if cmd == "clear":
                console.clear()
                console.print(f"[bold cyan]{banner}[/bold cyan]")
                continue

            if cmd == "stats":
                table = Table(title="📊 Thống kê Hệ thống RAG UET")
                table.add_column("Thông số", style="cyan")
                table.add_column("Giá trị", style="green")
                table.add_row("Cơ sở dữ liệu Vector", "ChromaDB (database/vector_db/)")
                table.add_row("Số lượng văn bản đã lập chỉ mục", str(db_count))
                table.add_row("Mô hình LLM", rag_pipeline.llm.model_name)
                table.add_row("Số chiều Embedding", str(rag_pipeline.embedding_model.dimension))
                console.print(table)
                console.print()
                continue

            if cmd == "help":
                console.print(
                    """
[bold]Các lệnh có sẵn:[/bold]
  • [bold yellow]stats[/bold yellow] : Xem số lượng vector và cấu hình hệ thống.
  • [bold yellow]clear[/bold yellow] : Xóa màn hình console.
  • [bold yellow]exit[/bold yellow]  : Thoát chương trình.
  • Nhập bất kỳ câu hỏi nào về quy chế đào tạo, học bổng, tốt nghiệp, học vụ UET.
"""
                )
                continue

            # Thực hiện truy vấn RAG
            with console.status("[bold cyan]🔍 Đang tra cứu văn bản liên quan và sinh câu trả lời qua AI...[/bold cyan]"):
                query = UserQuery(query_text=user_input)
                response = rag_pipeline.query(query)

            console.print()
            console.print(
                Panel(
                    Markdown(response.answer),
                    title=f"🤖 Trợ Lý UET ({response.model_name or 'AI Engine'})",
                    border_style="green",
                )
            )

            # Hiển thị trích dẫn nguồn
            if response.sources:
                sources_table = Table(title="📚 Tài liệu nguồn trích dẫn", show_header=True, header_style="bold magenta")
                sources_table.add_column("#", style="dim", width=4)
                sources_table.add_column("Độ khớp", style="yellow", width=10)
                sources_table.add_column("Tên tài liệu / Văn bản", style="cyan")
                sources_table.add_column("Trích đoạn nội dung", style="white")

                for idx, src in enumerate(response.sources[:3], 1):
                    title = src.metadata.get("title") or src.metadata.get("source") or "Tài liệu UET"
                    snippet = src.text.replace("\n", " ").strip()
                    if len(snippet) > 90:
                        snippet = snippet[:87] + "..."
                    score_str = f"{int(src.similarity_score * 100)}%"
                    sources_table.add_row(str(idx), score_str, str(title), snippet)

                console.print(sources_table)
            console.print()

        except (KeyboardInterrupt, EOFError):
            console.print("\n[cyan]Tạm biệt! Hẹn gặp lại.[/cyan]")
            break
        except Exception as e:
            console.print(f"[bold red]❌ Đã xảy ra lỗi:[/bold red] {e}\n")


if __name__ == "__main__":
    run_cli()
