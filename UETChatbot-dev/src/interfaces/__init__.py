"""
src/interfaces - Tầng giao diện người dùng của UET Chatbot.
Bao gồm:
- Giao diện Web & REST API (routes.py + static/)
- Giao diện dòng lệnh Terminal (cli.py)
"""
from src.interfaces.routes import create_app, ChatRequest, ChatResponse
from src.interfaces.cli import run_cli

__all__ = ["create_app", "ChatRequest", "ChatResponse", "run_cli"]
