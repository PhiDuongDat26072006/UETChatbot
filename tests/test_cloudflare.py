import re
import subprocess
import time

# Khởi chạy ngầm
tunnel_log = open("tunnel.log", "w", encoding="utf-8")
tunnel_process = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://127.0.0.1:5000"],
    stdout=tunnel_log,
    stderr=subprocess.STDOUT,
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
)

print(f"Tunnel PID: {tunnel_process.pid}. Đang lấy link public...")

# Đợi 5-10 giây để đọc link từ file log
for _ in range(15):
    time.sleep(1)
    with open("tunnel.log", "r", encoding="utf-8", errors="ignore") as f:
        match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", f.read())
        if match:
            print(f"\n=> LINK CỦA BẠN: {match.group(0)}\n")
            break