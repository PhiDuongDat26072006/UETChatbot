import requests

KEY = ""
WORKSPACE_2_ID = "8782a874-3ed4-41be-b6ab-c3c76f39a111"  # Thay bằng workspace_id trên URL của Workspace thứ 2

resp = requests.get(
    "https://openrouter.ai/api/v1/keys",
    headers={"Authorization": f"Bearer {KEY}"},
    params={"workspace_id": WORKSPACE_2_ID},
    timeout=15,
)
print("Status:", resp.status_code)
print("Body:", resp.text)