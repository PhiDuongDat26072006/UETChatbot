import time
import requests

url = "https://semester-agency-pirates-doll.trycloudflare.com/v1/systemone"
payload = {
    "state": "Help! My payments have been failing for 3 days.",
    "questions": {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "", "sales": "", "technical": ""},
        }
    },
}

with requests.Session() as session:
    for i in range(1, 20):
        t0 = time.perf_counter()
        r = session.post(url, json=payload, timeout=120)
        dt = time.perf_counter() - t0
        print(f"Run {i} | Status: {r.status_code} | Elapsed: {dt:.3f}s")