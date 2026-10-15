"""Minimal HTTP client for demoing (and smoke-testing) the Iris API.

Works as a fallback for the live demo if Swagger UI's CDN assets are
unreachable (Swagger UI's static JS/CSS loads from a CDN by default; this
script only talks to the local API and needs no internet after `pip install`).

Usage (from repo root, PowerShell, with the server already running):
    .venv\\Scripts\\python.exe scripts\\demo_client.py
    .venv\\Scripts\\python.exe scripts\\demo_client.py --base-url http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import sys

import requests

VALID_PAYLOAD = {
    "sepal_length_cm": 5.1,
    "sepal_width_cm": 3.5,
    "petal_length_cm": 1.4,
    "petal_width_cm": 0.2,
}

INVALID_PAYLOAD = {
    "sepal_length_cm": -1.0,  # nonpositive -> should be rejected
    "sepal_width_cm": 3.5,
    "petal_length_cm": 1.4,
    "petal_width_cm": 0.2,
}


def check(condition: bool, description: str) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {description}")
    return condition


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    all_ok = True

    print(f"Calling GET {base_url}/health ...")
    resp = requests.get(f"{base_url}/health", timeout=10)
    all_ok &= check(resp.status_code == 200, "GET /health returns 200")
    body = resp.json()
    all_ok &= check(body.get("model_loaded") is True, "health reports model_loaded=true")

    print(f"Calling GET {base_url}/model-info ...")
    resp = requests.get(f"{base_url}/model-info", timeout=10)
    all_ok &= check(resp.status_code == 200, "GET /model-info returns 200")
    info = resp.json()
    all_ok &= check("held_out_accuracy" in info, "model-info includes held_out_accuracy")
    print(f"  model_version={info.get('model_version')} held_out_accuracy={info.get('held_out_accuracy')}")

    print(f"Calling POST {base_url}/predict with a valid payload ...")
    resp = requests.post(f"{base_url}/predict", json=VALID_PAYLOAD, timeout=10)
    all_ok &= check(resp.status_code == 200, "valid /predict returns 200")
    prediction = resp.json()
    probs = prediction.get("probabilities", {})
    prob_sum = sum(probs.values()) if probs else 0.0
    all_ok &= check(bool(prediction.get("predicted_species")), "prediction includes predicted_species")
    all_ok &= check(abs(prob_sum - 1.0) < 1e-3, f"probabilities sum to ~1 (got {prob_sum:.6f})")
    print(f"  predicted_species={prediction.get('predicted_species')} probabilities={probs}")

    print(f"Calling POST {base_url}/predict with an invalid payload (nonpositive measurement) ...")
    resp = requests.post(f"{base_url}/predict", json=INVALID_PAYLOAD, timeout=10)
    all_ok &= check(resp.status_code == 422, f"invalid /predict returns 422 (got {resp.status_code})")

    if all_ok:
        print("\nAll demo client checks PASSED.")
        return 0
    print("\nSome demo client checks FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
