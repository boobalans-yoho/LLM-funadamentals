"""Send the first sample resume to the locally running FastAPI app."""

import json
import os
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
RESUMES_FILE = ROOT / "data" / "resumes.json"
API_URL = os.getenv(
    "RESUME_API_URL",
    "http://127.0.0.1:8001/api/resumes/extract",
)


def main() -> None:
    with RESUMES_FILE.open("r", encoding="utf-8") as file:
        resumes = json.load(file)

    if not resumes:
        raise ValueError(f"No resumes found in {RESUMES_FILE}")

    first = resumes[0]
    text = (
        first["text"]
        if isinstance(first, dict) and "text" in first
        else str(first)
    )

    try:
        response = requests.post(API_URL, json={"text": text}, timeout=120)
        print("Status:", response.status_code)
        response.raise_for_status()
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    except requests.HTTPError:
        # Show FastAPI's detail message when the endpoint returns an error.
        try:
            error = response.json()
        except ValueError:
            error = response.text
        raise SystemExit(f"API request failed: {error}")
    except requests.RequestException as exc:
        raise SystemExit(
            f"Could not reach {API_URL}. Start the API first. ({exc})"
        ) from exc


if __name__ == "__main__":
    main()
