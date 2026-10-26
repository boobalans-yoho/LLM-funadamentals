
import csv
import json
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

# Resolve .env from this file's location, so startup works from any cwd.
load_dotenv(dotenv_path=WORKSPACE_ROOT / ".env")

DATA_DIR = ROOT / "data"
PROMPT_DIR = ROOT / "prompts"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL = os.getenv("OPENROUTER_MODEL", "")

API_URL = "https://openrouter.ai/api/v1/chat/completions"

CATEGORIES = {"Billing", "Technical", "Account"}
PRIORITIES = {"High", "Medium", "Low"}


def load_prompt(filename):
    return (PROMPT_DIR / filename).read_text(encoding="utf-8")


def safe_json_parse(text):
    """Parse a model response containing a JSON object."""
    if not text:
        raise ValueError("Empty response from model")
    if not isinstance(text, str):
        raise ValueError("Model response must be text")

    text = text.strip()
    text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text,
        flags=re.MULTILINE,
    )
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)

    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    return result


def call_llm(system_prompt, user_prompt, temperature=0, top_p=1):
    """Call OpenRouter. API credentials are never printed."""
    if not API_KEY or not MODEL:
        raise RuntimeError("OPENROUTER_CONFIG_MISSING")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
    }

    last_error = None

    for attempt in range(3):
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=60,
            )

            if response.status_code in (429, 500, 502, 503, 504):
                response.raise_for_status()

            response.raise_for_status()
            data = response.json()

            return data["choices"][0]["message"]["content"]

        except requests.HTTPError as exc:
            last_error = exc
            status_code = (
                exc.response.status_code if exc.response is not None else None
            )
            if status_code not in (429, 500, 502, 503, 504):
                break
            if attempt < 2:
                time.sleep(2 ** attempt)
        except (
            requests.RequestException,
            KeyError,
            IndexError,
            TypeError,
            AttributeError,
        ) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)

    if isinstance(last_error, requests.HTTPError):
        status_code = (
            last_error.response.status_code
            if last_error.response is not None else "unknown"
        )
        provider_message = ""
        if last_error.response is not None:
            try:
                error_data = last_error.response.json().get("error", {})
                if isinstance(error_data, dict):
                    provider_message = str(error_data.get("message", ""))
            except (ValueError, AttributeError):
                pass
        detail = f": {provider_message[:300]}" if provider_message else ""
        raise RuntimeError(
            f"OpenRouter returned HTTP {status_code}{detail}"
        ) from last_error

    raise RuntimeError(
        f"LLM request failed after retries: {type(last_error).__name__}"
    )


def validate_ticket_prediction(result):
    """Validate the returned JSON fields and allowed labels."""
    category = result.get("category")
    priority = result.get("priority")

    if category not in CATEGORIES:
        raise ValueError(f"Invalid category: {category}")

    if priority not in PRIORITIES:
        raise ValueError(f"Invalid priority: {priority}")

    return {"category": category, "priority": priority}


def offline_ticket_prediction(ticket):
    """
    Simple demo baseline, not an LLM prediction.
    Use actual API results for the real prompt comparison.
    """
    text = ticket.lower()

    if any(word in text for word in [
        "payment", "refund", "invoice", "charge", "receipt",
        "subscription", "transaction"
    ]):
        category = "Billing"
    elif any(word in text for word in [
        "password", "account", "email", "profile", "verify",
        "locked", "login", "log in"
    ]):
        category = "Account"
    else:
        category = "Technical"

    if any(word in text for word in [
        "twice", "crashes", "locked", "server error", "cannot verify"
    ]):
        priority = "High"
    elif any(word in text for word in [
        "slowly", "incorrect", "does not work", "doesn't work",
        "missing", "cannot", "can't"
    ]):
        priority = "Medium"
    else:
        priority = "Low"

    return {"category": category, "priority": priority}


def run_ticket_tests():
    tickets_file = DATA_DIR / "tickets.csv"

    if not tickets_file.is_file():
        raise FileNotFoundError(f"Ticket data file not found: {tickets_file}")

    with tickets_file.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)
        required_columns = {"text", "category", "priority"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                "Ticket CSV is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )
        tickets = list(reader)

    if not tickets:
        raise ValueError(f"Ticket data file is empty: {tickets_file}")

    system_prompt = load_prompt("system_prompt.txt")
    zero_shot_template = load_prompt("zero_shot.txt")
    few_shot_template = load_prompt("few_shot.txt")

    versions = {
        "zero_shot": zero_shot_template,
        "few_shot": few_shot_template,
    }

    use_api = bool(API_KEY and MODEL)
    results = []

    print("Mode:", "OpenRouter API" if use_api else "Offline demo")
    print("Tickets:", len(tickets))

    output_fields = [
        "ticket", "expected_category", "expected_priority",
        "prompt_version", "predicted_category", "predicted_priority",
        "category_correct", "priority_correct", "status", "error",
    ]

    for row_number, ticket_row in enumerate(tickets, start=2):
        ticket_text = (ticket_row.get("text") or "").strip()
        expected_category = (ticket_row.get("category") or "").strip()
        expected_priority = (ticket_row.get("priority") or "").strip()
        if not ticket_text:
            raise ValueError(f"Ticket CSV row {row_number} has empty text")
        if expected_category not in CATEGORIES:
            raise ValueError(
                f"Ticket CSV row {row_number} has invalid category: "
                f"{expected_category!r}"
            )
        if expected_priority not in PRIORITIES:
            raise ValueError(
                f"Ticket CSV row {row_number} has invalid priority: "
                f"{expected_priority!r}"
            )

        for version, template in versions.items():
            # Replace only our placeholder; .format() treats the JSON braces
            # in the few-shot examples as formatting fields.
            prompt = template.replace("{ticket}", ticket_text)

            try:
                if use_api:
                    raw = call_llm(
                        system_prompt,
                        prompt,
                        temperature=0,
                        top_p=1,
                    )
                    prediction = validate_ticket_prediction(
                        safe_json_parse(raw)
                    )
                    status = "actual_api"
                else:
                    prediction = offline_ticket_prediction(ticket_text)
                    status = "offline_demo"

                error = ""

            except (RuntimeError, ValueError, TypeError) as exc:
                prediction = {"category": "", "priority": ""}
                status = "error"
                error = str(exc)

            results.append({
                "ticket": ticket_text,
                "expected_category": expected_category,
                "expected_priority": expected_priority,
                "prompt_version": version,
                "predicted_category": prediction["category"],
                "predicted_priority": prediction["priority"],
                "category_correct": (
                    prediction["category"] == expected_category
                ),
                "priority_correct": (
                    prediction["priority"] == expected_priority
                ),
                "status": status,
                "error": error,
            })

            print(
                f"{version}: {ticket_text[:35]} | "
                f"{prediction['category']} / {prediction['priority']}"
            )

    output_file = REPORT_DIR / "ticket_prompt_results.csv"

    with output_file.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=output_fields
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved:", output_file)


def run_resume_extraction():
    resumes_file = DATA_DIR / "resumes.json"

    if not resumes_file.is_file():
        raise FileNotFoundError(f"Resume data file not found: {resumes_file}")

    resumes = json.loads(
        resumes_file.read_text(encoding="utf-8-sig")
    )
    if not isinstance(resumes, list):
        raise ValueError("Resume data must be a JSON array")

    schema = {
        "candidate_name": "string or null",
        "skills": ["string"],
        "years_of_experience": "number or null",
        "education": ["string"],
    }

    system_prompt = (
        "Extract facts from the supplied resume. "
        "Do not invent missing details. Use null when a single "
        "value is unknown, and an empty array when no items are "
        "provided. Return valid JSON only with keys: "
        "candidate_name, skills, years_of_experience, education. "
        f"Follow this schema: {json.dumps(schema)}"
    )

    results = []
    use_api = bool(API_KEY and MODEL)

    for index, resume in enumerate(resumes, start=1):
        if not isinstance(resume, dict):
            raise ValueError(f"Resume record {index} must be a JSON object")
        text = resume.get("text")
        expected = resume.get("expected")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Resume record {index} has missing or empty text")
        if not isinstance(expected, dict):
            raise ValueError(f"Resume record {index} has invalid expected data")

        if not use_api:
            print(
                f"Resume {index}: skipped API extraction "
                "(offline demo mode)"
            )
            continue

        raw = None
        try:
            raw = call_llm(
                system_prompt,
                f"Extract the resume information:\n{text}",
                temperature=0,
                top_p=1,
            )

            extracted = safe_json_parse(raw)

            required = {
                "candidate_name",
                "skills",
                "years_of_experience",
                "education",
            }

            if not required.issubset(extracted):
                raise ValueError("Missing required JSON keys")

            if not (
                extracted["candidate_name"] is None
                or isinstance(extracted["candidate_name"], str)
            ):
                raise ValueError("Invalid candidate_name")

            if not isinstance(extracted["skills"], list):
                raise ValueError("skills must be an array")

            if not isinstance(extracted["education"], list):
                raise ValueError("education must be an array")

            years = extracted["years_of_experience"]
            if years is not None:
                if (
                    isinstance(years, bool)
                    or not isinstance(years, (int, float))
                    or not 0 <= years <= 50
                ):
                    raise ValueError("Experience must be between 0 and 50")

            extracted["resume_id"] = index
            extracted["expected"] = expected
            results.append(extracted)

            print(f"Resume {index}: extraction successful")

        except Exception as exc:
            results.append({
                "resume_id": index,
                "error": str(exc),
            })
            print(
                f"Resume {index}: extraction failed -> "
                f"{type(exc).__name__}: {exc}"
            )
            print("Raw response:", raw if raw is not None else "no response")

    output_file = REPORT_DIR / "resume_extraction_results.json"
    output_file.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Saved:", output_file)


if __name__ == "__main__":
    run_ticket_tests()
    run_resume_extraction()
