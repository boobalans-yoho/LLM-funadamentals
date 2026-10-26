"""Local HTTP API for ticket classification and resume extraction."""

import json
import re
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

if __package__:
    from .prompt_tests import (
        API_KEY,
        MODEL,
        call_llm,
        load_prompt,
        safe_json_parse,
        validate_ticket_prediction,
    )
else:  # Also support running this file directly from its folder.
    from prompt_tests import (
        API_KEY,
        MODEL,
        call_llm,
        load_prompt,
        safe_json_parse,
        validate_ticket_prediction,
    )

app = FastAPI(
    title="LLM Fundamentals Task 4 API",
    description="Ticket classification and resume extraction using OpenRouter.",
    version="1.0.0",
)


class TicketRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    prompt_version: Literal["zero_shot", "few_shot"] = "zero_shot"


class ResumeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)


def _nonempty_text(value: str, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must contain non-whitespace text.",
        )
    return value


def _request_model(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    try:
        raw = call_llm(system_prompt, user_prompt, temperature=0, top_p=1)
        return safe_json_parse(raw)
    except RuntimeError as exc:
        if str(exc) == "OPENROUTER_CONFIG_MISSING":
            raise HTTPException(
                status_code=503,
                detail=(
                    "OpenRouter is not configured in the API process. Set "
                    "OPENROUTER_API_KEY and OPENROUTER_MODEL in .env, then "
                    "restart uvicorn."
                ),
            ) from exc
        match = re.search(r"HTTP (\d{3})", str(exc))
        status = int(match.group(1)) if match else 502
        if status == 429:
            raise HTTPException(
                status_code=429,
                detail=f"OpenRouter rate limit reached: {exc}",
            ) from exc
        raise HTTPException(
            status_code=502,
            detail=f"OpenRouter request failed (HTTP {status}): {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail="OpenRouter returned invalid JSON.",
        ) from exc


@app.get("/health")
def health():
    return {
        "status": "ok",
        "openrouter_configured": bool(API_KEY and MODEL),
    }


@app.post("/api/tickets/classify")
def classify_ticket(request: TicketRequest):
    ticket_text = _nonempty_text(request.text, "text")
    template = load_prompt(f"{request.prompt_version}.txt")
    system_prompt = load_prompt("system_prompt.txt")
    result = _request_model(
        system_prompt,
        template.replace("{ticket}", ticket_text),
    )
    try:
        prediction = validate_ticket_prediction(result)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        "category": prediction["category"],
        "priority": prediction["priority"],
        "prompt_version": request.prompt_version,
    }


@app.post("/api/resumes/extract")
def extract_resume(request: ResumeRequest):
    resume_text = _nonempty_text(request.text, "text")
    schema = {
        "candidate_name": "string or null",
        "skills": ["string"],
        "years_of_experience": "number or null",
        "education": ["string"],
    }
    system_prompt = (
        "Extract facts from the supplied resume. Do not invent missing details. "
        "Use null for an unknown single value and an empty array when no items "
        "are present. Return valid JSON only with keys: candidate_name, skills, "
        "years_of_experience, education. Follow this schema: "
        f"{json.dumps(schema)}"
    )
    result = _request_model(
        system_prompt,
        f"Extract the resume information:\n{resume_text}",
    )

    required = (
        "candidate_name", "skills", "years_of_experience", "education"
    )
    if any(field not in result for field in required):
        raise HTTPException(status_code=502, detail="Missing resume fields.")
    if not (
        result["candidate_name"] is None
        or isinstance(result["candidate_name"], str)
    ):
        raise HTTPException(status_code=502, detail="Invalid candidate_name.")
    if not all(
        isinstance(result[field], list)
        and all(isinstance(item, str) for item in result[field])
        for field in ("skills", "education")
    ):
        raise HTTPException(
            status_code=502,
            detail="Skills and education must be arrays of strings.",
        )

    years = result["years_of_experience"]
    if years is not None and (
        isinstance(years, bool)
        or not isinstance(years, (int, float))
        or not 0 <= years <= 50
    ):
        raise HTTPException(
            status_code=502,
            detail="years_of_experience must be between 0 and 50 or null.",
        )
    return {field: result[field] for field in required}
