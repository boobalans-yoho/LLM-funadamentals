# LLM Fundamentals — Week 4

## Task 4 API server

Install the project requirements, then start the local API from this directory:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn task4_prompt_toolkit.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the interactive API page. The server
loads `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` from the workspace `.env`.

Endpoints:

- `GET /health` — local status and whether OpenRouter credentials are configured.
- `POST /api/tickets/classify` — JSON body: `{"text":"...","prompt_version":"zero_shot"}`.
- `POST /api/resumes/extract` — JSON body: `{"text":"resume text..."}`.

Ticket prompt version can be `zero_shot` or `few_shot`. API failures from
OpenRouter are returned as HTTP errors; rate limits are reported as `429`.
