import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
REPORT_DIR = PROJECT_ROOT / "reports"


def evaluate_tickets():
    path = REPORT_DIR / "ticket_prompt_results.csv"

    if not path.exists():
        print("Run prompt_tests.py first.")
        return

    df = pd.read_csv(path)
    required_columns = {
        "status", "prompt_version", "expected_category",
        "expected_priority", "predicted_category", "predicted_priority",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(
            "Ticket results are missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
        return

    # Exclude failed API requests from metric calculations.
    valid = df[
        (df["status"] == "actual_api")
        | (df["status"] == "offline_demo")
    ].copy()

    if valid.empty:
        print("No valid predictions available.")
        return

    summaries = []

    for version, group in valid.groupby("prompt_version"):
        category_accuracy = accuracy_score(
            group["expected_category"],
            group["predicted_category"],
        )

        priority_accuracy = accuracy_score(
            group["expected_priority"],
            group["predicted_priority"],
        )

        category_f1 = f1_score(
            group["expected_category"],
            group["predicted_category"],
            labels=["Billing", "Technical", "Account"],
            average="macro",
            zero_division=0,
        )

        priority_f1 = f1_score(
            group["expected_priority"],
            group["predicted_priority"],
            labels=["High", "Medium", "Low"],
            average="macro",
            zero_division=0,
        )

        summaries.append({
            "prompt_version": version,
            "sample_count": len(group),
            "category_accuracy": round(category_accuracy, 4),
            "priority_accuracy": round(priority_accuracy, 4),
            "category_macro_f1": round(category_f1, 4),
            "priority_macro_f1": round(priority_f1, 4),
            "evaluation_mode": ", ".join(
                sorted(group["status"].unique())
            ),
        })

    result = pd.DataFrame(summaries)

    print("\n--- TICKET EVALUATION ---")
    print(result.to_string(index=False))

    output = REPORT_DIR / "ticket_evaluation_summary.csv"
    result.to_csv(output, index=False)
    print("\nSaved:", output)


def evaluate_resumes():
    path = REPORT_DIR / "resume_extraction_results.json"

    if not path.exists():
        print("Resume extraction results not found.")
        return

    try:
        results = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        print(f"Resume results contain invalid JSON: {exc}")
        return
    if not isinstance(results, list):
        print("Resume results must be a JSON array.")
        return
    successful = [
        row for row in results
        if isinstance(row, dict)
        and "error" not in row
        and isinstance(row.get("expected"), dict)
    ]

    if not successful:
        print("No successful resume extractions to evaluate.")
        return

    exact_matches = 0
    field_matches = 0
    total_fields = 0

    for row in successful:
        expected = row["expected"]

        fields = [
            "candidate_name",
            "skills",
            "years_of_experience",
            "education",
        ]

        matches = 0

        for field in fields:
            actual_value = row.get(field)
            expected_value = expected.get(field)

            # Skill order does not matter for this comparison.
            if field == "skills" and isinstance(actual_value, list) and isinstance(expected_value, list):
                match = set(actual_value) == set(expected_value)
            elif field == "education" and isinstance(actual_value, list) and isinstance(expected_value, list):
                match = set(actual_value) == set(expected_value)
            else:
                match = actual_value == expected_value

            total_fields += 1
            field_matches += int(match)
            matches += int(match)

        exact_matches += int(matches == len(fields))

    print("\n--- RESUME EVALUATION ---")
    print("Successful extractions:", len(successful))
    print("Exact record accuracy:", round(
        exact_matches / len(successful), 4
    ))
    print("Field accuracy:", round(
        field_matches / total_fields, 4
    ))


if __name__ == "__main__":
    evaluate_tickets()
    evaluate_resumes()