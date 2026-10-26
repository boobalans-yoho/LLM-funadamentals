
import csv
from pathlib import Path

tests = [
    {
        "test": "Summarisation",
        "prompt": (
            "Summarise in one sentence: "
            "A customer was charged twice and requests a refund."
        ),
        "result": "A customer requests a refund for a duplicate charge."
    },
    {
        "test": "Classification",
        "prompt": (
            "Classify this ticket as Billing, Technical or Account: "
            "My payment was deducted twice."
        ),
        "result": "Billing"
    },
    {
        "test": "Extraction",
        "prompt": (
            'Extract JSON from: "Arun has 2 years of Python experience." '
            'Use keys name and years_experience.'
        ),
        "result": '{"name": "Arun", "years_experience": 2}'
    },
    {
        "test": "Translation",
        "prompt": "Translate into English: வணக்கம், எனக்கு உதவி வேண்டும்.",
        "result": "Hello, I need help."
    },
    {
        "test": "Code generation",
        "prompt": "Write a Python function that adds two numbers.",
        "result": "def add(a, b): return a + b"
    },
    {
        "test": "Knowledge cutoff",
        "prompt": (
            "What happened in the most recent major technology event? "
            "Give the date and a source."
        ),
        "result": "Record the actual model response and verify it using a source."
    },
    {
        "test": "Hallucination",
        "prompt": (
            "Summarise the fictional paper 'Quantum Support Networks by "
            "Dr. Xylor Example (2099)'."
        ),
        "result": "Check whether the model invents details."
    },
    {
        "test": "Hallucination safeguard",
        "prompt": (
            "Summarise the fictional paper 'Quantum Support Networks by "
            "Dr. Xylor Example (2099)'. Say I don't know if unsure."
        ),
        "result": "Check whether the model admits uncertainty."
    },
    {
        "test": "Context length",
        "prompt": (
            "In a long text, place this fact in the middle: "
            "The project code is BLUE-742. Ask the model for the code."
        ),
        "result": "Expected answer: BLUE-742"
    },
    {
        "test": "Arithmetic",
        "prompt": (
            "A company processes 125 tickets daily for 24 days. "
            "How many tickets does it process?"
        ),
        "result": "3000"
    },
    {
        "test": "Logic",
        "prompt": (
            "All engineers are employees. Ravi is an engineer. "
            "Is Ravi an employee?"
        ),
        "result": "Yes"
    }
]

output_file = (
    Path(__file__).parent / "results_template.csv"
)

with output_file.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["Test", "Prompt", "Result", "Rating"]
    )
    writer.writeheader()

    for item in tests:
        writer.writerow({
            "Test": item["test"],
            "Prompt": item["prompt"],
            "Result": item["result"],
            "Rating": "Pending actual test"
        })

print("Test template created:", output_file)
print("Tests:", len(tests))
print("Fill in actual model responses and ratings from 1 to 5.")
