"""Demonstrate token counts and estimate usage costs."""

from __future__ import annotations

import tiktoken


# This is a tokenizer demonstration, not a token count for every model.
ENCODING_NAME = "cl100k_base"
INPUT_PRICE_PER_MILLION = 1.00  # Illustrative USD price; update for your model.
OUTPUT_PRICE_PER_MILLION = 5.00  # Illustrative USD price; update for your model.
INPUT_TOKENS_PER_CHAT = 1_500
OUTPUT_TOKENS_PER_CHAT = 400
CHATS_PER_DAY = 1_000
DAYS_PER_MONTH = 30

SAMPLE_TEXTS = {
    "Sentence": "The customer wants a refund for a failed payment.",
    "Paragraph": (
        "Our support team helps customers with billing, technical "
        "issues, and account access. Each ticket is classified "
        "and assigned a priority for faster resolution."
    ),
    "Page": "Customer support report. " * 100,
    "JSON": (
        '{"name":"Arun","skills":["Python","FastAPI"],'
        '"experience_years":2}'
    ),
    "Tamil": "வணக்கம்! எனக்கு பணத்தை திரும்ப பெற வேண்டும்.",
}


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    input_price_per_million: float,
    output_price_per_million: float,
) -> float:
    """Return estimated input and output cost in USD."""
    input_cost = input_tokens / 1_000_000 * input_price_per_million
    output_cost = output_tokens / 1_000_000 * output_price_per_million
    return input_cost + output_cost


def print_token_report(encoding: tiktoken.Encoding) -> dict[str, int]:
    """Print token and context-window information for each sample."""
    token_counts: dict[str, int] = {}

    print("\n--- TOKEN COUNT REPORT ---")
    for name, text in SAMPLE_TEXTS.items():
        token_count = len(encoding.encode(text))
        word_count = len(text.split())
        words_per_token = word_count / token_count if token_count else 0
        token_counts[name] = token_count

        print(f"\nText: {name}")
        print(f"Words: {word_count}")
        print(f"Tokens: {token_count}")
        print(f"Words per token: {words_per_token:.3f}")
        print(f"Fits in 8K: {token_count <= 8_000}")
        print(f"Fits in 128K: {token_count <= 128_000}")

    return token_counts


def print_cost_report() -> None:
    """Print illustrative daily and monthly usage cost estimates."""
    daily_cost = CHATS_PER_DAY * estimate_cost(
        INPUT_TOKENS_PER_CHAT,
        OUTPUT_TOKENS_PER_CHAT,
        INPUT_PRICE_PER_MILLION,
        OUTPUT_PRICE_PER_MILLION,
    )
    monthly_cost = daily_cost * DAYS_PER_MONTH

    print("\n--- COST ESTIMATION ---")
    print(f"Input price per 1M tokens: ${INPUT_PRICE_PER_MILLION:.2f}")
    print(f"Output price per 1M tokens: ${OUTPUT_PRICE_PER_MILLION:.2f}")
    print(f"Chats per day: {CHATS_PER_DAY:,}")
    print(f"Input tokens per chat: {INPUT_TOKENS_PER_CHAT:,}")
    print(f"Output tokens per chat: {OUTPUT_TOKENS_PER_CHAT:,}")
    print(f"Daily cost: ${daily_cost:.4f}")
    print(f"Monthly cost ({DAYS_PER_MONTH} days): ${monthly_cost:.2f}")


def print_context_report(token_counts: dict[str, int]) -> None:
    """Print a compact context-window summary."""
    print("\n--- CONTEXT WINDOW REPORT ---")
    for name, count in token_counts.items():
        fits_8k = "YES" if count <= 8_000 else "NO"
        fits_128k = "YES" if count <= 128_000 else "NO"
        print(f"{name}: {count} tokens | 8K={fits_8k} | 128K={fits_128k}")


def main() -> None:
    """Run the tokenizer and cost demonstrations."""
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    token_counts = print_token_report(encoding)
    print_cost_report()
    print_context_report(token_counts)


if __name__ == "__main__":
    main()
