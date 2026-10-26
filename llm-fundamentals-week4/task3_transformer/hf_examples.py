
from transformers import pipeline

print("--- BERT fill-mask ---")
fill_mask = pipeline("fill-mask", model="google-bert/bert-base-uncased")
print(fill_mask("The capital of France is [MASK]."))

print("\n--- GPT-2 text generation ---")
generator = pipeline("text-generation", model="openai-community/gpt2")
print(
    generator(
        "Customer support teams should",
        max_new_tokens=30,
        do_sample=False
    )
)
