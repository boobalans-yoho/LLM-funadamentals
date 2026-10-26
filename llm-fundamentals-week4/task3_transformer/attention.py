"""A small NumPy demonstration of single- and multi-head self-attention."""

import numpy as np


def softmax(values: np.ndarray) -> np.ndarray:
    """Compute a numerically stable softmax along the final axis."""
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


def scaled_dot_product_attention(
    inputs: np.ndarray,
    query_weights: np.ndarray,
    key_weights: np.ndarray,
    value_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return self-attention output and attention weights for one head."""
    queries = inputs @ query_weights
    keys = inputs @ key_weights
    values = inputs @ value_weights

    scale = np.sqrt(queries.shape[-1])
    scores = queries @ keys.T / scale
    weights = softmax(scores)
    return weights @ values, weights


def main() -> None:
    """Run the single-head and two-head attention demonstrations."""
    rng = np.random.default_rng(seed=0)

    # Four tokens, each represented by three features.
    inputs = rng.random((4, 3))

    # Single-head attention with three-dimensional queries, keys, and values.
    query_weights = rng.random((3, 3))
    key_weights = rng.random((3, 3))
    value_weights = rng.random((3, 3))
    output, attention_weights = scaled_dot_product_attention(
        inputs,
        query_weights,
        key_weights,
        value_weights,
    )

    print("Input shape:", inputs.shape)
    print("\nSingle-head attention weights:")
    print(attention_weights)
    print("Row sums:", attention_weights.sum(axis=1))
    print("Output shape:", output.shape)

    # Two attention heads, each producing two features per token.
    head_dimension = 2
    head_outputs = []
    for _ in range(2):
        head_output, _ = scaled_dot_product_attention(
            inputs,
            rng.random((inputs.shape[1], head_dimension)),
            rng.random((inputs.shape[1], head_dimension)),
            rng.random((inputs.shape[1], head_dimension)),
        )
        head_outputs.append(head_output)

    concatenated = np.concatenate(head_outputs, axis=1)
    output_projection = rng.random((2 * head_dimension, 3))
    projected = concatenated @ output_projection

    print("\nTwo-head concatenated shape:", concatenated.shape)
    print("Output projection shape:", projected.shape)

    assert attention_weights.shape == (4, 4)
    assert np.allclose(attention_weights.sum(axis=1), 1.0)
    assert concatenated.shape == (4, 4)
    assert projected.shape == (4, 3)
    print("\nPASS: Attention rows sum to 1 and output shapes are correct.")


if __name__ == "__main__":
    main()
