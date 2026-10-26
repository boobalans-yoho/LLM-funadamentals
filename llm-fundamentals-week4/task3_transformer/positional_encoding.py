"""Generate and visualize sinusoidal positional encodings."""

import matplotlib.pyplot as plt
import numpy as np


def positional_encoding(
    max_positions: int = 10,
    dimensions: int = 8,
) -> np.ndarray:
    """Create sinusoidal position encodings.

    Args:
        max_positions: Number of token positions to encode.
        dimensions: Width of each position encoding.

    Returns:
        An array with shape ``(max_positions, dimensions)``.

    Raises:
        ValueError: If either argument is less than one.
    """
    if max_positions < 1:
        raise ValueError("max_positions must be at least 1")
    if dimensions < 1:
        raise ValueError("dimensions must be at least 1")

    positions = np.arange(max_positions, dtype=np.float64)[:, np.newaxis]
    dimension_indices = np.arange(dimensions)[np.newaxis, :]
    angle_rates = 1 / np.power(
        10_000,
        (2 * (dimension_indices // 2)) / dimensions,
    )
    angles = positions * angle_rates

    encoding = np.empty((max_positions, dimensions), dtype=np.float64)
    encoding[:, 0::2] = np.sin(angles[:, 0::2])
    encoding[:, 1::2] = np.cos(angles[:, 1::2])
    return encoding


def main() -> None:
    """Print positional encodings and save a visualization."""
    encodings = positional_encoding()

    print("Positional encoding shape:", encodings.shape)
    print(encodings.round(3))

    plt.figure(figsize=(8, 5))
    plt.imshow(encodings, aspect="auto", cmap="viridis")
    plt.xlabel("Embedding dimension")
    plt.ylabel("Token position")
    plt.title("Sinusoidal Positional Encoding")
    plt.colorbar(label="Value")
    plt.tight_layout()
    plt.savefig("positional_encoding.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
