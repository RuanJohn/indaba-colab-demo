from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import tyro

from colab_demo.utils import plot_results


@dataclass
class Args:
    num_elements: int = 100
    output_file: str = "cumulative_result_plot.png"


def compute_cumulative_result(array: np.ndarray) -> np.ndarray:
    cumulative_result = []

    for i, element in enumerate(array):
        if len(cumulative_result) == 0:
            cumulative_result.append(element)
        else:
            cumulative_result.append(cumulative_result[-1] + element)

        if i % 10 == 0:
            print(f"Current cumulative sum: {cumulative_result[-1]}")

    return np.array(cumulative_result)


def main(args: Args):
    array_to_sum = (
        np.random.randint(-100, 100, args.num_elements) + 1e-6
    )  # add a small value to exclude zeros.
    cumulative_arr_sum = compute_cumulative_result(array_to_sum)

    print(f"Final cumulative sum: {cumulative_arr_sum[-1]}")

    # Create a new figure
    plt.figure(figsize=(10, 6))
    plot_results(cumulative_arr_sum)

    # Save the figure
    plt.savefig(args.output_file)
    plt.close()  # Close the figure to free up memory

    print(f"Plot saved as '{args.output_file}'")


if __name__ == "__main__":
    tyro.cli(main)
