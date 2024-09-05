from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import tyro

from colab_demo.utils import plot_results


@dataclass
class Args:
    num_elements: int = 100  # Default value if not specified


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
    array_to_sum = np.random.randint(-100, 100, args.num_elements) + 1e-6 # add a small value to exclude zeros.
    cumulative_arr_sum = compute_cumulative_result(array_to_sum)

    print(f"Final cumulative sum: {cumulative_arr_sum[-1]}")
    plot_results(cumulative_arr_sum)
    plt.show()


if __name__ == "__main__":
    tyro.cli(main)
