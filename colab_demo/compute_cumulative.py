from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import tyro

from colab_demo.utils import plot_results


@dataclass
class Args:
    num_elements: int = 100  # Default value if not specified


def cumulative_sum(array: np.ndarray) -> np.ndarray:
    cumulative_sum = []

    for i, element in enumerate(array):
        if len(cumulative_sum) == 0:
            cumulative_sum.append(element)
        else:
            cumulative_sum.append(cumulative_sum[-1] + element)

        if i % 10 == 0:
            print(f"Current cumulative sum: {cumulative_sum[-1]}")

    return np.array(cumulative_sum)


def main(args: Args):
    array_to_sum = np.random.randint(0, 10, args.num_elements)
    cumulative_arr_sum = cumulative_sum(array_to_sum)

    print(f"Final cumulative sum: {cumulative_arr_sum[-1]}")
    plot_results(cumulative_arr_sum)
    plt.show()


if __name__ == "__main__":
    tyro.cli(main)
