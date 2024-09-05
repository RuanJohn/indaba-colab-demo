from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import tyro

from colab_demo.utils import plot_results

# Check if running in Colab
import sys
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    from google.colab import output
    output.enable_custom_widget_manager()

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
    
    # Create a new figure
    plt.figure(figsize=(10, 6))
    plot_results(cumulative_arr_sum)
    
    if IN_COLAB:
        # Display the plot inline in Colab
        from IPython.display import display
        display(plt.gcf())
    else:
        # Show the plot in a new window for local environments
        plt.show(block=True)
    
    print("Plot displayed. Close the plot window to end the program.")

if __name__ == "__main__":
    if IN_COLAB:
        import matplotlib_inline
        matplotlib_inline.backend_inline.set_matplotlib_formats('retina')
    tyro.cli(main)