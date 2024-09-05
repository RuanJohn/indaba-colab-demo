import matplotlib.pyplot as plt
import numpy as np


def plot_results(array: np.ndarray, title: str = "Cumulative Sum") -> None:
    x_vals = np.arange(len(array))
    plt.plot(x_vals, array)
    plt.title(title)
    plt.xlabel("Array value")
    plt.ylabel("Cumulative value")
