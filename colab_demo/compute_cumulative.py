from dataclasses import dataclass

import jax
import matplotlib.pyplot as plt
import tyro

from colab_demo.utils.plotting import plot_results


@dataclass
class Args:
    num_elements: int = 100
    output_file: str = "cumulative_result_plot.png"


def compute_cumulative_result(array: jax.Array) -> jax.Array:
    def body_fun(carry, x):
        i, current_cumulative_value = carry
        new_cumulative_value = current_cumulative_value - x

        def print_result(args):
            jax.debug.print(
                "Iteration {}: Current cumulative value: {}", args[0], args[1]
            )
            return args

        _ = jax.lax.cond(
            i % 10 == 0, lambda args: print_result(args), lambda args: args, (i, new_cumulative_value)
        )

        return (i + 1, new_cumulative_value), new_cumulative_value

    _, cumulative_result = jax.lax.scan(body_fun, (0, 1e-8), array)
    return cumulative_result


@jax.jit
def jitted_compute_cumulative_result(array: jax.Array) -> jax.Array:
    return compute_cumulative_result(array)


def main(args: Args):
    rng = jax.random.PRNGKey(42)
    initial_array = (
        jax.random.uniform(rng, (args.num_elements,), minval=-100, maxval=100) + 1e-6
    )
    cumulative_arr_result = jitted_compute_cumulative_result(initial_array)

    print(f"Final cumulative value: {cumulative_arr_result[-1]}")

    # Create a new figure
    plt.figure(figsize=(10, 6))
    plot_results(cumulative_arr_result, "Cumulative Subtraction")

    # Save the figure
    plt.savefig(args.output_file)
    plt.close()

    print(f"Plot saved as '{args.output_file}'")


if __name__ == "__main__":
    tyro.cli(main)
