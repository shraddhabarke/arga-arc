import datetime
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

COLORS = [
    # First 10 colors are from the original ARC dataset.
    "#000",  # Black
    "#0074D9",  # Blue
    "#FF4136",  # Red
    "#2ECC40",  # Green
    "#FFDC00",  # Yellow
    "#AAAAAA",  # Gray
    "#F012BE",  # Fuchsia
    "#FF851B",  # Orange
    "#7FDBFF",  # Teal
    "#870C25",  # Brown
    # These are new colors
    "#606060",  # Dark Gray
    "#8E6F00",  # Olive
    "#5E077A",  # Purple
]


def print_grid(grid, title=None, ticks=False, save=False, output_dir="out", dpi=300):
    """
    Renders `grid` using matplotlib and the COLORS defined above.
    """
    grid = np.array(grid)

    cmap = mpl.colors.ListedColormap(COLORS)
    norm = mpl.colors.BoundaryNorm(np.arange(-0.5, len(COLORS), 1), cmap.N)

    fig = plt.figure(frameon=False)

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off
    fig.add_axes(ax)

    if ticks:
        ax.grid(axis="both", linestyle="-", color=COLORS[10], linewidth=1)
        ax.set_xticks(np.arange(0.5, grid.shape[1], 1))
        ax.set_yticks(np.arange(0.5, grid.shape[0], 1))
    ax.tick_params(
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )

    if title is not None:
        ax.set_title(title)

    ax.imshow(grid, cmap=cmap, norm=norm)

    if save:
        timestamp = datetime.datetime.now().strftime("%H%M%S-%f")
        output_path = os.path.join(output_dir, f"{timestamp}_{title}.png")
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    else:
        plt.show()
    plt.close()


def pad_to_same_size(A, B, pad_value=10, axis=None):
    """
    Given two 2D arrays, pads the specified axis with `value` so that they have the same size.
    The padding is done toward the end of the array, i.e. right and bottom.
    If `axis` is None, pads both axes.
    """
    max_rows = max(A.shape[0], B.shape[0])
    max_cols = max(A.shape[1], B.shape[1])

    res = []
    for arr in [A, B]:
        rows = arr.shape[0] if axis != 1 else max_rows
        cols = arr.shape[1] if axis != 0 else max_cols
        base = np.full((rows, cols), pad_value)
        base[: arr.shape[0], : arr.shape[1]] = arr
        res.append(base)

    A_padded, B_padded = res
    return A_padded, B_padded


def extend(A, B, axis=0, padding=0, border=None):
    """
    Given A, B two 2D arrays, returns a 2D that results from appending B to A, either horizontally or vertically.
    verically, padding in case of conflicting dimensions.
    """
    A_padded, B_padded = pad_to_same_size(A, B, pad_value=0, axis=axis)
    # Check if the border value is present
    # If not, create a new array with the border value
    if border is None:
        return np.concatenate([A_padded, B_padded], axis=axis)
    if axis == 0:
        border = np.full((1, A_padded.shape[1]), border)
    else:
        border = np.full((A_padded.shape[0], 1), border)
    return np.concatenate([A_padded, border, B_padded], axis=axis)


def pad_grids(input, output):
    """
    Pads the input and output grids so that they have the same number of rows and columns.
    The padding is done with color 11.
    The padding is done on the right and bottom of the grids.
    """
    # Pad the input grid.
    input_rows, input_cols = input.shape
    output_rows, output_cols = output.shape
    rows = max(input_rows, output_rows)
    cols = max(input_cols, output_cols)
    input_padded = np.full((rows, cols), 10)
    input_padded[:input_rows, :input_cols] = input

    # Pad the output grid.
    output_padded = np.full((rows, cols), 10)
    output_padded[:output_rows, :output_cols] = output

    return input_padded, output_padded


def construct_example(example, show_output=True):
    """
    Renders the input and output grids of `example`.
    """
    # Take both the input and output grids from the example.
    input = np.array(example["input"])
    output = np.array(example["output"])
    if not show_output:
        output = np.full(output.shape, 10)
    example_grid = extend(input, output, axis=1, padding=1, border=11)
    # Pad on the sides with the same color as the border.
    example_grid = np.pad(example_grid, ((0, 0), (1, 1)), constant_values=11)
    return example_grid


def construct_task(task):
    """
    Renders a task by rendering all its examples.
    """
    task_grid = np.array([[]])
    for example in task["train"]:
        example_grid = construct_example(example)
        task_grid = extend(task_grid, example_grid, axis=0, padding=1, border=12)
    # Add a border between the train and test examples.
    task_grid = extend(
        task_grid, np.full((1, task_grid.shape[1]), 11), axis=0, padding=1, border=12
    )
    for example in task["test"]:
        example_grid = construct_example(example, show_output=False)
        task_grid = extend(task_grid, example_grid, axis=0, padding=1, border=12)
    return task_grid
