import numpy as np
from matplotlib import pyplot as plt

DATA_PATH = "../data/4bit_spikeset_PHASE_full.npy"
DEBUG = True

classes = [
    "walking",
    "jogging",
    "stairs",
    "sitting",
    "standing",
    "typing",
    "brushing_teeth",
    "eating_soup",
    "eating_chips",
    "eating_pasta",
    "drinking",
    "eating_sandwich",
    "kicking_soccer",
    "catch_tennis",
    "dribbling_basketball",
    "writing",
    "clapping",
    "folding_clothes",
]


def load_data(filename):
    x, y = None, None
    with open(filename, "rb") as file:
        data = np.load(file, allow_pickle=True)
        x = data[:, 0:2]
        y = data[:, 2]

    return x, y


def filter_data_by_class(x, y, c):
    return x[y == c]


def plot_sample(x):
    plt.scatter(x[1], x[0], marker="|")
    plt.yticks(range(12))
    plt.ylim((-1, 12))
    plt.show()


if __name__ == "__main__":
    x, y = load_data(DATA_PATH)
    x_c = filter_data_by_class(x, y, 2)
    plot_sample(x_c[0])
