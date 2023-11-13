import numpy as np
from matplotlib import pyplot as plt
from dtaidistance import dtw_ndim
import itertools as it

DATA_PATH = "../data/data_watch_full_40.npz"
DEBUG = True
N_CLASSES = 2
BEST = True
SHOW_MATRIX = False

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
    with open(filename, "rb") as file:
        data = np.load(file, allow_pickle=True)
        train_x, train_y = data["arr_0"], data["arr_3"]
        val_x, val_y = data["arr_1"], data["arr_4"]
        test_x, test_y = data["arr_2"], data["arr_5"]

        if DEBUG:
            print(f"n_timesteps: {train_x.shape[1]}")
            print(f"n_dim: {train_x.shape[2]}")
            print(f"n_samples_train: {train_x.shape[0]}")
            print(f"n_samples_val: {val_x.shape[0]}")
            print(f"n_samples_test: {test_x.shape[0]}")

    return train_x, train_y, val_x, val_y, test_x, test_y


# Plot class samples counts
def plot_class_samples_count(y):
    fig, ax = plt.subplots()

    ax.hist(np.argmax(y, axis=1), bins=np.arange(0, len(classes) + 1) - 0.5, rwidth=0.5)
    plt.xticks(rotation=-45, ticks=np.arange(0, len(classes)), labels=classes)
    plt.xlim((-1, len(classes)))
    plt.tight_layout()
    plt.show()


def calc_avg_sample_per_class(x, y, c):
    return np.median((x[np.argmax(y, axis=1) == c]), axis=0)


def plot_sample(x):
    plt.plot(x[:, :])
    plt.show()


def calc_mean_dtw_distance(x, y, c1, c2):
    x_c1 = x[np.argmax(y, axis=1) == c1]
    x_c2 = x[np.argmax(y, axis=1) == c2]

    scores = []

    for a in x_c1:
        for b in x_c2:
            scores.append(dtw_ndim.distance_fast(a, b))

    return np.mean(scores)


def plot_sample_distance_matrix(x, y):
    matrix = np.zeros((len(classes), len(classes)))
    values = []
    for c1 in range(len(classes)):
        for c2 in range(len(classes)):
            val = calc_mean_dtw_distance(x, y, c1, c2)
            matrix[c1][c2] = val
            if c1 <= c2:
                values.append((c1, c2, np.round(val, decimals=3)))

    if SHOW_MATRIX:
        plt.figure(figsize=(6, 6))
        plt.matshow(matrix, fignum=1, cmap="hot")
        plt.colorbar()
        plt.show()

    return matrix, sorted(values, key=lambda x: x[2], reverse=True)


if __name__ == "__main__":
    train_x, train_y, val_x, val_y, test_x, test_y = load_data(DATA_PATH)

    # Calculate average sample per class
    avg_x = []
    for c in range(len(classes)):
        avg_x.append(calc_avg_sample_per_class(train_x, train_y, c))
    avg_x = np.array(avg_x)

    avg_y = [
        [1 if c == i else 0 for i in range(len(classes))] for c in range(len(classes))
    ]

    # Calculate the DTW between class distance matrix
    matrix, ranking = plot_sample_distance_matrix(avg_x, avg_y)
    ranking = {(e[0], e[1]): e[2] for e in ranking}

    # Calculate the combinations of the classes with biggest distance
    combinations = list(it.combinations(range(len(classes)), N_CLASSES))
    result = []
    for comb in combinations:
        sum = 0
        for coup in list(it.combinations(comb, 2)):
            sum += ranking[coup]
        result.append(([classes[i] for i in comb], sum))

    result_sorted = sorted(result, key=lambda x: x[1], reverse=BEST)
    print(result_sorted[0:5])
