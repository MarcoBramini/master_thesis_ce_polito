import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy import stats
import itertools as it

DATA_PATH = "../data/data_watch_40.npz"
PROB_DIST_CLASSES_PATH = "prob_dist_classes.npy"
RUN_KDE = False
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


def filter_data_by_class(x, y, c):
    return x[np.argmax(y, axis=1) == c]


def calc_kde(x, x_plot):
    kde = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(x)
    return np.exp(kde.score_samples(x_plot))


def calc_kl_divergence(p, q):
    return stats.entropy(p, q) + stats.entropy(q, p)


def plot_class_kl_divergence_matrix(prob_dist_classes):
    matrix = np.zeros((len(classes), len(classes)))
    values = []
    for c1 in range(len(classes)):
        for c2 in range(len(classes)):
            val = 0
            for d in range(6):
                val += calc_kl_divergence(
                    prob_dist_classes[c1][d], prob_dist_classes[c2][d])
            matrix[c1][c2] = val
            if c1 <= c2:
                values.append((c1, c2, np.round(val, decimals=3)))

    plt.figure(figsize=(6, 6))
    plt.matshow(matrix, fignum=1, cmap="hot")
    plt.colorbar()
    plt.show()

    return matrix, sorted(values, key=lambda x: x[2], reverse=True)


if __name__ == "__main__":
    train_x, train_y, val_x, val_y, test_x, test_y = load_data(DATA_PATH)

    # Calculate KDE for each class
    prob_dist_classes = []
    if RUN_KDE:
        kde_x_plot = np.linspace(-20, 20, 1000)[:, np.newaxis]
        for c in range(len(classes)):
            prob_dist_class = []
            x_c = filter_data_by_class(train_x, train_y, c)
            for d in range(6):
                prob_dist_dim = calc_kde(x_c[:, :, d].flatten()[
                                         :, np.newaxis], kde_x_plot)
                prob_dist_class.append(prob_dist_dim)
            prob_dist_classes.append(prob_dist_class)
            if DEBUG:
                print(f"Class {c} done...")

        np.save(PROB_DIST_CLASSES_PATH, prob_dist_classes)
    else:
        prob_dist_classes = np.load(PROB_DIST_CLASSES_PATH)

    # Calculate KL divergence matrix
    matrix, ranking = plot_class_kl_divergence_matrix(prob_dist_classes)
    ranking = {(e[0], e[1]): e[2] for e in ranking}

    # Calculate the combinations of the classes with biggest distance
    n = 4
    combinations = list(it.combinations(range(len(classes)), n))
    result = []
    for comb in combinations:
        sum = 0
        for coup in list(it.combinations(comb, 2)):
            sum += ranking[coup]
        result.append(([classes[i] for i in comb], sum))

    result_sorted = sorted(result, key=lambda x: x[1], reverse=False)
    print(result_sorted[0:5])
