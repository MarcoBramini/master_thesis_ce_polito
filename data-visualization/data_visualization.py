import numpy as np
from matplotlib import pyplot as plt

DATA_PATH = "../data/data_watch_full_40.npz"
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


def plot_sample(x):
    x_val = np.linspace(0, 2, 40)
    plt.plot(
        x_val,
        x,
    )
    # plt.yticks(range(12))
    # plt.ylim((-1, 12))
    plt.show()


if __name__ == "__main__":
    train_x, train_y, val_x, val_y, test_x, test_y = load_data(DATA_PATH)
    x_c = filter_data_by_class(train_x, train_y, 2)
    plot_sample(x_c[0])
