import numpy as np
from matplotlib import pyplot as plt

DATA_PATH = "../data/data_watch_full_40.npz"
DEBUG = True

classes = [
    "Walking",
    "Jogging",
    "Stairs",
    "Sitting",
    "Standing",
    "Typing",
    "Brushing Teeth",
    "Eating Soup",
    "Eating Chips",
    "Eating Pasta",
    "Drinking",
    "Eating Sandwich",
    "Kicking Soccer",
    "Catch Tennis",
    "Dribbling Basketball",
    "Writing",
    "Clapping",
    "Folding Clothes",
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


    y = abs(np.array(sorted(np.argmax(y, axis=1), reverse=False))-17)
    ax.hist(y, bins=np.arange(0, len(classes) + 1) - 0.5, rwidth=0.2, orientation="horizontal")
    plt.grid(axis="x", linestyle='dotted', color="grey")
    plt.yticks(rotation=0, ticks=np.arange(0, len(classes)), labels=reversed(classes))
    plt.ylim((-1, len(classes)))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_x, train_y, val_x, val_y, test_x, test_y = load_data(DATA_PATH)
    plot_class_samples_count([*train_y,*val_y,*test_y])
