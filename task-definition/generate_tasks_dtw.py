from commons import load_data, calc_distance_matrix, search_class_combinations
from dtaidistance import dtw_ndim
import numpy as np

# OPTIONS
DATA_PATH = "../data/wisdm_watch_full_40.npz"
CLASS_LABELS_PATH = "../data/wisdm_watch_full_40_classes.json"
DEBUG = False
N_CLASSES = 2  # Controls the size of the class combinations to generate
BEST_COMBS = True  # Controls if the script must return the most separable combinations (True) or less separable ones (False).
DISPLAY_MATRIX = False  # Enables plotting the distance matrix on the screen


def calc_median_sample(x, y, c):
    """
    Calculates the median sample for the class provided through the c input parameter.
    Input Parameters:
      - x: Samples
      - y: Labels, must correspond with the samples
      - c: An integer selecting the class, must be in the interval 0-17
    """
    return np.median((x[np.argmax(y, axis=1) == c]), axis=0)


def calc_median_sample_all_classes(train_x, train_y, n_classes):
    """
    Calculates the median samples for each class in the training subset.
    Input Parameters:
      - train_x: Training samples
      - train_y: Training labels, must correspond with the samples
      - n_classes: The number of classes in the dataset
    """
    median_x = []
    for c in range(n_classes):
        median_x.append(calc_median_sample(train_x, train_y, c))
    median_y = [
        [1 if c == i else 0 for i in range(n_classes)] for c in range(n_classes)
    ]
    return np.array(median_x), median_y


def calc_dtw_distance(x, y, c1, c2):
    """
    Calculates the DTW distance between the median samples of two classes.
    Input Parameters:
      - x: List containing the median samples for all the classes (must contain 18 items)
      - y: Labels, must correspond with the samples
      - c1: An integer selecting the first class, must be in the interval 0-17
      - c2: An integer selecting the second class, must be in the interval 0-17
    """
    x_c1 = x[np.argmax(y, axis=1) == c1]
    x_c2 = x[np.argmax(y, axis=1) == c2]

    scores = []
    for a in x_c1:
        for b in x_c2:
            scores.append(dtw_ndim.distance_fast(a, b))

    return np.mean(scores)


def distance_fn(median_x, median_y):
    """
    Initialize the distance function to employ for class combination search.
    Input Parameters:
      - x: List containing the median samples for all the classes (must contain 18 items)
      - y: Labels, must correspond with the samples
    """

    def _fn(c1, c2) -> float:
        return calc_dtw_distance(median_x, median_y, c1, c2)

    return _fn


if __name__ == "__main__":
    train_x, train_y, _, _, _, _, data_prop = load_data(
        DATA_PATH, CLASS_LABELS_PATH, debug=DEBUG
    )

    # Calculate average sample for each class
    median_x, median_y = calc_median_sample_all_classes(
        train_x, train_y, data_prop["n_classes"]
    )

    # Calculate the DTW matrix
    matrix, pairs_distances_dict = calc_distance_matrix(
        distance_fn(median_x, median_y),
        data_prop,
        display_matrix=DISPLAY_MATRIX,
    )

    # Calculate the optimal combinations of the classes
    result_sorted = search_class_combinations(
        data_prop, pairs_distances_dict, N_CLASSES, BEST_COMBS
    )
    print(result_sorted[0:5])
