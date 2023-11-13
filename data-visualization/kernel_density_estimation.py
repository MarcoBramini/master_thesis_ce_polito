import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "../data/data_watch_full_40.npz"
classes = [0,5]


def load_wisdm2_data(filename):
    filepath = filename
    a = np.load(filepath, allow_pickle=True)
    return (a["arr_0"], a["arr_1"], a["arr_2"], a["arr_3"], a["arr_4"], a["arr_5"])


act_map = {             #       2cb     2cw     3cb     3cw     4cb     4cw     7cb     7cw
    "A": "walking",     #0              -               -
    "B": "jogging",     #1                                      *               *
    "C": "stairs",      #2              -               -
    "D": "sitting",     #3
    "E": "standing",    #4
    "F": "typing",      #5                              -
    "G": "teeth",       #6      *               *               *               *
    "H": "soup",        #7                                                      *
    "I": "chips",       #8      *               *               *               *
    "J": "pasta",       #9
    "K": "drinking",    #10
    "L": "sandwich",    #11                                                             -
    "M": "kicking",     #12                                                             -
    "O": "catch",       #13                     *               *       -       *       -
    "P": "dribbling",   #14                                                     *       -
    "Q": "writing",     #15                                             -               -
    "R": "clapping",    #16                                             -               -
    "S": "folding",     #17                                             -       *       -
}

(x_train, x_val, x_test, y_train_oh, y_val_oh, y_test_oh) = load_wisdm2_data(DATA_PATH)

# okay, the stuff we want to plot is in x_train.
# the labels are here;
y_train = np.argmax(y_train_oh, axis=-1)

# aight, here goes...

# fig = plt.figure(figsize=(8,4.5))
# fig, axs = plt.subplots(3,2)
cmap = plt.get_cmap("turbo")

# assign these to a dictionary for niceness
act_dict = dict(zip(np.unique(y_train), list(act_map.values())))
print(act_dict)

# print(x_train.shape)

# h = np.histogram(x_train[:,:,0])
# print(h)
# cmap = sns.color_palette("Spectral", as_cmap=True)(np.linspace(0,1,6))
cmap = sns.color_palette("husl", len(classes))
titles = [
    ["Accelerometer x-axis", "Accelerometer y-axis", "Accelerometer z-axis"],
    ["Gyroscope x-axis", "Gyroscope y-axis", "Gyroscope z-axis"],
]
x_labels = ["$m/{s^2}$", "$rad/s$"]
clips = [
    [(-25.0, +25.0), (-25.0, +25.0), (-2.5, +2.5)],
    [(-25.0, +25.0), (-2.5, +2.5), (-2.5, +2.5)],
]

fig, axs = plt.subplots(3, 2)
fig.set_figheight(10)
fig.set_figwidth(10)

for ci, _class in enumerate(classes):
    # select x for class
    class_indices = np.where(y_train == _class)
    class_samples = x_train[class_indices, :, :][0]
    for j in range(2):
        for i in range(3):
            ax_samples = class_samples[:, :, 2 * i + j].flatten()
            sns.kdeplot(
                data=ax_samples,
                fill=True,
                common_norm=False,
                alpha=0.5,
                # linewidth=0,
                ax=axs[i, j],
                color=cmap[ci],
                label=act_dict[_class],
                clip=clips[j][i],
            )
            axs[i][j].set_title(titles[j][i])
            if i == 2:
                axs[i][j].set_xlabel(x_labels[j])

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(
    by_label.values(),
    by_label.keys(),
    loc="lower center",
    prop={"size": "large"},
    ncol=len(by_label.keys()),
)
fig.tight_layout(rect=[0, 0.07, 1, 1])
plt.subplots_adjust(
    left=0.08, right=0.96, bottom=0.1, top=0.96, wspace=0.25, hspace=0.30
)
plt.savefig("kde")

