from commons import load_data, filter_data_by_class
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "../data/wisdm_watch_full_40.npz"
CLASS_LABELS_PATH = "../data/wisdm_watch_full_40_classes.json"
DEBUG = False
KDE_CLASSES = [0, 2]  # Controls the classes for which KDE must be calculated

if __name__ == "__main__":
    train_x, train_y, _, _, _, _, data_prop = load_data(
        DATA_PATH, CLASS_LABELS_PATH, debug=DEBUG
    )

    cmap = sns.color_palette("husl", len(KDE_CLASSES))

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

    for ci, c in enumerate(KDE_CLASSES):
        x_c = filter_data_by_class(train_x, train_y, c)
        for j in range(2):
            for i in range(3):
                d = 2 * i + j
                sns.kdeplot(
                    data=x_c[
                        :, :, d
                    ].flatten(),  # Use only data associated to dimension d
                    fill=True,
                    common_norm=False,
                    alpha=0.5,
                    ax=axs[i, j],
                    color=cmap[ci],
                    label=data_prop["class_labels"][c],
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

    filename = "kde_"
    for c in KDE_CLASSES:
        filename += str(c) + "_"
    filename = filename[:-1]
    plt.savefig(filename)
