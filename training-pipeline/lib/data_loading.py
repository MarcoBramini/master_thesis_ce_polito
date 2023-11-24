# Utils
import numpy as np
from sklearn.model_selection import train_test_split
import tonic
import logging

# Rockpool Imports
from rockpool.timeseries import TSEvent

# Torch Imports
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger("main")


def load_data_as_tsevents(
    file_path,
    n_channels,
    dt,
    sample_duration,
    sample_time_modifier=1,
    enabled_classes=None,
):
    with open(file_path, "rb") as f:
        data = np.array(np.load(f, allow_pickle=True), dtype=object)

        # Class filtering
        if enabled_classes is not None:
            mask = np.zeros(len(data), dtype=bool)
            for c in enabled_classes:
                mask += data[:, 2] == c
            data = data[list(mask)]

        tsevents = np.array(
            [
                TSEvent(
                    times=sample[1] * sample_time_modifier,
                    channels=sample[0],
                    t_start=0,
                    t_stop=sample_duration * sample_time_modifier,
                    num_channels=n_channels,
                ).raster(
                    dt,
                    t_start=0,
                    t_stop=sample_duration * sample_time_modifier,
                    add_events=True,
                )
                for sample in data
            ]
        )

        labels = np.array(
            [
                sample[2]
                if enabled_classes == None
                else enabled_classes.index(sample[2])
                for sample in data
            ]
        )

        logger.info(f"Loaded {len(tsevents)} events from file {file_path}")

        return tsevents, labels


def split_train_val_test(events, labels, partition_sizes=(0.6, 0.2, 0.2)):
    if np.sum(partition_sizes) != 1:
        e = "Partitions sizes must sum up to 1"
        logger.error(e)
        raise e

    x_train, x_val_test, y_train, y_val_test = train_test_split(
        events, labels, train_size=partition_sizes[0], random_state=42
    )

    relative_test_size = partition_sizes[2] / (partition_sizes[1] + partition_sizes[2])

    x_val, x_test, y_val, y_test = train_test_split(
        x_val_test, y_val_test, test_size=relative_test_size, random_state=42
    )

    logger.info(
        f"Generated events partitions: train({len(x_train)}), val({len(x_val)}, test({len(x_test)}))"
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], int(self.y[idx])


def convert_labels_one_hot(labels, n_classes, enabled_classes=None):
    return torch.tensor(
        [
            [
                1 if label == i else 0
                for i in range(len(enabled_classes) if enabled_classes else n_classes)
            ]
            for label in labels
        ]
    )


# - Load input data


def load_data(
    file_path,
    n_channels,
    n_classes,
    dt,
    sample_duration,
    batch_size=128,
    sample_time_modifier=1,
    enabled_classes=None,
    use_onehot_labels=False,
):
    """
    Loads an encoded data file, splits it into training, validation and test subsets and returns a Torch DataLoader for each of them.
    Input Params:
      - file_path: Path for the data file
      - n_channels: Number of the dataset's channels
      - n_classes: Number of labels in the dataset
      - dt: Duration of a timestep for the dataset
      - sample_duration: Duration of a single sample in seconds
      - batch_size: Size of a single batch of samples this function will output
      - sample_time_modifier: Dilates/Compresses a single sample to adapt to a different dt
      - enabled_classes: Filters the classes based on the indexes in this list
      - use_onehot_labels: Controls if the labels must be one-hot encoded or integer indexes
    """
    events, labels = load_data_as_tsevents(
        file_path,
        n_channels,
        dt,
        sample_duration,
        sample_time_modifier=sample_time_modifier,
        enabled_classes=enabled_classes,
    )

    x_train, x_val, x_test, y_train, y_val, y_test = split_train_val_test(
        events, labels
    )

    train_ds = CustomDataset(torch.Tensor(x_train), torch.Tensor(y_train))
    val_ds = CustomDataset(torch.Tensor(x_val), torch.Tensor(y_val))
    test_ds = CustomDataset(torch.Tensor(x_test), torch.Tensor(y_test))

    def collate_fn(batch):
        x, y = tonic.collation.PadTensors(batch_first=True)(batch)
        if use_onehot_labels:
            y = convert_labels_one_hot(y, n_classes, enabled_classes=enabled_classes)
        return x, y

    dataloader_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    train_dl = DataLoader(train_ds, **dataloader_kwargs)
    val_dl = DataLoader(val_ds, **dataloader_kwargs)
    test_dl = DataLoader(test_ds, **dataloader_kwargs)

    return train_dl, val_dl, test_dl
