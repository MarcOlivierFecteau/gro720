import gzip
import os
import pickle

from dnn_framework.dataset import Dataset
import sys


class MnistDataset(Dataset):
    def __init__(self, split):
        root = os.path.dirname(os.path.realpath(__file__))
        if split == "training":
            path = os.path.join(root, "mnist_training.pkl.gz")
        elif split == "validation":
            path = os.path.join(root, "mnist_validation.pkl.gz")
        elif split == "testing":
            path = os.path.join(root, "mnist_testing.pkl.gz")
        else:
            raise ValueError("Invalid split")

        with gzip.open(path, "rb") as file:
            data = pickle.load(file)

        self._images = data["images"].astype(float)
        self._labels = data["labels"]

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        return self._images[index], self._labels[index]


if __name__ == "__main__":
    # NOTE: Removes the deprecation warning
    if len(sys.argv) > 1 and sys.argv[1] == "fix":
        root = os.path.dirname(os.path.realpath(__file__))
        splits = ["training", "validation", "testing"]

        for split in splits:
            path = os.path.join(root, f"mnist_{split}.pkl.gz")
            with gzip.open(path, "rb") as file:
                data = pickle.load(file)
            with gzip.open(path, "wb") as file:
                pickle.dump(data, file)
            print(f"Saved mnist_{split}.pkl.gz")
