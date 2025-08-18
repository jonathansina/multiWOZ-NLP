import numpy as np


def find_zero_percentage(loader, name, max_length):
    return np.mean(
        [
            (loader.dataset[i][name] == 0).sum() / max_length
            for i in range(len(loader.dataset))
        ]
    )