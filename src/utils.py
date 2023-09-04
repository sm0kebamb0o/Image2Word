import numpy as np


def find_consecutive(row):
    """
    Finds consecutive elements in row and returns subsequences lens, 
    elements and indexes of thie ending
    """
    assert len(row.shape) == 1
    # pairwise unequal (string safe)
    mask = row[1:] != row[:-1]
    # ищем где несовпадают значения соседних пикселей
    # must include last element posi
    inds = np.append(np.where(mask), row.shape[0] - 1)
    # находим разницу между правым и левым элементом, то есть нужные нам промежутки
    lens = np.diff(np.append(-1, inds))       # run lengths
    return (lens, row[inds], inds)


def calc_kernel_size(anchor: int, percent: float) -> int:
    """
    Calculate kernel size for OpenCV operations
    """
    kernel_size = int(percent * anchor)
    kernel_size += ((kernel_size & 1) + 1) & 1
    return kernel_size


def calc_gaussian_kernel_size(anchor: int, percent: float) -> int:
    """
    Calculate kernel size for OpenCV operations with Gaussian filter
    """
    kernel_size = calc_kernel_size(anchor, percent)
    return kernel_size if kernel_size > 1 else 3
