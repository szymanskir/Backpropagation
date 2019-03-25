import struct
import gzip
import numpy as np


def read_idx(filename: str):
    """Reads idx file.

    Adapted from: https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40
    """
    with gzip.open(filename) as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def image2vector(image: np.array) -> np.array:
    return image.flatten()


def convert_images_to_training_samples(images: np.array) -> np.array:
    return np.array([
        image2vector(images[image, :, :]) / 255
        for image in range(images.shape[0])
    ])


def label2vector(label: int):
    x = np.zeros(10)
    x[label] = 1
    return(x)


def convert_image_labels_to_training_labels(
        labels: np.array) -> np.array:
    return np.array([label2vector(label) for label in labels])
