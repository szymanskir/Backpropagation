from .neural_network import NeuralNetwork
import numpy as np
from typing import List


def predict(nn: NeuralNetwork, sample: np.array) -> int:
    output = nn._feedforward(sample)
    predicted_class = np.argmax(output)

    return predicted_class


def test_single(nn: NeuralNetwork, sample: np.array, label: int) -> bool:
    predicted_class = predict(nn, sample)
    return predicted_class == label


def test_multiple(
    nn: NeuralNetwork, samples: List[np.array], labels: List[int]
) -> List[bool]:
    return [predict(nn, s) == l for s, l in zip(samples, labels)]
