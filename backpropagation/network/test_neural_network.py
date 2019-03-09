from .neural_network import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
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


def test_multiple_with_visualization(
    nn: NeuralNetwork, samples: List[np.array], labels: List[int]
) -> List[bool]:
    results = test_multiple(nn, samples, labels)
    misclassified = list()
    for i in range(0, len(results)):
        if not results[i]:
            misclassified.append(samples[i])

    print(len(misclassified))
    plt.imshow(misclassified[0].reshape(28, 28)[:, :], cmap='gray')
    plt.show()
