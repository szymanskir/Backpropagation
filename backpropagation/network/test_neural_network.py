from .neural_network import NeuralNetwork
import numpy as np
from typing import List, Tuple

def test_multiple(
    nn: NeuralNetwork, samples: List[np.array], labels: List[int]
) -> List[Tuple[int, float, int]]:
    results = list()
    for sample, label in zip(samples, labels):
        output = nn._feedforward(sample)
        predicted_class = np.argmax(output)
        confidence = np.max(output)
        results.append((predicted_class, confidence, label))

    return results