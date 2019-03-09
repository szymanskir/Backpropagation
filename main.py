import backpropagation
import backpropagation.network
import backpropagation.network.neural_network
import backpropagation.data
import backpropagation.data.utils

import click
import pickle


@click.group()
def main():
    pass


@main.command()
@click.argument('neurons_dist_filepath', type=click.Path(exists=True))
@click.option('--epochs', default=5)
@click.option('--mini-batch-size', default=10)
@click.option('--output', default=None)
def train(
    neurons_dist_filepath: str,
    epochs: int,
    mini_batch_size: int,
    output: str,
):
    """Train neural network.

    Args:
        neurons_dist_filepath (str): Path to file containing neural network
        definition in form of comma separted list: layer_1_neurons_count,
        layer_2_neurons_count, ..., layer_last_neurons_count.
        epochs (int): Number of epochs.
        mini_batch_size (int): Size of the mini batch.
        output (str): Path to a file where neural network should be stored.
    """
    with open(neurons_dist_filepath) as f:
        data = f.readline().strip().split(",")
        neurons_counts = [int(i) for i in data]

    images = backpropagation.data.utils.read_idx(
        "data/raw/train-images-idx3-ubyte.gz")
    labels = backpropagation.data.utils.read_idx(
        "data/raw/train-labels-idx1-ubyte.gz"
    )

    X_train = backpropagation.data.utils.convert_images_to_training_samples(
        images
    )

    y_train = backpropagation.data.utils.convert_image_labels_to_training_labels(
        labels
    )

    nn = backpropagation.network.neural_network.NeuralNetwork(
        neurons_count_per_layer=neurons_counts,
        activation_function=backpropagation.network.activation_function.SigmoidActivationFunction(),
        cost_function=backpropagation.network.cost_function.MSECostFunction()
    )

    nn._stochastic_gradient_descent(
        X_train, y_train, mini_batch_size=mini_batch_size, epochs_count=epochs)

    if output:
        with open(output, 'wb') as f:
            pickle.dump(nn, f)


@main.command()
@click.argument('model_path', type=click.Path(exists=True))
def test(
    model_path: str,
):
    """Test neural network.

    Args:
        model_path (str): Path to a file containing neural network.
    """
    with open(model_path, 'rb') as f:
        obj = pickle.load(f)


if __name__ == '__main__':
    main()
