import backpropagation
import backpropagation.network
import backpropagation.network.neural_network
import backpropagation.data
from backpropagation.data.utils import (
    read_idx, convert_images_to_training_samples,
    convert_image_labels_to_training_labels)
from backpropagation.network.test_neural_network import (
    test_multiple)
import click
import pickle
import matplotlib.pyplot as plt


@click.group()
def main():
    pass


@main.command()
@click.argument('neurons_dist_filepath', type=click.Path(exists=True))
@click.option('--epochs', default=5)
@click.option('--mini-batch-size', default=10)
@click.option('--output', default=None)
@click.option('--visualize-loss', is_flag=True)
def train(
    neurons_dist_filepath: str,
    epochs: int,
    mini_batch_size: int,
    output: str,
    visualize_loss: bool,
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

    train_images = read_idx("data/raw/train-images-idx3-ubyte.gz")
    train_labels = read_idx("data/raw/train-labels-idx1-ubyte.gz")

    X_train = convert_images_to_training_samples(train_images)
    y_train = convert_image_labels_to_training_labels(train_labels)

    nn = backpropagation.network.neural_network.NeuralNetwork(
        neurons_count_per_layer=neurons_counts,
        activation_function=backpropagation.network.activation_function.SigmoidActivationFunction(),
        cost_function=backpropagation.network.cost_function.MSECostFunction()
    )

    if visualize_loss:
        test_images = read_idx("data/raw/t10k-images-idx3-ubyte.gz")
        test_labels = read_idx("data/raw/t10k-labels-idx1-ubyte.gz")
        X_test = convert_images_to_training_samples(test_images)
        y_test = convert_image_labels_to_training_labels(test_labels)

        train_cost, test_cost = nn._stochastic_gradient_descent(
            X_train, y_train, mini_batch_size=mini_batch_size,
            epochs_count=epochs, test_samples=X_test, test_labels=y_test)

        plt.plot([str(i) for i in range(epochs)], train_cost, marker='o')
        plt.plot([str(i) for i in range(epochs)], test_cost, marker='o')
        plt.legend(['train_cost', 'test_cost'])
        plt.show()
    else:
        nn._stochastic_gradient_descent(
            X_train, y_train, mini_batch_size=mini_batch_size,
            epochs_count=epochs)

    if output:
        with open(output, 'wb') as f:
            pickle.dump(nn, f)


@main.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.option('--per-class-errors', is_flag=True)
@click.option('--all-errors', is_flag=True)
def test(
    model_path: str,
    per_class_errors: bool,
    all_errors: bool
):
    """Test neural network.

    Args:
        model_path (str): Path to a file containing neural network.
    """
    with open(model_path, 'rb') as f:
        nn = pickle.load(f)

    images = read_idx("data/raw/t10k-images-idx3-ubyte.gz")
    labels = read_idx("data/raw/t10k-labels-idx1-ubyte.gz")
    samples = convert_images_to_training_samples(images)

    results = test_multiple(nn, samples, labels)
    misclassified_samples = list()
    for i in range(len(results)):
        if not results[i]:
            misclassified_samples.append(samples[i])

    print(f"Success rate: {1-len(misclassified_samples)/len(samples):0.4f}")

    if all_errors:
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.1, wspace=0.001)
        dim = int(len(misclassified_samples) ** 0.5) + 1
        for i in range(len(misclassified_samples)):
            ax = fig.add_subplot(dim, dim, i+1)
            ax.set_axis_off()
            ax.imshow(misclassified_samples[i].reshape(
                28, 28)[:, :], cmap='gray')

        plt.show()

    if per_class_errors:
        error_percentage = list()
        for label in range(10):
            label_indexes = [i for i, v in enumerate(labels) if v == label]
            tp = sum(results[i] == True for i in label_indexes)
            error_percentage.append(1-tp/len(label_indexes))

        plt.bar([str(i) for i in range(10)],
                error_percentage, align='center', alpha=0.5)
        plt.ylabel('Error percentage')
        plt.xlabel('Label')

        plt.show()


if __name__ == '__main__':
    main()
