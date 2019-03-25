import backpropagation
import backpropagation.network
from backpropagation.data import (
    read_idx, convert_images_to_training_samples,
    convert_image_labels_to_training_labels)
from backpropagation.network.test_neural_network import (
test_multiple)
import click
import logging
import pickle
import random
import matplotlib.pyplot as plt
import Augmentor
from sklearn.metrics import roc_curve, auc

@click.group()
def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M:%S'
    )


@main.command()
@click.argument('neurons_dist_filepath', type=click.Path(exists=True))
@click.option('--epochs', default=10)
@click.option('--output', default=None)
@click.option('--visualize-loss', is_flag=True)
def train(
    neurons_dist_filepath: str,
    epochs: int,
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

    random.seed(44)
    p = Augmentor.Pipeline()
    p.rotate(probability=0.5, max_left_rotation=20, max_right_rotation=20)
    p.shear(probability=0.25, max_shear_left=10, max_shear_right=10)
    p.random_distortion(probability=0.25, grid_width=3, grid_height=3, magnitude=3)

    samples_count = int(len(train_images)*1.5)
    logging.info(f"Generating {samples_count} images...")
    g = p.keras_generator_from_array(train_images, Augmentor.Pipeline.categorical_labels(train_labels),
        batch_size=samples_count, scaled=False)
    X_aug, y_aug = next(g)
    X_train = convert_images_to_training_samples(X_aug.reshape(X_aug.shape[0], 28, 28))
    y_train = y_aug/1.0

    activation_functions = [backpropagation.network.SigmoidActivationFunction()] * (len(neurons_counts) - 2) + [backpropagation.network.SoftmaxActivationFunction()]
    nn = backpropagation.network.neural_network.NeuralNetwork(
        neurons_count_per_layer=neurons_counts,
        activation_functions=activation_functions,
        cost_function=backpropagation.network.cost_function.CrossEntropyCostFunction(),
        random_seed=44
    )

    if visualize_loss:
        test_images = read_idx("data/raw/t10k-images-idx3-ubyte.gz")
        test_labels = read_idx("data/raw/t10k-labels-idx1-ubyte.gz")
        X_test = convert_images_to_training_samples(test_images)
        y_test = convert_image_labels_to_training_labels(test_labels)

        train_cost, test_cost = nn._stochastic_gradient_descent(
            X_train,
            y_train,
            epochs_count=epochs,
            test_samples=X_test, 
            test_labels=y_test)

        plt.plot([str(i) for i in range(epochs)], train_cost, marker='o')
        plt.plot([str(i) for i in range(epochs)], test_cost, marker='o')
        plt.legend(['train_cost', 'test_cost'])
        plt.show()
    else:
        nn._stochastic_gradient_descent(X_train, y_train, epochs_count=epochs)

    if output:
        with open(output, 'wb') as f:
            pickle.dump(nn, f)


@main.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.option('--per-class-errors', is_flag=True)
@click.option('--misclassified', is_flag=True)
@click.option('--roc-curves', is_flag=True)
def test(
    model_path: str,
    per_class_errors: bool,
    misclassified: bool,
    roc_curves: bool
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
        if results[i][0] != results[i][2]:
            misclassified_samples.append(samples[i])

    logging.info(f"Success rate: {1-len(misclassified_samples)/len(samples):0.4f}")

    if per_class_errors:
        error_percentage = list()
        for label in range(10):
            current_results = [r for r in results if r[2] == label]
            tp = sum(r[0] == r[2] for r in current_results)
            error_percentage.append(1-tp/len(current_results))

        plt.bar([str(i) for i in range(10)],
                error_percentage, align='center', alpha=0.5)
        plt.ylabel('Error percentage')
        plt.xlabel('Label')

        plt.show()

    if misclassified:
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.1, wspace=0.001)
        dim = int(len(misclassified_samples) ** 0.5) + 1
        for i in range(len(misclassified_samples)):
            ax = fig.add_subplot(dim, dim, i+1)
            ax.set_axis_off()
            ax.imshow(misclassified_samples[i].reshape(
                28, 28)[:, :], cmap='gray')

        plt.show()


    if roc_curves:
        fig = plt.figure()
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for label in range(10):
            current_results = [r for r in results if r[2] == label]
            y_true = list()
            y_score = list()
            for r in current_results:
                if r[0] == r[2]:
                    y_true.append(1)
                    y_score.append(r[1])
                else:
                    y_true.append(0)
                    y_score.append(1-r[1])                    

            fpr[label], tpr[label], _ = roc_curve(y_true, y_score)
            roc_auc[label] = auc(fpr[label], tpr[label])
            
        for i in range(10):
            plt.plot(fpr[i], tpr[i], lw=2,
                    label='ROC curve of class {0} (area = {1:0.4f})'
                    ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic curve')
        plt.legend(loc="lower right")
        plt.show()


if __name__ == '__main__':
    main()
