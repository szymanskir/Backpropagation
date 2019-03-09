import backpropagation
import backpropagation.network
import backpropagation.network.neural_network
import backpropagation.data
import backpropagation.data.utils

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
    neurons_count_per_layer=[28*28, 100, 10],
    activation_function=backpropagation.network.activation_function.SigmoidActivationFunction(),
    cost_function=backpropagation.network.cost_function.MSECostFunction()
)

nn._stochastic_gradient_descent(X_train, y_train, mini_batch_size = 10, epocs_count = 30)
breakpoint()
