import json
import random
import time
import sklearn.datasets
from enum import Enum
from xgboost import XGBClassifier
import matplotlib.pyplot
import numpy


class Operation(Enum):
    PLUS = "+"
    MINUS = "-"
    TIMES = "*"
    DIVIDED_BY = "/"
    EXPONENT = "^"
    TANH = "tanh"
    SUM = "sum"
    BCE = "Binary-Cross Entropy"

    @staticmethod
    def operate(operation, values: list):
        if len(values) == 2:
            if operation is Operation.PLUS:
                return values[0] + values[1]
            elif operation is Operation.MINUS:
                return values[0] - values[1]
            elif operation is Operation.TIMES:
                return values[0] * values[1]
            elif operation is Operation.DIVIDED_BY:
                if values[1] == 0:
                    raise ValueError("Cannot divide by zero")
                return values[0] / values[1]
        elif operation is Operation.TANH:
            return numpy.tanh(values[0])
        elif operation is Operation.SUM:
            return sum(values)
        else:
            raise ValueError("Operation not defined")


class Value:
    def __init__(self, data, children=(), operation: Operation = None, label: str = None):
        self.data = data
        self.children = children
        self.operation: Operation = operation

        self._setup_label(label)

        self._calculate_child_gradients = lambda: None
        # The gradient with respect to the loss function (dL/d-self)
        # 'loss' will be the node on which back_propagation() will be called
        # gradients accumulate, so they need to be initialized at 0
        self.gradient_with_respect_to_loss = 0.0

    def _setup_label(self, label):
        if label is not None:
            self.label = label
        elif len(self.children) == 2:
            child1: Value = self.children[0]
            child2 = self.children[1]
            self.label = child1._label(child2, self.operation)
        else:
            self.label = None

    # The string that is print when you do print(object)
    def __repr__(self):
        return f"Value(data={self.data}, label={self.label}, operation={self.operation}, gradient={self.gradient_with_respect_to_loss})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other, label=f"{other}")
        result = Value(self.data + other.data, (self, other), Operation.PLUS)

        def _gradient_calculation():
            self.gradient_with_respect_to_loss += 1 * result.gradient_with_respect_to_loss
            other.gradient_with_respect_to_loss += 1 * result.gradient_with_respect_to_loss

        result._calculate_child_gradients = _gradient_calculation
        return result

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other, label=f"{other}")
        result = Value(self.data - other.data, (self, other), Operation.MINUS)

        def _gradient_calculation():
            self.gradient_with_respect_to_loss += 1 * result.gradient_with_respect_to_loss
            other.gradient_with_respect_to_loss += -1 * result.gradient_with_respect_to_loss

        result._calculate_child_gradients = _gradient_calculation
        return result


    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other, label=f"{other}")

        # the result is the parent during back prop
        result = Value(self.data * other.data, (self, other), Operation.TIMES)

        def _gradient_calculation():
            # since this is in a block the new_value.gradient_with_respect_to_loss
            # isn't called until it is calculated.
            self.gradient_with_respect_to_loss += other.data * result.gradient_with_respect_to_loss
            other.gradient_with_respect_to_loss += self.data * result.gradient_with_respect_to_loss

        # during backprop the parent (i.e. "new_value") sets the children's gradients
        result._calculate_child_gradients = _gradient_calculation
        return result

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other, label=f"{other}")
        result = Value(self.data / other.data, (self, other), Operation.DIVIDED_BY, self._label(other, Operation.DIVIDED_BY))

        def _gradient_calculation():
            self.gradient_with_respect_to_loss += 1 / other.data * result.gradient_with_respect_to_loss

            # d / d-other = - a / b^2 (because a/b = a*b^-1)
            other.gradient_with_respect_to_loss -= self.data / (other.data ** 2) * result.gradient_with_respect_to_loss

        result._calculate_child_gradients = _gradient_calculation
        return result

    def _label(self, other, operator: Operation) -> str:
        if (operator == Operation.TIMES or operator == Operation.DIVIDED_BY): # and len(self.label) == 1:
            new_label = f"({self.label}){operator.value}{other.label}"
        else:
            new_label = f"{self.label}{operator.value}{other.label}"
        return new_label

    def __rmul__(self, other):
        other = other if isinstance(other, Value) else Value(other, label=f"{other}")
        # reverse multiply
        # Handles the case: other.__mul__(self)
        # crashes because self.data is needed
        # example: 2 * value
        new_value = self * other
        new_value.label = other._label(self, Operation.TIMES)
        return new_value

    def __radd__(self, other):
        other = other if isinstance(other, Value) else Value(other, label=f"{other}")
        new_value = self + other
        new_value.label = f"{other.label}+{self.label}"
        return new_value

    def __rsub__(self, other):
        other = other if isinstance(other, Value) else Value(other, label=f"{other}")
        new_value = (self - other) * -1
        new_value.label = f"{other.label}-{self.label}"
        return new_value

    def exp(self, other):
        other = other if isinstance(other, Value) else Value(other, label=f"{other}")

        # the result is the parent during back prop
        result = Value(self.data ** other.data, (self, other), Operation.EXPONENT)

        def _gradient_calculation():
            # since this is in a block the new_value.gradient_with_respect_to_loss
            # isn't called until it is calculated.

            if self.data < 0:
                raise ValueError("Negative base not supported for logarithm in derivative calculation.")
                # Or, if using absolute value:
                # log_term = numpy.log(abs(self.data))
            else:
                log_term = numpy.log(self.data)

            # d / d-other = b * a ^ (b-1) (because the derivative of a/b with respect to a is a*b^-1)
            self.gradient_with_respect_to_loss += other.data * self.data ** (other.data - 1)

            # d / d-other = a^b * log(a) (because the derivative of a/b with respect to b is a^b*log(a) )
            other.gradient_with_respect_to_loss += self.data ** other.data * log_term

        # during backprop the parent (i.e. "result") sets the children's gradients
        result._calculate_child_gradients = _gradient_calculation
        return result

    def tanh(self, label=None):
        # y = (e^2x - 1) / (e^2x + 1)
        x = self.data
        y = numpy.tanh(x) #(numpy.e ** (2 * x) - 1) / (numpy.e ** (2 * x) + 1)

        # the result is the parent during back prop
        result = Value(y, (self, ), Operation.TANH, label)

        def _gradient_calculation():
            # since this is in a block the new_value.gradient_with_respect_to_loss
            # isn't called until it is calculated.

            # dd / dx = 1 - tanh(x)^2
            self.gradient_with_respect_to_loss += 1 - result.data ** 2

        # during backprop the parent (i.e. "result") sets the children's gradients
        result._calculate_child_gradients = _gradient_calculation
        return result

    def binary_cross_entropy(self, y_true):
        y_true = y_true if isinstance(y_true, Value) else Value(y_true, label=f"y_true")

        # Ensure predictions are in range (1e-7, 1 - 1e-7) to avoid log(0)
        y_pred = numpy.clip(self.data, 1e-7, 1 - 1e-7)
        loss = - (y_true.data * numpy.log(y_pred) + (1 - y_true.data) * numpy.log(1 - y_pred))

        result = Value(loss, (self, y_true), Operation.BCE)

        def _gradient_calculation():
            # Compute the gradients for backpropagation
            # dL/dp = - (y/p - (1-y)/(1-p))
            self.gradient_with_respect_to_loss += (- y_true.data / y_pred + (1 - y_true.data) / (
                        1 - y_pred)) * result.gradient_with_respect_to_loss

            # Gradient for the target is not needed as it's a fixed true label
            y_true.gradient_with_respect_to_loss += 0.0

        result._calculate_child_gradients = _gradient_calculation
        return result

    def _gradient_descent(self, step_size):
        self.data += step_size * self.gradient_with_respect_to_loss

    def back_propagation(self, perform_gradient_descent: bool, step_size):
        # Backpropagation: iterate over the graph in reverse (from outputs to inputs)

        # Sort the graph in topological order to ensure proper gradient propagation
        # Essential for correctly applying the chain rule in backpropagation.
        topologically_sorted_graph = topological_sort(self)

        # reset gradients
        for value in topologically_sorted_graph:
            value.gradient_with_respect_to_loss = 0

        # Initialize the gradient of the loss function as 1
        self.gradient_with_respect_to_loss = 1

        # At each node, apply the chain rule to calculate and accumulate gradients.
        for node in reversed(topologically_sorted_graph):
            node._calculate_child_gradients()

            if perform_gradient_descent and node is not self:
                # gradient descent
                node._gradient_descent(step_size)


def topological_sort(value: Value) -> ['Value']:
    # sort the graphy such that children get added after parents
    topologically_sorted_graph = []
    visited = set()

    def build_topological_sort(value):
        if value not in visited:
            visited.add(value)
            for child in value.children:
                build_topological_sort(child)
            topologically_sorted_graph.append(value)

    build_topological_sort(value)
    return topologically_sorted_graph


class Neuron:

    def __init__(self, inputs, state=None):
        if state:
            self.__setstate__(state)
        else:
            # initialize n weights as random numbers between -1 and 1
            # where n is the number of inputs
            self.weights = [Value(random.uniform(-0.15,0.15), label=f"w{i}") for i in range(inputs)]
            self.bias = Value(random.uniform(-0.15,0.15), label="b")

    def __call__(self, inputs):
        # w * x + b

        zipped_weights_and_input = zip(self.weights, inputs)
        stimulation = self.bias.data
        children = [self.bias]
        for weight, input_x in zipped_weights_and_input:
            input_x = input_x if isinstance(input_x, Value) else Value(input_x, label=f"{input_x}")
            stimulation_by_single_input = weight * input_x
            stimulation += stimulation_by_single_input.data
            children.append(stimulation_by_single_input)

        activation = Value(stimulation, children=tuple(children), operation=Operation.SUM, label="cell body stimulation")

        def _gradient_calculation():
            for child in children:
                child.gradient_with_respect_to_loss += 1 * activation.gradient_with_respect_to_loss

        activation._calculate_child_gradients = _gradient_calculation

        out = activation.tanh()
        out.label = "activation"
        return out

    def __getstate__(self):
        return {
            "weights": [value.data for value in self.weights],
            "bias": self.bias.data
        }

    def __setstate__(self, state):
        state_weights = state["weights"]
        self.weights = [Value(state_weights[i], label=f"w{i}") for i in range(len(state_weights))]
        self.bias = Value(state["bias"])

    def _state_equal_to(self, other):
        for i in range(len(self.weights)):
            if self.weights[i].data != other.weights[i].data:
                return False
        if self.bias.data != other.bias.data:
            return False
        return True

    def _test_state(self):
        state = self.__getstate__()
        copy = Neuron(None, state)
        if not self._state_equal_to(copy):
            raise ValueError(f"State instantiation error")



class FullyConnectedLayer:

    def __init__(self, number_of_inputs_to_layer, neurons_in_layer, state=None):
        if state:
            self.__setstate__(state)
        else:
            self.neurons = [Neuron(number_of_inputs_to_layer) for _ in range(neurons_in_layer)]

    def __call__(self, x_input_to_layer):
        outs = []
        for neuron in self.neurons:
            neuron_output = neuron(x_input_to_layer)
            outs.append(neuron_output)
        return outs[0] if len(outs) == 1 else outs

    def __getstate__(self):
        return {
            "neurons": [neuron.__getstate__() for neuron in self.neurons]
        }

    def __setstate__(self, state):
        neurons = state["neurons"]
        self.neurons = [Neuron(None, neurons[i]) for i in range(len(neurons))]

    def _state_equal_to(self, other):
        for i in range(len(self.neurons)):
            if not self.neurons[i]._state_equal_to(other.neurons[i]):
                return False
        return True

    def _test_state(self):
        state = self.__getstate__()
        copy = FullyConnectedLayer(None, None, state)
        if not self._state_equal_to(copy):
            raise ValueError(f"State instantiation error")


class MultilayerFullyConnectedNetwork:

    def __init__(self, number_of_inputs, list_of_layer_output_dimensions, state=None):
        if state:
            self.__setstate__(state)
        else:
            # layers will live in between the items in the array
            edges_between_layers = [number_of_inputs] + list_of_layer_output_dimensions

            self.layers = []
            # the layers are a list of input -> output pairs for that layer
            for i in range(len(list_of_layer_output_dimensions)):
                layer_input_size = edges_between_layers[i]
                layer_output_size = edges_between_layers[i+1]
                fully_connected_layer = FullyConnectedLayer(layer_input_size, layer_output_size)
                self.layers.append(fully_connected_layer)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        # convert to probability (between 0-1) and return
        return (x + 1) / 2


    def __getstate__(self):
        return {
            "layers": [fcl.__getstate__() for fcl in self.layers]
        }

    def __setstate__(self, state):
        layers = state["layers"]
        self.layers = [FullyConnectedLayer(None, None, layers[i]) for i in range(len(layers))]

    def _state_equal_to(self, other):
        for i in range(len(self.layers)):
            if not self.layers[i]._state_equal_to(other.layers[i]):
                return False
        return True

    def _test_state(self):
        state = self.__getstate__()
        copy = MultilayerFullyConnectedNetwork(None, None, state)
        if not self._state_equal_to(copy):
            raise ValueError(f"State instantiation error")

    def save_to_file(self, file_name):
        with open(file_name, 'w') as file:
            file.write(json.dumps(self.__getstate__()))

    @classmethod
    def load_from_file(self, file_name):
        with open(file_name) as f:
            txt = f.read()
            state = json.loads(txt)
            return MultilayerFullyConnectedNetwork(None, None, state)


def compute_validation_loss(network, validation_inputs, validation_targets):
    total_loss = 0
    for x, y in zip(validation_inputs, validation_targets):
        out = network(x)
        loss = out.binary_cross_entropy(y)
        total_loss += loss.data
    average_loss = total_loss / len(validation_inputs)
    return average_loss



def plot_losses(training_losses, validation_losses):
    matplotlib.pyplot.figure(figsize=(10, 6))
    matplotlib.pyplot.plot(training_losses, label='Training Loss')
    matplotlib.pyplot.plot(validation_losses, label='Validation Loss')
    matplotlib.pyplot.xlabel('Epoch')
    matplotlib.pyplot.ylabel('Loss')
    matplotlib.pyplot.title('Training and Validation Loss Over Epochs')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()


def xgboost(x_train, x_validate, y_train, y_validate):
    # create model instance
    model = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
    # fit model
    model.fit(x_train, y_train)
    # make predictions
    y_preds = model.predict(x_validate)

    number_correct = 0
    for y_pred, y_true in zip(y_preds, y_validate):
        if y_pred == y_true:
            number_correct += 1

    accuracy = number_correct / len(x_validate)
    print(f"XGBoost validation accuracy: {accuracy * 100}%")

    return model


def train(model, epochs, step_size):

    training_data_x = numpy.genfromtxt('kaggle_dsl_scikit_learn/train.csv', delimiter=',')
    training_data_y = numpy.genfromtxt('kaggle_dsl_scikit_learn/trainLabels.csv', delimiter=',')

    x_train, x_validate, y_train, y_validate = sklearn.model_selection.train_test_split(training_data_x, training_data_y, test_size=0.10, random_state=42)

    training_losses = []
    validation_losses = []

    for epoch in range(epochs):
        epoch_start = time.time()

        total_epoch_loss = 0

        # Training loop
        for (index, x) in enumerate(x_train):
            y_true = y_train[index]

            y_pred = model(x)
            loss = y_pred.binary_cross_entropy(y_true)

            loss.back_propagation(perform_gradient_descent=True, step_size=step_size)

            total_epoch_loss += loss.data

        # Calculate the average loss for this epoch
        average_training_loss = total_epoch_loss / len(x_train)

        # Append the average loss to the training_losses list
        training_losses.append(average_training_loss)

        # Calculate validation loss and append to validation_losses
        validation_loss = compute_validation_loss(model, x_validate, y_validate)
        validation_losses.append(validation_loss)

        model_name = f"network_{step_size}_{epoch}_{round(validation_loss,2)}.txt"
        print(f"Epoch {epoch + 1}, Training Loss: {total_epoch_loss}, Validation Loss: {validation_loss} in {time.time() - epoch_start} seconds - {model_name}")
        model.save_to_file(model_name)

    xgboost_model: XGBClassifier = xgboost(x_train, x_validate, y_train, y_validate)
    xgboost_model.save_model('xgb_model.json')

    plot_losses(training_losses, validation_losses)


def apply_binary_threshold(probability, threshold):
    if isinstance(probability, list):
        predictions = []
        for p in probability:
            predictions.append(apply_binary_threshold(p, threshold))
        return predictions
    else:
        if probability >= threshold:
            return 1
        else:
            return 0


def save_for_kaggle_submission(predictions, file_name):
    # Create a 2D array with indices and data
    indexed_data = numpy.column_stack((numpy.arange(1,len(predictions) + 1), predictions))
    numpy.savetxt(file_name, indexed_data, delimiter=',', fmt='%d', header='Id,Solution', comments='')


if __name__ == '__main__':

    start_time = time.time()

    # load the ANN from disk
    ann = MultilayerFullyConnectedNetwork.load_from_file("network_4e-05_0_1.22.txt")
    # or create your own
    # ann = MultilayerFullyConnectedNetwork(x_train.shape[0], [2, 3, 5, 4, 1])
    # ann = MultilayerFullyConnectedNetwork(x_train.shape[0], [50, 20, 1])

    train_first = False
    if train_first:
        epochs = 1
        step_size = 0.00001
        train(ann, epochs, step_size)

    cvs_to_test_path = 'kaggle_dsl_scikit_learn/test.csv'
    test_x = numpy.genfromtxt(cvs_to_test_path, delimiter=',')

    predictions = []
    low_confidence_indexes = []
    for index, x in enumerate(test_x):
        out = ann(x).data

        if 0.25 < abs(out) < 0.75:
            low_confidence_indexes.append(index)

        y = apply_binary_threshold(out, 0.5)
        predictions.append(y)

        if index % 1000 == 0:
            print(f"Testing {index / len(test_x) * 100}% done")

    save_for_kaggle_submission(predictions, f'nn_predictions{start_time}.csv')

    print(f"NN test completed in {time.time() - start_time} seconds")

    low_confidence_y = {}
    for index in low_confidence_indexes:
        low_confidence_y[index] = test_x[index]

    xgboost_model = XGBClassifier()
    xgboost_model.load_model('xgb_model.json')

    xgb_predictions = {}
    for index in low_confidence_y.keys():
        x = low_confidence_y[index]

        # x is (40, ), xgboost wants (1, 40)
        reshaped_x = x.reshape(1, -1)
        xgb_prediction = xgboost_model.predict(reshaped_x)[0]
        xgb_predictions[index] = xgb_prediction

        # compare the two models on low confidence examples
        nn_prediction = predictions[index]
        print(f"Index: {index} - nn: {nn_prediction}; xgb: {xgb_prediction}")

        predictions[index] = xgb_prediction

    print(predictions)
    save_for_kaggle_submission(predictions, f'xgb_predictions{start_time}.csv')

    print(f"Done in {time.time() - start_time} seconds")

