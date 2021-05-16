"""
CSC 578 Spring 2021

NN578_network2.py
==============

Modified from the NNDL book code "network2.py".

network2.py
~~~~~~~~~~~~~~

An improved version of network.py, implementing the stochastic
gradient descent learning algorithm for a feedforward neural network.
Improvements include the addition of the cross-entropy cost function,
regularization, and better initialization of network weights.  Note
that I have focused on making the code simple, easily readable, and
easily modifiable.  It is not optimized, and omits many desirable
features.

"""

#### Libraries
import json
import random
import sys
import numpy as np


#### Definitions of the cost functions (as function classes)
class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output ``y``.
        """
        return 0.5 * np.linalg.norm(y - a) ** 2

    # 578: addition
    @staticmethod
    def derivative(a, y):
        """Return the first derivative of the function."""
        return -(y - a)


class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).
        """
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def derivative(a, y):
        """Return the first derivative of the function."""
        return np.divide((a-y), (a*(1-a)), out=np.zeros_like(a), where=((a*(1-a))!=0))


class LogLikelihood(object):
    
    @staticmethod
    def fn(a, y):
        
        return -np.log(a[np.argmax(y)])[0]
    
    @staticmethod
    def derivative(a, y):  
        for i in range(len(a)):
            a[i] = y[i] * (-1/a[i])     
        return a


#### Definitions of the activation functions (as function classes)
class Sigmoid(object):
    @staticmethod
    def fn(z):
        """The sigmoid function."""
        return 1.0 / (1.0 + np.exp(-z))

    @classmethod
    def derivative(cls, z):
        """Derivative of the sigmoid function."""
        return cls.fn(z) * (1 - cls.fn(z))


class Softmax(object):
    @staticmethod
    # Parameter z is an array of shape (len(z), 1).
    def fn(z):
        """The softmax of vector z."""
        return np.divide(np.exp(z), np.sum(np.exp(z)), out=np.zeros_like(z), where=(np.sum(np.exp(z)!=0)))

    @classmethod
    def derivative(cls, z):
        """Derivative of the softmax.  
        REMEMBER the derivative is an N*N matrix.
        """
        a = cls.fn(z)  # obtain the softmax vector
        return np.diagflat(a) - np.dot(a, a.T)


class Tanh(object):
    @staticmethod
    def fn(z):
        """The tanh function."""
        return np.divide(np.exp(z)-np.exp(-z), (np.exp(z)+np.exp(-z)))

    @classmethod
    def derivative(cls, z):
        """Derivative of the tanh function."""
        return 1-(cls.fn(z))**2

class ReLU(object):
    
    @staticmethod
    def fn(z):
        relu = np.zeros(z.shape)
        for i in range(len(z)):
            relu[i] = np.maximum(relu[i], 0) 
        return relu
    
    @classmethod
    def derivative(cls, z):
        drelu = cls.fn(z)
        for i in range(len(z)):
            if i > 0:
                drelu[i] = 1
            else:
                drelu[i] = 0
        return drelu


#### Main Network class
class Network(object):

    ## Additional keyword arguments for hyper-parameters
    def __init__(
        self,
        sizes,
        cost=CrossEntropyCost,
        act_hidden=Sigmoid,
        act_output=None,
        regularization=None,
        lmbda=0.0,
        dropoutpercent=0.0,
    ):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that method).
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()

        self.set_parameters(
            cost, act_hidden, act_output, regularization, lmbda, dropoutpercent
        )


    ## 578: convenience function for setting network hyperparameters
    def set_parameters(
        self,
        cost=QuadraticCost,
        act_hidden=Sigmoid,
        act_output=None,
        regularization=None,
        lmbda=0.0,
        dropoutpercent=0.0,
    ):
        self.cost = cost
        self.act_hidden = act_hidden
        if act_output == None:
            self.act_output = self.act_hidden
        else:
            self.act_output = act_output

        if self.act_output == Tanh:
            if self.cost != QuadraticCost:
                self.act_output = Sigmoid

        if dropoutpercent != 0.0:
            self.dropout = True
            self.dropoutlayers = []
        else: 
            self.dropout = False

        self.regularization = regularization
        self.lmbda = lmbda
        self.dropoutpercent = dropoutpercent

    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [
            np.random.randn(y, x) / np.sqrt(x)
            for x, y in zip(self.sizes[:-1], self.sizes[1:])
        ]

    def large_weight_initializer(self):
        """Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        This weight and bias initializer uses the same approach as in
        Chapter 1, and is included for purposes of comparison.  It
        will usually be better to use the default weight initializer
        instead.

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [
            np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])
        ]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        idx = 0
        for b, w in zip(self.biases, self.weights):
            if idx < len(self.biases)-1:
                a = (self.act_hidden).fn(np.dot(w, a)+b)
                # apply dropout layer to hidden activation layer(s)
                if (self.dropout):
                    a *= self.dropoutlayers[idx-2]
                idx += 1
            else: 
                a = (self.act_output).fn(np.dot(w, a)+b)
        return a

    ## 578: additional parameter 'no_convert' to control the vectorization of the target.
    def SGD(
        self,
        training_data,
        epochs,
        mini_batch_size,
        eta,
        lmbda=0.0,
        evaluation_data=None,
        monitor_evaluation_cost=False,
        monitor_evaluation_accuracy=False,
        monitor_training_cost=False,
        monitor_training_accuracy=False,
        no_convert=True,
    ):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.

        """
        ## 578: additional lines to possibly change the dataset
        ##   in case the output layer's activation function is tanh.
        if self.act_output == Tanh:
            training_data = tanh_data_transform(training_data)
            if evaluation_data is not None:
                evaluation_data = tanh_data_transform(evaluation_data)

        ## 578: back to the original code..
        if evaluation_data:
            n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            # random.shuffle(training_data)
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))

            ## 578: from here, most lines are for printing purpose only.
            print("Epoch %s training complete" % j)
            if monitor_training_cost:
                cost = self.total_cost(
                    training_data, lmbda
                )  # 578: for cost, always NO convert (default) for training
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(
                    training_data, convert=True
                )  # 578: for accuracy, always _DO_ convert (argmax) for training
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(accuracy, n))
            if monitor_evaluation_cost:
                ## 578: changed the last parameter convert
                if no_convert:
                    cost = self.total_cost(
                        evaluation_data, lmbda
                    )  # 578: if test/val data is already vectorized for y
                else:
                    cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                ## 578: changed the last parameter convert
                if no_convert:
                    accuracy = self.accuracy(
                        evaluation_data, convert=True
                    )  # 578: _DO_ convert (argmax)
                else:
                    accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print(
                    "Accuracy on evaluation data: {} / {}".format(
                        ## 578: This seems like a bug!
                        # self.accuracy(evaluation_data), n_data))
                        accuracy,
                        n_data,
                    )
                )
            print("")
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    ##  CHANGES NEEDED.
    ##  This original code is hard-coding L2 norm.  You need to change
    ##  so that the parameter self.regularization is used and do the
    ##  appropriate regularization.
    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.

        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        ### Create dropout layers
        if (self.dropout):
            for h in self.sizes[:-1]:
                d = np.random.binomial(1, self.dropoutpercent, size=(h,1)) / self.dropoutpercent
                self.dropoutlayers.append(d)

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        ###only do it this way for L2
        if self.regularization == None or self.regularization == 'L2':
            ###provided code
            self.weights = [
                (1 - eta * (lmbda / n)) * w - (eta / len(mini_batch)) * nw
                for w, nw in zip(self.weights, nabla_w)
            ]
            self.biases = [
                b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)
            ]
        ###if not L2, assume L1
        else: 
            self.weights = [(w-eta*(lmbda/n))*abs(w)/w-(eta/len(mini_batch))*nw 
                            for w, nw in zip(self.weights, nabla_w)]            
            self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    ## CHANGES NEEDED.
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            ## 578: changed to use function class for
            ##   the activation function of hidden layer(s).
            # activation = sigmoid(z)
            activation = (self.act_hidden).fn(z)
            activations.append(activation)

        # backward pass
        ## 578: Cost and activation functions are parameterized now.
        ##     Call the activation function of the output layer with z.
        # delta = (self.cost).delta(zs[-1], activations[-1], y)
        a_prime = (self.act_output).derivative(zs[-1])  # 578: da/dz
        c_prime = (self.cost).derivative(activations[-1], y)  # 578: dC/da

        # 578: compute delta -- separate case for Softmax
        if self.act_output == Softmax:
            delta = np.dot(a_prime, c_prime)
        else:
            delta = c_prime * a_prime  # 578: dC/da * da/dz

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            ## 578: Changed to call the activation function of the
            ##  hidden layer with z.
            # sp = sigmoid_prime(z)
            sp = (self.act_hidden).derivative(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.

        """
        if convert:
            results = [
                (np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data
            ]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    ## CHANGES NEEDED.
    ##  This original code is hard-coding L2 norm.  You need to change
    ##  so that the parameter self.regularization is used and do the
    ##  appropriate regularization.
    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert:
                y = vectorized_result(y)
            cost += self.cost.fn(a, y) / len(data)
        cost += (
            0.5
            * (lmbda / len(data))
            * sum(np.linalg.norm(w) ** 2 for w in self.weights)
        )
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {
            "sizes": self.sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "cost": str(self.cost.__name__),
        }
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


#### Loading a Network from a json file
def load_network(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    # cost = getattr(sys.modules[__name__], data["cost"])
    # net = Network(data["sizes"], cost=cost)
    net = Network(data["sizes"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


#### Miscellaneous functions
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


## 578: new function to generate a new dataset compatible with tanh.
## updated with copy.deepcopy().  This ensures the right solution.
import copy


def tanh_data_transform(dataset):
    xlist = [x for (x, y) in dataset]
    ylist = copy.deepcopy([y for (x, y) in dataset])
    for lst in ylist:
        lst[lst == 0] = -1  # replace 0's by -1's
    return list(zip(xlist, ylist))
