import itertools
import numpy as np


def volterra_function(indices, x, t=-1):
    output = 1
    for i in indices:
        output *= x[t - i]
    return output


class Dictionary:
    """
    A class used to represent a dictionary of functions
    """

    def __init__(self):
        self.size = 0
        self.functions = []

    def append(self, f):
        self.functions.append(f)
        self.size += 1


class DictionaryBasedModel:
    """
    A class used to represent a linear-in-parameters model
    """

    def __init__(self, dictionary):
        self.dictionary = dictionary
        self.parameters = np.zeros(self.dictionary.size)

    def evaluate_output(self, x, t=None):
        if t is None:
            t_list = range(0, len(x))
        else:
            t_list = [t]

        y = np.zeros(len(t_list))

        i = 0
        for t in t_list:
            dict_output = [f(x, t) for f in self.dictionary.functions]
            y[i] = np.dot(self.parameters, dict_output)
            i += 1

        return y

    def set_parameters(self, parameters):
        assert len(parameters) == self.dictionary.size
        self.parameters = parameters


class VolterraDictionary(Dictionary):
    """
    A class used to represent a dictionary of Volterra polynomials
    """

    def __init__(self, kernels, include_constant_function=True):
        """
        :param kernels: contains an array of memory lengths of consecutive Volterra operators
        :param include_constant_function: if True, then the model 's dictionary contains a constant function,
        that always returns 1, regardless of an argument
        """
        super().__init__()
        self.kernels = kernels
        self.functions_indices = []
        self.include_constant_function = include_constant_function

        self.generate_dictionary()
        self.size = len(self.functions)

        if not kernels:
            self.memory_length = 1
        else:
            self.memory_length = np.max(kernels)

    @staticmethod
    def generate_indices(order, memory_length):
        return itertools.combinations_with_replacement(range(0, memory_length), order)

    def generate_dictionary(self):
        self.functions = []
        self.functions_indices = []

        if self.include_constant_function:
            self.functions.append(lambda x, t: 1)  # constant function
            self.functions_indices.append([])

        order = 1
        for memory_length in self.kernels:
            indices = self.generate_indices(order, memory_length)
            for ind in indices:
                # closure hack https://stackoverflow.com/questions/2295290/what-do-lambda-function-closures-capture
                f = (lambda i: lambda x, t: volterra_function(i, x, t))(ind)
                self.functions.append(f)
                self.functions_indices.append(ind)
            order += 1


class VolterraModel(DictionaryBasedModel):
    """
    A class used to represent a Volterra model
    """

    def __init__(self, order=None, memory_length=None, kernels=None, include_constant_function=True):
        """
        :param order: order of the model (denoted by P in article)
        :param memory_length: memory length of the Volterra model (denoted by M in article)
        :param kernels: if neither order nor memory_length is provided, this parameter contains an array of
        memory lengths of consecutive Volterra operators
        :param include_constant_function: if True, then the model's dictionary contains a constant function,
        that always returns 1, regardless of an argument
        """

        if kernels is None:
            assert order is not None
            assert memory_length is not None
            kernels = [memory_length] * order

        dictionary = VolterraDictionary(kernels, include_constant_function=include_constant_function)
        DictionaryBasedModel.__init__(self, dictionary)

        if not kernels:
            self.memory_length = 1
        else:
            self.memory_length = np.max(kernels)

        self.kernels = kernels
        self.D = self.dictionary.size

    def evaluate_output(self, x, x0=None, t=None):
        if x0 is None:
            x0 = np.zeros(self.memory_length - 1)
        else:
            x0 = np.array(x0)
            assert len(x0) == (self.memory_length - 1)

        if np.isscalar(t):
            t_list = [t]
        elif t is None:
            t_list = list(range(0, len(x)))
        else:
            t_list = t

        x = np.concatenate([x0, x])
        y = np.zeros(len(t_list))

        i = 0
        for t in t_list:
            y[i] = super().evaluate_output(x, t + self.memory_length - 1)
            i += 1

        return y
