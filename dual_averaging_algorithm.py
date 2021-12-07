from volterra_model import *
import copy
from tqdm import tqdm


def compute_gradient(model, x, y, t, x0=None):
    """
    :param model: an instance of volterra_model.DictionaryBasedModel based on an extended dictionary (containing
    redundant functions with opposite sign; see the article for details)
    :param x: vector of inputs
    :param y: vector of outputs
    :param t: index of current input in the vector of inputs
    :param x0: vector of model's initial conditions
    :return:
    """

    y_sys = y[t]

    if x0 is not None:
        x = np.concatenate([x0, x])
        t += model.dictionary.memory_length - 1

    D = int(model.dictionary.size / 2)
    dict_output = np.array([f(x, t) for f in model.dictionary.functions[0:D]])
    dict_output = np.concatenate((dict_output, -dict_output))

    y_mod = np.dot(model.parameters, dict_output)

    gradient = (y_mod - y_sys) * dict_output

    return gradient


class EntropicAlgorithm:
    """
    A class representing an algorithm constrained to a probabilistic simplex. It needs to extend and scale the
    dictionary provided by the user.
    """

    def __init__(self, dictionary, R):
        self.dictionary = copy.deepcopy(dictionary)

        if R is not 1:
            scale_dictionary(self.dictionary, R)

        extend_dictionary(self.dictionary)

        self.R = R
        self.D = self.dictionary.size


def scale_dictionary(dictionary, R):
    for f_idx, f in enumerate(dictionary.functions):
        scaled_f = (lambda func: lambda x, t: R * func(x, t))(f)
        dictionary.functions[f_idx] = scaled_f


def extend_dictionary(dictionary):
    redundant_functions = []

    for f in dictionary.functions:
        negative_fun = (lambda func: lambda x, t: -func(x, t))(f)
        redundant_functions.append(negative_fun)

    for f in redundant_functions:
        dictionary.append(f)


def map_parameters_to_simplex(parameters, R):
    assert len(parameters) % 2 == 0
    D = int(len(parameters) / 2)

    transformed_parameters = np.zeros(D)

    for i in range(D):
        transformed_parameters[i] = R * (parameters[i] - parameters[i + D])

    return transformed_parameters


class EntropicDualAveragingAlgorithm(EntropicAlgorithm):
    """
    A class representing (Adaptive) Entropic Dual Averaging algorithm. This implementation works in an "offline" regime
    """

    def __init__(self, dictionary, R=1):
        """
        :param dictionary: an instance of volterra_model.Dictionary
        :param R: radius of the l1 ball (the algorithm's parameter; see the article for details)
        """
        super().__init__(dictionary, R)

    def run(self, x, y, G_sq=0, x0=None, adaptive_stepsize=True):
        """
        :param x: vector of inputs
        :param y: vector of outputs
        :param G_sq: G squared constant (the algorithm's parameter; see the article for details)
        :param x0: vector of model's initial conditions
        :param adaptive_stepsize:
        :return: model's parameters
        """
        assert len(x) == len(y)

        model = DictionaryBasedModel(self.dictionary)
        theta_0 = np.ones(self.D) / self.D
        model.set_parameters(theta_0)

        gradient_sum = 0
        gradient_max_sq_sum = 0
        T = len(x)

        theta_avg = theta_0

        for i in tqdm(range(T)):
            gradient_i = compute_gradient(model, x, y, i, x0=x0)

            gradient_sum += gradient_i
            gradient_max_sq_sum += np.max(np.abs(gradient_i)) ** 2

            if adaptive_stepsize:
                stepsize = np.sqrt(np.log(self.D) / gradient_max_sq_sum)
            else:
                stepsize = np.sqrt(np.log(self.D) / (G_sq * (i+1)))

            theta_i = np.exp(-stepsize * gradient_sum)
            theta_i /= np.linalg.norm(theta_i, 1)

            model.set_parameters(theta_i)

            theta_avg = (theta_i + theta_avg * (i +1)) / (i + 2)

        return map_parameters_to_simplex(theta_avg, self.R)
