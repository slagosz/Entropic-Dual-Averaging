import numpy as np

from cvxpy import Problem, Minimize, Variable
from cvxpy import norm as cvx_norm


def aggregation(X, Y, R=1):
    """
    :param X: design matrix
    :param Y: system's outputs
    :param R: radius of l1 ball (feasible set)
    :return: vector of parameters
    """

    num_of_params = X.shape[1]
    A = Variable(num_of_params)
    o = Minimize(cvx_norm(X @ A - Y, 2))
    c = [cvx_norm(A, 1) <= R]
    p = Problem(o, c)
    p.solve()

    return A.value


def create_design_matrix(dictionary, x, x0=None):
    """
    :param dictionary: an instance of volterra_model.Dictionary
    :param x: vector of inputs
    :param x0: vector of initial conditions
    :return:
    """
    t_list = list(range(0, len(x)))
    X = np.zeros([len(x), dictionary.size])

    if x0 is None:
        x0 = np.zeros(dictionary.memory_length - 1)
    else:
        x0 = np.array(x0)
        assert len(x0) == (dictionary.memory_length - 1)

    x = np.concatenate([x0, x])

    row_idx = 0
    for t in t_list:
        X[row_idx, :] = [f(x, t + dictionary.memory_length - 1) for f in dictionary.functions]
        row_idx += 1

    return X


def aggregation_for_volterra(dictionary, x, y, x0=None, R=1):
    """
    :param dictionary: an instance of volterra_model.VolterraDictionary
    :param x: vector of inputs
    :param y: vector of outputs
    :param x0: vector of initial conditions
    :param R: radius of l1 ball (feasible set)
    :return:
    """

    X = create_design_matrix(dictionary, x, x0)

    return aggregation(X, y, R)
