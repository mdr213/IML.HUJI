from __future__ import annotations
from typing import Callable, NoReturn
import numpy as np

from IMLearn.base import BaseModule, BaseLR
from .learning_rate import FixedLR

OUTPUT_VECTOR_TYPE = ["last", "best", "average"]


def default_callback(model: GradientDescent, **kwargs) -> NoReturn:
    pass


class GradientDescent:
    """
    Gradient Descent algorithm

    Attributes:
    -----------
    learning_rate_: BaseLR
        Learning rate strategy for retrieving the learning rate at each iteration t of the algorithm

    tol_: float
        The stopping criterion. Training stops when the Euclidean norm of w^(t)-w^(t-1) is less than
        specified tolerance

    max_iter_: int
        The maximum number of GD iterations to be performed before stopping training

    out_type_: str
        Type of returned solution:
            - `last`: returns the point reached at the last GD iteration
            - `best`: returns the point achieving the lowest objective
            - `average`: returns the average point over the GD iterations

    callback_: Callable[[GradientDescent, ...], None]
        A callable function to be called after each update of the model while fitting to given data
        Callable function should receive as input a GradientDescent instance, and any additional
        arguments specified in the `GradientDescent.fit` function
    """
    def __init__(self,
                 learning_rate: BaseLR = FixedLR(1e-3),
                 tol: float = 1e-5,
                 max_iter: int = 1000,
                 out_type: str = "last",
                 callback: Callable[[GradientDescent, ...], None] = default_callback):
        """
        Instantiate a new instance of the GradientDescent class

        Parameters
        ----------
        learning_rate: BaseLR, default=FixedLR(1e-3)
            Learning rate strategy for retrieving the learning rate at each iteration t of the algorithm

        tol: float, default=1e-5
            The stopping criterion. Training stops when the Euclidean norm of w^(t)-w^(t-1) is less than
            specified tolerance

        max_iter: int, default=1000
            The maximum number of GD iterations to be performed before stopping training

        out_type: str, default="last"
            Type of returned solution. Supported types are specified in class attributes

        callback: Callable[[GradientDescent, ...], None], default=default_callback
            A callable function to be called after each update of the model while fitting to given data
            Callable function should receive as input a GradientDescent instance, and any additional
            arguments specified in the `GradientDescent.fit` function
        """
        self.learning_rate_ = learning_rate
        if out_type not in OUTPUT_VECTOR_TYPE:
            raise ValueError("output_type not supported")
        self.out_type_ = out_type
        self.tol_ = tol
        self.max_iter_ = max_iter
        self.callback_ = callback

    def fit(self, f: BaseModule, X: np.ndarray, y: np.ndarray):
        """
        Optimize module using Gradient Descent iterations over given input samples and responses

        Parameters
        ----------
        f : BaseModule
            Module of objective to optimize using GD iterations
        X : ndarray of shape (n_samples, n_features)
            Input data to optimize module over
        y : ndarray of shape (n_samples, )
            Responses of input data to optimize module over

        Returns
        -------
        solution: ndarray of shape (n_features)
            Obtained solution for module optimization, according to the specified self.out_type_

        Notes
        -----
        - Optimization is performed as long as self.max_iter_ has not been reached and that
        Euclidean norm of w^(t)-w^(t-1) is more than the specified self.tol_

        - At each iteration the learning rate is specified according to self.learning_rate_.lr_step

        - At the end of each iteration the self.callback_ function is called passing self and the
        following named arguments:
            - solver: GradientDescent
                self, the current instance of GradientDescent
            - weights: ndarray of shape specified by module's weights
                Current weights of objective
            - val: ndarray of shape specified by module's compute_output function
                Value of objective function at current point, over given data X, y
            - grad:  ndarray of shape specified by module's compute_jacobian function
                Module's jacobian with respect to the weights and at current point, over given data X,y
            - t: int
                Current GD iteration
            - eta: float
                Learning rate used at current iteration
            - delta: float
                Euclidean norm of w^(t)-w^(t-1)

        """
        w = f.weights
        w_lst = [w]
        if self.out_type_ == "best":
            erm_lst = [f.compute_output(X, y)]
        output_vector_dict = {"last": lambda x: x[-1],
                              "best": lambda x: x[np.argmin(erm_lst)],
                              "average": lambda x: np.sum(x) / len(x)}
        for t in range(self.max_iter_):
            w_t = w - self.learning_rate_.lr_step(t=t) * f.compute_jacobian(x=X, y=y)
            w_lst.append(w_t)
            f.weights = np.copy(w_t)  # todo maybe not needed
            if self.out_type_ == "best":
                erm_lst.append(f.compute_output(x=X, y=y))
            norm = np.linalg.norm(w_t - w)
            self.callback_(self, w_t, f.compute_output(x=X, y=y), f.compute_jacobian(x=X, y=y), t, self.learning_rate_.lr_step(t=t), norm)
            w = np.copy(w_t)
            if norm < self.tol_:
                break
        return output_vector_dict[self.out_type_](w_lst)
        # best_weight = sum_weights = np.copy(f.weights)
        # best_val = np.inf
        # iter_counter = 0
        #
        # for iter_ in range(self.max_iter_):
        #     curr_val = f.compute_output(X=X, y=y)
        #     prev_weights = np.copy(f.weights_)
        #     grad = f.compute_jacobian(X=X, y=y)
        #     eta = self.learning_rate_.lr_step(t=iter_)
        #     f.weights_ -= (eta * grad)
        #     print("weight diffrences: ",np.sum(np.abs(f.weights_ - prev_weights)))
        #     delta = np.linalg.norm(f.weights_ - prev_weights)
        #
        #     self.callback_(self, weights=prev_weights, val=curr_val, grad=grad, t=iter_, eta=eta, delta=delta)
        #     if curr_val < best_val:
        #         best_val = curr_val
        #         best_weight = prev_weights
        #     sum_weights = np.add(sum_weights, f.weights_)
        #     if delta <= self.tol_:
        #         break
        #     iter_counter += 1
        #
        # print(f" the iteration counter is {iter_counter}")
        # if self.out_type_ == 'last':
        #     return f.weights_
        # elif self.out_type_ == 'best':
        #     return best_weight
        # else:
        #     return sum_weights / iter_counter
