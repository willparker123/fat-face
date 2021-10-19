"""
The :mod:`fatf.interpretability.gradient_explainer` module holds the object and
functions relevant to performing counterfactual explanations.
"""

# Author: Rafael Poyiadzi <rp13102@bristol.ac.uk>
# License: BSD 3 clause

import numpy as np
from typing import Optional, Union, Callable, List, Dict
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.neighbors import KernelDensity
from copy import deepcopy
import math
import scipy

from cvxopt import matrix, solvers

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Explainer2(object):
    """Class for doing counterfactual explanations in the provided dataset.

    """
    def __init__(self,
                 model: Callable,
                 X: np.ndarray,
                 y: np.array,
                 sc_threshold = 0.05,
                 norm_bool = True,
                 alpha = 5,
                 n_steps = 50,
                 step = 0.0001,
                 step_size = 0.01,
                 sc_percentile = 20,
                 reg: Optional[Union[float, int]] = None,
                 threshold: Optional[float] = None,
                 boundaries: Optional[Dict] = None,
                 cost_func: Optional[Callable] = None,
                 distance_measure: Optional[Callable] = 'fro',
                 bool_return_grad = False,
                 rho = 1,
                 whattouse=0,
                 gamma=1,
                 norm_bool_2=False,
                 lambda_1 = 1,
                 lambda_2 = 1,
                 epsilon = 1,
                 delta = None,
                 radius = None,
                 sigma=None,
                 sim_step_size=None):

        self.sim_step_size = sim_step_size
        self.sigma = sigma
        self.radius = radius
        self.epsilon = epsilon
        self.delta = delta
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

        self.norm_bool_2 = norm_bool_2
        self.kernel = None
        self.gamma = gamma
        self.whattouse = whattouse
        self.rho = rho
        self.bool_return_grad = bool_return_grad
        self.sc_percentile = sc_percentile
        self.sc_threshold = sc_threshold
        self.norm_bool = norm_bool
        self.alpha = alpha
        self.n_steps = n_steps
        self.step_size = step_size
        self.distance_measure = distance_measure
        self.X = X
        self.y = y
        self.preds = model.predict(X)
        self.n_samples, self.n_ftrs = self.X.shape
        self.step = 0.00001
        self.cycle = True
        self.last_diff = 10000
        if reg is not None:
            self.reg = [reg]
        else:
            self.reg = [100, 50, 20, 10, 5, 1]

        self.model = model
        if not threshold:
            self.threshold = 0.49
        else:
            self.threshold = threshold
        self.boundaries = boundaries
        self.counter = 0
        self.stagnation_bound = 10

        self.own_func = False
        if cost_func is not None:
            self.cost_func = cost_func
            self.own_func = True

        self.density_estimator = None
        self.step = step

    def visualise(self,
                  X: np.array):
        n_samples, n_ftrs = X.shape

        pred = self.model.predict(X)
        targets = np.zeros((n_samples, 2))
        for i in range(n_samples):
            if pred[i] == 1:
                targets[i, :] = np.array([1, 0])
            else:
                targets[i, :] = np.array([0, 1])
        if n_ftrs != 2:
            raise ValueError('Only works for 2-dimensional data')
        newx = np.zeros((n_samples, n_ftrs))
        newx = []
        for i in range(n_samples):
            if pred[i] == 1:
                target = np.array([0.90, 0.10])
            else:
                target = np.array([0.10, 0.90])
            newx.append(self.explain(X[i, :], target))
        return newx

    def __estimate_gradient(self,
                            instance: np.array):
        idx = self.counter
        instance[idx] += self.step
        eval_up = self.model.predict_proba(instance.reshape(1, -1))
        instance[idx] -= 2 * self.step
        eval_down = self.model.predict_proba(instance.reshape(1, -1))
        instance[idx] += self.step
        grad_fx = (eval_up - eval_down) / (2 * self.step)

        return grad_fx.reshape(-1, 1)

    def __estimate_gradient_all(self,
                                instance: np.array,
                                get_hess = False):
        grad_fx = np.zeros((self.n_ftrs, 1))
        hess_fx = np.zeros((self.n_ftrs, self.n_ftrs))
        instance.reshape(-1, 1)
        eval_here = self.model.predict_proba(instance.reshape(1, -1))[0][self.target_class_idx]
        for idx in range(self.n_ftrs):
            instance[idx] += self.step
            eval_up = self.model.predict_proba(instance.reshape(1, -1))[0][self.target_class_idx]
            instance[idx] -= 2 * self.step
            eval_down = self.model.predict_proba(instance.reshape(1, -1))[0][self.target_class_idx]
            instance[idx] += self.step
            grad_fx[idx] = (eval_up - eval_down) / (2 * self.step)
            if get_hess:
                hess_fx[idx, idx] = (eval_up + eval_down - 2 * eval_here)/(self.step ** 2)

        if get_hess:
            instance[0] += self.step
            instance[1] += self.step
            eval_diag_up = self.model.predict_proba(instance.reshape(1, -1))[0][self.target_class_idx]

            instance[0] -= 2*self.step
            instance[1] -= 2*self.step
            eval_diag_down = self.model.predict_proba(instance.reshape(1, -1))[0][self.target_class_idx]

            hess_fx[0, 1] = (eval_diag_up + eval_diag_down - 2 * eval_here)/(self.step ** 2)
            hess_fx[1, 0] = hess_fx[0, 1]

        if get_hess:
            return grad_fx.reshape(-1, 1), hess_fx
        else:
            return grad_fx.reshape(-1, 1)

    def __eval_condition(self,
                         test_prediction: np.array) -> bool:
        diff = 1 - test_prediction[self.target_class_idx]
        return diff < self.threshold

    def __optimise(self,
                   instance: np.array,
                   reg: int) -> np.array:
        self.st = 0
        satisfied = self.__eval_condition(self.current_prediction)
        stagnated = 0
        while not satisfied:
            if self.cycle:
                self.counter += 1
                self.counter = self.counter%self.n_ftrs
                if self.counter in self.nottochange:
                    continue
            else:
                self.counter = np.random.choice(self.tosamplefrom)
            self.st += 1

            grad = self.__estimate_gradient(instance)[self.target_class_idx]
            pred_full = self.model.predict_proba(instance.reshape(1, -1))
            pred = pred_full[0][self.target_class_idx]
            den = grad**2 + 1
            num = grad * (1 - pred)

            delta = num / den
            #instance += delta
            #pred_full = self.model.predict_proba(instance.reshape(1, -1))
            #satisfied = self.__eval_condition(pred_full)


            if self.boundaries is not None:
                newx_i = instance[self.counter] + delta
                if (newx_i >= self.boundaries[self.counter]['min'] and
                    newx_i <= self.boundaries[self.counter]['max']):
                    instance[self.counter] = newx_i
                    stagnated = 0
                    part1a = self.model.predict_proba(instance.reshape(1, -1)).reshape(-1, 1)
                    satisfied = self.__eval_condition(part1a)
                else:
                    stagnated += 1
                    if stagnated == self.stagnation_bound:
                        return instance
            else:
                instance[self.counter] += delta
                satisfied = self.__eval_condition(pred_full)
            n_steps = 10
            colors=cycle(list((plt.cm.Greens(np.linspace(0,1,n_steps)))))

            plt.scatter(instance[0], instance[1], color=next(colors), marker='x')
        return instance

    def __estimate_gradient2(self,
                            instance: np.array,
                            get_hess: Optional[bool] = False):
        grad = np.zeros((self.n_ftrs, self.n_classes))
        if get_hess:
            hess = np.zeros(self.n_ftrs)
            eval_here = self.model.predict_proba(instance.reshape(1, -1))
        for idx in range(self.n_ftrs):
            instance[idx] += self.step
            eval_up = self.model.predict_proba(instance.reshape(1, -1))
            instance[idx] -= 2 * self.step
            eval_down = self.model.predict_proba(instance.reshape(1, -1))
            instance[idx] += self.step
            grad_fx = (eval_up - eval_down) / (2 * self.step)
            grad[idx, :] = grad_fx

            if get_hess:
                hess[idx] = (eval_up + eval_down - 2 * eval_here) / (self.step**2)
        if not get_hess:
            return grad
        else:
            return grad, hess

    def chop_image(self, instance):
        for idx, val in enumerate(instance):
            if val < 0:
                instance[idx] = 0
            if val > 1:
                instance[idx] = 1
        return instance

    def __optimise_lasso(self,
                   input_instance: np.array,
                   reg: int) -> np.array:
        self.st = 0
        satisfied = self.__eval_condition(self.current_prediction)
        restart = 0
        prev_diff = 1
        distances_list = None
        instance = input_instance.copy(order='K').reshape(-1, 1)

        while not satisfied:
            if self.cycle:
                self.counter += 1
                self.counter = self.counter%self.n_ftrs
                if self.counter in self.nottochange:
                    continue
            else:
                self.counter = np.random.choice(self.tosamplefrom)
            self.st += 1

            grad = self.__estimate_gradient(instance)[self.target_class_idx]
            pred_full = self.model.predict_proba(instance.reshape(1, -1))
            pred = pred_full[0][self.target_class_idx]
            den = grad**2
            diff_pred = 1 - pred
            num = grad * diff_pred
            if (num > reg or num < -reg):
                delta = num / den
                instance[self.counter] += delta
            #instance += delta
            #instance = self.chop_image(instance)
            pred_full = self.model.predict_proba(instance.reshape(1, -1)).reshape(-1, 1)
            satisfied = self.__eval_condition(pred_full)
            #print(satisfied)
            pred = pred_full[self.target_class_idx]
            diff_pred = 1 - pred
            if diff_pred > prev_diff - 0.000001:
                restart += 1
                instance = input_instance.copy(order='K').reshape(-1, 1)
                if distances_list is None:
                    distances_list = []
                    for i in range(self.n_samples):
                        if self.preds[i] == self.target_class_idx:
                            distances_list.append((i,
                                                   np.linalg.norm(self.X[i, :].reshape(-1, 1) - instance,
                                                                  self.distance_measure)))
                best = np.argmin([item[1] for item in distances_list])
# =============================================================================
#                 plt.scatter(self.X[distances_list[best][0], 0],
#                             self.X[distances_list[best][0], 1],
#                             color='k', marker='x')
# =============================================================================
                target = self.X[distances_list[best][0], :].reshape(-1, 1)
                instance = instance + restart * 0.01 * (target - instance)
                #print('renewed')
                prev_diff = 1
                pred_full = self.model.predict_proba(instance.reshape(1, -1)).reshape(-1, 1)
                satisfied = self.__eval_condition(pred_full)
            else:
                prev_diff = diff_pred
            n_steps = 10
            colors=cycle(list((plt.cm.rainbow(np.linspace(0,1,n_steps)))))

            #plt.scatter(instance[0], instance[1], color=next(colors), marker='x')
        return instance, restart

    def __optimise5(self,
                   input_instance: np.array,
                   reg: int) -> np.array:
        self.st = 0
        satisfied = self.__eval_condition(self.current_prediction)
        restart = 0
        prev_diff = 1
        distances_list = None
        instance = input_instance.copy(order='K').reshape(-1, 1)

        while not satisfied:
            if self.cycle:
                self.counter += 1
                self.counter = self.counter%self.n_ftrs
                if self.counter in self.nottochange:
                    continue
            else:
                self.counter = np.random.choice(self.tosamplefrom)
            self.st += 1

            grad = self.__estimate_gradient(instance)[self.target_class_idx]
            pred_full = self.model.predict_proba(instance.reshape(1, -1))
            pred = pred_full[0][self.target_class_idx]
            den = grad**2 + reg
            diff_pred = 1 - pred
            num = grad * diff_pred
            delta = num / den
            instance[self.counter] += delta
            #instance += delta
            pred_full = self.model.predict_proba(instance.reshape(1, -1)).reshape(-1, 1)
            #print(pred)
            satisfied = self.__eval_condition(pred_full)
            #print(satisfied)
            if diff_pred > prev_diff - 0.000001:
                restart += 1
                instance = input_instance.copy(order='K').reshape(-1, 1)
                if distances_list is None:
                    distances_list = []
                    for i in range(self.n_samples):
                        if self.y[i] == self.target_class_idx:
                            distances_list.append((i,
                                                   np.linalg.norm(self.X[i, :] - instance,
                                                                  self.distance_measure)))
                best = np.argmin([item[1] for item in distances_list])
                target = self.X[distances_list[best][0], :].reshape(-1, 1)
                instance = instance + restart * 0.001 * (target - instance)
                #print('renewed')
                prev_diff = 1
                pred_full = self.model.predict_proba(instance.reshape(1, -1)).reshape(-1, 1)
                satisfied = self.__eval_condition(pred_full)
            else:
                prev_diff = diff_pred

        return instance

    def eval_density(self):
# =============================================================================
#         bandwidths = 10 ** np.linspace(-1, 1, 10)
#
#         my_cv = LeaveOneOut()
#         grid = GridSearchCV(KernelDensity(kernel='gaussian'),
#                     {'bandwidth': [1]},
#                     cv=1,
#                     iid=True)
#
#         grid.fit(self.X)
#         mdl = grid.best_estimator_
# =============================================================================
        mdl = KernelDensity(bandwidth=0.30)
        mdl.fit(self.X)
        #self.sc_lim = np.percentile(np.exp(mdl.score_samples(self.X)), self.sc_percentile)
        self.density_estimator = mdl

    def get_density_grad(self, instance, get_hess=False):
        grad_fx = np.zeros((self.n_ftrs, 1))
        eval_here = np.exp(self.density_estimator.score_samples(instance.reshape(1, -1)))
        hess = np.zeros((self.n_ftrs, self.n_ftrs))
        for idx in range(self.n_ftrs):
            instance[idx] += self.step
            eval_up = np.exp(self.density_estimator.score_samples(instance.reshape(1, -1)))
            instance[idx] -= 2 * self.step
            eval_down = np.exp(self.density_estimator.score_samples(instance.reshape(1, -1)))
            instance[idx] += self.step
            grad_fx[idx] = (eval_up - eval_down) / (2 * self.step)
            if get_hess:
                hess[idx, idx] = (eval_up + eval_down - 2 * eval_here) / (self.step**2)
        if get_hess:
            self.step /= 2
            instance[0] += self.step
            instance[1] += self.step
            eval_up_both = np.exp(self.density_estimator.score_samples(instance.reshape(1, -1)))

            instance[0] -= self.step
            eval_up_1 = np.exp(self.density_estimator.score_samples(instance.reshape(1, -1)))

            instance[0] += self.step
            instance[1] -= self.step
            eval_up_0 = np.exp(self.density_estimator.score_samples(instance.reshape(1, -1)))

            instance[0] -= self.step

            hess[0, 1] = (eval_up_both - eval_up_0 - eval_up_1 + eval_here) / (self.step**2)
            hess[1, 0] = hess[0, 1]
            self.step *= 2
            return grad_fx.reshape(-1, 1), hess
        else:
            return grad_fx.reshape(-1, 1)

    def apply_mod_func(self, v):
        #return np.exp(1 - sigmoid(v)) ** 1/4
        #return v
        return -np.log(1 + np.exp(self.delta * v))
        #return np.exp(-self.delta * v)
        #return 1 - sigmoid(-self.delta * v)

    def get_density_mat_modified(self, instance):
        hess = np.zeros((self.n_ftrs, self.n_ftrs))
        eval_here = np.exp(self.density_estimator.score_samples(instance.reshape(1, -1)))

        for idx in range(self.n_ftrs):
            instance[idx] += self.step
            eval_up = np.exp(self.density_estimator.score_samples(instance.reshape(1, -1)))
            instance[idx] -= 2 * self.step
            eval_down = np.exp(self.density_estimator.score_samples(instance.reshape(1, -1)))
            instance[idx] += self.step

            hess[idx, idx] = self.apply_mod_func((eval_up - eval_down) / (2 * self.step))

        instance[0] += self.step
        instance[1] += self.step
        eval_up_both = np.exp(self.density_estimator.score_samples(instance.reshape(1, -1)))

        instance[0] -= self.step
        eval_up_1 = np.exp(self.density_estimator.score_samples(instance.reshape(1, -1)))

        instance[0] += self.step
        instance[1] -= self.step
        eval_up_0 = np.exp(self.density_estimator.score_samples(instance.reshape(1, -1)))

        instance[0] -= self.step

        hess[0, 1] = 0#(eval_up_both - eval_up_0 - eval_up_1 + eval_here) / (self.step**2)
        hess[1, 0] = hess[0, 1]

        return hess

    def get_prediction_mat_modified(self, instance):
        grad_fx = np.zeros((self.n_ftrs, self.n_ftrs))
        for idx in range(self.n_ftrs):
            instance[idx] += self.step
            eval_up = self.model.predict_proba(instance.reshape(1, -1))[0][self.target_class_idx]
            instance[idx] -= 2 * self.step
            eval_down = self.model.predict_proba(instance.reshape(1, -1))[0][self.target_class_idx]
            instance[idx] += self.step

            grad_fx[idx, idx] = self.apply_mod_func((eval_up - eval_down) / (2 * self.step))

        self.step /= 2
        instance[0] += self.step
        instance[1] += self.step
        eval_up = self.model.predict_proba(instance.reshape(1, -1))[0][self.target_class_idx]
        instance[0] -= 2 * self.step
        instance[1] -= 2 * self.step
        eval_down = self.model.predict_proba(instance.reshape(1, -1))[0][self.target_class_idx]
        instance[0] += self.step
        instance[1] += self.step
        #grad_fx[0, 1] = self.apply_mod_func((eval_up - eval_down) / (2 * self.step))
        self.step *= 2
        grad_fx[1, 0] = grad_fx[0, 1]
        return grad_fx

    def modify_density_grad(self, grad):
        for idx in range(self.n_ftrs):
            grad[idx] = 1 - sigmoid(grad[idx])
        return grad

    def __optimise_density(self,
                         input_instance: np.array,
                         reg: int) -> np.array:
        if self.density_estimator is None:
            self.eval_density()

        self.st = 0
        #satisfied = self.__eval_condition(self.current_prediction)
        instance = input_instance.copy(order='K').reshape(-1, 1)

        #while not satisfied:
        n_steps = self.n_steps
        colors=cycle(list((plt.cm.rainbow(np.linspace(0,1,n_steps)))))
        t = 1
        for i in range(n_steps):
            self.st += 1

            pr = self.model.predict_proba(instance.reshape(1, -1))[0][self.target_class_idx]

            if (pr >= 0.70):
                break

            fx_grad = self.__estimate_gradient_all(instance)
            #fx_grad /= np.linalg.norm(fx_grad)

            density_mat = self.get_density_mat_modified(instance)
            #print(density_mat)
            inv_density_mat = np.linalg.inv(density_mat)


            M = np.dot(inv_density_mat + t * self.sigma * np.identity(2), fx_grad)
            #instance -= 0.002 * M

# =============================================================================
#             dx_grad = self.get_density_grad(instance)
#             fx_mat = self.get_prediction_mat_modified(instance)
#             sigma = 1e-6
#             inv_density_mat = np.linalg.inv(fx_mat + sigma * np.identity(2))
#             M = np.dot(inv_density_mat, dx_grad)
# =============================================================================
            if self.bool_return_grad:
                return M
            n = np.linalg.norm(M)
            if n <= 1e-3:
                t *= 1.10
                plt.scatter(instance[0], instance[1], color='k',
                        alpha=0.50, marker='o', s=100)
            if n != 0:
                if self.norm_bool:
                    M /= np.linalg.norm(M)
            instance += self.step_size * M
            #print(M)

            plt.scatter(instance[0], instance[1], color=next(colors),
                        alpha=0.50, marker='x', s=100)
        return instance

    def __optimise_combined(self,
                            input_instance: np.array,
                            reg: int) -> np.array:
        if self.density_estimator is None:
            self.eval_density()

        self.st = 0
        #satisfied = self.__eval_condition(self.current_prediction)
        instance = input_instance.copy(order='K').reshape(-1, 1)

        #while not satisfied:
        n_steps = self.n_steps
        colors=cycle(list((plt.cm.Greens(np.linspace(0,1,n_steps)))))
        prev_grad = np.zeros((self.n_ftrs, 1))
        for i in range(n_steps):
            self.st += 1

            pr = self.model.predict_proba(instance.reshape(1, -1))[0][self.target_class_idx]
            density_est = np.exp(self.density_estimator.score_samples(instance.reshape(1, -1)))

            if (pr >= 0.70):
                break


            fx_grad = self.__estimate_gradient_all(instance)
            df_grad = self.get_density_grad(instance)
            if np.linalg.norm(df_grad) <= 0.05:
                comb_grad = fx_grad + prev_grad
            else:
                comb_grad = density_est**self.rho * fx_grad + self.rho * density_est ** (self.rho - 1) *self.alpha * pr * df_grad

            prev_grad = comb_grad
            if self.norm_bool:
                comb_grad /= np.linalg.norm(comb_grad)

            if self.bool_return_grad:
                return comb_grad
            instance += self.step_size * comb_grad

            self.ax.scatter(instance[0], instance[1], color=next(colors), marker='x', s=100)
        return instance

    def __optimise_new(self,
                       input_instance: np.array,
                       reg: int) -> np.array:
        if self.density_estimator is None:
            self.eval_density()

        self.st = 0
        instance = input_instance.copy(order='K').reshape(-1, 1)

        n_steps = self.n_steps
        colors=cycle(list((plt.cm.Greens(np.linspace(0,1,n_steps)))))
        for i in range(n_steps):
            self.st += 1

            pr = self.model.predict_proba(instance.reshape(1, -1))[0][self.target_class_idx]
            density_est = np.exp(self.density_estimator.score_samples(instance.reshape(1, -1)))
            fx_grad = self.__estimate_gradient_all(instance)
            df_grad = self.get_density_grad(instance)

            #fx_grad_norm = np.linalg.norm(fx_grad)
            #print(grad)
            grad = density_est**self.alpha * fx_grad + self.alpha * density_est**(self.alpha - 1) * pr * df_grad

            if self.bool_return_grad:
                return -grad
            grad /= np.linalg.norm(grad)
            instance += self.step_size * grad
            self.ax.scatter(instance[0], instance[1], color=next(colors), marker='x', s=100)
        return instance

    def __optimise_new2(self,
                       input_instance: np.array,
                       reg: int) -> np.array:
        ## product

        if self.density_estimator is None:
            self.eval_density()

        self.st = 0
        instance = input_instance.copy(order='K').reshape(-1, 1)

        n_steps = self.n_steps
        colors=cycle(list((plt.cm.Greens(np.linspace(0,1,n_steps)))))
        for i in range(n_steps):
            self.st += 1

            pr = self.model.predict_proba(instance.reshape(1, -1))[0][self.target_class_idx]
            density_est = np.exp(self.density_estimator.score_samples(instance.reshape(1, -1)))
            fx_grad = self.__estimate_gradient_all(instance).reshape(-1, 1)
            df_grad = self.get_density_grad(instance).reshape(-1, 1)

            L = np.hstack((df_grad, fx_grad))
            P = np.array([[0, 1], [1, 0]])
            Q = np.dot(np.dot(L, P), L.T)
            v = np.linalg.inv(Q)
            b = np.dot(v, L)
            c = np.array([pr, density_est]).reshape(-1, 1)
            grad = -2 * np.dot(b, c)

            if self.bool_return_grad:
                return -grad
            #fx_grad_norm = np.linalg.norm(fx_grad)
            #grad = density_est**self.alpha * fx_grad + self.alpha * density_est**(self.alpha - 1) * pr * df_grad
            #grad /= np.linalg.norm(grad)
            instance -= self.step_size * grad
            self.ax.scatter(instance[0], instance[1], color=next(colors), marker='x', s=100)
        return instance


    def __optimise_combined_withtarget(self,
                                       input_instance: np.array,
                                       reg: int) -> np.array:
        if self.density_estimator is None:
            self.eval_density()

        self.st = 0
        #satisfied = self.__eval_condition(self.current_prediction)
        instance = input_instance.copy(order='K').reshape(-1, 1)

        #while not satisfied:
        n_steps = self.n_steps
        colors=cycle(list((plt.cm.Greens(np.linspace(0,1,n_steps)))))
        for i in range(n_steps):
            self.st += 1


            #fx_grad = self.__estimate_gradient_all(instance)
            fx_grad = self.target_instance.reshape(-1, 1) - instance.reshape(-1, 1)
            df_grad = self.get_density_grad(instance)

            fx_grad /= np.linalg.norm(fx_grad)
            df_grad /= np.linalg.norm(df_grad)
            alpha = self.alpha

            counter = 0
            while True:
                counter += 1
                if counter > 10:
                    break
                comb_grad = fx_grad + alpha * df_grad
                if self.norm_bool:
                    comb_grad /= np.linalg.norm(comb_grad)
                test_instance = instance + self.step_size * comb_grad
                sc = np.exp(self.density_estimator.score_samples(test_instance.reshape(1, -1)))
                if sc >= self.sc_lim:
                    instance = test_instance
                else:
                    alpha *= 2
            #instance += self.step_size * comb_grad

            self.ax.scatter(instance[0], instance[1], color=next(colors), marker='x', s=100)
        return instance

    def __optimise_new_tijl_(self,
                       input_instance: np.array,
                       reg: int) -> np.array:
        if self.density_estimator is None:
            self.eval_density()

        self.st = 0
        instance = input_instance.copy(order='K').reshape(-1, 1)

        n_steps = self.n_steps
        colors=cycle(list((plt.cm.Greens(np.linspace(0,1,n_steps)))))
        for i in range(n_steps):
            self.st += 1

            pr = self.model.predict_proba(instance.reshape(1, -1))[0][self.target_class_idx]
            density_est = np.exp(self.density_estimator.score_samples(instance.reshape(1, -1)))
            fx_grad = self.__estimate_gradient_all(instance)
            df_grad = self.get_density_grad(instance)

            fx_grad_norm = np.linalg.norm(fx_grad)
            diff = 1 - pr
            part0 = - density_est * (fx_grad_norm + diff / fx_grad_norm)
            part1 = - diff * fx_grad_norm
            grad = part0 * fx_grad + part1 * df_grad

            den = fx_grad_norm**2 * density_est**2
            grad /= den
            #print(grad)
            if self.bool_return_grad:
                return -grad
            if self.norm_bool:
                grad /= np.linalg.norm(grad)
            instance -= self.step_size * grad
            self.ax.scatter(instance[0], instance[1], color=next(colors), marker='x', s=100)
        return instance

    def __optimise_new_tijl(self,
                            input_instance: np.array,
                            reg: int) -> np.array:
        if self.density_estimator is None:
            self.eval_density()

        self.st = 0
        instance = input_instance.copy(order='K').reshape(-1, 1)

        n_steps = self.n_steps
        colors=cycle(list((plt.cm.Greens(np.linspace(0,1,n_steps)))))
        for i in range(n_steps):
            self.st += 1

            pr = self.model.predict_proba(instance.reshape(1, -1))[0][self.target_class_idx]
            density_est = np.exp(self.density_estimator.score_samples(instance.reshape(1, -1)))
            fx_grad = self.__estimate_gradient_all(instance)
            df_grad = self.get_density_grad(instance)

            fx_grad_norm = np.linalg.norm(fx_grad)
            diff = 1 - pr
            part0 = - density_est
            part1 = - diff
            grad = part0 * fx_grad + part1 * df_grad

            den = density_est**2
            grad /= den
            #print(grad)
            if self.bool_return_grad:
                return grad
            if self.norm_bool:
                grad /= np.linalg.norm(grad)
            instance -= self.step_size * grad
            self.ax.scatter(instance[0], instance[1], color=next(colors), marker='x', s=100)
        return instance

    def kernel_func(self, x0, x1):
        return np.exp(-self.gamma * np.linalg.norm(x0 - x1)**2)

    def get_kernel(self):
        self.kernel = np.ones((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            v0 = self.X[i, :].reshape(-1, 1)
            for j in range(i):
                self.kernel[i, j] = self.kernel_func(v0, self.X[j, :].reshape(-1, 1))
                self.kernel[j, i] = self.kernel[i, j]

    def eval_grad(self, x0):
        num = np.zeros((self.n_ftrs, 1))
        den = 0
        for i in range(self.n_samples):
            x1 = self.X[i, :].reshape(-1, 1)
            k = self.kernel_func(x0, x1)
            den += k
            d = x1 - x0
            num += self.gamma * k * d
        return num/self.n_samples, den/self.n_samples

    def __optimise_lap(self,
                       input_instance: np.array,
                       reg: int) -> np.array:
        if self.kernel is None:
            self.get_kernel()

        self.st = 0
        instance = input_instance.copy(order='K').reshape(-1, 1)

        n_steps = self.n_steps
        colors=cycle(list((plt.cm.Greens(np.linspace(0,1,n_steps)))))
        for i in range(n_steps):
            pr = self.model.predict_proba(instance.reshape(1, -1))[0][self.target_class_idx]
            if pr > 0.60:
                break
            fx_grad = self.__estimate_gradient_all(instance)
            lap_grad, lap_eval = self.eval_grad(instance)

            if self.norm_bool_2:
                fx_grad /= np.linalg.norm(fx_grad)
                lap_grad /= np.linalg.norm(lap_grad)

            grad = lap_eval * lap_grad + self.alpha * fx_grad * lap_eval
            if self.bool_return_grad:
                return grad
            if self.norm_bool:
                grad /= np.linalg.norm(grad)
            instance += self.step_size * grad
            self.ax.scatter(instance[0], instance[1], color=next(colors), marker='x', s=100)
        return instance

    def get_grad_lap2(self, z):
        L = sum([self.kernel_func(z, self.X[i, :].reshape(-1, 1)) for i in range(self.n_samples)])
        L /= self.n_samples

        part0 = self.gamma * (self.instance - z) * self.kernel_func(z, self.instance)

        part1 = np.zeros((self.n_ftrs, 1))
        for i in range(self.n_samples):
            x = self.X[i, :].reshape(-1, 1)
            part1 += self.gamma * (x - z) * self.kernel_func(z, x)
        part1 /= self.n_samples

        part0_mult = self.C * self.lambda_1 / np.sqrt(L)

        part1_mult = 1 - self.C * self.lambda_1 * self.kernel_func(z, self.instance) / (2 * L**(3/2))

        return part0 * part0_mult + part1 * part1_mult


    def __optimise_lap2(self,
                       input_instance: np.array,
                       reg: int) -> np.array:
        if self.kernel is None:
            self.get_kernel()

        self.st = 0
        instance = input_instance.copy(order='K').reshape(-1, 1)
        self.instance = instance
# =============================================================================
#         C = 0
#         for i in range(self.n_samples):
#             C += self.kernel_func(instance, self.X[i, :].reshape(-1, 1))
# =============================================================================
        s = sum([self.kernel_func(instance, self.X[i, :].reshape(-1, 1)) for i in range(self.n_samples)])
        s /= self.n_samples
        s = np.sqrt(s)
        self.C = 1 / (self.n_samples * s)

        n_steps = self.n_steps
        colors=cycle(list((plt.cm.Greens(np.linspace(0,1,n_steps)))))
        for i in range(n_steps):

            fx_grad = self.__estimate_gradient_all(instance)
            lap_grad_2 = self.get_grad_lap2(instance)

            if self.norm_bool_2:
                fx_grad /= np.linalg.norm(fx_grad)
                lap_grad_2 /= np.linalg.norm(lap_grad_2)

            grad = lap_grad_2 + self.lambda_2 * fx_grad
            if self.bool_return_grad:
                return grad
            if self.norm_bool:
                grad /= np.linalg.norm(grad)
            instance += self.step_size * grad
            self.ax.scatter(instance[0], instance[1], color=next(colors), marker='x', s=100)
        return instance

    def compute_sample_similarity(self, instance):
        s = 0
        for i in range(self.n_samples):
            s += self.kernel_func(instance, self.X[i, :].reshape(-1, 1))
        return s / self.n_samples

    def get_lap_mat_modified(self, instance):
        grad_fx = np.zeros((self.n_ftrs, self.n_ftrs))

        s = np.sqrt(self.compute_sample_similarity(instance))
        C = 1 / (self.n_samples * s)

        for idx in range(self.n_ftrs):
            instance[idx] += self.step
            den = self.compute_sample_similarity(instance)
            eval_up = C * self.kernel_func(self.original, instance) / np.sqrt(den)

            instance[idx] -= 2 * self.step
            den = self.compute_sample_similarity(instance)
            eval_down = C * self.kernel_func(self.original, instance) / np.sqrt(den)
            instance[idx] += self.step

            grad_fx[idx, idx] = self.apply_mod_func((eval_up - eval_down) / (2 * self.step))

        self.step /= 2
        instance[0] += self.step
        instance[1] += self.step
        den = self.compute_sample_similarity(instance)
        eval_up = C * self.kernel_func(self.original, instance) / np.sqrt(den)

        instance[0] -= 2 * self.step
        instance[1] -= 2 * self.step
        den = self.compute_sample_similarity(instance)
        eval_down = C * self.kernel_func(self.original, instance) / np.sqrt(den)


        instance[0] += self.step
        instance[1] += self.step
        grad_fx[0, 1] = self.apply_mod_func((eval_up - eval_down) / (2 * self.step))
        self.step *= 2
        grad_fx[1, 0] = grad_fx[0, 1]
        return grad_fx

    def get_lap_mat_modified_2(self, instance):
        grad_fx = np.zeros((self.n_ftrs, self.n_ftrs))

        for idx in range(self.n_ftrs):
            instance[idx] += self.step
            eval_up = 0
            for i in range(self.n_samples):
                x = self.X[i, :].reshape(-1, 1)
                eval_up += self.gamma * (x[idx] - instance[idx]) * self.kernel_func(x, instance)

            instance[idx] -= 2 * self.step
            eval_down = 0
            for i in range(self.n_samples):
                x = self.X[i, :].reshape(-1, 1)
                eval_down += self.gamma * (x[idx] - instance[idx]) * self.kernel_func(x, instance)

            instance[idx] += self.step

            grad_fx[idx, idx] = self.apply_mod_func((eval_up - eval_down) / (2 * self.step))

        self.step /= 2
        instance[0] += self.step
        instance[1] += self.step
        eval_up = 0
        for i in range(self.n_samples):
            x = self.X[i, :].reshape(-1, 1)
            eval_up += self.gamma * (x[idx] - instance[idx]) * self.kernel_func(x, instance)

        instance[0] -= 2 * self.step
        instance[1] -= 2 * self.step
        eval_down = 0
        for i in range(self.n_samples):
            x = self.X[i, :].reshape(-1, 1)
            eval_down += self.gamma * (x[idx] - instance[idx]) * self.kernel_func(x, instance)


        instance[0] += self.step
        instance[1] += self.step
        grad_fx[0, 1] = self.apply_mod_func((eval_up - eval_down) / (2 * self.step))
        self.step *= 2
        grad_fx[1, 0] = grad_fx[0, 1]
        return grad_fx

    def get_lap_mat_modified_nodiag(self, instance):
        grad_fx = np.zeros((self.n_ftrs, self.n_ftrs))

        for idx in range(self.n_ftrs):
            instance[idx] += self.step
            eval_up = 0
            for i in range(self.n_samples):
                x = self.X[i, :].reshape(-1, 1)
                eval_up += self.gamma * (x[idx] - instance[idx]) * self.kernel_func(x, instance)

            instance[idx] -= 2 * self.step
            eval_down = 0
            for i in range(self.n_samples):
                x = self.X[i, :].reshape(-1, 1)
                eval_down += self.gamma * (x[idx] - instance[idx]) * self.kernel_func(x, instance)

            instance[idx] += self.step

            grad_fx[idx, idx] = self.apply_mod_func((eval_up - eval_down) / (2 * self.step))

        self.step /= 2
        instance[0] += self.step
        instance[1] += self.step
        eval_up = 0
        for i in range(self.n_samples):
            x = self.X[i, :].reshape(-1, 1)
            eval_up += self.gamma * (x[idx] - instance[idx]) * self.kernel_func(x, instance)

        instance[0] -= 2 * self.step
        instance[1] -= 2 * self.step
        eval_down = 0
        for i in range(self.n_samples):
            x = self.X[i, :].reshape(-1, 1)
            eval_down += self.gamma * (x[idx] - instance[idx]) * self.kernel_func(x, instance)


        instance[0] += self.step
        instance[1] += self.step
        grad_fx[0, 1] = 0#self.apply_mod_func((eval_up - eval_down) / (2 * self.step))
        self.step *= 2
        grad_fx[1, 0] = grad_fx[0, 1]
        return grad_fx

    def __optimise_lap3(self,
                        input_instance: np.array,
                        reg: int) -> np.array:

        self.st = 0
        instance = input_instance.copy(order='K').reshape(-1, 1)
        self.original = deepcopy(instance)

        n_steps = self.n_steps
        colors=cycle(list((plt.cm.rainbow(np.linspace(0,1,n_steps)))))
        for i in range(n_steps):
            self.st += 1


            fx_grad = self.__estimate_gradient_all(instance)

            #density_grad = self.get_density_grad(instance)
            #mod_density_grad = self.modify_density_grad(density_grad)
            #M = np.multiply(np.apply_along_axis(lambda x: 1/x, 0, mod_density_grad), fx_grad)

            density_mat = self.get_lap_mat_modified_nodiag(instance)
            sigma = 1e-6
            inv_density_mat = np.linalg.inv(density_mat + sigma * np.identity(2))


            M = np.dot(inv_density_mat, fx_grad)
            #instance -= 0.002 * M

# =============================================================================
#             dx_grad = self.get_density_grad(instance)
#             fx_mat = self.get_prediction_mat_modified(instance)
#             sigma = 1e-6
#             inv_density_mat = np.linalg.inv(fx_mat + sigma * np.identity(2))
#             M = np.dot(inv_density_mat, dx_grad)
# =============================================================================
            if self.bool_return_grad:
                return M
            M /= np.linalg.norm(M)
            #print(M)
            instance -= self.step_size * M
            #print(M)
            #print(instance)
            plt.scatter(instance[0], instance[1], color=next(colors),
                        alpha=0.50, marker='x', s=100)
        return instance

    def __optimise_lap4(self,
                        input_instance: np.array,
                        reg: int) -> np.array:

        self.st = 0
        instance = input_instance.copy(order='K').reshape(-1, 1)
        self.original = deepcopy(instance)

        n_steps = self.n_steps
        colors=cycle(list((plt.cm.Greens(np.linspace(0,1,100)))))
        for i in range(n_steps):
            self.st += 1

            pr = self.model.predict_proba(instance.reshape(1, -1))[0][self.target_class_idx]
            if pr > 0.60:
                break
                color = 'k'
            else:
                color = next(colors)

            fx_grad = self.__estimate_gradient_all(instance)
            fx_grad /= np.linalg.norm(fx_grad)
            #density_grad = self.get_density_grad(instance)
            #mod_density_grad = self.modify_density_grad(density_grad)
            #M = np.multiply(np.apply_along_axis(lambda x: 1/x, 0, mod_density_grad), fx_grad)

            density_mat = self.get_lap_mat_modified_2(instance)
            sigma = 1e-6
            inv_density_mat = np.linalg.inv(density_mat + sigma * np.identity(2))


            M = np.dot(inv_density_mat, fx_grad)
            #instance -= 0.002 * M

# =============================================================================
#             dx_grad = self.get_density_grad(instance)
#             fx_mat = self.get_prediction_mat_modified(instance)
#             sigma = 1e-6
#             inv_density_mat = np.linalg.inv(fx_mat + sigma * np.identity(2))
#             M = np.dot(inv_density_mat, dx_grad)
# =============================================================================
            if self.bool_return_grad:
                return M
            M /= np.linalg.norm(M)
            instance -= self.step_size * M
            #print(M)
            if self.toplot:
                plt.scatter(instance[0], instance[1], color=color,
                            alpha=0.75, marker='x', s=100)
        return instance

    def __optimise_lp_2(self,
                        input_instance: np.array,
                        reg: int) -> np.array:

        self.st = 0
        instance = input_instance.copy(order='K').reshape(-1, 1)

        n_steps = self.n_steps
        colors=cycle(list((plt.cm.Greens(np.linspace(0,1,100)))))

        if self.kernel is None:
            self.get_kernel()
        m = np.mean(self.kernel, axis=1)
        self.epsilon = 0.05 #np.median(m) /4

        for i in range(n_steps):

            pr = self.model.predict_proba(instance.reshape(1, -1))[0][self.target_class_idx]
            if pr > 0.60:
                break
                color = 'k'
            else:
                color = next(colors)

            fx_grad = self.__estimate_gradient_all(instance)
            #fx_grad /= np.linalg.norm(fx_grad)

            # solve LP
            #c = matrix(fx_grad, (self.n_ftrs, 1), tc='d')
            s = self.compute_sample_similarity(instance)
            L = np.sqrt(s)
            #bound = -self.epsilon + 1/(self.n_samples * L)
            G = np.zeros((self.n_ftrs, 1))
            G_prime = np.zeros((self.n_ftrs, self.n_ftrs))

            for i in range(self.n_samples):
                x = self.X[i, :].reshape(-1, 1)
                diff0 = (x - instance)
                f = self.kernel_func(x, instance)
                G += diff0 * f

                G_prime += (self.gamma * np.dot(diff0, diff0.T) - np.identity(self.n_ftrs)) *  f

            G_prime *= (self.gamma / self.n_samples)

            G *= (self.gamma / self.n_samples)

            G_sq = np.dot(G, G.T)

            part0 = -self.gamma * np.identity(self.n_ftrs)
            part1 = -0.50 * (G_prime * L - 0.50 * G_sq / L) / L**3
            part2 = -0.50 * G_sq / L**4
            #print(part0.shape, part1.shape, part2.shape)
            const = 1 / (self.n_samples * L**2)
            hess = const * (part0 + part1 + part2)
            hess_inv = np.linalg.inv(hess)

            sim_grad = - G / (2 * self.n_samples * s**2)

            #sim_grad /= np.linalg.norm(sim_grad)
            #fx_grad /= np.linalg.norm(fx_grad)

            t = 1
            for i in range(1):# True:
                comb = -fx_grad - t * self.delta * sim_grad
                M = np.dot(hess_inv, comb)
                #M = comb
                test_instance = instance + self.step_size * M
                s = self.compute_sample_similarity(test_instance)
                if s >= self.epsilon:
                    break
                else:
                    t *= 1.50
            #print(i)
            #sim_grad /= np.linalg.norm(sim_grad)
# =============================================================================
#             sim_grad = - sim_grad
#             c = fx_grad.ravel().tolist()
#             A = [sim_grad.ravel().tolist()]
#             b = [bound]
#
#             from scipy.optimize import linprog
#             lb = -10
#             ub = 10
#             res = linprog(c, A_ub=A, b_ub=b, bounds=((lb, ub), (lb, ub)))
#             M = res['x']
# =============================================================================

            if self.bool_return_grad:
                return M
            M /= np.linalg.norm(M)
            instance += self.step_size * M
            #print(M)
            if self.toplot:
                plt.scatter(instance[0], instance[1], color=color,
                            alpha=0.75, marker='x', s=100)
        return instance

    def __optimise_lp(self,
                        input_instance: np.array,
                        reg: int) -> np.array:

        self.st = 0
        instance = input_instance.copy(order='K').reshape(-1, 1)

        n_steps = self.n_steps
        colors=cycle(list((plt.cm.Greens(np.linspace(0,1,100)))))

        self.epsilon_0 = 1
        self.epsilon_1 = 0

        S = 0
        mu = np.zeros((self.n_ftrs, 1))
        for i in range(self.n_samples):
            v0 = self.X[i, :].reshape(-1, 1)
            mu += v0
            for j in range(i):
                S += 2 * np.dot(self.X[j, :].reshape(1, -1), v0)
            S += np.dot(v0.T, v0)
        S /= self.n_samples**2
        mu /= self.n_samples

        for i in range(n_steps):

            pr = self.model.predict_proba(instance.reshape(1, -1))[0][self.target_class_idx]
            if pr > 0.60:
                break
                color = 'k'
            else:
                color = next(colors)

            fx_grad = self.__estimate_gradient_all(instance)
            fx_grad /= np.linalg.norm(fx_grad)

            # solve LP
            c = matrix(fx_grad, (self.n_ftrs, 1), tc='d')
            bound = -self.epsilon_1 - S/2 + np.linalg.norm(instance)**2 #+ 2 * np.dot(mu.T, instance)

            a = -(instance / self.n_samples - mu).reshape(1, -1)
            A = matrix(a, (1, self.n_ftrs))
            b = matrix(bound, (1, 1))
            sol=solvers.lp(c,A,b)

# =============================================================================
#             l_00 = 4 * self.epsilon_0
#             l_11 = np.linalg.norm(a) **2
#             l_01 = 2 * bound[0, 0]
#
#             Q = 2*matrix([[l_00, l_01],
#                            [l_01, l_11]])
#
#             l_1 = 2 * np.dot(a, fx_grad)[0, 0]
#             p = matrix([0.0, l_1])
#             G = matrix([[-1.0,0.0],[0.0,-1.0]])
#             h = matrix([0.0,0.0])
#             A = matrix([0.0, 0.0], (1,2))
#             b = matrix(0.0)
#             sol=solvers.qp(Q, p, G, h)
# =============================================================================
# =============================================================================
#             c = fx_grad.ravel().tolist()
#             A = [a.ravel().tolist()]
#             b = [bound[0, 0]]
#
#             print(c, A, b)
#             from scipy.optimize import linprog
#             lb = -100
#             ub = 100
#             res = linprog(c, A_ub=A, b_ub=b, bounds=((lb, ub), (lb, ub)))
#             print(res)
# =============================================================================

            v0, v1 = sol['x']
            M = -0.50/v0 * (c + v1 * fx_grad)
            M = np.asarray(sol['x']).reshape(-1, 1)
            if self.bool_return_grad:
                return -M
            M /= np.linalg.norm(M)
            instance += self.step_size * M
            #print(M)
            if self.toplot:
                plt.scatter(instance[0], instance[1], color=color,
                            alpha=0.75, marker='x', s=100)
        return instance

    def __optimise_lp_3(self,
                        input_instance: np.array,
                        reg: int) -> np.array:

        self.st = 0
        instance = input_instance.copy(order='K').reshape(-1, 1)

        n_steps = self.n_steps
        colors=cycle(list((plt.cm.Greens(np.linspace(0,1,100)))))

        if self.density_estimator is None:
            self.eval_density()

        C = self.epsilon / (1 - self.epsilon)
        for i in range(n_steps):

            pr = self.model.predict_proba(instance.reshape(1, -1))[0][self.target_class_idx]
            if pr > 0.60:
                break
                color = 'k'
            else:
                color = next(colors)

            fx_grad = self.__estimate_gradient_all(instance)
            #fx_grad /= np.linalg.norm(fx_grad)

            G = self.get_density_grad(instance)
            #sim_grad /= np.linalg.norm(sim_grad)
            p = np.exp(self.density_estimator.score_samples(instance.reshape(1, -1)))[0]


# =============================================================================
#             p = self.compute_sample_similarity(instance)
#             G = np.zeros((self.n_ftrs, 1))
#             for i in range(self.n_samples):
#                 x = self.X[i, :].reshape(-1, 1)
#                 diff0 = (x - instance)
#                 f = self.kernel_func(x, instance)
#                 G += diff0 * f
#             G *= (self.gamma / self.n_samples)
#
# =============================================================================
            l = 0.50
            bound = C * p
            counter = 0
            #print(fx_grad)
            #print(G)
            #print(bound)
            while True:
                counter += 1
                if counter > 25:
                    break
                M = -(fx_grad - l * G)
                M /= np.linalg.norm(M)
                val = np.dot(M.T, G)[0, 0]
                #print(val)
                if val < bound:
                    l += 0.50
                else:
                    break
            if self.bool_return_grad:
                return M

            instance += self.step_size * M
            #print(M)
            if self.toplot:
                plt.scatter(instance[0], instance[1], color=color,
                            alpha=0.75, marker='x', s=100)
        return instance

    def __optimise_qp(self,
                        input_instance: np.array,
                        reg: int) -> np.array:

        self.st = 0
        instance = input_instance.copy(order='K').reshape(-1, 1)

        n_steps = self.n_steps
        colors=cycle(list((plt.cm.Greens(np.linspace(0,1,100)))))

        self.epsilon_0 = 1
        self.epsilon_1 = 0

        S = 0
        mu = np.zeros((self.n_ftrs, 1))
        for i in range(self.n_samples):
            v0 = self.X[i, :].reshape(-1, 1)
            mu += v0
            for j in range(i):
                S += 2 * np.dot(self.X[j, :].reshape(1, -1), v0)
            S += np.dot(v0.T, v0)
        S /= self.n_samples**2
        mu /= self.n_samples

        for i in range(n_steps):

            pr = self.model.predict_proba(instance.reshape(1, -1))[0][self.target_class_idx]
            if pr > 0.60:
                break
                color = 'k'
            else:
                color = next(colors)

            fx_grad = self.__estimate_gradient_all(instance)
            fx_grad /= np.linalg.norm(fx_grad)

            # solve LP
            c = matrix(fx_grad, (self.n_ftrs, 1), tc='d')
            bound = -self.epsilon_1 - S/2 + np.linalg.norm(instance)**2 #+ 2 * np.dot(mu.T, instance)

            a = -(instance / self.n_samples - mu).reshape(1, -1)
            A = matrix(a, (1, self.n_ftrs))
            b = matrix(bound, (1, 1))
            sol=solvers.lp(c,A,b)

# =============================================================================
#             l_00 = 4 * self.epsilon_0
#             l_11 = np.linalg.norm(a) **2
#             l_01 = 2 * bound[0, 0]
#
#             Q = 2*matrix([[l_00, l_01],
#                            [l_01, l_11]])
#
#             l_1 = 2 * np.dot(a, fx_grad)[0, 0]
#             p = matrix([0.0, l_1])
#             G = matrix([[-1.0,0.0],[0.0,-1.0]])
#             h = matrix([0.0,0.0])
#             A = matrix([0.0, 0.0], (1,2))
#             b = matrix(0.0)
#             sol=solvers.qp(Q, p, G, h)
# =============================================================================
# =============================================================================
#             c = fx_grad.ravel().tolist()
#             A = [a.ravel().tolist()]
#             b = [bound[0, 0]]
#
#             print(c, A, b)
#             from scipy.optimize import linprog
#             lb = -100
#             ub = 100
#             res = linprog(c, A_ub=A, b_ub=b, bounds=((lb, ub), (lb, ub)))
#             print(res)
# =============================================================================

            v0, v1 = sol['x']
            M = -0.50/v0 * (c + v1 * fx_grad)
            M = np.asarray(sol['x']).reshape(-1, 1)
            if self.bool_return_grad:
                return -M
            M /= np.linalg.norm(M)
            instance += self.step_size * M
            #print(M)
            if self.toplot:
                plt.scatter(instance[0], instance[1], color=color,
                            alpha=0.75, marker='x', s=100)
        return instance

    def __optimise_qp_2(self,
                        input_instance: np.array,
                        reg: int) -> np.array:

        self.st = 0
        instance = input_instance.copy(order='K').reshape(-1, 1)

        n_steps = self.n_steps
        colors=cycle(list((plt.cm.Greens(np.linspace(0,1,100)))))
        solvers.options['show_progress'] = False

        if self.density_estimator is None:
            self.eval_density()

        C = self.epsilon / (1 - self.epsilon)
        for i in range(n_steps):

            pr = self.model.predict_proba(instance.reshape(1, -1))[0][self.target_class_idx]
            if pr > 0.60:
                break
                color = 'k'
            else:
                color = next(colors)

            fx_grad = self.__estimate_gradient_all(instance)
            #fx_grad /= np.linalg.norm(fx_grad)

            sim_grad = self.get_density_grad(instance)
            #sim_grad /= np.linalg.norm(sim_grad)
            p = np.exp(self.density_estimator.score_samples(instance.reshape(1, -1)))[0]

            # solve QP dual

            #p = self.compute_sample_similarity(instance)
# =============================================================================
#             sim_grad = np.zeros((self.n_ftrs, 1))
#             for i in range(self.n_samples):
#                 x = self.X[i, :].reshape(-1, 1)
#                 diff0 = (x - instance)
#                 f = self.kernel_func(x, instance)
#                 sim_grad += diff0 * f
#             sim_grad *= (self.gamma / self.n_samples)
#
# =============================================================================
            #sim_grad /= np.linalg.norm(sim_grad)
            bound = -C * p


            # min c.Tx + l * (x.Tx - 1) + v * (a.Tx - b)
            # a = -grad p(x)
            # b = - c * p(x)

            l_00 = 4 * self.radius**2## l
            l_11 = np.linalg.norm(sim_grad) **2 # v
            l_01 = -2 * bound

            Q = 2*matrix([[l_00, l_01],
                           [l_01, l_11]])

            l_1 = -2 * np.dot(sim_grad.T, fx_grad)[0, 0]
            p = matrix([0.0, l_1])
            G = matrix([[-1.0,0.0],[0.0,-1.0]])
            h = matrix([0.0,0.0])

            sol=solvers.qp(Q, p, G, h)

            v0, v1 = sol['x'][0], sol['x'][1]

            #M = -(fx_grad - v1 * sim_grad)
            M = -v1 * sim_grad + fx_grad
            #M = -0.50/v0 * (c + v1 * fx_grad)
            #M = np.asarray(sol['x']).reshape(-1, 1)
            #print(M)
            #print(sol['s'])
            if self.bool_return_grad:
                return M
            M /= np.linalg.norm(M)
            instance += self.step_size * M

            if self.toplot:
                plt.scatter(instance[0], instance[1], color=color,
                            alpha=0.75, marker='x', s=100)
        return instance

    def __optimise_density_threshold(self,
                                     input_instance: np.array,
                                     reg: int) -> np.array:
        if self.density_estimator is None:
            self.eval_density()

        self.st = 0
        #satisfied = self.__eval_condition(self.current_prediction)
        instance = input_instance.copy(order='K').reshape(-1, 1)

        #while not satisfied:

        n_steps = self.n_steps
        colors=cycle(list((plt.cm.Greens(np.linspace(0,1,n_steps)))))
        for i in range(n_steps):
            self.st += 1
            pr = self.model.predict_proba(instance.reshape(1, -1))[0][self.target_class_idx]
            density_est = np.exp(self.density_estimator.score_samples(instance.reshape(1, -1)))

            if (pr >= 0.60 and density_est >=self.epsilon):
                break

            fx_grad = self.__estimate_gradient_all(instance)
            df_grad, df_hess = self.get_density_grad(instance, get_hess=True)



            hess_inv = np.linalg.inv(df_hess + 1e-6 * np.identity(2))

# =============================================================================
#             num = np.dot(np.dot(df_grad.T, hess_inv), fx_grad)
#             den = np.dot(np.dot(df_grad.T, hess_inv), df_grad) - 2 * bound
#             l = -num / den
# =============================================================================

            t = 1
            counter = 0
            while True:
                bound = density_est - t * self.epsilon
                num = np.dot(np.dot(fx_grad.T, hess_inv), fx_grad)
                den = np.dot(np.dot(df_grad.T, hess_inv), df_grad) - 2 * bound
                rat = num/den
                if rat < 0:
                    t /= 2
                    counter += 1
                    if counter > 5:
                        M = fx_grad
                        break
                else:
                    l = np.sqrt(rat[0][0])
                    M = -np.dot(hess_inv, df_grad + fx_grad/l)
                    break

             #  needs /l
            if self.norm_bool:
                M /= np.linalg.norm(M)

            if self.bool_return_grad:
                return -M
            #print(M)
            instance += self.step_size * M
# =============================================================================
#             fx_grad /= np.linalg.norm(fx_grad)
#             df_grad /= np.linalg.norm(df_grad)
#             alpha = self.alpha
#
#             counter = 0
#             t = 1
#             while True:
#                 counter += 1
#                 if counter > 3:
#                     t *= 0.50
#                     if t < 0.01:
#                         break
#                 comb_grad = fx_grad - alpha * df_grad
#                 if self.norm_bool:
#                     comb_grad /= np.linalg.norm(comb_grad)
#                 test_instance = instance - t * self.step_size * comb_grad
#                 sc = np.exp(self.density_estimator.score_samples(test_instance.reshape(1, -1)))
#                 if sc >= self.sc_lim:
#                     instance = test_instance
#                 else:
#                     if alpha == 0:
#                         alpha = 1
#                     else:
#                         alpha *= 1.5
# =============================================================================

            #instance += self.step_size * comb_grad

            self.ax.scatter(instance[0], instance[1], color=next(colors), marker='x', s=100)
        return instance

    def solve_sdp(self, c, P):
        import cvxpy as cp

        # Construct the problem.
        x = cp.Variable(self.n_ftrs + 1)
        objective = cp.Minimize(c.T*x)
        constraints = [cp.quad_form(x, np.identity(self.n_ftrs + 1)) <= 2,
                       cp.quad_form(x, P) <= 0,
                       x[2]**2 == 1]
        prob = cp.Problem(objective, constraints)

        # The optimal objective value is returned by `prob.solve()`.
        prob.solve(gp=True)
        return x.value()

    def __optimise_sdp(self,
                       input_instance: np.array,
                       reg: int) -> np.array:

        if self.density_estimator is None:
            self.eval_density()

        instance = input_instance.copy(order='K').reshape(-1, 1)


        n_steps = self.n_steps
        colors=cycle(list((plt.cm.Greens(np.linspace(0,1,n_steps)))))
        for i in range(n_steps):

            fx_grad = self.__estimate_gradient_all(instance)
            df_grad, df_hess = self.get_density_grad(instance, get_hess=True)

            #pr = self.model.predict_proba(instance.reshape(1, -1))[0][self.target_class_idx]
            density_est = np.exp(self.density_estimator.score_samples(instance.reshape(1, -1)))


            c = -np.vstack((fx_grad, 0))
            H = (df_hess + 1e-6*np.identity(self.n_ftrs))
            p = df_grad.reshape(-1, 1)
            e = density_est - self.epsilon

            P = np.hstack((H, 0.50 * p))
            q = np.vstack((0.50*p, e))
            P = np.vstack((P, q.T))

            M = self.solve_sdp(c, P)
            if self.norm_bool:
                M /= np.linalg.norm(M)

            if self.bool_return_grad:
                return -M
            print(M)
            instance += self.step_size * M
# =============================================================================
#             fx_grad /= np.linalg.norm(fx_grad)
#             df_grad /= np.linalg.norm(df_grad)
#             alpha = self.alpha
#
#             counter = 0
#             t = 1
#             while True:
#                 counter += 1
#                 if counter > 3:
#                     t *= 0.50
#                     if t < 0.01:
#                         break
#                 comb_grad = fx_grad - alpha * df_grad
#                 if self.norm_bool:
#                     comb_grad /= np.linalg.norm(comb_grad)
#                 test_instance = instance - t * self.step_size * comb_grad
#                 sc = np.exp(self.density_estimator.score_samples(test_instance.reshape(1, -1)))
#                 if sc >= self.sc_lim:
#                     instance = test_instance
#                 else:
#                     if alpha == 0:
#                         alpha = 1
#                     else:
#                         alpha *= 1.5
# =============================================================================

            #instance += self.step_size * comb_grad

            self.ax.scatter(instance[0], instance[1], color=next(colors), marker='x', s=100)
        return instance

    def __optimise_combined_lap(self,
                                input_instance: np.array,
                                reg: int) -> np.array:

        self.st = 0
        instance = input_instance.copy(order='K').reshape(-1, 1)

        n_steps = self.n_steps
        colors=cycle(list((plt.cm.Greens(np.linspace(0,1,100)))))

        if self.kernel is None:
            self.get_kernel()

        prev_grad = np.zeros((2, 1))
        for i in range(n_steps):

            pr = self.model.predict_proba(instance.reshape(1, -1))[0][self.target_class_idx]
            if pr > 0.60:
                break
                color = 'k'
            else:
                color = next(colors)

            fx_grad = self.__estimate_gradient_all(instance)
            #fx_grad /= np.linalg.norm(fx_grad)

            # solve LP
            #c = matrix(fx_grad, (self.n_ftrs, 1), tc='d')
            s = self.compute_sample_similarity(instance)
            L = np.sqrt(s)

            G = np.zeros((self.n_ftrs, 1))
            G_prime = np.zeros((self.n_ftrs, self.n_ftrs))

            for i in range(self.n_samples):
                x = self.X[i, :].reshape(-1, 1)
                diff0 = (x - instance)
                f = self.kernel_func(x, instance)
                G += diff0 * f

                G_prime += (self.gamma * np.dot(diff0, diff0.T) - np.identity(self.n_ftrs)) *  f

            G_prime *= (self.gamma / self.n_samples)

            G *= (self.gamma / self.n_samples)

            G_sq = np.dot(G, G.T)

            part0 = -self.gamma * np.identity(self.n_ftrs)
            #part1 = -0.50 * (G_prime * L - 0.50 * G_sq / L) / L**3
            #part2 = -0.50 * G_sq / L**4

            part1 = 0.50*G_prime / L**2
            part2 = 0.75 * G_sq / L**4
            #print(part0.shape, part1.shape, part2.shape)
            const = 1 / (self.n_samples * L**2)
            hess = const * (part0 + part1 + part2)
            hess_inv = np.linalg.pinv(hess)

            sim_grad = G / (2 * self.n_samples * s**2)

            sim_eval = 1 / (self.n_samples * s)

            t = 1
            counter = 0
            while True:
                bound = sim_eval - t * self.epsilon
                num = np.dot(np.dot(sim_grad.T, hess_inv), fx_grad)
                den = np.dot(np.dot(sim_grad.T, hess_inv), sim_grad) - 2 * bound
                rat = num/den
                if rat < 0:
                    t /= 2
                    counter += 1
                    if counter > 5:
                        M = fx_grad #+ prev_grad
                        break
                else:
                    l = np.sqrt(rat[0][0])
                    M = np.dot(hess_inv, sim_grad*l + fx_grad)
                    break

            if self.bool_return_grad:
                return M

            if self.norm_bool:
                M /= np.linalg.norm(M)
            instance += self.step_size * M
            #print(M)
            if self.toplot:
                plt.scatter(instance[0], instance[1], color=color,
                            alpha=0.75, marker='x', s=100)
        return instance

    def compute_sim_grad(self, instance, s):
        G = np.zeros((self.n_ftrs, 1))

        for i in range(self.n_samples):
            x = self.X[i, :].reshape(-1, 1)
            diff0 = (x - instance)
            f = self.kernel_func(x, instance)
            G += diff0 * f

        G *= self.gamma
        return G / (2 * self.n_samples * s**2)

    def compute_sim_grad_and_hess(self, instance, s):
        G = np.zeros((self.n_ftrs, 1))
        G_prime = np.zeros((self.n_ftrs, self.n_ftrs))
        L = np.sqrt(s)

        for i in range(self.n_samples):
            x = self.X[i, :].reshape(-1, 1)
            diff0 = (x - instance)
            f = self.kernel_func(x, instance)
            G += diff0 * f

            G_prime += (self.gamma * np.dot(diff0, diff0.T) - np.identity(self.n_ftrs)) *  f

        G_prime *= (self.gamma / self.n_samples)

        G *= (self.gamma / self.n_samples)

        G_sq = np.dot(G, G.T)

        part0 = -self.gamma * np.identity(self.n_ftrs)
        part1 = -0.50 * (G_prime * L - 0.50 * G_sq / L) / L**3
        part2 = -0.50 * G_sq / L**4
        #print(part0.shape, part1.shape, part2.shape)
        const = 1 / (self.n_samples * L**2)
        hess = const * (part0 + part1 + part2)

        return G / (2 * self.n_samples * s**2), hess

    def plot_ellipse_1(self, instance):
        G = np.zeros((self.n_ftrs, 1))
        s = self.compute_sample_similarity(instance) * self.n_samples
        for i in range(self.n_samples):
            x = self.X[i, :].reshape(-1, 1)
            diff0 = (x - instance)
            f = self.kernel_func(x, instance)
            G += diff0 * f

        G *= (self.gamma ** 2)

        G_sq = np.dot(G, G.T)
        const = 2 * self.gamma * s**2
        cov_ = (const * np.identity(2) - G_sq) / s**3
        cov = np.linalg.inv(cov_) / 500
        #cov = cov_
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)
        from matplotlib.patches import Ellipse
        import matplotlib.pyplot as plt
        for j in range(1, 4):
            ell = Ellipse(xy=(instance[0], instance[1]),
                          width=lambda_[0]*j*2, height=lambda_[1]*j*2,
                          angle=np.rad2deg(np.arccos(v[0, 0] / v[0, 1])))
            ell.set_edgecolor('red')
            ell.set_facecolor('none')
            self.ax.add_artist(ell)
            ell = Ellipse(xy=(instance[0], instance[1]),
                          width=1, height=1)
            ell.set_edgecolor('blue')
            ell.set_facecolor('none')
            self.ax.add_artist(ell)

    def plot_ellipse_3(self, instance):
        G = np.zeros((self.n_ftrs, 1))
        s = self.compute_sample_similarity(instance) * self.n_samples
        for i in range(self.n_samples):
            x = self.X[i, :].reshape(-1, 1)
            diff0 = (x - instance)
            f = self.kernel_func(x, instance)
            G += diff0 * f

        G *= (self.gamma)

        G_sq = np.dot(G, G.T)
        const = 2 * self.gamma * s**2
        cov_ = (const * np.identity(2) - G_sq) / s**3
        cov = np.linalg.inv(cov_) / 500
        t = np.linspace(0, 2*math.pi, 1000)
        Z = np.vstack((self.radius**2 * np.cos(t), self.radius**2 * np.sin(t))).T #+ instance.reshape(1, -1)
        X = np.dot(Z, cov)
        plt.scatter(instance[0], instance[1], color='red')
        plt.plot(X[:, 0] + instance[0], X[:, 1] + instance[1], color='red')
        plt.plot(Z[:, 0] + instance[0], Z[:, 1] + instance[1], color='blue')


    def __optimise_projected(self,
                             input_instance: np.array,
                             reg: int) -> np.array:

        self.st = 0
        instance = input_instance.copy(order='K').reshape(-1, 1)
        instance_original = input_instance.copy(order='K').reshape(-1, 1)

        n_steps = self.n_steps
        colors=cycle(list((plt.cm.RdBu(np.linspace(0,1,100)))))

        L_original = np.sqrt(self.compute_sample_similarity(instance))
        C = 1/(L_original * self.n_samples)

        bound = self.epsilon / C

        for i in range(n_steps):

            pr = self.model.predict_proba(instance.reshape(1, -1))[0][self.target_class_idx]
            if pr > 0.70:
                break

            fx_grad = self.__estimate_gradient_all(instance)

            s = self.compute_sample_similarity(instance)

            similarity = self.kernel_func(instance_original, instance) / np.sqrt(s)
            if self.norm_bool:
                fx_grad /= np.linalg.norm(fx_grad)

            instance += self.step_size * fx_grad

            counter = 0
            while similarity < bound:
                counter += 1
                if counter > 10:
                    break
                sim_grad = self.compute_sim_grad(instance, s)
                instance += self.sim_step_size * sim_grad / np.linalg.norm(sim_grad)

            #print(M)
            if self.toplot:
                plt.scatter(instance[0], instance[1], color=next(colors),
                            alpha=0.75, marker='x', s=100)
        return instance

    def __optimise_projected_hess(self,
                                  input_instance: np.array,
                                  reg: int) -> np.array:

        self.st = 0
        instance = input_instance.copy(order='K').reshape(-1, 1)
        instance_original = input_instance.copy(order='K').reshape(-1, 1)

        n_steps = self.n_steps
        colors=cycle(list((plt.cm.Greens(np.linspace(0,1,100)))))

        L_original = np.sqrt(self.compute_sample_similarity(instance))
        C = 1/(L_original * self.n_samples)

        bound = self.epsilon / C

        for i in range(n_steps):

            pr = self.model.predict_proba(instance.reshape(1, -1))[0][self.target_class_idx]
            if pr > 0.70:
                break

            fx_grad = self.__estimate_gradient_all(instance)

            s = self.compute_sample_similarity(instance)

            similarity = self.kernel_func(instance_original, instance) / np.sqrt(s)
            if self.norm_bool:
                fx_grad /= np.linalg.norm(fx_grad)

            instance += self.step_size * fx_grad

            counter = 0
            while similarity < bound:
                counter += 1
                if counter > 5:
                    break
                sim_grad, hess = self.compute_sim_grad_and_hess(instance, s)
                H = np.linalg.inv(hess)
                M = -np.dot(H, sim_grad)
                instance += 0.025 * M / np.linalg.norm(M)

            #print(M)
            if self.toplot:
                plt.scatter(instance[0], instance[1], color=next(colors),
                            alpha=0.75, marker='x', s=100)
        return instance

    def plot_ellipse_2(self, instance, fx_grad = None, plot_ellipse=False):
        if self.kernel is None:
            self.get_kernel()
            self.volume = np.sum(self.kernel) - self.n_samples

        G = np.zeros((self.n_ftrs, 1))
        G_prime = np.zeros((self.n_ftrs, self.n_ftrs))
        s = self.compute_sample_similarity(instance) * self.n_samples
        for i in range(self.n_samples):
            x = self.X[i, :].reshape(-1, 1)
            diff0 = (x - instance).reshape(-1, 1)
            f = self.kernel_func(x, instance)
            G += diff0 * f
            G_prime += (self.gamma * np.dot(diff0, diff0.T) - np.identity(self.n_ftrs)) *  f

        G_prime *= (self.gamma)
        G *= (self.gamma)
        N = 100
        epsilon = self.radius**2 - 1/s #self.radius**2/self.volume - 1/s #np.sqrt(self.radius/self.volume) - 1/s

        G_sq = np.dot(G, G.T)
        #part0 = 0#G_prime / s
        part1 = - G_sq / s**2
        part2 = 2 * self.gamma * np.identity(2)
        cov_ = (part1 + part2)
        cov = np.linalg.inv(cov_) * s

# =============================================================================
#         if plot_ellipse:
#             lambda_, v = np.linalg.eig(cov)
#             lambda_ = np.sqrt(lambda_)
#             from matplotlib.patches import Ellipse
#             import matplotlib.pyplot as plt
#             for j in range(1, 2):
#                 ell = Ellipse(xy=(instance[0], instance[1]),
#                               width=lambda_[0]*j*2, height=lambda_[1]*j*2,
#                               angle=np.rad2deg(np.arctan(v[1, 0] / v[0, 0])))
#                 ell.set_edgecolor('red')
#                 ell.set_facecolor('none')
#                 self.ax.add_artist(ell)
#                 ell = Ellipse(xy=(instance[0], instance[1]),
#                               width=1, height=1)
#                 ell.set_edgecolor('blue')
#                 ell.set_facecolor('none')
#                 self.ax.add_artist(ell)
# =============================================================================
        if plot_ellipse:
            t = np.linspace(0, 2*math.pi, N)
            Z = np.vstack((self.radius**2 * np.cos(t), self.radius**2 * np.sin(t))).T #+ instance.reshape(1, -1)
            X = np.zeros(Z.shape)
            for i in range(N):
                v0 = Z[i, :].reshape(-1, 1) #+ instance.reshape(-1, 1)
                l = np.sqrt(np.dot(np.dot(v0.T, cov), v0) / (4 * epsilon))
                X[i, :] = -np.dot(cov, v0).ravel() / l

            plt.scatter(instance[0], instance[1], color='red')
            plt.plot(X[:, 0] + instance[0], X[:, 1] + instance[1], color='red')
            plt.plot(Z[:, 0] + instance[0], Z[:, 1] + instance[1], color='blue')

            if fx_grad is not None:
                v0 = fx_grad.reshape(-1, 1) #+ instance.reshape(-1, 1)
                l = np.sqrt(np.dot(np.dot(v0.T, cov), v0) / (4 * epsilon))
                return cov / l
        else:
            return cov, epsilon


    def __optimise_lap_dist(self,
                            input_instance: np.array):
        self.st = 0
        instance = input_instance.copy(order='K').reshape(-1, 1)

        n_steps = self.n_steps
        colors=cycle(list((plt.cm.Greens(np.linspace(0,1,100)))))
        if self.kernel is None:
            self.get_kernel()
            self.volume = np.sum(self.kernel) - self.n_samples

        t = 0
        for i in range(n_steps):

            pr = self.model.predict_proba(instance.reshape(1, -1))[0][self.target_class_idx]
            if pr > 0.70:
                print('success')
                break

            fx_grad = self.__estimate_gradient_all(instance)
            fx_grad /= np.linalg.norm(fx_grad)
            if self.bool_return_grad:
                return self.plot_ellipse_2(instance, fx_grad, 0)
            else:
                #instance += fx_grad * self.step_size
                hess_inv, epsilon = self.plot_ellipse_2(instance, fx_grad, 0)
                num = np.dot(hess_inv + t * self.sigma * np.identity(2), fx_grad)
                part0 = np.dot(fx_grad.T, num)
                den = np.sqrt(part0)
                M = num * 2 * epsilon / den
                if np.linalg.norm(M) < 1e-3:
                    pass
                    #M = fx_grad
                instance += self.step_size * M #/np.linalg.norm(M)

            if self.toplot:
                plt.scatter(instance[0], instance[1], color=next(colors),
                            alpha=0.75, marker='x', s=100)
        print(t, np.linalg.norm(fx_grad))
        return instance

    def __optimise_qp_both_hess(self,
                                input_instance: np.array) -> np.array:

        self.st = 0
        instance = input_instance.copy(order='K').reshape(-1, 1)

        n_steps = self.n_steps
        colors=cycle(list((plt.cm.Greens(np.linspace(0,1,100)))))
        if self.kernel is None:
            self.get_kernel()
            self.volume = np.sum(self.kernel) - self.n_samples

        t = 0
        for i in range(n_steps):

            pr = self.model.predict_proba(instance.reshape(1, -1))[0][self.target_class_idx]
            if pr > 0.70:
                print('hi')
                break

            fx_grad, fx_hess = self.__estimate_gradient_all(instance, 1)
            hess_inv, epsilon = self.plot_ellipse_2(instance, fx_grad, 0)
            P, L, U = scipy.linalg.lu(hess_inv)
            Q = np.dot(np.dot(L, fx_hess), U)
            M = np.dot(Q, fx_grad)
            M /= np.linalg.norm(M)
            M *= epsilon

            instance -= self.step_size * M

            if self.toplot:
                plt.scatter(instance[0], instance[1], color=next(colors),
                            alpha=0.75, marker='x', s=100)
        return instance

    def explain_lasso_path(self,
                           subject_instance: np.array,
                           target_prediction: np.array,
                           nottochange: Optional[List[int]] = None):

        self.subject_instance = subject_instance
        self.target_prediction = target_prediction.reshape(-1, 1)
        self.target_class_idx = np.argmax(self.target_prediction)
        self.n_classes = self.target_prediction.shape[0]
        self.n_ftrs = len(self.subject_instance)
        self.current_prediction = self.model.predict_proba(
                                    self.subject_instance.reshape(1, -1)).reshape(-1, 1)
        self.counter = 0
        if not nottochange:
            self.nottochange = []
            self.tosamplefrom = self.n_ftrs
        else:
            self.nottochange = nottochange
            self.tosamplefrom = list(range(self.n_ftrs))[:]
            for item in self.nottochange:
                self.tosamplefrom.pop(self.tosamplefrom.index(item))

        instance = self.subject_instance.copy(order='K')
        vvv = self.subject_instance.copy(order='K').reshape(-1, 1)
        xs = []
        ts = []
        regs = np.linspace(0.001, 0.50, 50)
        breakpoint = 0
        c = 0
        for reg in regs:
            modified_x, restart = self.__optimise_lasso(instance, reg)
            xs.append(modified_x.reshape(-1, 1) - vvv)
            ts.append(np.argmax(self.model.predict_proba(modified_x.reshape(1, -1))).reshape(-1, 1))
            if (restart > 0 and c == 0):
                breakpoint = reg
                c += 1
        return xs, regs, breakpoint, ts

    def explain(self,
                subject_instance: np.array,
                ax,
                target_prediction: np.array,
                target_instance = None,
                toplot=False,
                nottochange: Optional[List[int]] = None):

        self.toplot = toplot
        self.ax = ax
        self.subject_instance = subject_instance
        self.target_instance = target_instance
        if isinstance(self.target_instance, np.ndarray):
            self.target_instance.reshape(-1, 1)
            if self.toplot:
                self.ax.scatter(target_instance[0], target_instance[1],
                                color='k', marker='x', s=100)

        self.target_prediction = target_prediction.reshape(-1, 1)
        self.target_class_idx = np.argmax(self.target_prediction)
        self.n_classes = self.target_prediction.shape[0]
        self.current_prediction = self.model.predict_proba(
                                    self.subject_instance.reshape(1, -1)).reshape(-1, 1)
        self.counter = 0
        if not nottochange:
            self.nottochange = []
            self.tosamplefrom = self.n_ftrs
        else:
            self.nottochange = nottochange
            self.tosamplefrom = list(range(self.n_ftrs))
            for item in self.nottochange:
                self.tosamplefrom.pop(self.tosamplefrom.index(item))

        instance = self.subject_instance.copy(order='K')
        previous_modified_x = 0
        for reg in [0]: #[0.001, 0.01, 0.05, 0.10, 0.20]:


            if self.whattouse == 1:
                modified_x = self.__optimise_lap(instance, reg)
            elif self.whattouse == 0:
                modified_x = self.__optimise_new_tijl_(instance, reg)
            elif self.whattouse == 2:
                modified_x = self.__optimise_lap2(instance, reg)
            elif self.whattouse == 3:
                modified_x = self.__optimise_lap3(instance, reg)
            elif self.whattouse == 'combined':
                modified_x = self.__optimise_combined(instance, reg)
            elif self.whattouse == 'density':
                modified_x = self.__optimise_density(instance, reg)
            elif self.whattouse == 'lap':
                modified_x = self.__optimise_lap(instance, reg)
            elif self.whattouse == 'lp':
                modified_x = self.__optimise_lp_2(instance, reg)
            elif self.whattouse == 'lp_new':
                modified_x = self.__optimise_lp_3(instance, reg)
            elif self.whattouse == 'qp_dual':
                modified_x = self.__optimise_qp_2(instance, reg)
            elif self.whattouse == 'qp_threshold':
                modified_x = self.__optimise_density_threshold(instance, reg)
            elif self.whattouse == 'sdp':
                modified_x = self.__optimise_sdp(instance, reg)
            elif self.whattouse == 'lap_comb':
                modified_x = self.__optimise_combined_lap(instance, reg)
            elif self.whattouse == 'projected':
                modified_x = self.__optimise_projected(instance, reg)
            elif self.whattouse == 'projected_hessian':
                modified_x = self.__optimise_projected_hess(instance, reg)
            elif self.whattouse == 'plot_ellipse':
                modified_x = self.plot_ellipse_2(instance.reshape(-1, 1),
                                                 plot_ellipse=1)
            elif self.whattouse == 'lap_dist':
                modified_x = self.__optimise_lap_dist(instance)
            elif self.whattouse == 'hess_both':
                modified_x = self.__optimise_qp_both_hess(instance)
            return modified_x

            pred = self.model.predict(modified_x.reshape(1, -1))

            if pred != np.argmax(self.target_prediction):
                return previous_modified_x
            else:
                previous_modified_x = modified_x

        return modified_x
