#  MIT License
#  Copyright (c) 2022. Ruslan Guseinov.

from abc import ABC, abstractmethod

import numpy as np
from scipy.sparse import linalg as sparse_linalg


class NewtonSolver(ABC):
    """Newton solver."""

    @property
    @abstractmethod
    def _var(self):
        pass

    def __init__(self):
        super(NewtonSolver, self).__init__()
        self._max_iter = 50
        self._print_iter = True

    def set_max_iter(self, value: int) -> None:
        """Set maximum number of iterations for solver.

        :param value: Maximum number of iterations
        :return: None
        """
        self._max_iter = value

    def set_print_iter(self, value: bool) -> None:
        """Set whether to print solver iterations.

        :param value: If True, solver iterations are printed.
        :return: None
        """
        self._print_iter = value

    @abstractmethod
    def eval_f(self, var):
        pass

    @abstractmethod
    def eval_g(self, var):
        pass

    @abstractmethod
    def eval_H(self, var):
        pass

    def optimize(self) -> None:
        """Run optimization routine.

        :return: None.
        """
        var = self._var
        f = self.eval_f(var)
        for i in range(self._max_iter):
            g = self.eval_g(var)
            g_norm = np.linalg.norm(g)
            if self._print_iter:
                print(f'{i:3}  {f:.8e}  {g_norm:.8e}')
            if np.isclose(g_norm, 0., atol=1.e-6):
                stopping_message = 'gradient norm'
                break
            h = sparse_linalg.spsolve(self.eval_H(var), g)
            next_var, next_f = None, None
            while (h_norm := (1.e-8 < np.linalg.norm(h))) and (f < (next_f := self.eval_f(next_var := var - h))):
                h *= 0.5
            if not h_norm:
                stopping_message = 'step size'
                break
            var, f = next_var, next_f
        else:
            stopping_message = 'iteration count'
        if self._print_iter:
            print(f'Stopping: {stopping_message}.')
