"""
Components for specifying dynamic linear (state space) models

Notes
-----
"""
from __future__ import division

import numpy as np
import numpy.linalg as npl
import scipy.linalg as L
import statlib.tools as tools

class Component(object):
    """
    Constant DLM component, can be combined with other components via
    superposition
    """
    def __init__(self, F, G, discount=None):
        if F.ndim == 1:
            F = np.atleast_2d(F)

        self.F = F
        self.G = G
        self.discount = discount

    def __add__(self, other):
        if not isinstance(other, Component):
            raise Exception('Can only add other DLM components!')

        return Superposition(self, other)

    def __radd__(self, other):
        return Superposition(other, self)

class ConstantComponent(Component):
    """
    F matrix is the same at each time t
    """
    pass

class Regression(Component):

    def __init__(self, F, discount=None):
        if F.ndim == 1:
            F = np.atleast_2d(F).T

        G = np.eye(F.shape[1])

        super(Regression, self).__init__(F, G, discount=discount)

class AR(Component):
    pass

class VectorAR(Component):

    def __init__(self, X, lags=1, intercept=True, discount=None):
        nobs = len(X) - lags

        X = np.asarray(X)
        F = np.concatenate([X[lags - i:-i] for i in range(1, lags + 1)], axis=1)
        G = None

        if intercept:
            F = np.c_[np.ones((nobs, 1)), F]

        super(VectorAR, self).__init__(F, G, discount=discount)

class ARMA(Component):
    """
    DLM for ARMA(p, q) Component

    Parameters
    ----------


    """
    def __init__(self, X, ar=None, ma=None, discount=None):
        pass



class Polynomial(ConstantComponent):
    """
    nth order Polynomial DLM using Jordan form system matrix

    Parameters
    ----------
    order : int
    lam : float, default 1.
    """
    def __init__(self, order, lam=1., discount=None):
        self.order = order

        F = _e_vector(order)
        G = tools.jordan_form(order, lam)
        ConstantComponent.__init__(self, F, G, discount=discount)

    def __repr__(self):
        return 'Polynomial(%d)' % self.order

class Superposition(object):
    """

    """

    def __init__(self, *comps):
        self.comps = list(comps)

    def is_observable(self):
        pass

    @property
    def F(self):
        length = None
        for c in self.comps:
            if not isinstance(c, ConstantComponent):
                if length is None:
                    length = len(c.F)
                elif length != len(c.F):
                    raise Exception('Length mismatch in dynamic components')


        if length is None:
            # all constant components
            return np.concatenate([c.F for c in self.comps], axis=1)

        to_concat = []
        for c in self.comps:
            F = c.F
            if isinstance(c, ConstantComponent):
                F = np.repeat(F, length, axis=0)

            to_concat.append(F)

        return np.concatenate(to_concat, axis=1)

    @property
    def G(self):
        return L.block_diag(*[c.G for c in self.comps])

    @property
    def discount(self):
        # TODO: FIX ME, LAZY-ness needed above

        # W&H p. 198, case of multiple discount factors
        k = len(self.G)
        disc_matrix = np.ones((k, k))
        j = 0

        need_matrix = False
        seen_factor = self.comps[0].discount
        for c in self.comps:
            if c.discount is None:
                raise Exception("Must specify discount factor for all "
                                "components or none of them")

            if seen_factor != c.discount:
                need_matrix = True

            i = len(c.G)
            disc_matrix[j : j + i, j : j + i] = c.discount
            j += i

        if need_matrix:
            return disc_matrix
        else:
            return seen_factor

    def __repr__(self):
        reprs = ', '.join(repr(c) for c in self.comps)
        return 'Superposition: [%s]' % reprs

    def __add__(self, other):
        if isinstance(other, Component):
            new_comps = self.comps + [other]
        elif isinstance(other, Superposition):
            new_comps = self.comps + other.comps

        return Superposition(*new_comps)

    def __radd__(self):
        pass

class SeasonalFactors(ConstantComponent):
    """

    """
    def __init__(self, period, discount=None):
        F = _e_vector(period)
        P = tools.perm_matrix(period)
        self.period = period
        ConstantComponent.__init__(self, F, P, discount=discount)

    def __repr__(self):
        return 'SeasonalFree(period=%d)' % self.period

class FourierForm(ConstantComponent):
    """

    """
    def __init__(self, theta=None, discount=None):
        self.theta = theta
        F = _e_vector(2)
        G = tools.fourier_matrix(theta)
        ConstantComponent.__init__(self, F, G, discount=discount)

    def __repr__(self):
        return 'FourierForm(%.4f)' % self.theta

class FullEffectsFourier(ConstantComponent):
    """
    Full effects Fourier form DLM rep'n

    Parameters
    ----------
    period : int
    harmonics : sequence, default None
        Optionally specify a subset of harmonics to use
    discount : float

    Notes
    -----
    W&H pp. 252-254
    """

    def __init__(self, period, harmonics=None, discount=None):
        period = int(period)
        theta = 2 * np.pi / period
        h = period // 2

        self.period = period
        self.comps = []
        self.model = None

        for j in np.arange(1, h + 1):
            if harmonics and j not in harmonics:
                continue

            comp = FourierForm(theta=theta * j)

            if j == h and period % 2 == 0:
                comp = Polynomial(1, lam=-1.)

            if self.model is None:
                self.model = comp
            else:
                self.model += comp

            self.comps.append(comp)

        ConstantComponent.__init__(self, self.model.F, self.model.G,
                                   discount=discount)

    @property
    def L(self):
        # W&H p. 254
        return np.vstack([np.dot(self.F, npl.matrix_power(self.G, i))
                          for i in range(self.period)])

    @property
    def H(self):
        # p. 254. Transformation to convert seasonal effects to equivalent full
        # effects Fourier form states
        el = self.L
        return np.dot(npl.inv(np.dot(el.T, el)), el.T)


def _e_vector(n):
    result = np.zeros(n)
    result[0] = 1
    return result

