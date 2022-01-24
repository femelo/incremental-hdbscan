import numpy as np

DTYPE = np.double
ITYPE = np.intp
INF = np.inf

from math import sqrt

######################################################################
# newObj function
#  this is a helper function for pickling
def newObj(obj):
    return obj.__new__(obj)

METRIC_MAPPING = {'euclidean': 'EuclideanDistance',
                  'l2': 'EuclideanDistance',
                  'minkowski': 'MinkowskiDistance',
                  'p': 'MinkowskiDistance',
                  'manhattan': 'ManhattanDistance',
                  'cityblock': 'ManhattanDistance',
                  'l1': 'ManhattanDistance',
                  'chebyshev': 'ChebyshevDistance',
                  'infinity': 'ChebyshevDistance',
                  'seuclidean': 'SEuclideanDistance',
                  'mahalanobis': 'MahalanobisDistance',
                  'wminkowski': 'WMinkowskiDistance',
                  'hamming': 'HammingDistance',
                  'canberra': 'CanberraDistance',
                  'braycurtis': 'BrayCurtisDistance',
                  'matching': 'MatchingDistance',
                  'jaccard': 'JaccardDistance',
                  'dice': 'DiceDistance',
                  'kulsinski': 'KulsinskiDistance',
                  'rogerstanimoto': 'RogersTanimotoDistance',
                  'russellrao': 'RussellRaoDistance',
                  'sokalmichener': 'SokalMichenerDistance',
                  'sokalsneath': 'SokalSneathDistance',
                  'haversine': 'HaversineDistance',
                  'cosine': 'ArccosDistance',
                  'arccos': 'ArccosDistance',
                  'pyfunc': 'PyFuncDistance'}

def get_valid_metric_ids(L):
    """Given an iterable of metric class names or class identifiers,
    return a list of metric IDs which map to those classes.

    Examples
    --------
    >>> L = get_valid_metric_ids([EuclideanDistance, 'ManhattanDistance'])
    >>> sorted(L)
    ['cityblock', 'euclidean', 'l1', 'l2', 'manhattan']
    """
    return [key for (key, val) in METRIC_MAPPING.items()
            if (val.__name__ in L) or (val in L)]

class DistanceMetric:
    """DistanceMetric class

    This class provides a uniform interface to fast distance metric
    functions.  The various metrics can be accessed via the `get_metric`
    class method and the metric string identifier (see below).

    Examples
    --------

    For example, to use the Euclidean distance:

    >>> dist = DistanceMetric.get_metric('euclidean')
    >>> X = [[0, 1, 2],
             [3, 4, 5]])
    >>> dist.pairwise(X)
    array([[ 0.        ,  5.19615242],
           [ 5.19615242,  0.        ]])

    Available Metrics
    The following lists the string metric identifiers and the associated
    distance metric classes:

    **Metrics intended for real-valued vector spaces:**

    ==============  ====================  ========  ===============================
    identifier      class name            args      distance function
    --------------  --------------------  --------  -------------------------------
    "euclidean"     EuclideanDistance     -         ``sqrt(sum((x - y)^2))``
    "manhattan"     ManhattanDistance     -         ``sum(|x - y|)``
    "chebyshev"     ChebyshevDistance     -         ``sum(max(|x - y|))``
    "minkowski"     MinkowskiDistance     p         ``sum(|x - y|^p)^(1/p)``
    "wminkowski"    WMinkowskiDistance    p, w      ``sum(w * |x - y|^p)^(1/p)``
    "seuclidean"    SEuclideanDistance    V         ``sqrt(sum((x - y)^2 / V))``
    "mahalanobis"   MahalanobisDistance   V or VI   ``sqrt((x - y)' V^-1 (x - y))``
    ==============  ====================  ========  ===============================

    **Metrics intended for two-dimensional vector spaces:**  Note that the haversine
    distance metric requires data in the form of [latitude, longitude] and both
    inputs and outputs are in units of radians.

    ============  ==================  ========================================
    identifier    class name          distance function
    ------------  ------------------  ----------------------------------------
    "haversine"   HaversineDistance   2 arcsin(sqrt(sin^2(0.5*dx)
                                             + cos(x1)cos(x2)sin^2(0.5*dy)))
    ============  ==================  ========================================


    **Metrics intended for integer-valued vector spaces:**  Though intended
    for integer-valued vectors, these are also valid metrics in the case of
    real-valued vectors.

    =============  ====================  ========================================
    identifier     class name            distance function
    -------------  --------------------  ----------------------------------------
    "hamming"      HammingDistance       ``N_unequal(x, y) / N_tot``
    "canberra"     CanberraDistance      ``sum(|x - y| / (|x| + |y|))``
    "braycurtis"   BrayCurtisDistance    ``sum(|x - y|) / (sum(|x|) + sum(|y|))``
    =============  ====================  ========================================

    **Metrics intended for boolean-valued vector spaces:**  Any nonzero entry
    is evaluated to "True".  In the listings below, the following
    abbreviations are used:

     - N  : number of dimensions
     - NTT : number of dims in which both values are True
     - NTF : number of dims in which the first value is True, second is False
     - NFT : number of dims in which the first value is False, second is True
     - NFF : number of dims in which both values are False
     - NNEQ : number of non-equal dimensions, NNEQ = NTF + NFT
     - NNZ : number of nonzero dimensions, NNZ = NTF + NFT + NTT

    =================  =======================  ===============================
    identifier         class name               distance function
    -----------------  -----------------------  -------------------------------
    "jaccard"          JaccardDistance          NNEQ / NNZ
    "maching"          MatchingDistance         NNEQ / N
    "dice"             DiceDistance             NNEQ / (NTT + NNZ)
    "kulsinski"        KulsinskiDistance        (NNEQ + N - NTT) / (NNEQ + N)
    "rogerstanimoto"   RogersTanimotoDistance   2 * NNEQ / (N + NNEQ)
    "russellrao"       RussellRaoDistance       NNZ / N
    "sokalmichener"    SokalMichenerDistance    2 * NNEQ / (N + NNEQ)
    "sokalsneath"      SokalSneathDistance      NNEQ / (NNEQ + 0.5 * NTT)
    =================  =======================  ===============================

    **User-defined distance:**

    ===========    ===============    =======
    identifier     class name         args
    -----------    ---------------    -------
    "pyfunc"       PyFuncDistance     func
    ===========    ===============    =======

    Here ``func`` is a function which takes two one-dimensional numpy
    arrays, and returns a distance.  Note that in order to be used within
    the BallTree, the distance must be a true metric:
    i.e. it must satisfy the following properties

    1) Non-negativity: d(x, y) >= 0
    2) Identity: d(x, y) = 0 if and only if x == y
    3) Symmetry: d(x, y) = d(y, x)
    4) Triangle Inequality: d(x, y) + d(y, z) >= d(x, z)

    Because of the Python object overhead involved in calling the python
    function, this will be fairly slow, but it will have the same
    scaling as other distances.
    """
    def __cinit__(self):
        self.p = 2
        self.vec = np.zeros(1, dtype=DTYPE, order='c')
        self.mat = np.zeros((1, 1), dtype=DTYPE, order='c')
        self.vec_ptr = self.vec
        self.mat_ptr = self.mat
        self.size = 1

    def __reduce__(self):
        """
        reduce method used for pickling
        """
        return (newObj, (self.__class__,), self.__getstate__())

    def __getstate__(self):
        """
        get state for pickling
        """
        if self.__class__.__name__ == "PyFuncDistance":
            return (float(self.p), self.vec, self.mat, self.func, self.kwargs)
        return (float(self.p), self.vec, self.mat)

    def __setstate__(self, state):
        """
        set state for pickling
        """
        self.p = state[0]
        self.vec = state[1]
        self.mat = state[2]
        if self.__class__.__name__ == "PyFuncDistance":
            self.func = state[3]
            self.kwargs = state[4]
        self.vec_ptr = self.vec
        self.mat_ptr = self.mat
        self.size = 1

    @classmethod
    def get_metric(cls, metric, **kwargs):
        """Get the given distance metric from the string identifier.

        See the docstring of DistanceMetric for a list of available metrics.

        Parameters
        ----------
        metric : string or class name
            The distance metric to use
        **kwargs
            additional arguments will be passed to the requested metric
        """
        if isinstance(metric, DistanceMetric):
            return metric

        if callable(metric):
            return PyFuncDistance(metric, **kwargs)

        # Map the metric string ID to the metric class
        if isinstance(metric, type) and issubclass(metric, DistanceMetric):
            pass
        else:
            try:
                metric = METRIC_MAPPING[metric]
            except:
                raise ValueError("Unrecognized metric '%s'" % metric)

        # # In Minkowski special cases, return more efficient methods
        # if metric is MinkowskiDistance:
        #     p = kwargs.pop('p', 2)
        #     if p == 1:
        #         return ManhattanDistance(**kwargs)
        #     elif p == 2:
        #         return EuclideanDistance(**kwargs)
        #     elif np.isinf(p):
        #         return ChebyshevDistance(**kwargs)
        #     else:
        #         return MinkowskiDistance(p, **kwargs)
        # else:
        #     return metric(**kwargs)
        return metric(**kwargs)

    def __init__(self):
        if self.__class__ is DistanceMetric:
            raise NotImplementedError("DistanceMetric is an abstract class")

    def dist(self, x1, x2, size):
        """Compute the distance between vectors x1 and x2

        This should be overridden in a base class.
        """
        return -999

    def rdist(self, x1, x2, size):
        """Compute the reduced distance between vectors x1 and x2.

        This can optionally be overridden in a base class.

        The reduced distance is any measure that yields the same rank as the
        distance, but is more efficient to compute.  For example, for the
        Euclidean metric, the reduced distance is the squared-euclidean
        distance.
        """
        return self.dist(x1, x2, size)

    def pdist(self, X, D):
        """compute the pairwise distances between points in X"""
        for i1 in range(X.shape[0]):
            for i2 in range(i1, X.shape[0]):
                D[i1, i2] = self.dist(X[i1, 0], X[i2, 0], X.shape[1])
                D[i2, i1] = D[i1, i2]
        return 0

    def cdist(self, X, Y, D):
        """compute the cross-pairwise distances between arrays X and Y"""
        if X.shape[1] != Y.shape[1]:
            raise ValueError('X and Y must have the same second dimension')
        for i1 in range(X.shape[0]):
            for i2 in range(Y.shape[0]):
                D[i1, i2] = self.dist(X[i1, 0], Y[i2, 0], X.shape[1])
        return 0

    def _rdist_to_dist(self, rdist):
        """Convert the reduced distance to the distance"""
        return rdist

    def _dist_to_rdist(self, dist):
        """Convert the distance to the reduced distance"""
        return dist

    def rdist_to_dist(self, rdist):
        """Convert the Reduced distance to the true distance.

        The reduced distance, defined for some metrics, is a computationally
        more efficent measure which preserves the rank of the true distance.
        For example, in the Euclidean distance metric, the reduced distance
        is the squared-euclidean distance.
        """
        return rdist

    def dist_to_rdist(self, dist):
        """Convert the true distance to the reduced distance.

        The reduced distance, defined for some metrics, is a computationally
        more efficent measure which preserves the rank of the true distance.
        For example, in the Euclidean distance metric, the reduced distance
        is the squared-euclidean distance.
        """
        return dist

    def pairwise(self, X, Y=None):
        """Compute the pairwise distances between X and Y

        This is a convenience routine for the sake of testing.  For many
        metrics, the utilities in scipy.spatial.distance.cdist and
        scipy.spatial.distance.pdist will be faster.

        Parameters
        ----------
        X : array_like
            Array of shape (Nx, D), representing Nx points in D dimensions.
        Y : array_like (optional)
            Array of shape (Ny, D), representing Ny points in D dimensions.
            If not specified, then Y=X.
        Returns
        -------
        dist : ndarray
            The shape (Nx, Ny) array of pairwise distances between points in
            X and Y.
        """
        Xarr = np.asarray(X, dtype=DTYPE, order='C')
        if Y is None:
            Darr = np.zeros((Xarr.shape[0], Xarr.shape[0]),
                            dtype=DTYPE, order='C')
            self.pdist(Xarr,
                       Darr)
        else:
            Yarr = np.asarray(Y, dtype=DTYPE, order='C')
            Darr = np.zeros((Xarr.shape[0], Yarr.shape[0]),
                            dtype=DTYPE, order='C')
            self.cdist(Xarr,
                       Yarr,
                       Darr)
        return Darr

class EuclideanDistance(DistanceMetric):
    """Euclidean Distance metric

    .. math::
       D(x, y) = \sqrt{ \sum_i (x_i - y_i) ^ 2 }
    """
    def __init__(self):
        self.p = 2

    def dist(self, x1, x2, size):
        return self._dist(x1, x2, size)

    def rdist(self, x1, x2, size):
        return self._rdist(x1, x2, size)

    def _rdist_to_dist(self, rdist):
        return sqrt(rdist)

    def _dist_to_rdist(self, dist):
        return dist * dist

    def rdist_to_dist(self, rdist):
        return np.sqrt(rdist)

    def dist_to_rdist(self, dist):
        return dist ** 2

class PyFuncDistance(DistanceMetric):
    """PyFunc Distance
    A user-defined distance
    Parameters
    ----------
    func : function
        func should take two numpy arrays as input, and return a distance.
    """
    def __init__(self, func, **kwargs):
        self.func = func
        self.kwargs = kwargs

    # in cython < 0.26, GIL was required to be acquired during definition of
    # the function and inside the body of the function. This behaviour is not
    # allowed in cython >= 0.26 since it is a redundant GIL acquisition. The
    # only way to be back compatible is to inherit `dist` from the base class
    # without GIL and called an inline `_dist` which acquire GIL.
    def dist(self, x1, x2, size):
        return self._dist(x1, x2, size)

    def _dist(self, x1, x2, size):
        d = self.func(x1, x2, **self.kwargs)
        try:
            # Cython generates code here that results in a TypeError
            # if d is the wrong type.
            return d
        except TypeError:
            raise TypeError("Custom distance function must accept two "
                            "vectors and return a float.")

def fmax(a, b):
    return max(a, b)