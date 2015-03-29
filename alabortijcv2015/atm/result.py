from __future__ import division

from alabortijcv2015.result import (AlgorithmResult, FitterResult,
                                    SerializableResult)


# Concrete Implementations of ATM Algorithm Results ---------------------------

class ATMAlgorithmResult(AlgorithmResult):

    def __init__(self, image, fitter, shape_parameters, costs,
                 gt_shape=None):
        super(ATMAlgorithmResult, self).__init__()
        self.image = image
        self.fitter = fitter
        self.shape_parameters = shape_parameters
        self._costs = costs
        self._gt_shape = gt_shape

    def costs(self, normalize=False):
        costs = self._costs
        if normalize:
            costs /= self._costs[0]
        return list(costs)

    @property
    def final_cost(self):
        return self.costs[-1]

    @property
    def initial_cost(self):
        return self.costs[0]


class LinearATMAlgorithmResult(ATMAlgorithmResult):

    def __init__(self, image, fitter, shape_parameters, cost,
                 gt_shape=None):
        super(LinearATMAlgorithmResult, self).__init__(
            image, fitter, shape_parameters, cost,
            gt_shape=gt_shape)

    def shapes(self, as_points=False, sparse=True):
        if as_points:
            if sparse:
                return [self.fitter.transform.from_vector(p).sparse_target.points
                        for p in self.shape_parameters]
            else:
                return [self.fitter.transform.from_vector(p).dense_target.points
                        for p in self.shape_parameters]

        else:
            if sparse:
                return [self.fitter.transform.from_vector(p).sparse_target
                        for p in self.shape_parameters]
            else:
                return [self.fitter.transform.from_vector(p).dense_target
                        for p in self.shape_parameters]

    @property
    def final_shape(self):
        return self.final_transform.sparse_target

    @property
    def final_dense_shape(self):
        return self.final_transform.dense_target

    @property
    def initial_shape(self):
        return self.initial_transform.sparse_target

    @property
    def initial_dense_shape(self):
        return self.initial_transform.dense_target


# Concrete Implementations of ATM Fitter Results ------------------------------

class ATMFitterResult(FitterResult):

    def costs(self):
        costs = []
        for j, alg in enumerate(self.algorithm_results):
            costs += alg.costs()

        return costs

    @property
    def final_cost(self):
        r"""
        Returns the final fitting cost.

        :type: `float`
        """
        return self.algorithm_results[-1].final_cost

    @property
    def initial_cost(self):
        r"""
        Returns the initial fitting cost.

        :type: `float`
        """
        return self.algorithm_results[0].initial_cost


# Serializable ATM Fitter Result ----------------------------------------------

class SerializableATMFitterResult(SerializableResult):

    def __init__(self, image_path, shapes, costs, n_iters, algorithm,
                 gt_shape=None):
        super(SerializableATMFitterResult, self).__init__(
            image_path, shapes, n_iters, algorithm, gt_shape=gt_shape)

        self._costs = costs

    def costs(self):
        return self._costs
