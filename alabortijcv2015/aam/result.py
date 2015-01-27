from __future__ import division
import numpy as np

from alabortijcv2015.result import (AlgorithmResult, FitterResult,
                                    SerializableResult)


# Concrete Implementations of AAM Algorithm Results ---------------------------

class AAMAlgorithmResult(AlgorithmResult):

    def __init__(self, image, algorithm, shape_params, linearized_costs,
                 appearance_params=None, gt_shape=None):
        super(AAMAlgorithmResult, self).__init__()
        self.image = image
        self.algorithm = algorithm
        self.shape_params = shape_params
        self._linearized_costs = linearized_costs
        self.appearance_params = appearance_params
        self._gt_shape = gt_shape

    @property
    def n_iters(self):
        return len(self.shape_params) - 1

    def warped_images(self):
        if not hasattr(self, '_warped_images'):
            self._warped_images = []
            for p in self.shape_params:
                self.algorithm.transform.from_vector_inplace(p)
                self._warped_images.append(
                    self.algorithm.interface.warp(self.image))
        return self._warped_images

    def templates(self):
        if not hasattr(self, '_templates'):
            if self.appearance_params:
                self._templates = []
                for c in self.appearance_params:
                    self._templates.append(
                        self.algorithm.appearance_model.instance(c))
            else:
                 self._templates = self.algorithm.appearance_model.mean()
        return self._templates

    def residuals(self):
        if not hasattr(self, '_residuals'):
            reference_frame = self.algorithm.template
            self._residuals = [reference_frame.from_vector(i.as_vector() -
                                                           t.as_vector())
                               for (i, t) in zip(self.warped_images(),
                                                 self.templates())]
        return self._residuals

    def costs(self, normalize=False):
        # if not hasattr(self, '_costs'):
        #     self._costs = [self.algorithm._cost(r.as_vector())
        #                    for r in self.residuals()]
        costs = [self._linearized_costs(k)(0) for k in xrange(self.n_iters)]
        if normalize:
            costs = list(np.asarray(costs) / costs[0])

        return costs

    @property
    def final_cost(self, normalize=False):
        return self.costs(normalize=normalize)[-1]

    @property
    def initial_cost(self, normalize=False):
        return self.costs(normalize=normalize)[0]

    def linearized_costs(self, n_iter):
        return self._linearized_costs(n_iter)


class LinearAAMAlgorithmResult(AAMAlgorithmResult):

    def shapes(self, as_points=False):
        if as_points:
            return [self.algorithm.transform.from_vector(p).sparse_target.points
                    for p in self.shape_params]

        else:
            return [self.algorithm.transform.from_vector(p).sparse_target
                    for p in self.shape_params]

    @property
    def final_shape(self):
        return self.final_transform.sparse_target

    @property
    def initial_shape(self):
        return self.initial_transform.sparse_target


# Concrete Implementations of AAM Fitter Results ------------------------------

class AAMFitterResult(FitterResult):

    def costs(self, normalize=False):
        costs = []
        for j, alg in enumerate(self.algorithm_results):
            costs += alg.costs(normalize=normalize)
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


# Serializable AAM Fitter Result ----------------------------------------------

class SerializableAAMFitterResult(SerializableResult):

    def __init__(self, image_path, shapes, costs, n_iters, algorithm,
                 gt_shape=None):
        super(SerializableAAMFitterResult, self).__init__(
            image_path, shapes, n_iters, algorithm, gt_shape=gt_shape)

        self._costs = costs

    def costs(self):
        return self._costs
