from __future__ import division

from alabortijcv2015.result import AlgorithmResult, FitterResult


# Concrete Implementations of AAM Algorithm Results #--------------------------

class AAMAlgorithmResult(AlgorithmResult):

    def __init__(self, image, fitter, shape_parameters,
                 appearance_parameters=None, gt_shape=None):
        super(AAMAlgorithmResult, self).__init__()
        self.image = image
        self.fitter = fitter
        self.shape_parameters = shape_parameters
        self.appearance_parameters = appearance_parameters
        self._gt_shape = gt_shape


class LinearAAMAlgorithmResult(AAMAlgorithmResult):

    def __init__(self, image, fitter, shape_parameters,
                 appearance_parameters=None, gt_shape=None):
        super(LinearAAMAlgorithmResult, self).__init__(
            image, fitter, shape_parameters=shape_parameters,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)

    def shapes(self, as_points=False):
        if as_points:
            return [self.fitter.transform.from_vector(p).sparse_target.points
                    for p in self.shape_parameters]

        else:
            return [self.fitter.transform.from_vector(p).sparse_target
                    for p in self.shape_parameters]

    @property
    def final_shape(self):
        return self.final_transform.sparse_target

    @property
    def initial_shape(self):
        return self.initial_transform.sparse_target


# Concrete Implementations of AAM Fitter Results # ----------------------------

class AAMFitterResult(FitterResult):

    @property
    def costs(self):
        r"""
        Returns a list containing the cost at each fitting iteration.

        :type: `list` of `float`
        """
        raise ValueError('costs not implemented yet.')

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
