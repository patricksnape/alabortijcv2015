from __future__ import division

from alabortijcv2015.fitter import Fitter
from alabortijcv2015.pdm import OrthoPDM
from alabortijcv2015.transform import OrthoMDTransform, OrthoLinearMDTransform

from .algorithm import (StandardAAMInterface, LinearAAMInterface,
                        PartsAAMInterface, AIC, CIC)
from .result import AAMFitterResult


# Abstract Interface for AAM Fitters ------------------------------------------

class AAMFitter(Fitter):

    def _check_n_appearance(self, n_appearance):
        if n_appearance is not None:
            if type(n_appearance) is int or type(n_appearance) is float:
                for am in self.dm.appearance_models:
                    am.n_active_components = n_appearance
            elif len(n_appearance) == 1 and self.dm.n_levels > 1:
                for am in self.dm.appearance_models:
                    am.n_active_components = n_appearance[0]
            elif len(n_appearance) == self.dm.n_levels:
                for am, n in zip(self.dm.appearance_models, n_appearance):
                    am.n_active_components = n
            else:
                raise ValueError('n_appearance can be an integer or a float '
                                 'or None or a list containing 1 or {} of '
                                 'those'.format(self.dm.n_levels))

    def _fitter_result(self, image, algorithm_results, affine_correction,
                       gt_shape=None):
        return AAMFitterResult(image, self, algorithm_results,
                               affine_correction, gt_shape=gt_shape)


# Concrete Implementations of AAM Fitters -------------------------------------

class StandardAAMFitter(AAMFitter):

    def __init__(self, global_aam, algorithm_cls=AIC,
                 n_shape=None, n_appearance=None, **kwargs):

        super(StandardAAMFitter, self).__init__()

        self.dm = global_aam
        self._algorithms = []
        self._check_n_shape(n_shape)
        self._check_n_appearance(n_appearance)

        for j, (am, sm) in enumerate(zip(self.dm.appearance_models,
                                         self.dm.shape_models)):

            md_transform = OrthoMDTransform(
                sm, self.dm.transform,
                source=am.mean().landmarks['source'].lms,
                sigma2=am.noise_variance())

            algorithm = algorithm_cls(StandardAAMInterface, am,
                                      md_transform, **kwargs)

            self._algorithms.append(algorithm)


class LinearAAMFitter(AAMFitter):

    def __init__(self, global_aam, algorithm_cls=AIC,
                 n_shape=None, n_appearance=None, **kwargs):

        super(LinearAAMFitter, self).__init__()

        self.dm = global_aam
        self._algorithms = []
        self._check_n_shape(n_shape)
        self._check_n_appearance(n_appearance)

        for j, (am, sm) in enumerate(zip(self.dm.appearance_models,
                                         self.dm.shape_models)):

            md_transform = OrthoLinearMDTransform(
                sm, self.dm.n_landmarks,
                sigma2=am.noise_variance())

            algorithm = algorithm_cls(LinearAAMInterface, am,
                                      md_transform, **kwargs)

            self._algorithms.append(algorithm)


class PartsAAMFitter(AAMFitter):

    def __init__(self, parts_aam, algorithm_cls=AIC,
                 n_shape=None, n_appearance=None, **kwargs):

        super(PartsAAMFitter, self).__init__()

        self.dm = parts_aam
        self._algorithms = []
        self._check_n_shape(n_shape)
        self._check_n_appearance(n_appearance)

        for j, (am, sm) in enumerate(zip(self.dm.appearance_models,
                                         self.dm.shape_models)):

            pdm = OrthoPDM(sm, sigma2=am.noise_variance())

            am.parts_shape = self.dm.parts_shape
            am.normalize_parts = self.dm.normalize_parts
            algorithm = algorithm_cls(PartsAAMInterface, am, pdm, **kwargs)

            self._algorithms.append(algorithm)


# -----------------------------------------------------------------------------

class CombinedGlobalAAMFitter(AAMFitter):

    def _check_n_combined(self, n_combined):
        if n_combined is not None:
            if type(n_combined) is int or type(n_combined) is float:
                for cm in self.dm.combined_models:
                    cm.n_active_components = n_combined
            elif len(n_combined) == 1 and self.dm.n_levels > 1:
                for cm in self.dm.combined_models:
                    cm.n_active_components = n_combined[0]
            elif len(n_combined) == self.dm.n_levels:
                for cm, n in zip(self.dm.combined_models, n_combined):
                    cm.n_active_components = n
            else:
                raise ValueError('n_combined can be an integer or a float '
                                 'or None or a list containing 1 or {} of '
                                 'those'.format(self.dm.n_levels))

    def __init__(self, global_aam, algorithm_cls=CIC,
                 n_combined=None, **kwargs):

        super(CombinedGlobalAAMFitter, self).__init__()

        self.dm = global_aam
        self._algorithms = []
        self._check_n_combined(n_combined)

        for j, (am, sm, cm) in enumerate(zip(self.dm.appearance_models,
                                             self.dm.shape_models,
                                             self.dm.combined_models)):

            md_transform = OrthoMDTransform(
                sm, self.dm.transform,
                source=am.mean().landmarks['source'].lms,
                sigma2=am.noise_variance())

            algorithm = algorithm_cls(StandardAAMInterface, am,
                                      md_transform, cm, **kwargs)

            self._algorithms.append(algorithm)