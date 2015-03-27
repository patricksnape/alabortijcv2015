from __future__ import division
import numpy as np

from menpo.shape import TriMesh
from alabortijcv2015.builder import build_reference_frame


# Abstract Interface for ATM Objects ------------------------------------------

class ATM(object):

    @property
    def n_levels(self):
        """
        The number of scale levels of the ATM.

        :type: `int`
        """
        return len(self.scales)

    def instance(self, shape_weights=None, level=-1):
        r"""
        Generates a novel ATM instance given a set of shape and appearance
        weights. If no weights are provided, the mean ATM instance is
        returned.

        Parameters
        -----------
        shape_weights : ``(n_weights,)`` `ndarray` or `float` list
            Weights of the shape model that will be used to create
            a novel shape instance. If ``None``, the mean shape
            ``(shape_weights = [0, 0, ..., 0])`` is used.
        level : `int`, optional
            The pyramidal level to be used.

        Returns
        -------
        image : :map:`Image`
            The novel ATM instance.
        """
        sm = self.shape_models[level]

        if shape_weights is None:
            shape_weights = [0]
        n_shape_weights = len(shape_weights)
        shape_weights *= sm.eigenvalues[:n_shape_weights] ** 0.5
        shape_instance = sm.instance(shape_weights)

        return self._instance(level, shape_instance, self.templates[level])

    def random_instance(self, level=-1):
        r"""
        Generates a novel random instance of the ATM.

        Parameters
        -----------
        level : `int`, optional
            The pyramidal level to be used.

        Returns
        -------
        image : :map:`Image`
            The novel ATM instance.
        """
        sm = self.shape_models[level]

        # TODO: this bit of logic should to be transferred down to PCAModel
        shape_weights = (np.random.randn(sm.n_active_components) *
                         sm.eigenvalues[:sm.n_active_components]**0.5)
        shape_instance = sm.instance(shape_weights)

        return self._instance(level, shape_instance, self.templates[level])


# Concrete Implementations of ATM Objects -------------------------------------

class GlobalATM(ATM):

    def __init__(self, shape_models, templates, reference_shape,
                 transform, features, sigma, scales, scale_shapes,
                 scale_features):
        self.shape_models = shape_models
        self.templates = templates
        self.transform = transform
        self.features = features
        self.reference_shape = reference_shape
        self.sigma = sigma
        self.scales = scales
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features

    def _instance(self, level, shape_instance, template):
        landmarks = template.landmarks['source'].lms

        reference_frame = self._build_reference_frame(
            shape_instance, landmarks)

        transform = self.transform(
            reference_frame.landmarks['source'].lms, landmarks)

        instance = template.as_unmasked().warp_to_mask(reference_frame.mask,
                                                       transform)
        instance.landmarks = reference_frame.landmarks

        return instance

    def _build_reference_frame(self, reference_shape, landmarks):
        if type(landmarks) == TriMesh:
            trilist = landmarks.trilist
        else:
            trilist = None
        return build_reference_frame(reference_shape, trilist=trilist)


class LinearGlobalATM(ATM):

    def __init__(self, shape_models, templates, reference_shape,
                 transform, features, sigma, scales, scale_shapes,
                 scale_features, n_landmarks):

        self.shape_models = shape_models
        self.templates = templates
        self.transform = transform
        self.features = features
        self.reference_shape = reference_shape
        self.sigma = sigma
        self.scales = scales
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features
        self.n_landmarks = n_landmarks

    def _instance(self, level, shape_instance, template):
        landmarks = template.landmarks['source'].lms

        reference_frame = self._build_reference_frame(
            shape_instance, landmarks)

        transform = self.transform(
            reference_frame.landmarks['source'].lms, landmarks)

        instance = template.as_unmasked().warp_to_mask(
            reference_frame.mask, transform, batch_size=3000)
        instance.landmarks = reference_frame.landmarks

        return instance

    def _build_reference_frame(self, reference_shape, landmarks):
        if type(landmarks) == TriMesh:
            trilist = landmarks.trilist
        else:
            trilist = None
        return build_reference_frame(reference_shape, trilist=trilist)
