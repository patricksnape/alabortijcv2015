from __future__ import division
import abc
import numpy as np

# from serializablecallable import SerializableCallable

from menpo.shape import TriMesh
from menpo.transform import Translation
from menpo.image import MaskedImage
from alabortijcv2015.builder import build_reference_frame

from menpofit.transform import DifferentiableThinPlateSplines


# Abstract Interface for ATM Objects ------------------------------------------

class AAM(object):

    __metaclass__ = abc.ABCMeta

    # def __getstate__(self):
    #     import menpofast.feature as menpofast_feature
    #     d = self.__dict__.copy()
    #
    #     features = d.pop('features')
    #     d['features'] = SerializableCallable(features, [menpofast_feature])
    #
    #     return d
    #
    # def __setstate__(self, state):
    #     state['features'] = state['features'].callable
    #     self.__dict__.update(state)

    @property
    def n_levels(self):
        """
        The number of scale levels of the ATM.

        :type: `int`
        """
        return len(self.scales)

    def instance(self, shape_weights=None, appearance_weights=None, level=-1):
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

        appearance_weights : ``(n_weights,)`` `ndarray` or `float` list
            Weights of the appearance model that will be used to create
            a novel appearance instance. If ``None``, the mean appearance
            ``(appearance_weights = [0, 0, ..., 0])`` is used.

        level : `int`, optional
            The pyramidal level to be used.

        Returns
        -------
        image : :map:`Image`
            The novel ATM instance.
        """
        sm = self.shape_models[level]
        am = self.appearance_models[level]

        # TODO: this bit of logic should to be transferred down to PCAModel
        if shape_weights is None:
            shape_weights = [0]
        if appearance_weights is None:
            appearance_weights = [0]
        n_shape_weights = len(shape_weights)
        shape_weights *= sm.eigenvalues[:n_shape_weights] ** 0.5
        shape_instance = sm.instance(shape_weights)
        n_appearance_weights = len(appearance_weights)
        appearance_weights *= am.eigenvalues[:n_appearance_weights] ** 0.5
        appearance_instance = am.instance(appearance_weights)

        return self._instance(level, shape_instance, appearance_instance)

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
        am = self.appearance_models[level]

        # TODO: this bit of logic should to be transferred down to PCAModel
        shape_weights = (np.random.randn(sm.n_active_components) *
                         sm.eigenvalues[:sm.n_active_components]**0.5)
        shape_instance = sm.instance(shape_weights)
        appearance_weights = (np.random.randn(am.n_active_components) *
                              am.eigenvalues[:am.n_active_components]**0.5)
        appearance_instance = am.instance(appearance_weights)

        return self._instance(level, shape_instance, appearance_instance)


# Concrete Implementations of ATM Objects -------------------------------------

class GlobalAAM(AAM):

    def __init__(self, shape_models, appearance_models, reference_shape,
                 transform, features, sigma, scales, scale_shapes,
                 scale_features):
        self.shape_models = shape_models
        self.appearance_models = appearance_models
        self.transform = transform
        self.features = features
        self.reference_shape = reference_shape
        self.sigma = sigma
        self.scales = scales
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features

    def _instance(self, level, shape_instance, appearance_instance):
        template = self.appearance_models[level].mean()
        landmarks = template.landmarks['source'].lms

        reference_frame = self._build_reference_frame(
            shape_instance, landmarks)

        transform = self.transform(
            reference_frame.landmarks['source'].lms, landmarks)

        instance = appearance_instance.as_unmasked().warp_to_mask(
            reference_frame.mask, transform)
        instance.landmarks = reference_frame.landmarks

        return instance

    def _build_reference_frame(self, reference_shape, landmarks):
        if type(landmarks) == TriMesh:
            trilist = landmarks.trilist
        else:
            trilist = None
        return build_reference_frame(reference_shape, trilist=trilist)


class PatchAAM(AAM):

    def __init__(self, shape_models, appearance_models, reference_shape,
                 patch_shape, features, sigma, scales, scale_shapes,
                 scale_features):

        self.shape_models = shape_models
        self.appearance_models = appearance_models
        self.transform = DifferentiableThinPlateSplines
        self.patch_shape = patch_shape
        self.features = features
        self.reference_shape = reference_shape
        self.sigma = sigma
        self.scales = scales
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features

    def _instance(self, level, shape_instance, appearance_instance):
        template = self.appearance_models[level].mean()
        landmarks = template.landmarks['source'].lms

        reference_frame = self._build_reference_frame(
            shape_instance, landmarks)

        transform = self.transform(
            reference_frame.landmarks['source'].lms, landmarks)

        instance = appearance_instance.as_unmasked().warp_to_mask(
            reference_frame.mask, transform)
        instance.landmarks = reference_frame.landmarks

        return instance

    def _build_reference_frame(self, reference_shape, _):
        return build_patch_reference_frame(
            reference_shape, patch_shape=self.patch_shape)


class LinearGlobalAAM(AAM):

    def __init__(self, shape_models, appearance_models, reference_shape,
                 transform, features, sigma, scales, scale_shapes,
                 scale_features, n_landmarks):

        self.shape_models = shape_models
        self.appearance_models = appearance_models
        self.transform = transform
        self.features = features
        self.reference_shape = reference_shape
        self.sigma = sigma
        self.scales = scales
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features
        self.n_landmarks = n_landmarks

    def _instance(self, level, shape_instance, appearance_instance):
        template = self.appearance_models[level].mean()
        landmarks = template.landmarks['source'].lms

        reference_frame = self._build_reference_frame(
            shape_instance, landmarks)

        transform = self.transform(
            reference_frame.landmarks['source'].lms, landmarks)

        instance = appearance_instance.as_unmasked().warp_to_mask(
            reference_frame.mask, transform, batch_size=3000)
        instance.landmarks = reference_frame.landmarks

        return instance

    def _build_reference_frame(self, reference_shape, landmarks):
        if type(landmarks) == TriMesh:
            trilist = landmarks.trilist
        else:
            trilist = None
        return build_reference_frame(reference_shape, trilist=trilist)


class LinearPatchAAM(AAM):

    def __init__(self, shape_models, appearance_models, reference_shape,
                 patch_shape, features, sigma, scales, scale_shapes,
                 scale_features, n_landmarks):

        self.shape_models = shape_models
        self.appearance_models = appearance_models
        self.transform = DifferentiableThinPlateSplines
        self.patch_shape = patch_shape
        self.features = features
        self.reference_shape = reference_shape
        self.sigma = sigma
        self.scales = scales
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features
        self.n_landmarks = n_landmarks


class PartsAAM(AAM):

    def __init__(self, shape_models, appearance_models, reference_shape,
                 parts_shape, features, normalize_parts, sigma, scales,
                 scale_shapes, scale_features):

        self.shape_models = shape_models
        self.appearance_models = appearance_models
        self.parts_shape = parts_shape
        self.features = features
        self.normalize_parts = normalize_parts
        self.sigma = sigma
        self.reference_shape = reference_shape
        self.scales = scales
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features
