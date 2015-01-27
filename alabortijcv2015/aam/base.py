from __future__ import division
import abc
import numpy as np

# from serializablecallable import SerializableCallable

from menpo.shape import TriMesh
from menpo.transform import Translation
from menpo.image import MaskedImage

from menpofit.transform import DifferentiableThinPlateSplines


# Abstract Interface for AAM Objects ------------------------------------------

class AAM(object):

    __metaclass__ = abc.ABCMeta

    @property
    def n_levels(self):
        """
        The number of scale levels of the AAM.

        :type: `int`
        """
        return len(self.scales)

    def instance(self, shape_weights=None, appearance_weights=None, level=-1):
        r"""
        Generates a novel AAM instance given a set of shape and appearance
        weights. If no weights are provided, the mean AAM instance is
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
            The novel AAM instance.
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
        Generates a novel random instance of the AAM.

        Parameters
        -----------
        level : `int`, optional
            The pyramidal level to be used.

        Returns
        -------
        image : :map:`Image`
            The novel AAM instance.
        """
        sm = self.shape_models[level]
        am = self.appearance_models[level]

        # TODO: this bit of logic should to be transferred down to PCAModel
        shape_weights = (np.random.randn(sm.n_active_components) *
                         sm.eigenvalues[:sm.n_active_components]**0.5)
        appearance_weights = (np.random.randn(am.n_active_components) *
                              am.eigenvalues[:am.n_active_components]**0.5)
        return self.instance(shape_weights=shape_weights,
                             appearance_weights=appearance_weights,
                             level=level)


# Concrete Implementations of AAM Objects -------------------------------------

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

        instance = appearance_instance.warp_to_mask(
            reference_frame.mask, transform)
        instance.landmarks = reference_frame.landmarks

        return instance

    def _build_reference_frame(self, reference_shape, landmarks):
        if type(landmarks) == TriMesh:
            trilist = landmarks.trilist
        else:
            trilist = None
        return build_reference_frame(reference_shape, trilist=trilist)


class CombinedGlobalAAM(GlobalAAM):

    def __init__(self, shape_models, appearance_models, combined_models,
                 reference_shape, transform, features, sigma, scales,
                 scale_shapes, scale_features):
        super(CombinedGlobalAAM, self).__init__(
            shape_models, appearance_models, reference_shape, transform,
            features, sigma, scales, scale_shapes, scale_features)
        self.combined_models = combined_models

    def instance(self, combined_weights=None, level=-1):
        r"""
        Generates a novel AAM instance given a set of combined
        shape and appearance weights. If no weights are provided, the mean
        AAM instance is returned.

        Parameters
        -----------
        weights : ``(n_weights,)`` `ndarray` or `float` list
            Weights of the combined model that will be used to create
            novel shape and appearance instance. If ``None``,
            ``(combined_weights = [0, 0, ..., 0])`` is used.

        level : `int`, optional
            The pyramidal level to be used.

        Returns
        -------
        image : :map:`Image`
            The novel AAM instance.
        """
        cm = self.combined_models[level]

        if combined_weights is None:
            combined_weights = [0]
        combined_instance = cm.instance(combined_weights)

        sm = self.shape_models[level]
        combined_vector = combined_instance.as_vector().copy()
        shape_weights = combined_vector[:sm.n_active_components]
        appearance_weights = combined_vector[sm.n_active_components:]

        return super(CombinedGlobalAAM, self).instance(
            shape_weights=shape_weights,
            appearance_weights=appearance_weights,
            level=level)

    def random_instance(self, level=-1):
        r"""
        Generates a novel random instance of the AAM.

        Parameters
        -----------
        level : `int`, optional
            The pyramidal level to be used.

        Returns
        -------
        image : :map:`Image`
            The novel AAM instance.
        """
        cm = self.combined_models[level]
        combined_weights = np.random.randn(cm.n_active_components)
        return self.instance(combined_weights=combined_weights, level=level)


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
        template = self.appearance_models[level].mean
        landmarks = template.landmarks['source'].lms

        reference_frame = self._build_reference_frame(
            shape_instance, landmarks)

        transform = self.transform(
            reference_frame.landmarks['source'].lms, landmarks)

        instance = appearance_instance.warp_to_mask(reference_frame.mask,
                                                transform)
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


# Functions -------------------------------------------------------------------

def build_reference_frame(landmarks, boundary=3, group='source',
                          trilist=None):
    r"""
    Builds a reference frame from a particular set of landmarks.

    Parameters
    ----------
    landmarks : :map:`PointCloud`
        The landmarks that will be used to build the reference frame.

    boundary : `int`, optional
        The number of pixels to be left as a safe margin on the boundaries
        of the reference frame (has potential effects on the gradient
        computation).

    group : `string`, optional
        Group that will be assigned to the provided set of landmarks on the
        reference frame.

    trilist : ``(t, 3)`` `ndarray`, optional
        Triangle list that will be used to build the reference frame.

        If ``None``, defaults to performing Delaunay triangulation on the
        points.

    Returns
    -------
    reference_frame : :map:`Image`
        The reference frame.
    """
    reference_frame = _build_reference_frame(landmarks, boundary=boundary,
                                             group=group)
    if trilist is not None:
        reference_frame.landmarks[group] = TriMesh(
            reference_frame.landmarks['source'].lms.points, trilist=trilist)

    # TODO: revise kwarg trilist in method constrain_mask_to_landmarks,
    # perhaps the trilist should be directly obtained from the group landmarks
    reference_frame.constrain_mask_to_landmarks(group=group, trilist=trilist)

    return reference_frame


def build_patch_reference_frame(landmarks, boundary=3, group='source',
                                patch_shape=(16, 16)):
    r"""
    Builds a reference frame from a particular set of landmarks.

    Parameters
    ----------
    landmarks : :map:`PointCloud`
        The landmarks that will be used to build the reference frame.

    boundary : `int`, optional
        The number of pixels to be left as a safe margin on the boundaries
        of the reference frame (has potential effects on the gradient
        computation).

    group : `string`, optional
        Group that will be assigned to the provided set of landmarks on the
        reference frame.

    patch_shape : tuple of ints, optional
        Tuple specifying the shape of the patches.

    Returns
    -------
    patch_based_reference_frame : :map:`Image`
        The patch based reference frame.
    """
    boundary = np.max(patch_shape) + boundary
    reference_frame = _build_reference_frame(landmarks, boundary=boundary,
                                             group=group)

    # mask reference frame
    reference_frame.build_mask_around_landmarks(patch_shape, group=group)

    return reference_frame


def _build_reference_frame(landmarks, boundary=3, group='source'):
    # translate landmarks to the origin
    minimum = landmarks.bounds(boundary=boundary)[0]
    landmarks = Translation(-minimum).apply(landmarks)

    resolution = landmarks.range(boundary=boundary)
    reference_frame = MaskedImage.blank(resolution)
    reference_frame.landmarks[group] = landmarks

    return reference_frame

