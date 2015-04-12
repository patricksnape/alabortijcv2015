from __future__ import division
import itertools
from matplotlib.collections import TriMesh
import numpy as np

from menpo.transform import (Translation, GeneralizedProcrustesAnalysis,
                             UniformScale)
from menpo.image import MaskedImage
from menpo.model import PCAModel
from menpo.shape import mean_pointcloud
from alabortijcv2015.utils import fsmooth
from menpo.visualize import print_dynamic, progress_bar_str


def batch(iterable, n):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def align_shapes(shapes):
    centered_shapes = [Translation(-s.centre()).apply(s) for s in shapes]
    if len(shapes) == 1:
        return centered_shapes[0]
    # align centralized shape using Procrustes Analysis
    gpa = GeneralizedProcrustesAnalysis(centered_shapes)
    return [s.aligned_source() for s in gpa.transforms]


def scale_shape_diagonal(ref_shape, diagonal):
    x, y = ref_shape.range()
    scale = diagonal / np.sqrt(x**2 + y**2)
    return UniformScale(scale, ref_shape.n_dims).apply(ref_shape)


def compute_reference_shape_from_shapes(shapes, verbose=True, diagonal=None):
    if verbose:
        print_dynamic('- Computing reference shape')
    ref_shape = mean_pointcloud(align_shapes(shapes))
    # fix the reference_shape's diagonal length if specified
    if diagonal:
        ref_shape = scale_shape_diagonal(ref_shape, diagonal)
    return ref_shape


def compute_reference_shape(images, group, label, verbose=True, diagonal=None):
    # the reference_shape is the mean shape of the images' landmarks
    shapes = [i.landmarks[group][label] for i in images]
    return compute_reference_shape_from_shapes(shapes, verbose=verbose,
                                               diagonal=diagonal)


def normalize_images(images, group, label, ref_shape, verbose=True, sigma=None):
    # normalize the scaling of all images wrt the reference_shape size
    norm_images = []
    for c, i in enumerate(images):
        if verbose:
            print_dynamic('- Normalizing images size: {}'.format(
                progress_bar_str((c + 1.) / len(images), show_bar=False)))
        i = i.rescale_to_reference_shape(ref_shape, group=group,
                                         label=label)
        if sigma:
            i.pixels = fsmooth(i.pixels, sigma)
        norm_images.append(i)
    return norm_images


def compute_features(images, verbose=True, features=None):
    feature_images = []
    for c, i in enumerate(images):
        if verbose:
            print_dynamic(
                '- Computing feature space: {}'.format(
                    progress_bar_str((c + 1.) / len(images), show_bar=False)))
        if features:
            i = features(i)
        feature_images.append(i)

    return feature_images


def scale_images(images, s, verbose=True):
    scaled_images = []
    for c, i in enumerate(images):
        if verbose:
            print_dynamic(
                '- Scaling features: {}'.format(
                    progress_bar_str((c + 1.) / len(images), show_bar=False)))
        scaled_images.append(i.rescale(s))
    return scaled_images


def build_shape_model(shapes, max_components):
    r"""
    Builds a shape model given a set of shapes.

    Parameters
    ----------
    shapes: list of :map:`PointCloud`
        The set of shapes from which to build the model.
    max_components: None or int or float
        Specifies the number of components of the trained shape model.
        If int, it specifies the exact number of components to be retained.
        If float, it specifies the percentage of variance to be retained.
        If None, all the available components are kept (100% of variance).

    Returns
    -------
    shape_model: :class:`menpo.model.pca`
        The PCA shape model.
    """
    # build shape model
    shape_model = PCAModel(align_shapes(shapes))
    if max_components is not None:
        # trim shape model if required
        shape_model.trim_components(max_components)

    return shape_model


def build_appearance_model(images, max_components):
    r"""
    Builds a shape model given a set of shapes.

    Parameters
    ----------
    images: list of :map:`PointCloud`
        The set of images from which to build the model.
    max_components: None or int or float
        Specifies the number of components of the trained shape model.
        If int, it specifies the exact number of components to be retained.
        If float, it specifies the percentage of variance to be retained.
        If None, all the available components are kept (100% of variance).

    Returns
    -------
    image_model: :class:`menpo.model.pca`
        The PCA image model.
    """
    appearance_model = PCAModel(images)
    # trim appearance model if required
    if max_components is not None:
        appearance_model.trim_components(max_components)
    return appearance_model


def build_reference_frame(landmarks, boundary=3, group='source',
                          point_in_pointcloud='pwa', batch_size=None):
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

    Returns
    -------
    reference_frame : :map:`Image`
        The reference frame.
    """
    reference_frame = _build_reference_frame(landmarks, boundary=boundary,
                                             group=group)

    reference_frame.constrain_mask_to_landmarks(
        group=group, point_in_pointcloud=point_in_pointcloud,
        batch_size=batch_size)

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
    reference_frame = MaskedImage.init_blank(resolution)
    reference_frame.landmarks[group] = landmarks

    return reference_frame
