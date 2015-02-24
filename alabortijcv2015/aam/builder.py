from __future__ import division
import abc
from copy import deepcopy
import numpy as np

from menpo.shape import PointCloud
from menpo.transform import Scale, Translation, GeneralizedProcrustesAnalysis
from menpo.model import PCAModel
from menpo.shape import mean_pointcloud
from menpo.image import Image
from menpo.visualize import print_dynamic, progress_bar_str

from menpofit.transform import (DifferentiablePiecewiseAffine,
                                DifferentiableThinPlateSplines)

from alabortijcv2015.utils import fsmooth

from .base import build_reference_frame, build_patch_reference_frame


def scale_shape_to_diagonal(shape, diagonal):
    x, y = shape.range()
    scale = diagonal / np.sqrt(x**2 + y**2)
    return Scale(scale, shape.n_dims).apply(shape)

def normalize_image_scale(image, reference_shape,
                          group=None, label=None, smoothing_sigma=None):
    rescaled_image = image.rescale_to_reference_shape(reference_shape,
                                                      group=group, label=label)
    if smoothing_sigma:
        rescaled_image.pixels = fsmooth(rescaled_image.pixels, smoothing_sigma)
    return image

# Abstract Interface for AAM Builders -----------------------------------------
def _compute_reference_shape(images, group, label, diagonal, verbose):
    # the reference_shape is the mean shape of the images' landmarks
    if verbose:
        print_dynamic('- Computing reference shape')
    shapes = [i.landmarks[group][label] for i in images]
    ref_shape = mean_pointcloud(shapes)
    # fix the reference_shape's diagonal length if specified
    if diagonal:
        ref_shape = scale_shape_to_diagonal(ref_shape, diagonal)
    return ref_shape


def _normalize_images(images, group, label, ref_shape, sigma, verbose):
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


def _compute_features(features, images, level_str, verbose):
    feature_images = []
    for c, i in enumerate(images):
        if verbose:
            print_dynamic(
                '{}Computing feature space: {}'.format(
                    level_str, progress_bar_str((c + 1.) / len(images),
                                                show_bar=False)))
        if features:
            i = features(i)
        feature_images.append(i)

    return feature_images


def _scale_images(images, s, level_str, verbose):
    scaled_images = []
    for c, i in enumerate(images):
        if verbose:
            print_dynamic(
                '{}Scaling features: {}'.format(
                    level_str, progress_bar_str((c + 1.) / len(images),
                                                show_bar=False)))
        scaled_images.append(i.rescale(s))
    return scaled_images


def _build_shape_model(cls, shapes, max_components):
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

    # centralize shapes
    centered_shapes = [Translation(-s.centre()).apply(s) for s in shapes]
    # align centralized shape using Procrustes Analysis
    gpa = GeneralizedProcrustesAnalysis(centered_shapes)
    aligned_shapes = [s.aligned_source() for s in gpa.transforms]
    # build shape model
    shape_model = PCAModel(aligned_shapes)
    if max_components is not None:
        # trim shape model if required
        shape_model.trim_components(max_components)

    return shape_model


def feature_then_scale(image, feature, scales):
    feature_image = feature(image)
    for s in scales:
        yield feature_image.rescale(s)


def scale_then_feature(image, feature, scales):
    for s in scales:
        yield feature(image.rescale(s))


def build_appearance_model(images, reference_shape, diagonal=None,
                           normalization_sigma=None, scale_features=True,
                           group=None, label=None):
    if diagonal is not None:
        reference_shape = scale_shape_to_diagonal(reference_shape, diagonal)

    for image in images:
        norm_image = normalize_image_scale(image, reference_shape,
                                           group=group, label=label,
                                           smoothing_sigma=normalization_sigma)
        if scale_features:
            norm_image = features[(norm_image)
        for scale in downscale_image(image, smooth):


    if first_iteration:
        feature_image = features(image)
    elif scale_features:
        scaled_image = image.rescale(feature_image)
    else:
        scaled_image = image.rescale(image)
        feature_image = features(scaled_image)

#for image in images:
#   normalize image
#   feature?
#   for scale in scales:
#       features?
#       warp
#       PCA

class AAMBuilder(object):
    def build(self, images, group=None, label=None, verbose=False):
        # compute reference shape
        reference_shape = self._compute_reference_shape(images, group, label,
                                                        verbose)
        # normalize images
        images = self._normalize_images(images, group, label, reference_shape,
                                        verbose)

        # build models at each scale
        if verbose:
            print_dynamic('- Building models\n')
        shape_models = []
        appearance_models = []
        # for each pyramid level (high --> low)
        for j, s in enumerate(self.scales):
            if verbose:
                if len(self.scales) > 1:
                    level_str = '  - Level {}: '.format(j)
                else:
                    level_str = '  - '

            # obtain image representation
            if j == 0:
                # compute features at highest level
                feature_images = self._compute_features(images, level_str,
                                                        verbose)
                level_images = feature_images
            elif self.scale_features:
                # scale features at other levels
                level_images = self._scale_images(feature_images, s,
                                                  level_str, verbose)
            else:
                # scale images and compute features at other levels
                scaled_images = self._scale_images(images, s, level_str,
                                                   verbose)
                level_images = self._compute_features(scaled_images,
                                                      level_str, verbose)

            # extract potentially rescaled shapes ath highest level
            level_shapes = [i.landmarks[group][label]
                            for i in level_images]

            # obtain shape representation
            if j == 0 or self.scale_shapes:
                # obtain shape model
                if verbose:
                    print_dynamic('{}Building shape model'.format(level_str))
                shape_model = self._build_shape_model(
                    level_shapes, self.max_shape_components)
                # add shape model to the list
                shape_models.append(shape_model)
            else:
                # copy precious shape model and add it to the list
                shape_models.append(deepcopy(shape_model))

            # obtain warped images
            warped_images = self._warp_images(level_images, level_shapes,
                                              shape_model.mean(), level_str,
                                              verbose)

            # obtain appearance model
            if verbose:
                print_dynamic('{}Building appearance model'.format(level_str))
            appearance_model = PCAModel(warped_images)
            # trim appearance model if required
            if self.max_appearance_components is not None:
                appearance_model.trim_components(
                    self.max_appearance_components)
            # add appearance model to the list
            appearance_models.append(appearance_model)

            if verbose:
                print_dynamic('{}Done\n'.format(level_str))

        # reverse the list of shape and appearance models so that they are
        # ordered from lower to higher resolution
        shape_models.reverse()
        appearance_models.reverse()
        self.scales.reverse()

        aam = self._build_aam(shape_models, appearance_models, reference_shape)

        return aam


    @abc.abstractmethod
    def _build_aam(self, shape_models, appearance_models, reference_shape):
        pass


# Concrete Implementations of AAM Builders ------------------------------------

class GlobalAAMBuilder(AAMBuilder):

    def __init__(self, features=None, transform=DifferentiablePiecewiseAffine,
                 trilist=None, diagonal=None, sigma=None, scales=(1, .5),
                 scale_shapes=True, scale_features=True,
                 max_shape_components=None, max_appearance_components=None,
                 boundary=3):

        self.features = features
        self.transform = transform
        self.trilist = trilist
        self.diagonal = diagonal
        self.sigma = sigma
        self.scales = list(scales)
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features
        self.max_shape_components = max_shape_components
        self.max_appearance_components = max_appearance_components
        self.boundary = boundary

    def _build_reference_frame(self, mean_shape):
        return build_reference_frame(mean_shape, boundary=self.boundary,
                                     trilist=self.trilist)

    def _warp_images(self, images, shapes, ref_shape, level_str, verbose):
        # compute transforms
        ref_frame = self._build_reference_frame(ref_shape)
        # warp images to reference frame
        warped_images = []
        for c, (i, s) in enumerate(zip(images, shapes)):
            if verbose:
                print_dynamic('{}Warping images - {}'.format(
                    level_str,
                    progress_bar_str(float(c + 1) / len(images),
                                     show_bar=False)))
            # compute transforms
            t = self.transform(ref_frame.landmarks['source'].lms, s)
            # warp images
            warped_i = i.warp_to_mask(ref_frame.mask, t)
            # attach reference frame landmarks to images
            warped_i.landmarks['source'] = ref_frame.landmarks['source']
            warped_images.append(warped_i)
        return warped_images

    def _build_aam(self, shape_models, appearance_models, reference_shape):
        return GlobalAAM(shape_models, appearance_models, reference_shape,
                         self.transform, self.features, self.sigma,
                         self.scales, self.scale_shapes, self.scale_features)


class PatchAAMBuilder(AAMBuilder):

    def __init__(self, patch_shape=(16, 16), features=None,
                 diagonal=None, sigma=None, scales=(1, .5), scale_shapes=True,
                 scale_features=True, max_shape_components=None,
                 max_appearance_components=None, boundary=3):

        self.patch_shape = patch_shape
        self.features = features
        self.transform = DifferentiableThinPlateSplines
        self.diagonal = diagonal
        self.sigma = sigma
        self.scales = list(scales)
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features
        self.max_shape_components = max_shape_components
        self.max_appearance_components = max_appearance_components
        self.boundary = boundary

    def _build_reference_frame(self, mean_shape):
        return build_patch_reference_frame(mean_shape, boundary=self.boundary,
                                           patch_shape=self.patch_shape)

    def _warp_images(self, images, shapes, ref_shape, level_str, verbose):
        # compute transforms
        ref_frame = self._build_reference_frame(ref_shape)
        # warp images to reference frame
        warped_images = []
        for c, (i, s) in enumerate(zip(images, shapes)):
            if verbose:
                print_dynamic('{}Warping images - {}'.format(
                    level_str,
                    progress_bar_str(float(c + 1) / len(images),
                                     show_bar=False)))
            # compute transforms
            t = self.transform(ref_frame.landmarks['source'].lms, s)
            # warp images
            warped_i = i.warp_to_mask(ref_frame.mask, t)
            # attach reference frame landmarks to images
            warped_i.landmarks['source'] = ref_frame.landmarks['source']
            warped_images.append(warped_i)
        return warped_images

    def _build_aam(self, shape_models, appearance_models, reference_shape):
        return PatchAAM(shape_models, appearance_models, reference_shape,
                        self.patch_shape, self.features, self.sigma,
                        self.scales, self.scale_shapes, self.scale_features)


class LinearGlobalAAMBuilder(GlobalAAMBuilder):

    def __init__(self, features=None, transform=DifferentiablePiecewiseAffine,
                 trilist=None, diagonal=None, sigma=None, scales=(1, .5),
                 scale_shapes=False, scale_features=True,
                 max_shape_components=None, max_appearance_components=None,
                 boundary=3):

        self.features = features
        self.transform = transform
        self.trilist = trilist
        self.diagonal = diagonal
        self.sigma = sigma
        self.scales = list(scales)
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features
        self.max_shape_components = max_shape_components
        self.max_appearance_components = max_appearance_components
        self.boundary = boundary

    def _build_shape_model(self, shapes, max_components):

        shape_model = GlobalAAMBuilder._build_shape_model(
            shapes, max_components)

        self.n_landmarks = shape_model.mean().n_points

        self.reference_frame = self._build_reference_frame(shape_model.mean())

        # compute non-linear transforms
        transforms = (
            [self.transform(self.reference_frame.landmarks['source'].lms, s)
             for s in shapes])

        # build dense shapes
        dense_shapes = []
        for (t, s) in zip(transforms, shapes):
            warped_points = t.apply(self.reference_frame.mask.true_indices())
            dense_shape = PointCloud(np.vstack((s.points, warped_points)))
            dense_shapes.append(dense_shape)

        # build dense shape mode3l
        shape_model = GlobalAAMBuilder._build_shape_model(
            dense_shapes, max_components)

        return shape_model

    def _warp_images(self, images, shapes, reference_shape, level_str, verbose):

        # warp images to reference frame
        warped_images = []
        for c, (i, s) in enumerate(zip(images, shapes)):
            if verbose:
                print_dynamic('{}Warping images - {}'.format(
                    level_str,
                    progress_bar_str(float(c + 1) / len(images),
                                     show_bar=False)))
            # compute transforms
            t = self.transform(PointCloud(self.reference_frame.landmarks[
                'source'].lms.points[:self.n_landmarks]), s)
            # warp images
            warped_i = i.warp_to_mask(self.reference_frame.mask, t)
            # attach reference frame landmarks to images
            warped_i.landmarks['source'] = \
                self.reference_frame.landmarks['source']
            warped_images.append(warped_i)
        return warped_images

    def _build_aam(self, shape_models, appearance_models, reference_shape):
        return LinearGlobalAAM(shape_models, appearance_models,
                               reference_shape, self.transform,
                               self.features, self.sigma, self.scales,
                               self.scale_shapes, self.scale_features,
                               self.n_landmarks)


class LinearPatchAAMBuilder(PatchAAMBuilder):

    def __init__(self, patch_shape=(16, 16), features=None,
                 diagonal=None, sigma=None, scales=(1, .5), scale_shapes=False,
                 scale_features=True, max_shape_components=None,
                 max_appearance_components=None, boundary=3):

        self.patch_shape = patch_shape
        self.features = features
        self.transform = DifferentiableThinPlateSplines
        self.diagonal = diagonal
        self.sigma = sigma
        self.scales = list(scales)
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features
        self.max_shape_components = max_shape_components
        self.max_appearance_components = max_appearance_components
        self.boundary = boundary

    def _build_shape_model(self, shapes, max_components):

        shape_model = GlobalAAMBuilder._build_shape_model(
            shapes, max_components)

        self.n_landmarks = shape_model.mean().n_points

        self.reference_frame = self._build_reference_frame(shape_model.mean())

        # compute non-linear transforms
        transforms = (
            [self.transform(self.reference_frame.landmarks['source'].lms, s)
             for s in shapes])

        # build dense shapes
        dense_shapes = []
        for (t, s) in zip(transforms, shapes):
            warped_points = t.apply(self.reference_frame.mask.true_indices())
            dense_shape = PointCloud(np.vstack((s.points, warped_points)))
            dense_shapes.append(dense_shape)

        # build dense shape mode3l
        shape_model = GlobalAAMBuilder._build_shape_model(
            dense_shapes, max_components)

        return shape_model

    def _warp_images(self, images, shapes, reference_shape, level_str,
                     verbose):

        # warp images to reference frame
        warped_images = []
        for c, (i, s) in enumerate(zip(images, shapes)):
            if verbose:
                print_dynamic('{}Warping images - {}'.format(
                    level_str,
                    progress_bar_str(float(c + 1) / len(images),
                                     show_bar=False)))
            # compute transforms
            t = self.transform(PointCloud(self.reference_frame.landmarks[
                'source'].lms.points[:self.n_landmarks]), s)
            # warp images
            warped_i = i.warp_to_mask(self.reference_frame.mask, t)
            # attach reference frame landmarks to images
            warped_i.landmarks['source'] = \
                self.reference_frame.landmarks['source']
            warped_images.append(warped_i)
        return warped_images

    def _build_aam(self, shape_models, appearance_models, reference_shape):
        return LinearPatchAAM(shape_models, appearance_models,
                              reference_shape, self.patch_shape,
                              self.features, self.sigma, self.scales,
                              self.scale_shapes, self.scale_features,
                              self.n_landmarks)


class PartsAAMBuilder(AAMBuilder):

    def __init__(self, parts_shape=(16, 16), features=None,
                 normalize_parts=False, diagonal=None, sigma=None,
                 scales=(1, .5), scale_shapes=False, scale_features=True,
                 max_shape_components=None, max_appearance_components=None):

        self.parts_shape = parts_shape
        self.features = features
        self.normalize_parts = normalize_parts
        self.diagonal = diagonal
        self.sigma = sigma
        self.scales = list(scales)
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features
        self.max_shape_components = max_shape_components
        self.max_appearance_components = max_appearance_components

    def _warp_images(self, images, shapes, _, level_str, verbose):

        # extract parts
        parts_images = []
        for c, (i, s) in enumerate(zip(images, shapes)):
            if verbose:
                print_dynamic('{}Warping images - {}'.format(
                    level_str,
                    progress_bar_str(float(c + 1) / len(images),
                                     show_bar=False)))
            parts_image = Image(i.extract_patches(
                s, patch_size=self.parts_shape, as_single_array=True))
            parts_images.append(parts_image)

        return parts_images

    def _build_aam(self, shape_models, appearance_models, reference_shape):
        return PartsAAM(shape_models, appearance_models, reference_shape,
                        self.parts_shape, self.features,
                        self.normalize_parts, self.sigma, self.scales,
                        self.scale_shapes, self.scale_features)


from .base import (GlobalAAM, PatchAAM,
                   LinearGlobalAAM, LinearPatchAAM,
                   PartsAAM)


