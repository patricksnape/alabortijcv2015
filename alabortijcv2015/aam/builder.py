from __future__ import division
import abc
from copy import deepcopy
import collections
import itertools
import numpy as np

from menpo.shape import PointCloud
from menpo.transform import Scale, Translation, GeneralizedProcrustesAnalysis
from menpo.feature import no_op
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


def build_shape_model(shapes, max_components=None):
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
    aligned_shapes = (s.aligned_source() for s in gpa.transforms)
    # build shape model
    shape_model = PCAModel(aligned_shapes, n_samples=gpa.n_sources)
    if max_components is not None:
        # trim shape model if required
        shape_model.trim_components(max_components)

    return shape_model


def build_appearance_model(images, max_components=None):
    app_model = PCAModel(images)
    if max_components is not None:
        # trim shape model if required
        app_model.trim_components(max_components)
    return app_model


def feature_then_scale(image, feature, scales, sigmas):
    feature_image = feature(image)
    for s in scales:
        yield feature_image.rescale(s)


def scale_then_features(image, features, scales):
    if not isinstance(features, collections.Iterable):
        features = [features] * len(scales)
    for k, s in enumerate(scales):
        yield features[k](image.rescale(s))


def pyramid_then_features(image, features, n_levels=3, downscale=2,
                          gaussian=False):
    if not isinstance(features, collections.Iterable):
        features = [features] * n_levels
    if gaussian:
        image_pyramid = image.gaussian_pyramid(n_levels=n_levels,
                                               downscale=downscale)
    else:
        image_pyramid = image.pyramid(n_levels=n_levels, downscale=downscale)

    for k, level in enumerate(image_pyramid):
        yield features[k](level)


def feature_then_pyramid(image, feature, n_levels=3, downscale=2,
                         gaussian=False):
    feature_image = feature(image)

    if gaussian:
        fimage_pyramid = feature_image.gaussian_pyramid(n_levels=n_levels,
                                                        downscale=downscale)
    else:
        fimage_pyramid = feature_image.pyramid(n_levels=n_levels,
                                               downscale=downscale)

    for level in fimage_pyramid:
        yield level

def reference_frame_warp(image, reference_frame, transform, group=None,
                         label=None, ref_frame_group='source', boundary=3):
    t = transform.set_target(image.landmarks[group][label])
    warped_im = image.warp_to_mask(ref_im.mask, t)

    # Attach reference frame landmarks to image
    warped_im.landmarks[ref_frame_group] = ref_im.landmarks[ref_frame_group]
    return warped_im


def process_image_per_scale(image, reference_shape, diagonal=None,
                            normalization_sigma=None,
                            image_scaling_callable,
                            features=no_op,
                            image_warping_callable,
                            group=None, label=None):
    norm_image = normalize_image_scale(image, reference_shape,
                                       group=group, label=label,
                                       smoothing_sigma=normalization_sigma)

    scale_imgs = []
    for k, scale_image in enumerate(image_scaling_callable(norm_image, features)):
        ref_frame = build_reference_frame(reference_shape,
                                          boundary=boundary,
                                          group='source')
        scale_imgs.append(image_warping_callable(scale_image,
                                                 ref_frame))
    return scale_imgs

def batch(iterable, n):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def build_batch_aam(images, batch_size=None, reference_shape=None):
    if batch_size is None:
        batch_size = len(images)

    provided_ref_shape = reference_shape is not None
    # Break the images into a set of batches (this can handle infinite
    # generators and will only yield batch_size at a time)
    image_batches = batch(images, batch_size)

    first_image_batch = next(image_batches)

    app_models = []
    shape_models = []

    next_reference_shape = reference_shape if provided_ref_shape else None

    # Consume the rest of the batches
    for image_batch in image_batches:
        app_models, shape_models = build_appearance_and_shape_models(image_batch,
                                                                     reference_shape=next_reference_shape,
                                                                     app_models=app_models,
                                                                     shape_models=shape_models)
        if not provided_ref_shape:
            next_reference_shape = shape_models[-1].mean()

# compute transforms
ref_lmark_group = 'source'
ref_im =

def build_appearance_and_shape_models(images, group=None, label=None,
                                      diagonal=None, reference_shape=None,
                                      app_models=None, shape_models=None):
    exist_app_model = app_models is not None
    exist_shape_model = shape_models is not None

    if not exist_app_model:
        app_models = []
    else:
        app_models.reverse()
    if not exist_shape_model:
        shape_models = []
    else:
        shape_models.reverse()

    # Use the mean as the reference shape if none is passed. This is the most
    # mathematically correct behaviour if you have all the images in memory
    # at training time.
    if reference_shape is None:
        reference_shape = mean_pointcloud((i.landmarks[group][label]
                                           for i in images))

    # Scale the reference shape to a given diagonal. Since the reference frame
    # (the mask each image is warped into for correspondance), is directly
    # related to the reference shape size, this scaling will affect how many
    # pixels are in the top level of the appearance model.
    if diagonal is not None:
        reference_shape = scale_shape_to_diagonal(reference_shape, diagonal)

    # Process every image and every scale
    top_lvl_ref_frame =
    reference_frames = process_image_per_scale(top_lvl_ref_frame, )
    for image in images:
        # This processing applies a data pipeline, where each stage prepares
        # each image for building the PCA model at every scale specified.
        all_scaled_images = process_image_per_scale(image, reference_frame)
        # Build the appearance and shape model for each scale, separately
        for k, scale_images in enumerate(zip(*all_scaled_images)):
            if exist_app_model:
                app_models[k].increment(scale_images)
            else:
                app_models.append(build_appearance_model(scale_images))

            scale_shapes = (i.landmarks[group][label] for i in scale_images)
            if exist_shape_model:
                shape_models[k].increment(scale_images)
            else:
                shape_models.append(build_shape_model(scale_shapes))

    # Reverse the models as we want to proceed the from smallest to largest
    # at test time.
    shape_models.reverse()
    app_models.reverse()

    return app_models, shape_models


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
        return

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


