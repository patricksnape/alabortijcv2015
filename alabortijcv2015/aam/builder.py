from __future__ import division
import abc
import numpy as np

from menpo.shape import PointCloud
from menpo.transform import Translation, UniformScale, AlignmentSimilarity
from menpo.image import Image
from menpo.feature import no_op
from menpo.visualize import print_dynamic, progress_bar_str

from menpofit.transform import (DifferentiablePiecewiseAffine,
                                DifferentiableThinPlateSplines)

from ..builder import batch, scale_shape_diagonal, compute_reference_shape, \
    normalize_images, compute_features, scale_images, build_appearance_model, \
    build_shape_model, build_reference_frame, build_patch_reference_frame

# Abstract Interface for ATM Builders -----------------------------------------


class AAMBuilder(object):

    def increment_aam(self, aam, images, group=None,
                      label=None, app_forgetting_factor=1.0,
                      shape_forgetting_factor=1.0, verbose=False):
        shape_models = aam.shape_models
        appearance_models = aam.appearance_models
        reference_shape = aam.reference_shape

        if verbose:
            print('Incrementing ATM with batch {} images'.format(len(images)))

        self._increment_models(
            images, shape_models, appearance_models,
            reference_shape, group=group,
            label=label, app_forgetting_factor=app_forgetting_factor,
            shape_forgetting_factor=shape_forgetting_factor,
            verbose=verbose)

        return aam

    def build_batched(self, images, reference_shape, batch_size=100, group=None,
                      label=None, app_forgetting_factor=1.0,
                      shape_forgetting_factor=1.0, verbose=False):
        if self.diagonal:
            reference_shape = scale_shape_diagonal(reference_shape,
                                                   self.diagonal)

        # Create a generator of fixed sized batches. Will still work even
        # on an infinite list.
        image_batches = batch(images, batch_size)

        if verbose:
            print('Building ATM using batches of size {}'.format(batch_size))

        shape_models = []
        appearance_models = []
        for k, image_batch in enumerate(image_batches):
            curr_batch_size = len(image_batch)

            if k == 0:
                if verbose:
                    print('Creating batch 1 - initial models '
                          'with {} images'.format(curr_batch_size))
                data_prepare = self._prepare_data(image_batch, reference_shape,
                                                  group=group, label=label,
                                                  verbose=verbose)
                for j, (warped_imgs, scaled_shapes) in enumerate(data_prepare):
                    s_app_model, s_shape_model = self._build_models(
                        warped_imgs, scaled_shapes, verbose=verbose)
                    appearance_models.append(s_app_model)
                    shape_models.append(s_shape_model)
                    if verbose:
                        print_dynamic(' - Level {} - Done\n'.format(j))
            else:
                if verbose:
                    print('Increment with batch {} - '
                          '{} images'.format(k + 1, curr_batch_size))
                self._increment_models(
                    image_batch, shape_models, appearance_models,
                    reference_shape, group=group,
                    label=label, app_forgetting_factor=app_forgetting_factor,
                    shape_forgetting_factor=shape_forgetting_factor,
                    verbose=verbose)
        # reverse the list of shape and appearance models so that they are
        # ordered from lower to higher resolution
        shape_models.reverse()
        appearance_models.reverse()

        return self._build_aam(shape_models, appearance_models, reference_shape)

    def build(self, images, group=None, label=None,
              reference_shape=None,
              verbose=False):
        # compute reference shape
        if reference_shape is None:
            reference_shape = compute_reference_shape(images, group, label,
                                                      diagonal=self.diagonal,
                                                      verbose=verbose)

        shape_models = []
        appearance_models = []

        if verbose:
            print_dynamic('Building models\n')

        data_prepare = self._prepare_data(images, reference_shape,
                                          group=group, label=label,
                                          verbose=verbose)
        for k, (warped_images, scaled_shapes) in enumerate(data_prepare):
            s_app_model, s_shape_model = self._build_models(warped_images,
                                                            scaled_shapes,
                                                            verbose=verbose)
            appearance_models.append(s_app_model)
            shape_models.append(s_shape_model)
            if verbose:
                print_dynamic(' - Level {} - Done\n'.format(k))

        # reverse the list of shape and appearance models so that they are
        # ordered from lower to higher resolution
        shape_models.reverse()
        appearance_models.reverse()

        return self._build_aam(shape_models, appearance_models, reference_shape)

    def _increment_models(self, image_batch, shape_models, appearance_models,
                          reference_shape, group=None,
                          label=None, app_forgetting_factor=1.0,
                          shape_forgetting_factor=1.0, verbose=False):
        curr_batch_size = len(image_batch)

        data_prepare = self._prepare_data(
            image_batch, reference_shape, group=group, label=label,
            verbose=verbose)

        for j, (warped_imgs, scaled_shapes) in enumerate(data_prepare):
            if verbose:
                print_dynamic(' - Incrementing Appearance Model with '
                              '{} images.'.format(len(warped_imgs)))
            appearance_models[j].increment(
                warped_imgs, forgetting_factor=app_forgetting_factor,
                verbose=False)
            if self.max_appearance_components is not None:
                appearance_models[j].trim_components(
                    self.max_appearance_components)

            if verbose:
                print_dynamic(' - Incrementing Shape Model with {} '
                              'shapes.'.format(curr_batch_size))
            # Before incrementing the shape model, we need to remove
            # similarity differences between the new shapes and the
            # model
            aligned_shapes = [
                AlignmentSimilarity(s, shape_models[j].mean()).apply(s)
                for s in scaled_shapes
            ]
            shape_models[j].increment(
                aligned_shapes,
                forgetting_factor=shape_forgetting_factor,
                verbose=False)
            if self.max_shape_components is not None:
                shape_models[j].trim_components(
                    self.max_shape_components)
            if verbose:
                print_dynamic(' - Level {} - Done\n'.format(j))

    def _prepare_data(self, images, reference_shape, group=None, label=None,
                      verbose=False):
        # normalize images
        images = normalize_images(images, group, label, reference_shape,
                                  verbose, sigma=self.sigma)
        original_shapes = [i.landmarks[group][label] for i in images]

        # build models at each scale
        # for each pyramid level (high --> low)
        for j, s in enumerate(self.scales):
            # obtain image representation
            if j == 0:
                # compute features at highest level
                feature_images = compute_features(images,
                                                  verbose=verbose,
                                                  features=self.features)
                level_images = feature_images
            elif self.scale_features:
                # scale features at other levels
                level_images = scale_images(feature_images, s,
                                            verbose=verbose)
            else:
                # scale images and compute features at other levels
                scaled_images = scale_images(images, s, verbose=verbose)
                level_images = compute_features(scaled_images, verbose=verbose,
                                                features=self.features)

            # Rescaled shapes
            level_shapes = [i.landmarks[group][label] for i in level_images]

            # obtain warped images
            ref_frame_scale = UniformScale(s, reference_shape.n_dims)
            scaled_ref_frame = ref_frame_scale.apply(reference_shape)
            self.current_scale = s
            warped_images = self._warp_images(level_images, level_shapes,
                                              scaled_ref_frame,
                                              verbose=verbose)

            if self.scale_shapes:
                shape_model_shapes = level_shapes
            else:
                shape_model_shapes = original_shapes

            yield warped_images, shape_model_shapes

    def _build_models(self, warped_images, scaled_shapes, verbose=False):
        if verbose:
            print_dynamic(' - Building shape model')
        shape_model = build_shape_model(
            scaled_shapes, self.max_shape_components)

        # obtain appearance model
        if verbose:
            print_dynamic(' - Building appearance model')
        appearance_model = build_appearance_model(
            warped_images, self.max_appearance_components)

        return appearance_model, shape_model

    @abc.abstractmethod
    def _build_aam(self, shape_models, appearance_models, reference_shape):
        pass


# Concrete Implementations of ATM Builders ------------------------------------

class GlobalAAMBuilder(AAMBuilder):

    def __init__(self, features=no_op, transform=DifferentiablePiecewiseAffine,
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

    def _warp_images(self, images, shapes, ref_shape, verbose=True):
        # compute transforms
        ref_frame = self._build_reference_frame(ref_shape)
        # warp images to reference frame
        warped_images = []
        t = self.transform(ref_frame.landmarks['source'].lms,
                           ref_frame.landmarks['source'].lms)
        for c, (i, s) in enumerate(zip(images, shapes)):
            if verbose:
                print_dynamic(' - Warping images - {}'.format(
                    progress_bar_str(float(c + 1) / len(images),
                                     show_bar=False)))
            # compute transforms
            t.set_target(s)
            # warp images
            warped_i = i.warp_to_mask(ref_frame.mask, t)
            # attach reference frame landmarks to images
            warped_i.landmarks['source'] = ref_frame.landmarks['source']
            warped_images.append(warped_i)
        return warped_images

    def _build_aam(self, shape_models, appearance_models, reference_shape):
        return GlobalAAM(shape_models, appearance_models, reference_shape,
                         self.transform, self.features, self.sigma,
                         list(reversed(self.scales)),
                         self.scale_shapes, self.scale_features)


class PatchAAMBuilder(AAMBuilder):

    def __init__(self, patch_shape=(16, 16), features=no_op,
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

    def _warp_images(self, images, shapes, ref_shape, verbose=True):
        # compute transforms
        ref_frame = self._build_reference_frame(ref_shape)
        # warp images to reference frame
        warped_images = []
        t = self.transform(ref_frame.landmarks['source'].lms,
                           ref_frame.landmarks['source'].lms)
        for c, (i, s) in enumerate(zip(images, shapes)):
            if verbose:
                print_dynamic(' - Warping images - {}'.format(
                              progress_bar_str(float(c + 1) / len(images),
                                               show_bar=False)))
            # compute transforms
            t.set_target(s)
            # warp images
            warped_i = i.warp_to_mask(ref_frame.mask, t)
            # attach reference frame landmarks to images
            warped_i.landmarks['source'] = ref_frame.landmarks['source']
            warped_images.append(warped_i)
        return warped_images

    def _build_aam(self, shape_models, appearance_models, reference_shape):
        return PatchAAM(shape_models, appearance_models, reference_shape,
                        self.patch_shape, self.features, self.sigma,
                        list(reversed(self.scales)), self.scale_shapes,
                        self.scale_features)


class LinearGlobalAAMBuilder(GlobalAAMBuilder):

    def __init__(self, features=no_op, transform=DifferentiablePiecewiseAffine,
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

        return shape_model

    def _warp_images(self, images, _, ref_shape, verbose=True):
        from ..builder import _build_reference_frame

        ref_frame = GlobalAAMBuilder._build_reference_frame(self, ref_shape)
        dense_landmarks = self.reference_frame.landmarks['source'].lms.copy()
        trans = Translation(-dense_landmarks.centre())
        trans = trans.compose_before(UniformScale(self.current_scale,
                                                  ref_shape.n_dims))
        scaled_source = _build_reference_frame(
            trans.apply(dense_landmarks),
            boundary=self.boundary).landmarks['source'].lms

        # warp images to reference frame
        warped_images = []
        t = self.transform(ref_frame.landmarks['source'].lms,
                           ref_frame.landmarks['source'].lms)
        for c, i in enumerate(images):
            if verbose:
                print_dynamic(' - Warping images - {}'.format(
                    progress_bar_str(float(c + 1) / len(images),
                                     show_bar=False)))
            # compute transforms
            t.set_target(i.landmarks[None].lms)
            # warp images
            warped_i = i.warp_to_mask(ref_frame.mask, t)
            # attach reference frame landmarks to images
            warped_i.landmarks['source'] = scaled_source
            warped_images.append(warped_i)
        return warped_images

    def _build_aam(self, shape_models, appearance_models, reference_shape):
        return LinearGlobalAAM(shape_models, appearance_models,
                               self.reference_frame.landmarks['source'].lms,
                               self.transform,
                               self.features, self.sigma,
                               list(reversed(self.scales)),
                               self.scale_shapes, self.scale_features,
                               0)


class LinearPatchAAMBuilder(PatchAAMBuilder):

    def __init__(self, patch_shape=(16, 16), features=no_op,
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

    def _warp_images(self, images, shapes, reference_shape, verbose=True):

        # warp images to reference frame
        warped_images = []
        for c, (i, s) in enumerate(zip(images, shapes)):
            if verbose:
                print_dynamic(' - Warping images - {}'.format(
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
                              self.features, self.sigma,
                              list(reversed(self.scales)),
                              self.scale_shapes, self.scale_features,
                              self.n_landmarks)


class PartsAAMBuilder(AAMBuilder):

    def __init__(self, parts_shape=(16, 16), features=no_op,
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

    def _warp_images(self, images, shapes, _, verbose=True):

        # extract parts
        parts_images = []
        for c, (i, s) in enumerate(zip(images, shapes)):
            if verbose:
                print_dynamic(' - Warping images - {}'.format(
                    progress_bar_str(float(c + 1) / len(images),
                                     show_bar=False)))
            parts_image = Image(i.extract_patches(
                s, patch_size=self.parts_shape, as_single_array=True))
            parts_images.append(parts_image)

        return parts_images

    def _build_aam(self, shape_models, appearance_models, reference_shape):
        return PartsAAM(shape_models, appearance_models, reference_shape,
                        self.parts_shape, self.features,
                        self.normalize_parts, self.sigma,
                        list(reversed(self.scales)),
                        self.scale_shapes, self.scale_features)


from .base import (GlobalAAM, PatchAAM,
                   LinearGlobalAAM, LinearPatchAAM,
                   PartsAAM)


