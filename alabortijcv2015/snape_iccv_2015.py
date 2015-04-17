from alabortijcv2015.aam import LinearGlobalAAM, GlobalAAM
from alabortijcv2015.atm import LinearGlobalATM
from menpo.transform import PiecewiseAffine, ThinPlateSplines
from menpo.visualize import print_dynamic, progress_bar_str
from menpo.feature import no_op
from menpo.image import MaskedImage, Image
from alabortijcv2015.atm.builder import (compute_features, scale_images,
                                         sparse_landmark_indices_from_dense,
                                         zero_flow_grid_pcloud)
from alabortijcv2015.aam.builder import normalize_images
from menpo.feature.base import winitfeature
from cyvlfeat.sift.dsift import dsift as cyvlfeat_dsift
import numpy as np


@winitfeature
def dsift(pixels, step=1, size=6, bounds=None, norm=False,
          fast=True, window_size=6, geometry=(1, 1, 8), float_descriptors=True):
    centers, output = cyvlfeat_dsift(np.rot90(pixels[0, ..., ::-1]),
                                     step=step, size=size, bounds=bounds,
                                     window_size=window_size, norm=norm,
                                     fast=fast,
                                     float_descriptors=float_descriptors,
                                     geometry=geometry)
    shape = pixels.shape[1:] - 2 * centers[:2, 0]
    return (np.require(output.reshape((-1, shape[0], shape[1])),
                       dtype=np.double),
            np.require(centers[:2, ...].T[..., ::-1].reshape(
                (shape[0], shape[1], 2)), dtype=np.int))


def build_atm(template_im, shape_models, reference_frames, scales, feature=no_op,
              sparse_group=None, scale_features=True, verbose=True):

    normalized_template = template_im.rescale_to_reference_shape(
        reference_frames[0].landmarks[sparse_group].lms,
        group=sparse_group)

    templates = []
    dense_indices = []
    sparse_masks = []
    for j, (s, ref_frame) in enumerate(zip(scales, reference_frames)):
        # obtain image representation
        if j == 0:
            # compute features at highest level
            feature_image = compute_features([normalized_template],
                                             verbose=verbose,
                                             features=feature)[0]
            level_template = feature_image
        elif scale_features:
            # scale features at other levels
            level_template = scale_images([feature_image], s,
                                          verbose=verbose)[0]
        else:
            # scale images and compute features at other levels
            scaled_image = scale_images([normalized_template], s,
                                        verbose=verbose)[0]
            level_template = compute_features([scaled_image],
                                              verbose=verbose,
                                              features=feature)[0]

        transform = PiecewiseAffine(ref_frame.landmarks[sparse_group].lms,
                                    level_template.landmarks[sparse_group].lms)

        # warp template to reference frame
        if isinstance(level_template, MaskedImage):
            level_template = level_template.as_unmasked(copy=False)
        warped_template = level_template.warp_to_mask(ref_frame.mask,
                                                      transform)
        warped_template.landmarks['source'] = ref_frame.landmarks[sparse_group]
        templates.append(warped_template)

        zero_flow = zero_flow_grid_pcloud(ref_frame.shape,
                                          mask=ref_frame.mask)
        dense_ind, sparse_mask = sparse_landmark_indices_from_dense(
            zero_flow, ref_frame.landmarks[sparse_group].lms)
        dense_indices.append(dense_ind)
        sparse_masks.append(sparse_mask)

    return LinearGlobalATM(list(reversed(shape_models)),
                           list(reversed(templates)),
                           list(reversed(reference_frames)),
                           feature,
                           None,
                           list(reversed(scales)),
                           scale_features,
                           dense_indices=list(reversed(dense_indices)),
                           sparse_masks=list(reversed(sparse_masks)))


def build_sparse_aam(images, shape_models, reference_frames, scales, feature=no_op,
                     sparse_group=None, scale_features=True, max_appearance_components=None,
                     verbose=True):


    normalized_images = normalize_images(images, sparse_group, None,
                                         reference_frames[-1].landmarks[sparse_group].lms,
                                         verbose)

    dense_indices = []
    sparse_masks = []
    appearance_models = []
    for j, (s, ref_frame) in enumerate(zip(scales, reference_frames)):
        # obtain image representation
        if j == 0:
            # compute features at highest level
            feature_images = compute_features(normalized_images,
                                              verbose=verbose,
                                              features=feature)
            level_images = feature_images
        elif scale_features:
            # scale features at other levels
            level_images = scale_images(feature_images, s,
                                        verbose=verbose)
        else:
            # scale images and compute features at other levels
            scaled_images = scale_images(normalized_images, s,
                                         verbose=verbose)
            level_images = compute_features(scaled_images,
                                            verbose=verbose,
                                            features=feature)

        transform = PiecewiseAffine(ref_frame.landmarks[sparse_group].lms,
                                    ref_frame.landmarks[sparse_group].lms)

        N = len(images)
        warped_images = []
        for k, im in enumerate(level_images):
            # warp template to reference frame
            if isinstance(im, MaskedImage):
                im = im.as_unmasked()
            transform.set_target(im.landmarks[sparse_group].lms)
            warped_im = im.warp_to_mask(ref_frame.mask,
                                        transform)
            warped_im.landmarks['source'] = ref_frame.landmarks[sparse_group]
            warped_images.append(warped_im)
            if verbose:
                print_dynamic('- Warping Images - {}'.format(
                    progress_bar_str((k + 1.) / N, show_bar=False)))

        if verbose:
            print_dynamic('- Building Appearance Model')
        appearance_model = PCAModel(warped_images)
        # trim appearance model if required
        if max_appearance_components is not None:
            appearance_model.trim_components(max_appearance_components)
        appearance_models.append(appearance_model)

        # Zero flow
        zero_flow = zero_flow_grid_pcloud(ref_frame.shape,
                                          mask=ref_frame.mask)
        dense_ind, sparse_mask = sparse_landmark_indices_from_dense(
            zero_flow, ref_frame.landmarks[sparse_group].lms)
        dense_indices.append(dense_ind)
        sparse_masks.append(sparse_mask)

    return GlobalAAM(list(reversed(shape_models)),
                     list(reversed(appearance_models)),
                     reference_frames[-1].landmarks[sparse_group].lms,
                     DifferentiablePiecewiseAffine,
                     feature,
                     None,
                     list(reversed(scales)),
                     True,
                     scale_features)


def build_aam(images, shape_models, reference_frames, scales, feature=no_op,
              sparse_group=None, scale_features=True, max_appearance_components=None,
              verbose=True):

    normalized_images = normalize_images(images, sparse_group, None,
                                         reference_frames[-1].landmarks[sparse_group].lms,
                                         verbose)

    dense_indices = []
    sparse_masks = []
    appearance_models = []
    for j, (s, ref_frame) in enumerate(zip(scales, reference_frames)):
        # obtain image representation
        if j == 0:
            # compute features at highest level
            feature_images = compute_features(normalized_images,
                                              verbose=verbose,
                                              features=feature)
            level_images = feature_images
        elif scale_features:
            # scale features at other levels
            level_images = scale_images(feature_images, s,
                                        verbose=verbose)
        else:
            # scale images and compute features at other levels
            scaled_images = scale_images(normalized_images, s,
                                         verbose=verbose)
            level_images = compute_features(scaled_images,
                                            verbose=verbose,
                                            features=feature)

        transform = PiecewiseAffine(ref_frame.landmarks[sparse_group].lms,
                                    ref_frame.landmarks[sparse_group].lms)

        N = len(images)
        warped_images = []
        for k, im in enumerate(level_images):
            # warp template to reference frame
            if isinstance(im, MaskedImage):
                im = im.as_unmasked()
            transform.set_target(im.landmarks[sparse_group].lms)
            warped_im = im.warp_to_mask(ref_frame.mask,
                                        transform)
            warped_im.landmarks['source'] = ref_frame.landmarks[sparse_group]
            warped_images.append(warped_im)
            if verbose:
                print_dynamic('- Warping Images - {}'.format(
                    progress_bar_str((k + 1.) / N, show_bar=False)))

        if verbose:
            print_dynamic('- Building Appearance Model')
        appearance_model = PCAModel(warped_images)
        # trim appearance model if required
        if max_appearance_components is not None:
            appearance_model.trim_components(max_appearance_components)
        appearance_models.append(appearance_model)

        # Zero flow
        zero_flow = zero_flow_grid_pcloud(ref_frame.shape,
                                          mask=ref_frame.mask)
        dense_ind, sparse_mask = sparse_landmark_indices_from_dense(
            zero_flow, ref_frame.landmarks[sparse_group].lms)
        dense_indices.append(dense_ind)
        sparse_masks.append(sparse_mask)

    return LinearGlobalAAM(list(reversed(shape_models)),
                           list(reversed(appearance_models)),
                           list(reversed(reference_frames)),
                           feature,
                           None,
                           list(reversed(scales)),
                           scale_features,
                           dense_indices=list(reversed(dense_indices)),
                           sparse_masks=list(reversed(sparse_masks)))
