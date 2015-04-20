from __future__ import division
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.spatial.distance import cdist

from alabortijcv2015.fitter import Fitter
from alabortijcv2015.transform import OrthoMDTransform, OrthoLinearMDTransform
from alabortijcv2015.utils import fsmooth

from .algorithm import StandardATMInterface, LinearATMInterface, TIC
from menpo.transform import (Scale, AlignmentAffine, UniformScale,
                             AlignmentSimilarity, ThinPlateSplines)
from menpo.visualize import print_dynamic
from menpo.shape import PointCloud
from menpo.image import BooleanImage
from menpofit.base import noisy_align
from .result import ATMFitterResult, LinearATMFitterResult


# Abstract Interface for ATM Fitters ------------------------------------------


class ATMFitter(Fitter):

    def fit_sequence(self, images, initial_shapes, max_iters=50, gt_shapes=None,
                     crop_image=None, **kwargs):
        # generate the list of images to be fitted
        prepared_objs = []
        for k, (im, ish) in enumerate(zip(images, initial_shapes)):
            if gt_shapes:
                gt = gt_shapes[k]
            else:
                gt = None
            prepared_objs.append(self._prepare_image(im, ish,
                                                     gt_shape=gt,
                                                     crop_image=crop_image))
        # Group into the three types (scaled images, scaled initial shapes,
        # scaled ground truth)
        for k in range(len(prepared_objs)):
            o = prepared_objs[k]
            # No ground truth shapes
            if o[-1] is None:
                prepared_objs[k] = (o[0], o[1], [])

        seq_images, seq_initial_shapes, seq_gt_shapes = [
            zip(*obj) for obj in zip(*prepared_objs)]

        # work out the affine transform between the initial shape of the
        # highest pyramidal level and the initial shape of the original image
        affine_corrections = [AlignmentAffine(sish, ish)
                              for ish, sish in zip(initial_shapes,
                                                   seq_initial_shapes[-1])]

        # run multilevel fitting
        algorithm_results = self._fit_sequence(seq_images,
                                               seq_initial_shapes[0],
                                               max_iters=max_iters,
                                               seq_gt_shapes=seq_gt_shapes,
                                               **kwargs)

        # build multilevel fitting results
        fitter_results = []
        for k, (im, alg_res, aff_corr) in enumerate(zip(images,
                                                        algorithm_results,
                                                        affine_corrections)):
            if gt_shapes:
                gt_shape = gt_shapes[k]
            else:
                gt_shape = None
            fitter_results.append(self._fitter_result(im, alg_res, aff_corr,
                                                      gt_shape=gt_shape))

        return fitter_results

    def _fit_sequence(self, seq_images, seq_initial_shapes, seq_gt_shapes=None,
                      max_iters=50, **kwargs):
        max_iters = self._prepare_max_iters(max_iters)

        seq_algorithm_results = []
        shapes = seq_initial_shapes
        print_dynamic('Beginning Fitting...'.format(j))
        for j, (ims, alg, it, s) in enumerate(zip(seq_images,
                                                  self._algorithms,
                                                  max_iters, self.scales)):
            if seq_gt_shapes:
                gt_shapes = seq_gt_shapes[j]
            else:
                gt_shapes = None

            algorithm_results = alg.run(ims, shapes,
                                        gt_shapes=gt_shapes,
                                        max_iters=it, **kwargs)
            seq_algorithm_results.append(algorithm_results)

            if s != self.scales[-1]:
                shapes = []
                for a in algorithm_results:
                    sh = a.final_shape
                    Scale(self.scales[j + 1] / s,
                          n_dims=sh.n_dims).apply_inplace(sh)
                    shapes.append(sh)
            print_dynamic('Finished Scale {}'.format(j))

        return [r for r in zip(*seq_algorithm_results)]

    def _fitter_result(self, image, algorithm_results, affine_correction,
                       gt_shape=None):
        return ATMFitterResult(image, self, algorithm_results,
                               affine_correction, gt_shape=gt_shape)


# Concrete Implementations of ATM Fitters -------------------------------------

class StandardATMFitter(ATMFitter):

    def __init__(self, global_atm, algorithm_cls=TIC, n_shape=None, **kwargs):
        super(StandardATMFitter, self).__init__()

        self.dm = global_atm
        self._algorithms = []
        self._check_n_shape(n_shape)

        for j, (template, sm) in enumerate(zip(self.dm.templates,
                                               self.dm.shape_models)):
            md_transform = OrthoMDTransform(
                sm, self.dm.transform,
                source=template.landmarks['source'].lms)

            algorithm = algorithm_cls(StandardATMInterface, template,
                                      md_transform, **kwargs)

            self._algorithms.append(algorithm)


class LinearATMFitter(ATMFitter):

    def __init__(self, global_atm, algorithm_cls=TIC, n_shape=None, **kwargs):
        super(LinearATMFitter, self).__init__()

        self.dm = global_atm
        self._algorithms = []
        self._check_n_shape(n_shape)

        for j, (template, sm) in enumerate(zip(self.dm.templates,
                                               self.dm.shape_models)):
            md_transform = OrthoLinearMDTransform(
                sm, dense_indices=self.dm.dense_indices[j],
                sparse_mask=self.dm.sparse_masks[j])

            algorithm = algorithm_cls(LinearATMInterface, template,
                                      md_transform, **kwargs)

            self._algorithms.append(algorithm)

    def _mask_gt_shape(self, alg, gt_shape):
        if alg.transform.sparse_mask is not None:
            # Assume the GT is sparse
            if gt_shape.n_points == alg.transform.sparse_mask.shape[0]:
                gt_shape = gt_shape.from_mask(alg.transform.sparse_mask)
        return gt_shape

    def _interpolate_shape(self, shape, level, scale):
        # Create an image from the final shape for interpolation
        c_ref = self.dm.reference_frames[level]
        current_shape_im = c_ref.from_vector(
            shape.points.T.ravel(), n_channels=2)

        # TODO: This should probably be done at model building time - build
        # the reference frame with a mask this is "too big" and then use a
        # mask that is smaller so that sampling out of that mask will not
        # result in incorrect values. This is a kind of manual "nearest
        # neighbour" interpolation for non-rectangular masks.

        # Create a 5 pixel wide "buffer" mask around the correct mask
        d_mask = binary_dilation(c_ref.mask.mask, iterations=5)
        np.logical_and(d_mask, ~c_ref.mask.mask, out=d_mask)
        # Create a one pixel wide mask that is the boundary of the correct mask
        one_pixel_b = binary_erosion(c_ref.mask.mask, iterations=1)
        np.logical_and(~one_pixel_b, c_ref.mask.mask, out=one_pixel_b)

        # Get the indices of these masks
        o_indices = BooleanImage(one_pixel_b, copy=False).true_indices()
        d_indices = BooleanImage(d_mask, copy=False).true_indices()

        # Get the pairwise euclidean distances
        dists = cdist(o_indices, d_indices)
        # Choose the minimum entries
        closest_index = np.argmin(dists, axis=0)
        # Sample at the closest values
        samples = current_shape_im.sample(o_indices[closest_index])
        # Update the boundary area so that out of mask samples are interpolated
        # correctly
        current_shape_im.pixels[:, d_indices[:, 0], d_indices[:, 1]] = samples

        # Warp the image up to interpolate
        current_shape_im = current_shape_im.as_unmasked().warp_to_mask(
            self.dm.reference_frames[level + 1].mask,
            UniformScale(scale / self.scales[level + 1], 2))
        # Back to pointcloud.
        shape = PointCloud(current_shape_im.as_vector(
            keep_channels=True).T)
        # But the values haven't changed! So we scale them as well.
        Scale(self.scales[level + 1] / scale,
              n_dims=shape.n_dims).apply_inplace(shape)
        return shape

    @property
    def reference_shape(self):
        return self.dm.reference_shapes()[-1]

    def perturb_sparse_shape(self, gt_shape, noise_std=0.04, rotation=False):
        reference_shape = self.dm.reference_shapes()[0]
        transform = self._algorithms[0].transform
        gt_to_m = AlignmentSimilarity(gt_shape.from_mask(transform.sparse_mask),
                                      transform.sparse_target)
        transform.set_target(gt_to_m.apply(gt_shape))
        scaled_target = gt_to_m.pseudoinverse().apply(transform.dense_target)
        return noisy_align(reference_shape, scaled_target,
                           noise_std=noise_std,
                           rotation=rotation).apply(reference_shape)

    def _tps_from_gt(self, gt_lmarks, min_singular_val=1.0):
        tps_warp = ThinPlateSplines(
            self.dm.reference_frames[0].landmarks['source'].lms,
            gt_lmarks, min_singular_val=min_singular_val)
        true_indices = self.dm.reference_frames[0].mask.true_indices()
        warped_grid = tps_warp.apply(true_indices)
        return PointCloud(warped_grid)

    def perturb_sparse_shape_tps(self, gt_shape, noise_std=0.04, rotation=False,
                                 min_singular_val=1.0):
        reference_shape = self.dm.reference_shapes()[0]
        transform = self._algorithms[0].transform
        scaled_target = self._tps_from_gt(gt_shape,
                                          min_singular_val=min_singular_val)
        return noisy_align(reference_shape, scaled_target,
                           noise_std=noise_std,
                           rotation=rotation).apply(reference_shape)

    def _fitter_result(self, image, algorithm_results, affine_correction,
                       gt_shape=None):
        return LinearATMFitterResult(image, self, algorithm_results,
                                     affine_correction, gt_shape=gt_shape)

###############################Fitting Algorithms###############################

    def _fit(self, images, initial_shape, gt_shapes=None, max_iters=50,
             **kwargs):
        max_iters = self._prepare_max_iters(max_iters)

        shape = initial_shape
        gt_shape = None
        algorithm_results = []
        for j, (i, alg, it, s) in enumerate(zip(images, self._algorithms,
                                                max_iters, self.scales)):
            if gt_shapes:
                gt_shape = self._mask_gt_shape(alg, gt_shapes[j])

            algorithm_result = alg.run(i, shape, gt_shape=gt_shape,
                                       max_iters=it, **kwargs)
            algorithm_results.append(algorithm_result)

            shape = algorithm_result.final_dense_shape
            if s != self.scales[-1]:
                shape = self._interpolate_shape(shape, j, s)

        return algorithm_results

    def _fit_sequence(self, seq_images, seq_initial_shapes, seq_gt_shapes=None,
                      max_iters=50, **kwargs):
        max_iters = self._prepare_max_iters(max_iters)

        seq_algorithm_results = []
        shapes = seq_initial_shapes
        print_dynamic('Beginning Fitting...')
        for j, (ims, alg, it, s) in enumerate(zip(seq_images,
                                                  self._algorithms,
                                                  max_iters, self.scales)):
            if seq_gt_shapes:
                gt_shapes = seq_gt_shapes[j]
                gt_shapes = [self._mask_gt_shape(alg, g) for g in gt_shapes]
            else:
                gt_shapes = None

            algorithm_results = alg.run(ims, shapes,
                                        gt_shapes=gt_shapes,
                                        max_iters=it, **kwargs)
            seq_algorithm_results.append(algorithm_results)

            if s != self.scales[-1]:
                shapes = []
                for a in algorithm_results:
                    sh = self._interpolate_shape(a.final_dense_shape, j, s)
                    shapes.append(sh)
            print_dynamic('Finished Scale {}'.format(j))

        return [r for r in zip(*seq_algorithm_results)]

    def _prepare_image(self, image, initial_shape, gt_shape=None,
                       crop_image=None):
        # attach landmarks to the image
        image.landmarks['initial_shape'] = initial_shape
        if gt_shape:
            image.landmarks['gt_shape'] = gt_shape

        if crop_image:
            image = image.copy()
            image.crop_to_landmarks_proportion_inplace(crop_image,
                                                       group='initial_shape')

        # rescale image w.r.t the scale factor between reference_shape and
        # initial_shape
        # This is a bit more complicated because we have a different number
        # of points in each reference shape per scale. Since this is supposed
        # to normalize BEFORE computing the scales, we really want to normalize
        # it to the scale space of the finest (largest) level, but we need
        # to use the POINTS of the lowest. Therefore, we need to scaled the
        # lowest points up to the SCALE of the finest (largest).
        scaled_lowest_reference = UniformScale(
            self.scales[-1] / self.scales[0],
            n_dims=initial_shape.n_dims).apply(self.dm.reference_shapes()[0])

        image = image.rescale_to_reference_shape(scaled_lowest_reference,
                                                 group='initial_shape')
        if self.sigma:
            image.pixels = fsmooth(image.pixels, self.sigma)

        # obtain image representation
        from copy import deepcopy
        scales = deepcopy(self.scales)
        scales.reverse()
        images = []
        for j, s in enumerate(scales):
            if j == 0:
                # compute features at highest level
                feature_image = self.features(image)
            elif self.scale_features:
                # scale features at other levels
                feature_image = images[0].rescale(s)
            else:
                # scale image and compute features at other levels
                scaled_image = image.rescale(s)
                feature_image = self.features(scaled_image)
            images.append(feature_image)
        images.reverse()

        # get initial shapes per level
        initial_shapes = [i.landmarks['initial_shape'].lms for i in images]

        # get ground truth shapes per level
        if gt_shape:
            gt_shapes = [i.landmarks['gt_shape'].lms for i in images]
        else:
            gt_shapes = None

        return images, initial_shapes, gt_shapes
