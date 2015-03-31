from __future__ import division
from skimage.morphology import binary_erosion, disk

from alabortijcv2015.fitter import Fitter
from alabortijcv2015.transform import OrthoMDTransform, OrthoLinearMDTransform

from .algorithm import StandardATMInterface, LinearATMInterface, TIC
from menpo.transform import (Scale, AlignmentAffine, UniformScale,
                             AlignmentSimilarity)
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
        # Group into the three types
        # Then group each scale together for each type
        for k in range(len(prepared_objs)):
            o = prepared_objs[k]
            if o[-1] is None:
                prepared_objs[k] = (prepared_objs[k][0], prepared_objs[k][1], [])

        seq_images, seq_initial_shapes, seq_gt_shapes = [zip(*obj)
                                                         for obj in zip(*prepared_objs)]

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
                for alg in algorithm_results:
                    sh = alg.final_shape
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
            if gt_shape.n_points < alg.transform.dense_target.n_points:
                gt_shape = gt_shape.from_mask(alg.transform.sparse_mask)
        return gt_shape

    def _interpolate_shape(self, shape, level, scale):
        # Create an image from the final shape for interpolation
        current_shape_im = self.dm.reference_frames[level].from_vector(
            shape.points.T.ravel(), n_channels=2)
        # TODO: lazily 'zoom' into the image to stop interpolation
        # issues at the boundaries. Really the image should have a
        # mask that is slightly too small to deal with this, or
        # model based interpolation should be performed using the
        # next shape model
        current_shape_im = current_shape_im.zoom(1.1)
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
        return PointCloud(self.dm.reference_frames[0].as_vector(
            keep_channels=True).T)

    def perturb_sparse_shape(self, gt_shape, noise_std=0.04, rotation=False):
        reference_shape = self.reference_shape
        transform = self._algorithms[0].transform
        gt_to_m = AlignmentSimilarity(gt_shape.from_mask(transform.sparse_mask),
                                      transform.sparse_target)
        transform.set_target(gt_to_m.apply(gt_shape))
        scaled_target = gt_to_m.pseudoinverse().apply(transform.dense_target)
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
                for alg in algorithm_results:
                    sh = self._interpolate_shape(alg.final_dense_shape, j, s)
                    shapes.append(sh)
            print_dynamic('Finished Scale {}'.format(j))

        return [r for r in zip(*seq_algorithm_results)]

    def fit_constrained_sequence(self, images, sparse_shapes, initial_shapes,
                                 max_iters=50, gt_shapes=None, crop_image=None,
                                 **kwargs):
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
        # Group into the three types
        # Then group each scale together for each type
        for k in range(len(prepared_objs)):
            o = prepared_objs[k]
            if o[-1] is None:
                prepared_objs[k] = (prepared_objs[k][0], prepared_objs[k][1], [])

        seq_images, seq_initial_shapes, seq_gt_shapes = [zip(*obj)
                                                         for obj in zip(*prepared_objs)]

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

    def _fit_constrained_sequence(self, seq_images, seq_initial_shapes,
                                  seq_gt_shapes=None, max_iters=50, **kwargs):
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
                for alg in algorithm_results:
                    sh = alg.final_shape
                    Scale(self.scales[j + 1] / s,
                          n_dims=sh.n_dims).apply_inplace(sh)
                    shapes.append(sh)
            print_dynamic('Finished Scale {}'.format(j))

        return [r for r in zip(*seq_algorithm_results)]
