from __future__ import division

from alabortijcv2015.fitter import Fitter
from alabortijcv2015.pdm import OrthoPDM
from alabortijcv2015.transform import OrthoMDTransform, OrthoLinearMDTransform

from .algorithm import (StandardAAMInterface, LinearAAMInterface,
                        PartsAAMInterface, AIC)
from menpo.transform import (AlignmentSimilarity, Scale, UniformScale,
                             ThinPlateSplines)
from menpo.shape import PointCloud
from alabortijcv2015.snape_iccv_2015 import noisy_align
from .result import AAMFitterResult, LinearAAMFitterResult


# Abstract Interface for ATM Fitters ------------------------------------------

class AAMFitter(Fitter):

    def _check_n_appearance(self, n_appearance):
        if n_appearance is not None:
            if type(n_appearance) is int or type(n_appearance) is float:
                for am in self.dm.appearance_models:
                    am.n_active_components = n_appearance
            elif len(n_appearance) == 1 and self.dm.n_levels > 1:
                for am in self.dm.appearance_models:
                    am.n_active_components = n_appearance[0]
            elif len(n_appearance) == self.dm.n_levels:
                for am, n in zip(self.dm.appearance_models, n_appearance):
                    am.n_active_components = n
            else:
                raise ValueError('n_appearance can be an integer or a float '
                                 'or None or a list containing 1 or {} of '
                                 'those'.format(self.dm.n_levels))

    def _fitter_result(self, image, algorithm_results, affine_correction,
                       gt_shape=None):
        return AAMFitterResult(image, self, algorithm_results,
                               affine_correction, gt_shape=gt_shape)


# Concrete Implementations of ATM Fitters -------------------------------------

class StandardAAMFitter(AAMFitter):

    def __init__(self, global_aam, algorithm_cls=AIC,
                 n_shape=None, n_appearance=None, **kwargs):

        super(StandardAAMFitter, self).__init__()

        self.dm = global_aam
        self._algorithms = []
        self._check_n_shape(n_shape)
        self._check_n_appearance(n_appearance)

        for j, (am, sm) in enumerate(zip(self.dm.appearance_models,
                                         self.dm.shape_models)):

            md_transform = OrthoMDTransform(
                sm, self.dm.transform,
                source=am.mean().landmarks['source'].lms,
                sigma2=am.noise_variance())

            algorithm = algorithm_cls(StandardAAMInterface, am,
                                      md_transform, **kwargs)

            self._algorithms.append(algorithm)


class LinearAAMFitter(AAMFitter):

    def __init__(self, global_aam, algorithm_cls=AIC, n_shape=None,
                 n_appearance=None, **kwargs):
        super(LinearAAMFitter, self).__init__()

        self.dm = global_aam
        self._algorithms = []
        self._check_n_shape(n_shape)
        self._check_n_appearance(n_appearance)

        for j, (am, sm) in enumerate(zip(self.dm.appearance_models,
                                         self.dm.shape_models)):
            md_transform = OrthoLinearMDTransform(
                sm, dense_indices=self.dm.dense_indices[j],
                sparse_mask=self.dm.sparse_masks[j])

            algorithm = algorithm_cls(LinearAAMInterface, am,
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
            shape.points.T.ravel(), n_channels=shape.n_dims)
        # TODO: lazily 'zoom' into the image to stop interpolation
        # issues at the boundaries. Really the image should have a
        # mask that is slightly too small to deal with this, or
        # model based interpolation should be performed using the
        # next shape model
        current_shape_im = current_shape_im.zoom(1.1)
        # Warp the image up to interpolate
        current_shape_im = current_shape_im.as_unmasked().warp_to_mask(
            self.dm.reference_frames[level + 1].mask,
            UniformScale(scale / self.scales[level + 1], shape.n_dims),
            warp_landmarks=False)
        # Back to pointcloud.
        new_shape = PointCloud(current_shape_im.as_vector(
            keep_channels=True).T)
        # But the values haven't changed! So we scale them as well.
        Scale(self.scales[level + 1] / scale,
              n_dims=new_shape.n_dims)._apply_inplace(new_shape)
        return new_shape

    @property
    def reference_shape(self):
        return self.dm.reference_shape

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

    def _tps_from_gt(self, gt_lmarks, min_singular_val=1.0):
        tps_warp = ThinPlateSplines(
            self.dm.reference_frames[0].landmarks['source'].lms,
            gt_lmarks, min_singular_val=min_singular_val)
        true_indices = self.dm.reference_frames[0].mask.true_indices()
        warped_grid = tps_warp.apply(true_indices)
        return PointCloud(warped_grid)

    def perturb_sparse_shape_tps(self, gt_shape, noise_std=0.04, rotation=False,
                                 min_singular_val=1.0):
        reference_shape = self.reference_shape
        dense_target = self._tps_from_gt(gt_shape,
                                         min_singular_val=min_singular_val)
        return noisy_align(reference_shape, dense_target,
                           noise_std=noise_std,
                           rotation=rotation).apply(reference_shape)

    def perturb_sparse_shape_exact(self, gt_shape, noise_std=0.04,
                                   rotation=False):
        transform = self._algorithms[0].transform
        sparse_ref_shape = PointCloud(
            self.reference_shape.points[transform.dense_indices])
        gt_to_m = gt_shape.from_mask(transform.sparse_mask)
        noisy_sparse = noisy_align(sparse_ref_shape, gt_to_m,
                                   noise_std=noise_std,
                                   rotation=rotation).apply(sparse_ref_shape)
        sim = AlignmentSimilarity(sparse_ref_shape, noisy_sparse)
        return sim.apply(self.reference_shape)

    def _fitter_result(self, image, algorithm_results, affine_correction,
                       gt_shape=None):
        return LinearAAMFitterResult(image, self, algorithm_results,
                                     affine_correction, gt_shape=gt_shape)

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


class PartsAAMFitter(AAMFitter):

    def __init__(self, parts_aam, algorithm_cls=AIC,
                 n_shape=None, n_appearance=None, **kwargs):

        super(PartsAAMFitter, self).__init__()

        self.dm = parts_aam
        self._algorithms = []
        self._check_n_shape(n_shape)
        self._check_n_appearance(n_appearance)

        for j, (am, sm) in enumerate(zip(self.dm.appearance_models,
                                         self.dm.shape_models)):

            pdm = OrthoPDM(sm, sigma2=am.noise_variance())

            am.parts_shape = self.dm.parts_shape
            am.normalize_parts = self.dm.normalize_parts
            algorithm = algorithm_cls(PartsAAMInterface, am, pdm, **kwargs)

            self._algorithms.append(algorithm)
