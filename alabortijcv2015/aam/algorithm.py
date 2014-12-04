from __future__ import division
import abc

import numpy as np

from menpofast.image import Image
from menpofast.feature import gradient as fast_gradient
from menpofast.utils import build_parts_image

from .result import AAMAlgorithmResult, LinearAAMAlgorithmResult


# Abstract Interfaces for AAM Algorithms --------------------------------------

class AAMInterface(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, aam_algorithm):
        self.algorithm = aam_algorithm

    @abc.abstractmethod
    def dw_dp(self):
        pass

    @abc.abstractmethod
    def warp(self, image):
        pass

    @abc.abstractmethod
    def gradient(self, image):
        pass

    @abc.abstractmethod
    def steepest_descent_images(self, gradient, dw_dp):
        pass

    @abc.abstractmethod
    def partial_newton_hessian(self, gradient2, dw_dp):
        pass

    @abc.abstractmethod
    def solve(self, h, j, e, prior):
        pass


class StandardAAMInterface(AAMInterface):

    def __init__(self, aam_algorithm, sampling_step=None):
        super(StandardAAMInterface, self). __init__(aam_algorithm)

        n_true_pixels = self.algorithm.template.n_true_pixels()
        n_channels = self.algorithm.template.n_channels
        n_parameters = self.algorithm.transform.n_parameters
        sampling_mask = np.require(np.zeros(n_true_pixels), dtype=np.bool)

        if sampling_step is None:
            sampling_step = 1
        sampling_pattern = xrange(0, n_true_pixels, sampling_step)
        sampling_mask[sampling_pattern] = 1

        self.image_vec_mask = np.nonzero(np.tile(
            sampling_mask[None, ...], (n_channels, 1)).flatten())[0]
        self.dw_dp_mask = np.nonzero(np.tile(
            sampling_mask[None, ..., None], (2, 1, n_parameters)))
        self.gradient_mask = np.nonzero(np.tile(
            sampling_mask[None, None, ...], (2, n_channels, 1)))
        self.gradient2_mask = np.nonzero(np.tile(
            sampling_mask[None, None, None, ...], (2, 2, n_channels, 1)))

        # self.eigenvalues = self.algorithm.transform.pdm.model.eigenvalues

    def dw_dp(self):
        dw_dp = np.rollaxis(self.algorithm.transform.d_dp(
            self.algorithm.template.mask.true_indices()), -1)
        return dw_dp[self.dw_dp_mask].reshape((dw_dp.shape[0], -1,
                                               dw_dp.shape[2]))

    def warp(self, image):
        return image.warp_to_mask(self.algorithm.template.mask,
                                  self.algorithm.transform)

    def gradient(self, image):
        return image.gradient(
            nullify_values_at_mask_boundaries=True).as_vector().reshape(
                (2, image.n_channels, -1))

    def steepest_descent_images(self, gradient, dw_dp):
        # reshape gradient
        # gradient: n_dims x n_channels x n_pixels
        gradient = gradient[self.gradient_mask].reshape(
            gradient.shape[:2] + (-1,))
        # compute steepest descent images
        # gradient: n_dims x n_channels x n_pixels
        # dw_dp:    n_dims x            x n_pixels x n_params
        # sdi:               n_channels x n_pixels x n_params
        sdi = 0
        a = gradient[..., None] * dw_dp[:, None, ...]
        for d in a:
            sdi += d

        # reshape steepest descent images
        # sdi: (n_channels x n_pixels) x n_params
        return sdi.reshape((-1, sdi.shape[2]))

    def partial_newton_hessian(self, gradient2, dw_dp):
        # reshape gradient
        # gradient: n_dims x n_dims x n_channels x n_pixels
        gradient2 = gradient2[self.gradient2_mask].reshape(
            (2,) + gradient2.shape[:2] + (-1,))

        # compute partial hessian
        # gradient: n_dims x n_dims x n_channels x n_pixels
        # dw_dp:    n_dims x                     x n_pixels x n_params
        # h:                 n_dims x n_channels x n_pixels x n_params
        h1 = 0
        aux = gradient2[..., None] * dw_dp[:, None, None, ...]
        for d in aux:
            h1 += d
        # compute partial hessian
        # h:     n_dims x n_channels x n_pixels x n_params
        # dw_dp: n_dims x            x n_pixels x          x n_params
        # h:
        h2 = 0
        aux = h1[..., None] * dw_dp[..., None, :, None, :]
        for d in aux:
            h2 += d

        # reshape hessian
        # 2:  (n_channels x n_pixels) x n_params x n_params
        return h2.reshape((-1, h2.shape[3] * h2.shape[4]))

    def solve(self, h, j, e, prior):
        t = self.algorithm.transform
        jp = t.jp()

        if prior:
            inv_h = np.linalg.inv(h)
            dp = inv_h.dot(j.T.dot(e))
            dp = -np.linalg.solve(t.h_prior + jp.dot(inv_h.dot(jp.T)),
                                  t.j_prior * t.as_vector() - jp.dot(dp))
        else:
            dp = np.linalg.solve(h, j.T.dot(e))
            dp = jp.dot(dp)

        return dp

    def algorithm_result(self, image, shape_parameters,
                         appearance_parameters=None, gt_shape=None):
        return AAMAlgorithmResult(
            image, self.algorithm, shape_parameters,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class LinearAAMInterface(StandardAAMInterface):

    def solve(self, h, j, e, prior):
        t = self.algorithm.transform

        if prior:
            dp = -np.linalg.solve(t.h_prior + h,
                                  t.j_prior * t.as_vector() - j.T.dot(e))
        else:
            dp = np.linalg.solve(h, j.T.dot(e))

        return dp

    def algorithm_result(self, image, shape_parameters,
                         appearance_parameters=None, gt_shape=None):
        return LinearAAMAlgorithmResult(
            image, self.algorithm, shape_parameters,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class PartsAAMInterface(AAMInterface):

    def __init__(self, aam_algorithm, sampling_mask=None):
        super(PartsAAMInterface, self). __init__(aam_algorithm)

        if sampling_mask is None:
            parts_shape = self.algorithm.appearance_model.parts_shape
            sampling_mask = np.require(np.ones((parts_shape)), dtype=np.bool)

        image_shape = self.algorithm.template.pixels.shape
        image_mask = np.tile(sampling_mask[None, None, None, ...],
                             image_shape[:3] + (1, 1))
        self.image_vec_mask = np.nonzero(image_mask.flatten())[0]
        self.gradient_mask = np.nonzero(np.tile(
            image_mask[None, ...], (2, 1, 1, 1, 1, 1)))

    def dw_dp(self):
        return np.rollaxis(self.algorithm.transform.d_dp(None), -1)

    def warp(self, image):
        return build_parts_image(
            image, self.algorithm.transform.target,
            parts_shape=self.algorithm.appearance_model.parts_shape)

    def gradient(self, image):
        pixels = image.pixels
        parts_shape = self.algorithm.appearance_model.parts_shape
        g = fast_gradient(pixels.reshape((-1,) + parts_shape))
        # remove 1st dimension gradient which corresponds to the gradient
        # between parts
        return g.reshape((2,) + pixels.shape)

    def steepest_descent_images(self, gradient, dw_dp):
        # reshape gradient
        # gradient: n_dims x n_parts x offsets x n_ch x (h x w)
        gradient = gradient[self.gradient_mask].reshape(
            gradient.shape[:-2] + (-1,))
        # compute steepest descent images
        # gradient: n_dims x n_parts x offsets x n_ch x (h x w)
        # ds_dp:    n_dims x n_parts x                          x n_params
        # sdi:               n_parts x offsets x n_ch x (h x w) x n_params
        sdi = 0
        a = gradient[..., None] * dw_dp[..., None, None, None, :]
        for d in a:
            sdi += d

        # reshape steepest descent images
        # sdi: (n_parts x n_offsets x n_ch x w x h) x n_params
        return sdi.reshape((-1, sdi.shape[-1]))

    def partial_newton_hessian(self, gradient2, dw_dp):
        # reshape gradient
        # gradient: n_dims x n_dims x n_channels x n_parts x (w x h)
        gradient2 = gradient2[self.gradient2_mask].reshape(
            gradient2.shape[:-2] + (-1,))

        # compute partial hessian
        # gradient: n_dims x n_dims x n_channels x n_parts x (w x h)
        # dw_dp:    n_dims x                     x n_parts x         x n_params
        # h:                 n_dims x n_channels x n_parts x (w x h) x n_params
        h1 = 0
        aux = gradient2[..., None] * dw_dp[..., None, None, :, None, :]
        for d in aux:
            h1 += d
        # compute partial hessian
        # h:     n_dims x n_channels x n_parts x (w x h) x n_params
        # dw_dp: n_dims x            x n_parts x                    x n_params
        # h:
        h2 = 0
        aux = h1[..., None] * dw_dp[..., None, :, None, None, :]
        for d in aux:
            h2 += d

        # reshape hessian
        # 2:  (n_channels x n_parts x w x h) x n_params x n_params
        return h2.reshape((-1, h2.shape[3] * h2.shape[4]))

    def solve(self, h, j, e, prior):
        t = self.algorithm.transform

        if prior:
            dp = -np.linalg.solve(t.h_prior + h,
                                  t.j_prior * t.as_vector() - j.T.dot(e))
        else:
            dp = np.linalg.solve(h, j.T.dot(e))

        return dp

    def algorithm_result(self, image, shape_parameters,
                         appearance_parameters=None, gt_shape=None):
        return AAMAlgorithmResult(
            image, self.algorithm, shape_parameters,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


# Abstract Interfaces for AAM Algorithms  -------------------------------------

class AAMAlgorithm(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, aam_interface, appearance_model, transform,
                 eps=10**-5, **kwargs):

        # set common state for all AAM algorithms
        self.appearance_model = appearance_model
        self.template = appearance_model.mean()
        self.transform = transform
        self.eps = eps

        # set interface
        self.interface = aam_interface(self, **kwargs)

        self._U = self.appearance_model.components.T
        self._pinv_U = np.linalg.pinv(
            self._U[self.interface.image_vec_mask, :]).T


    @abc.abstractmethod
    def _precompute(self, **kwargs):
        pass

    @abc.abstractmethod
    def run(self, image, initial_shape, max_iters=20, gt_shape=None, **kwargs):
        pass


class ProjectOut(AAMAlgorithm):

    def __init__(self, aam_interface, appearance_model, transform,
                 eps=10**-5, **kwargs):
        # call super constructor
        super(ProjectOut, self).__init__(
            aam_interface, appearance_model, transform, eps, **kwargs)

        # set common state for all Project Out AAM algorithms
        self._masked_U = self._U[self.interface.image_vec_mask, :]

        # pre-compute
        self._precompute()
        
    def project_out(self, j):
        return j - self._masked_U.dot(self._pinv_U.T.dot(j))


class FastSimultaneous(AAMAlgorithm):

    def __init__(self, aam_interface, appearance_model, transform,
                 eps=10**-5, **kwargs):
        # call super constructor
        super(FastSimultaneous, self).__init__(
            aam_interface, appearance_model, transform, eps, **kwargs)
        
        # set common state for all Fast Simultaneous AAM algorithms
        self._masked_U = self._U[self.interface.image_vec_mask, :]

        # pre-compute
        self._precompute()

    def project_out(self, j):
        return j - self._masked_U.dot(self._pinv_U.T.dot(j))


class Alternating(AAMAlgorithm):

    def __init__(self, aam_interface, appearance_model, transform,
                 eps=10**-5, **kwargs):
        # call super constructor
        super(Alternating, self).__init__(
            aam_interface, appearance_model, transform, eps, **kwargs)

        # pre-compute
        self._precompute()


class Bayesian(AAMAlgorithm):

    def __init__(self, aam_interface, appearance_model, transform,
                 eps=10**-5, **kwargs):

        # call super constructor
        super(Bayesian, self).__init__(
            aam_interface, appearance_model, transform, eps, **kwargs)

        # set common state for all Bayesian AAM algorithms
        self._U = self._U[self.interface.image_vec_mask, :]
        sigma2 = self.appearance_model.noise_variance()
        self._inv_sigma2 = 1 / sigma2
        self._inv_D = 1 / (self.appearance_model.eigenvalues + sigma2)

        # pre-compute
        self._precompute()

    def project_out(self, j, l=0.5):
        l_inv_sigma2 = l * self._inv_sigma2
        A = l_inv_sigma2 - (1 - l) * self._inv_D
        return 2 * (l_inv_sigma2 * j -
                    self._U.dot(A[..., None] * self._pinv_U.T.dot(j)))


# Concrete implementations of AAM Algorithms  ---------------------------------

# Project Out Compositional Algorithms ----------------------------------------

class PIC_SD(ProjectOut):
    r"""
    Project-Out Inverse Compositional Steepest-Descent Algorithm
    """

    def _precompute(self):

        # compute model's gradient
        nabla_t = self.interface.gradient(self.template)

        # compute warp jacobian
        dw_dp = self.interface.dw_dp()

        # compute steepest descent images
        j = self.interface.steepest_descent_images(nabla_t, dw_dp)

        # project out appearance model from J
        self._j = self.project_out(j)

    def run(self, image, initial_shape, gt_shape=None, max_iters=20, **kwargs):

        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]

        for _ in xrange(max_iters):

            target = self.transform.target

            # compute warped image with current weights
            i_p = image.warp_to(self.mask, self.transform)

            # compute error image
            e = self.template.as_vector() - i_p.as_vector()

            # compute steepest-descent parameter updates
            delta_p = self._j.dot(e)

            # update transform weights
            self.transform.compose_after_from_vector_inplace(delta_p)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(target.points -
                                          self.transform.target.points))
            if error < self.eps:
                break

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters,
                                               gt_shape=gt_shape)


class PIC_GN(ProjectOut):
    r"""
    Project-Out Inverse Compositional Gauss-Newton Algorithm
    """

    def _precompute(self):

        # compute model's gradient
        nabla_t = self.interface.gradient(self.template)

        # compute warp jacobian
        dw_dp = self.interface.dw_dp()

        # compute steepest descent images
        j = self.interface.steepest_descent_images(nabla_t, dw_dp)

        # project out appearance model from J
        self._j_po = self.project_out(j)

        # compute inverse hessian
        self._h = self._j_po.T.dot(j)

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # masked model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # compute error image
            e = masked_m - masked_i

            # compute gauss-newton parameter updates
            dp = self.interface.solve(self._h, self._j_po, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(
                target.points - self.transform.target.points))
            if error < self.eps:
                break

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters,
                                               gt_shape=gt_shape)


class PIC_N(ProjectOut):
    r"""
    Project-Out Inverse Compositional Newton Algorithm
    """

    def _precompute(self):

        # compute model gradient
        nabla_t = self.interface.gradient(self.template)

        # compute model second gradient
        nabla2_t = self.interface.gradient(Image(nabla_t))

        # compute warp jacobian
        dw_dp = self.interface.dw_dp()

        # compute steepest descent images
        j = self.interface.steepest_descent_images(nabla_t, dw_dp)

        # project out appearance model from J
        self._j_po = self.project_out(j)

        # compute gauss-newton hessian
        self._h_gn = self._j_po.T.dot(j)

        # compute newton hessian
        self._h_pn = self.interface.partial_newton_hessian(nabla2_t, dw_dp)

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

         # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # masked model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # compute error image
            e = masked_m - masked_i

            # project out appearance model from error
            e_po = self.project_out(e)
            # compute full newton hessian
            h = e_po.dot(self._h_pn).reshape(self._h_gn.shape) + self._h_gn

            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, self._j_po, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(target.points -
                                          self.transform.target.points))
            if error < self.eps:
                break

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters,
                                               gt_shape=gt_shape)


class PFC_SD(ProjectOut):
    r"""
    Project-Out Forward Compositional Steepest-Descent Algorithm
    """
    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20, **kwargs):

        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]

        for _ in xrange(max_iters):

            target = self.transform.target

            # compute warped image with current weights
            i_p = image.warp_to(self.mask, self.transform)

            # compute model's gradient
            nabla_i = i_p.gradient(nullify_values_at_mask_boundaries=True)

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_i, self._dw_dp)

            # project out appearance model from model jacobian
            j = self._project_out(j)

            # compute error image
            e = self.template.as_vector() - i_p.as_vector()

            # compute steepest-descent parameter updates
            delta_p = j.dot(e)

            # update transform weights
            self.transform.compose_after_from_vector_inplace(delta_p)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(target.points -
                                          self.transform.target.points))
            if error < self.eps:
                break

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters,
                                               gt_shape=gt_shape)


class PFC_GN(ProjectOut):
    r"""
    Project-Out Forward Compositional Gauss-Newton Algorithm
    """

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # masked model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # compute error image
            e = masked_m - masked_i

            # compute image gradient
            nabla_i = self.interface.gradient(i)

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_i, self._dw_dp)
            # project out appearance model from model jacobian
            j_po = self.project_out(j)

            # compute hessian
            h = j_po.T.dot(j)

            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, j_po, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(
                target.points - self.transform.target.points))
            if error < self.eps:
                break

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters,
                                               gt_shape=gt_shape)


class PFC_N(ProjectOut):
    r"""
    Project-Out Forward Compositional Newton Algorithm
    """

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # mask model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # compute error image
            e = masked_m - masked_i

            # compute image gradient
            nabla_i = self.interface.gradient(i)
            # compute image second order gradient
            nabla2_i = self.interface.gradient(Image(nabla_i))

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_i, self._dw_dp)
            # project out appearance model from model jacobian
            j_po = self.project_out(j)

            # compute gauss-newton hessian
            h_gn = j_po.T.dot(j)
            # compute partial newton hessian
            h_pn = self.interface.partial_newton_hessian(nabla2_i, self._dw_dp)
            # project out appearance model from error
            e_po = self.project_out(e)
            # compute full newton hessian
            h = e_po.dot(h_pn).reshape(h_gn.shape) + h_gn

            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, j_po, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(target.points -
                                          self.transform.target.points))
            if error < self.eps:
                break

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters,
                                               gt_shape=gt_shape)


class PSC_GN(ProjectOut):
    r"""
    Project-Out Symmetric Compositional ESM Algorithm
    """

    def _precompute(self):

        # compute model gradient
        self._nabla_t = self.interface.gradient(self.template)

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False, a=0.5):

        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # masked model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # compute error image
            e = masked_m - masked_i

            # combine image and model gradient
            nabla = a * self.interface.gradient(i) + (1-a) * self._nabla_t

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla, self._dw_dp)
            # project out appearance model from model jacobian
            j_po = self.project_out(j)

            # compute hessian
            h = j_po.T.dot(j)

            # compute symmetric esm parameter updates
            dp = self.interface.solve(h, j_po, e, prior)

            # update transform
            target = self.transform.target
            dt = self.transform.from_vector(a * dp)
            dt.from_vector_inplace(dt.as_vector() + (1-a) * dp)
            self.transform.from_vector_inplace(
                self.transform.as_vector() + dt.as_vector())
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(
                target.points - self.transform.target.points))
            if error < self.eps:
                break

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters,
                                               gt_shape=gt_shape)


class PBC_GN(ProjectOut):
    r"""
    Project-Out Bidirectional Compositional ESM Algorithm
    """

    def _precompute(self):

        # compute model gradient
        self._nabla_t = self.interface.gradient(self.template)

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        n_shape_params = self.transform.n_parameters
        # masked model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # compute error image
            e = masked_m - masked_i

            # compute image gradient
            nabla_i = self.interface.gradient(i)

            # compute jacobian
            j_i = self.interface.steepest_descent_images(nabla_i, self._dw_dp)
            j_t = self.interface.steepest_descent_images(self._nabla_t,
                                                         self._dw_dp)
            j = np.hstack((j_i, j_t))
            # project out appearance model from  jacobian
            j_po = self.project_out(j)

            # compute hessian
            h = j_po.T.dot(j)

             # compute symmetric esm parameter updates
            dp = self.interface.solve(h, j_po, e, prior)

            # update transform
            target = self.transform.target
            dt = self.transform.from_vector(dp[:n_shape_params])
            dt.from_vector_inplace(dt.as_vector() + dp[n_shape_params:])
            self.transform.from_vector_inplace(
                self.transform.as_vector() + dt.as_vector())
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(
                target.points - self.transform.target.points))
            if error < self.eps:
                break

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters,
                                               gt_shape=gt_shape)


# Fast Simultaneous Compositional Algorithms ----------------------------------

class FastSIC_SD(AAMAlgorithm):
    r"""
    Fast Simultaneous Inverse Compositional Steepest-Descent Algorithm
    """


class FastSIC_GN(FastSimultaneous):
    r"""
    Fast Simultaneous Inverse Compositional Gauss-Newton Algorithm
    """

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # warp image
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            if _ == 0:
                # project image onto the model bases
                c = self._pinv_U.T.dot(masked_i - masked_m)
            else:
                # compute gauss-newton appearance parameters updates
                masked_t = self.template.as_vector()[
                    self.interface.image_vec_mask]
                dc = self._pinv_U.T.dot(masked_i - masked_t + j.dot(dp))
                c += dc

            # reconstruct appearance
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = masked_m - masked_i

            # compute model gradient
            nabla_t = self.interface.gradient(self.template)

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_t, self._dw_dp)
            # project out appearance model from model jacobian
            j_po = self.project_out(j)

            # compute hessian
            h = j_po.T.dot(j)

            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, j_po, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(
                target.points - self.transform.target.points))
            if error < self.eps:
                break

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class FastSIC_N(FastSimultaneous):

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

        # compute U jacobian
        n_pixels = len(self.template.as_vector()[
            self.interface.image_vec_mask])
        self._j_U = np.zeros((self.appearance_model.n_active_components,
                              n_pixels, self.transform.n_parameters))
        for k, u in enumerate(self._U.T):
            nabla_u = self.interface.gradient(Image(u.reshape(
                self.template.pixels.shape)))
            j_u = self.interface.steepest_descent_images(nabla_u, self._dw_dp)
            self._j_U[k, ...] = j_u

        # compute U inverse hessian
        self._inv_h_U = np.linalg.inv(self._masked_U.T.dot(self._masked_U))

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

         # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # warp image
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            if _ == 0:
                # project image onto the model bases
                c = self._pinv_U.T.dot(masked_i - masked_m)
            else:
                # compute gauss-newton appearance parameters updates
                masked_e = (masked_i -
                            self.template.as_vector()[
                                self.interface.image_vec_mask])
                dc = (self._pinv_U.T.dot(masked_e - j.dot(dp)) -
                      masked_e.dot(self._j_U).dot(dp))
                c += dc

            # reconstruct appearance
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = masked_i - masked_m

            # compute model gradient
            nabla_t = self.interface.gradient(self.template)
            # compute model second order gradient
            nabla2_t = self.interface.gradient(Image(nabla_t))

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_t, self._dw_dp)
            # project out appearance model from model jacobian
            j_po = self.project_out(j)

            # compute gauss-newton hessian
            h_gn = j_po.T.dot(j)
            # compute partial newton hessian
            h_pn = self.interface.partial_newton_hessian(nabla2_t, self._dw_dp)
            # project out appearance model from error
            e_po = self.project_out(e)
            # compute cp hessian
            h_cp = self._pinv_U.T.dot(j_po) + e_po.dot(self._j_U)

            # compute full newton hessian
            h = (e_po.dot(h_pn).reshape(h_gn.shape) + h_gn -
                 h_cp.T.dot(self._inv_h_U.dot(h_cp)))
            # compute full newton jacobian
            j = - j_po + self._pinv_U.dot(h_cp)

            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, j, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(target.points -
                                          self.transform.target.points))
            if error < self.eps:
                break

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class FastSSC_GN(FastSimultaneous):

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False, a=0.5):

        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # warp image
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            if _ == 0:
                # project image onto the model bases
                c = self._pinv_U.T.dot(masked_i - masked_m)
            else:
                # compute gauss-newton appearance parameters updates
                masked_t = self.template.as_vector()[
                    self.interface.image_vec_mask]
                dc = self._pinv_U.T.dot(masked_i - masked_t + j.dot(2 * dp))
                c += dc

            # reconstruct appearance
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = masked_m - masked_i

            # combine image and model gradient
            nabla = (a * self.interface.gradient(i) +
                     (1-a) * self.interface.gradient(self.template))

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla, self._dw_dp)
            # project out appearance model from model jacobian
            j_po = self.project_out(j)

            # compute hessian
            h = j_po.T.dot(j)

            # compute symmetric esm parameter updates
            dp = self.interface.solve(h, j_po, e, prior)

            # update transform
            target = self.transform.target
            dt = self.transform.from_vector(a * dp)
            dt.from_vector_inplace(dt.as_vector() + (1-a) * dp)
            self.transform.from_vector_inplace(
                self.transform.as_vector() + dt.as_vector())
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(
                target.points - self.transform.target.points))
            if error < self.eps:
                break

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class FastSBC_GN(FastSimultaneous):

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        n_shape_params = self.transform.n_parameters
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # warp image
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            if _ == 0:
                # project image onto the model bases
                c = self._pinv_U.T.dot(masked_i - masked_m)
            else:
                # compute gauss-newton appearance parameters updates
                masked_t = self.template.as_vector()[
                    self.interface.image_vec_mask]
                dc = self._pinv_U.T.dot(masked_i - masked_t + j.dot(dp))
                c += dc

            # reconstruct appearance
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = masked_m - masked_i

           # combine image and model gradient
            nabla_i = self.interface.gradient(i)
            nabla_t = self.interface.gradient(self.template)

            # compute model jacobian
            # compute jacobian
            j_i = self.interface.steepest_descent_images(nabla_i, self._dw_dp)
            j_t = self.interface.steepest_descent_images(nabla_t, self._dw_dp)
            j = np.hstack((j_i, j_t))
            # project out appearance model from model jacobian
            j_po = self.project_out(j)

            # compute hessian
            h = j_po.T.dot(j)

            # compute symmetric esm parameter updates
            dp = self.interface.solve(h, j_po, e, prior)

            # update transform
            target = self.transform.target
            dt = self.transform.from_vector(dp[n_shape_params:])
            dt.from_vector_inplace(dt.as_vector() + dp[:n_shape_params])
            self.transform.from_vector_inplace(
                self.transform.as_vector() + dt.as_vector())
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(
                target.points - self.transform.target.points))
            if error < self.eps:
                break

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


# Alternating Compositional Algorithms ----------------------------------------

class AIC_SD(Alternating):
    r"""
    Alternating Inverse Compositional Steepest-Descent Algorithm
    """


class AIC_GN(Alternating):
    r"""
    Alternating Inverse Compositional Gauss-Newton Algorithm
    """

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # warp image
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # reconstruct appearance
            c = self._pinv_U.T.dot(masked_i - masked_m)
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = (self.template.as_vector()[self.interface.image_vec_mask] -
                 masked_i)

            # compute model gradient
            nabla_t = self.interface.gradient(self.template)

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_t, self._dw_dp)

            # compute hessian
            h = j.T.dot(j)

            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, j, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(
                target.points - self.transform.target.points))
            if error < self.eps:
                break

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class AIC_N(Alternating):
    r"""
    Alternating Inverse Compositional Newton Algorithm
    """

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            i = i.as_vector()[self.interface.image_vec_mask]
            c = self._pinv_U.T.dot(i - masked_m)
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = self.template.as_vector()[self.interface.image_vec_mask] - i

            # compute model gradient
            nabla_t = self.interface.gradient(self.template)
            # compute model second order gradient
            nabla2_t = self.interface.gradient(Image(nabla_t))

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_t, self._dw_dp)

            # compute gauss-newton hessian
            h_gn = j.T.dot(j)
            # compute partial newton hessian
            h_pn = self.interface.partial_newton_hessian(nabla2_t, self._dw_dp)
            # compute full newton hessian
            h = e.dot(h_pn).reshape(h_gn.shape) + h_gn

            # compute newton parameter updates
            dp = self.interface.solve(h, j, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(
                target.points - self.transform.target.points))
            if error < self.eps:
                break

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class ASC_GN(Alternating):
    r"""
    Alternating Symmetric Compositional ESM Algorithm
    """

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False, a=0.5):

        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.image_vec_mask]
            c = self._pinv_U.T.dot(masked_i - masked_m)
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = (self.template.as_vector()[self.interface.image_vec_mask] -
                 masked_i)

            # combine image and model gradient
            nabla = (a * self.interface.gradient(i) +
                     (1-a) * self.interface.gradient(self.template))

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla, self._dw_dp)

            # compute hessian
            h = j.T.dot(j)

            # compute symmetric esm parameter updates
            dp = self.interface.solve(h, j, e, prior)

            # update transform
            target = self.transform.target
            dt = self.transform.from_vector(a * dp)
            dt.from_vector_inplace(dt.as_vector() + (1 - a) * dp)
            self.transform.from_vector_inplace(
                self.transform.as_vector() + dt.as_vector())
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(
                target.points - self.transform.target.points))
            if error < self.eps:
                break

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class ABC_GN(Alternating):
    r"""
    Alternating Bidirectional Compositional ESM Algorithm
    """

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        n_shape_params = self.transform.n_parameters
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.image_vec_mask]
            c = self._pinv_U.T.dot(masked_i - masked_m)
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = (self.template.as_vector()[self.interface.image_vec_mask] -
                 masked_i)

            # combine image and model gradient
            nabla_i = self.interface.gradient(i)
            nabla_t = self.interface.gradient(self.template)

            # compute model jacobian
            # compute jacobian
            j_i = self.interface.steepest_descent_images(nabla_i, self._dw_dp)
            j_t = self.interface.steepest_descent_images(nabla_t, self._dw_dp)
            j = np.hstack((j_i, j_t))

            # compute hessian
            h = j.T.dot(j)

            # compute symmetric esm parameter updates
            dp = self.interface.solve(h, j, e, prior)

            # update transform
            target = self.transform.target
            dt = self.transform.from_vector(dp[:n_shape_params])
            dt.from_vector_inplace(dt.as_vector() + dp[n_shape_params:])
            self.transform.from_vector_inplace(
                self.transform.as_vector() + dt.as_vector())
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(
                target.points - self.transform.target.points))
            if error < self.eps:
                break

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


# Bayesian Compositional Algorithms -------------------------------------------

class BIC_GN(Bayesian):
    r"""
    Project-Out Inverse Compositional Gauss-Newton Algorithm
    """

    def _precompute(self):

        # compute model's gradient
        nabla_t = self.interface.gradient(self.template)

        # compute warp jacobian
        dw_dp = self.interface.dw_dp()

        # compute steepest descent images
        j = self.interface.steepest_descent_images(nabla_t, dw_dp)

        # project out appearance model from J
        self._j_po = self.project_out(j)

        # compute inverse hessian
        self._h = self._j_po.T.dot(j)

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # masked model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # compute error image
            e = masked_m - masked_i

            # compute gauss-newton parameter updates
            dp = self.interface.solve(self._h, self._j_po, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(
                target.points - self.transform.target.points))
            if error < self.eps:
                break

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters,
                                               gt_shape=gt_shape)


class BIC_N(Bayesian):
    r"""
    Project-Out Inverse Compositional Newton Algorithm
    """

    def _precompute(self):

        # compute model gradient
        nabla_t = self.interface.gradient(self.template)

        # compute model second gradient
        nabla2_t = self.interface.gradient(Image(nabla_t))

        # compute warp jacobian
        dw_dp = self.interface.dw_dp()

        # compute steepest descent images
        j = self.interface.steepest_descent_images(nabla_t, dw_dp)

        # project out appearance model from J
        self._j_po = self.project_out(j)

        # compute gauss-newton hessian
        self._h_gn = self._j_po.T.dot(j)

        # compute newton hessian
        self._h_pn = self.interface.partial_newton_hessian(nabla2_t, dw_dp)

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

         # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # masked model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # compute error image
            e = masked_m - masked_i

            # project out appearance model from error
            e_po = self.project_out(e)
            # compute full newton hessian
            h = e_po.dot(self._h_pn).reshape(self._h_gn.shape) + self._h_gn

            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, self._j_po, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(target.points -
                                          self.transform.target.points))
            if error < self.eps:
                break

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters,
                                               gt_shape=gt_shape)


class BIC2_GN(Bayesian):
    r"""
    Fast Simultaneous Inverse Compositional Gauss-Newton Algorithm
    """

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # warp image
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            if _ == 0:
                # project image onto the model bases
                c = self._pinv_U.T.dot(masked_i - masked_m)
            else:
                # compute gauss-newton appearance parameters updates
                masked_t = self.template.as_vector()[
                    self.interface.image_vec_mask]
                dc = self._pinv_U.T.dot(masked_i - masked_t + j.dot(dp))
                c += dc

            # reconstruct appearance
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = masked_m - masked_i

            # compute model gradient
            nabla_t = self.interface.gradient(self.template)

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_t, self._dw_dp)
            # project out appearance model from model jacobian
            j_po = self.project_out(j)

            # compute hessian
            h = j_po.T.dot(j)

            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, j_po, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(
                target.points - self.transform.target.points))
            if error < self.eps:
                break

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class BFC_SD(Bayesian):
    r"""
    Bayesian Forward Compositional Steepest-Descent Algorithm
    """


class BFC_GN(Bayesian):
    r"""
    Bayesian Forward Compositional Gauss-Newton Algorithm
    """

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False, l=0.5):

        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # masked model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # compute error image
            e = masked_m - masked_i

            # compute image gradient
            nabla_i = self.interface.gradient(i)

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_i, self._dw_dp)
            # project out appearance model from model jacobian
            j_po = self.project_out(j, l)

            # compute hessian
            h = j_po.T.dot(j)

            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, j_po, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(
                target.points - self.transform.target.points))
            if error < self.eps:
                break

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters,
                                               gt_shape=gt_shape)


class BFC_N(Bayesian):
    r"""
    Bayesian Forward Compositional Gauss-Newton Algorithm
    """

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False, l=0.5):

        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # mask model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # compute error image
            e = masked_m - masked_i

            # compute image gradient
            nabla_i = self.interface.gradient(i)
            # compute image second order gradient
            nabla2_i = self.interface.gradient(Image(nabla_i))

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_i, self._dw_dp)
            # project out appearance model from model jacobian
            j_po = self.project_out(j, l)

            # compute gauss-newton hessian
            h_gn = j_po.T.dot(j)
            # compute partial newton hessian
            h_pn = self.interface.partial_newton_hessian(nabla2_i, self._dw_dp)
            # project out appearance model from error
            e_po = self.project_out(e[..., None], l)[..., 0]
            # compute full newton hessian
            h = e_po.dot(h_pn).reshape(h_gn.shape) + h_gn

            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, j, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(target.points -
                                          self.transform.target.points))
            if error < self.eps:
                break

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters,
                                               gt_shape=gt_shape)


class BSC_GN(Bayesian):
    r"""
    Bayesian Symmetric Compositional ESM Algorithm
    """

    def _precompute(self):

        # compute model gradient
        self._nabla_t = self.interface.gradient(self.template)

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False, l=0.5, a=0.5):

        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # masked model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # compute error image
            e = masked_m - masked_i

            # combine image and model gradient
            nabla = a * self.interface.gradient(i) + (1-a) * self._nabla_t

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla, self._dw_dp)
            # project out appearance model from model jacobian
            j_po = self.project_out(j, l)

            # compute hessian
            h = j_po.T.dot(j)

            # compute symmetric esm parameter updates
            dp = self.interface.solve(h, j_po, e, prior)

            # update transform
            target = self.transform.target
            dt = self.transform.from_vector(a * dp)
            dt.from_vector_inplace(dt.as_vector() + (1-a) * dp)
            self.transform.from_vector_inplace(
                self.transform.as_vector() + dt.as_vector())
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(
                target.points - self.transform.target.points))
            if error < self.eps:
                break

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters,
                                               gt_shape=gt_shape)


class BBC_GN(Bayesian):
    r"""
    Bayesian Bidirectional Compositional ESM Algorithm
    """

    def _precompute(self):

        # compute model gradient
        self._nabla_t = self.interface.gradient(self.template)

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False, l=0.5):

        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        n_shape_params = self.transform.n_parameters
        # masked model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # compute error image
            e = masked_m - masked_i

            # compute image gradient
            nabla_i = self.interface.gradient(i)

            # compute jacobian
            j_i = self.interface.steepest_descent_images(nabla_i, self._dw_dp)
            j_t = self.interface.steepest_descent_images(self._nabla_t,
                                                         self._dw_dp)
            j = np.hstack((j_i, j_t))
            # project out appearance model from  jacobian
            j_po = self.project_out(j, l)

            # compute hessian
            h = j_po.T.dot(j)

             # compute symmetric esm parameter updates
            dp = self.interface.solve(h, j_po, e, prior)

            # update transform
            target = self.transform.target
            dt = self.transform.from_vector(dp[:n_shape_params])
            dt.from_vector_inplace(dt.as_vector() + dp[n_shape_params:])
            self.transform.from_vector_inplace(
                self.transform.as_vector() + dt.as_vector())
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(
                target.points - self.transform.target.points))
            if error < self.eps:
                break

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters,
                                               gt_shape=gt_shape)
