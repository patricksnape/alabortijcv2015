from __future__ import division
import abc

import numpy as np
import scipy

from menpo.image import Image
from menpo.feature import gradient as fast_gradient

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
    def solve(self, h, j, e, prior, fixed_p):
        pass

    @abc.abstractmethod
    def algorithm_result(self, image, shape_parameters, lin_cost_functions,
                         appearance_parameters=None, gt_shape=None):
        pass


class StandardAAMInterface(AAMInterface):

    def __init__(self, aam_algorithm, sampling_step=None):
        super(StandardAAMInterface, self). __init__(aam_algorithm)

        self.template = self.algorithm.template
        self.n_true_pixels = self.template.n_true_pixels()
        self.n_channels = self.template.n_channels

        self.transform = self.algorithm.transform
        self.n_params = self.transform.n_parameters
        self.n_dims = self.transform.n_dims
        self.noise_variance = self.transform.pdm.model.noise_variance()

        if sampling_step is None:
            sampling_step = 1
        sampling_mask = np.zeros(self.n_true_pixels, dtype=np.bool)
        sampling_pattern = xrange(0, self.n_true_pixels, sampling_step)
        sampling_mask[sampling_pattern] = 1
        self.i_mask = np.nonzero(np.tile(
            sampling_mask[None, ...], (self.n_channels, 1)).flatten())[0]
        self.dw_dp_mask = np.nonzero(np.tile(
            sampling_mask[None, ..., None], (2, 1, self.n_params)))
        self.g_mask = np.nonzero(np.tile(
            sampling_mask[None, None, ...], (2, self.n_channels, 1)))
        self.g2_mask = np.nonzero(np.tile(
            sampling_mask[None, None, None, ...], (2, 2, self.n_channels, 1)))

    def dw_dp(self):
        dw_dp = self.transform.d_dp(self.template.mask.true_indices())
        dw_dp = np.rollaxis(dw_dp, -1)[self.dw_dp_mask]
        return dw_dp.reshape((self.n_dims, -1, self.n_params))

    def warp(self, img):
        return img.warp_to_mask(self.template.mask, self.transform)

    def gradient(self, image):
        g = fast_gradient(image)
        g.set_boundary_pixels()
        g = g.as_vector()
        return g.reshape((self.n_dims, self.n_channels, -1))

    def steepest_descent_images(self, g, dw_dp):
        # reshape gradient
        # g: n_dims x n_channels x n_pixels
        g = g[self.g_mask]
        g = g.reshape((self.n_dims, self.n_channels, -1))

        # compute steepest descent images
        # g:     n_dims x n_channels x n_pixels
        # dw_dp: n_dims x            x n_pixels x n_params
        # sdi:            n_channels x n_pixels x n_params
        sdi = 0
        a = g[..., None] * dw_dp[:, None, ...]
        for d in a:
            sdi += d

        # reshape steepest descent images
        # sdi: (n_channels x n_pixels) x n_params
        return sdi.reshape((-1, self.n_params))

    def partial_newton_hessian(self, g2, dw_dp):
        # reshape second order gradient
        # g2: n_dims x n_dims x n_channels x n_pixels
        g2 = g2[self.g2_mask]
        g2 = g2.reshape((self.n_dims, self.n_dims, self.n_channels, -1))

        # compute partial hessian
        # g2:    n_dims x n_dims x n_channels x n_pixels
        # dw_dp: n_dims x                     x n_pixels x n_params
        # h1:             n_dims x n_channels x n_pixels x n_params
        h1 = 0
        aux = g2[..., None] * dw_dp[:, None, None, ...]
        for d in aux:
            h1 += d
        # compute partial hessian
        # h1:     n_dims x n_channels x n_pixels x n_params
        # dw_dp:  n_dims x            x n_pixels x          x n_params
        # h2:              n_channels x n_pixels x n_params x n_params
        h2 = 0
        aux = h1[..., None] * dw_dp[..., None, :, None, :]
        for d in aux:
            h2 += d

        # reshape hessian
        # h2:  (n_channels x n_pixels) x (n_params x n_params)
        return h2.reshape((-1, self.n_params**2))

    def solve(self, h, j, e, prior, fixed_p):
        t = self.algorithm.transform
        jp = np.asarray(t.jp()[fixed_p, fixed_p])

        if prior:
            inv_h = np.linalg.inv(h)
            dp = inv_h.dot(j.T.dot(e))
            dp = -np.linalg.solve(t.h_prior + jp.dot(inv_h.dot(jp.T)),
                                  t.j_prior * t.as_vector() - jp.dot(dp))
        else:
            1
            # if isinstance(h, np.ndarray):
            #     dp = np.linalg.solve(h, j.T.dot(e))
            # else:
            #     dp = j.T.dot(e) / h
            #
            # if not isinstance(dp, np.ndarray) or jp.shape[0] is dp.shape[0]:
            #     dp = jp.dot(dp)
            # else:
            #     dp = scipy.linalg.block_diag(jp, jp).dot(dp)

        dp = np.linalg.solve(h, j.T.dot(e))

        return dp

    def algorithm_result(self, image, shape_params, linearized_costs,
                         appearance_params=None, gt_shape=None):
        return AAMAlgorithmResult(
            image, self.algorithm, shape_params, linearized_costs,
            appearance_params=appearance_params, gt_shape=gt_shape)


class LinearAAMInterface(StandardAAMInterface):

    def solve(self, h, j, e, prior):
        t = self.algorithm.transform

        if prior:
            dp = -np.linalg.solve(t.h_prior + h,
                                  t.j_prior * t.as_vector() - j.T.dot(e))
        else:
            dp = np.linalg.solve(h, j.T.dot(e))

        return dp

    def algorithm_result(self, image, shape_params,
                         appearance_params=None, gt_shape=None):
        return LinearAAMAlgorithmResult(
            image, self.algorithm, shape_params,
            appearance_params=appearance_params, gt_shape=gt_shape)


class PartsAAMInterface(AAMInterface):

    def __init__(self, aam_algorithm, sampling_mask=None):
        super(PartsAAMInterface, self). __init__(aam_algorithm)

        if sampling_mask is None:
            parts_shape = self.algorithm.appearance_model.parts_shape
            sampling_mask = np.require(np.ones((parts_shape)), dtype=np.bool)

        image_shape = self.algorithm.template.pixels.shape
        image_mask = np.tile(sampling_mask[None, None, None, ...],
                             image_shape[:3] + (1, 1))
        self.i_mask = np.nonzero(image_mask.flatten())[0]
        self.gradient_mask = np.nonzero(np.tile(
            image_mask[None, ...], (2, 1, 1, 1, 1, 1)))
        self.gradient2_mask = np.nonzero(np.tile(
            image_mask[None, None, ...], (2, 2, 1, 1, 1, 1, 1)))

    def dw_dp(self):
        return np.rollaxis(self.algorithm.transform.d_dp(None), -1)

    def warp(self, image):
        return Image(image.extract_patches(
            self.algorithm.transform.target,
            patch_size=self.algorithm.appearance_model.parts_shape,
            as_single_array=True))

    def gradient(self, image):
        pixels = image.pixels
        parts_shape = self.algorithm.appearance_model.parts_shape
        g = fast_gradient(pixels.reshape((-1,) + parts_shape))
        # remove 1st dimension gradient which corresponds to the gradient
        # between parts
        return g.reshape((2,) + pixels.shape)

    def steepest_descent_images(self, gradient, dw_dp):
        # reshape gradient
        # gradient: dims x parts x off x ch x (h x w)
        gradient = gradient[self.gradient_mask].reshape(
            gradient.shape[:-2] + (-1,))
        # compute steepest descent images
        # gradient: dims x parts x off x ch x (h x w)
        # ds_dp:    dims x parts x                             x params
        # sdi:             parts x off x ch x (h x w) x params
        sdi = 0
        a = gradient[..., None] * dw_dp[..., None, None, None, :]
        for d in a:
            sdi += d

        # reshape steepest descent images
        # sdi: (parts x offsets x ch x w x h) x params
        return sdi.reshape((-1, sdi.shape[-1]))

    def partial_newton_hessian(self, gradient2, dw_dp):
        # reshape gradient
        # gradient: dims x dims x parts x off x ch x (h x w)
        gradient2 = gradient2[self.gradient2_mask].reshape(
            gradient2.shape[:-2] + (-1,))

        # compute partial hessian
        # gradient: dims x dims x parts x off x ch x (h x w)
        # dw_dp:    dims x      x parts x                    x params
        # h:               dims x parts x off x ch x (h x w) x params
        h1 = 0
        aux = gradient2[..., None] * dw_dp[:, None, :, None, None, None, ...]
        for d in aux:
            h1 += d
        # compute partial hessian
        # h:     dims x parts x off x ch x (h x w) x params
        # dw_dp: dims x parts x                             x params
        # h:
        h2 = 0
        aux = h1[..., None] * dw_dp[..., None, None, None, None, :]
        for d in aux:
            h2 += d

        # reshape hessian
        # 2:  (parts x off x ch x w x h) x params x params
        return h2.reshape((-1, h2.shape[-2] * h2.shape[-1]))

    def solve(self, h, j, e, prior):
        t = self.algorithm.transform

        if prior:
            dp = -np.linalg.solve(t.h_prior + h,
                                  t.j_prior * t.as_vector() - j.T.dot(e))
        else:
            dp = np.linalg.solve(h, j.T.dot(e))

        return dp

    def algorithm_result(self, image, shape_params,
                         appearance_params=None, gt_shape=None):
        return AAMAlgorithmResult(
            image, self.algorithm, shape_params,
            appearance_params=appearance_params, gt_shape=gt_shape)


# Abstract Interfaces for AAM Algorithms  -------------------------------------

class AAMAlgorithm(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, aam_interface, appearance_model, transform,
                 eps=None, **kwargs):

        # set common state for all AAM algorithms
        self.appearance_model = appearance_model
        self.template = appearance_model.mean()
        self.transform = transform

        # set interface
        self.interface = aam_interface(self, **kwargs)

        self.eps = eps if eps else self.interface.noise_variance

        self._U = self.appearance_model.components.T
        self._pinv_U = np.linalg.pinv(
            self._U[self.interface.i_mask, :]).T

        # extract appearance model mean
        self.m = self.appearance_model.mean().as_vector()
        # masked appearance model mean
        self.masked_m = self.m[self.interface.i_mask]


    @abc.abstractmethod
    def _precompute(self, **kwargs):
        pass

    @abc.abstractmethod
    def run(self, image, initial_shape, max_iters=20, gt_shape=None, **kwargs):
        pass


class ProjectOut(AAMAlgorithm):

    def __init__(self, aam_interface, appearance_model, transform,
                 eps=None, **kwargs):
        # call super constructor
        super(ProjectOut, self).__init__(
            aam_interface, appearance_model, transform, eps, **kwargs)

        # set common state for all Project Out AAM algorithms
        self._masked_U = self._U[self.interface.i_mask, :]

        # pre-compute
        self._precompute()
        
    def project_out(self, j):
        return j - self._masked_U.dot(self._pinv_U.T.dot(j))


class Simultaneous(AAMAlgorithm):

    def __init__(self, aam_interface, appearance_model, transform,
                 eps=None, **kwargs):
        # call super constructor
        super(Simultaneous, self).__init__(
            aam_interface, appearance_model, transform, eps, **kwargs)

        # set common state for all Fast Simultaneous AAM algorithms
        self._masked_U = self._U[self.interface.i_mask, :]

        # pre-compute
        self._precompute()

    def project_out(self, j):
        return j - self._masked_U.dot(self._pinv_U.T.dot(j))


class Alternating(AAMAlgorithm):

    def __init__(self, aam_interface, appearance_model, transform,
                 eps=None, **kwargs):
        # call super constructor
        super(Alternating, self).__init__(
            aam_interface, appearance_model, transform, eps, **kwargs)

        # pre-compute
        self._precompute()


class Bayesian(AAMAlgorithm):

    def __init__(self, aam_interface, appearance_model, transform,
                 eps=None, **kwargs):

        # call super constructor
        super(Bayesian, self).__init__(
            aam_interface, appearance_model, transform, eps, **kwargs)

        # set common state for all Bayesian AAM algorithms
        self._U = self._U[self.interface.i_mask, :]
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

class PIC(ProjectOut):
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

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # masked model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.i_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.i_mask]

            # compute error image
            e = masked_m - masked_i

            # compute gauss-newton parameter updates
            dp = self.interface.solve(self._h, self._j_po, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(self.project_out(e)))

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters, cost,
                                               gt_shape=gt_shape)


class PICN(ProjectOut):
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

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # masked model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.i_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.i_mask]

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
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e_po))

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters, cost,
                                               gt_shape=gt_shape)


class PFC(ProjectOut):
    r"""
    Project-Out Forward Compositional Gauss-Newton Algorithm
    """

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # masked model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.i_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.i_mask]

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
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(self.project_out(e)))

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters, cost,
                                               gt_shape=gt_shape)


class PFCN(ProjectOut):
    r"""
    Project-Out Forward Compositional Newton Algorithm
    """

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # mask model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.i_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.i_mask]

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
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e_po))

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters, cost,
                                               gt_shape=gt_shape)


class PSC(ProjectOut):
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

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # masked model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.i_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.i_mask]

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
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(self.project_out(e)))

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters, cost,
                                               gt_shape=gt_shape)


class PBC(ProjectOut):
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

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        n_shape_params = self.transform.n_parameters
        # masked model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.i_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.i_mask]

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
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(self.project_out(e)))

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters, cost,
                                               gt_shape=gt_shape)


# Simultaneous Compositional Algorithms ----------------------------------

class SIC(Simultaneous):
    r"""
    Fast Simultaneous Inverse Compositional Gauss-Newton Algorithm
    """
    def _residual(self, masked_i):
        return masked_i - self.masked_m

    def _cost(self, r):
        return r.dot(r) / r.shape[0]

    def _shape_cost(self, r):
        r_po = self.project_out(r)
        return r.dot(r_po) / r.shape[0]

    def _precompute(self):
        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False, fixed_p=None):
        # initialize transform
        self.transform.set_target(initial_shape)

        # initialize lists
        shape_params = [self.transform.as_vector()]
        appearance_params = []
        residuals = []
        costs = []
        jacobians = []

        for n_iter in xrange(max_iters+1):
            # warp image
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.i_mask]

            if n_iter == 0:
                # project image onto the model bases
                c = self._pinv_U.T.dot(masked_i - self.masked_m)
            else:
                # compute gauss-newton appearance parameters updates
                masked_t = self.template.as_vector()[self.interface.i_mask]
                dc = self._pinv_U.T.dot(masked_i - masked_t + j.dot(dp))
                c += dc
            # reconstruct appearance
            t = self._U.dot(c) + self.m
            self.template.from_vector_inplace(t)
            # compute error image
            e = self.masked_m - masked_i

            # compute model gradient
            nabla_t = self.interface.gradient(self.template)
            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_t, self._dw_dp)
            if fixed_p:
                j = j[:, fixed_p]
            # project out appearance model from model jacobian
            j_po = self.project_out(j)
            # compute hessian
            h = j_po.T.dot(j)
            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, j_po, e, prior, fixed_p)

            # update transform
            if fixed_p:
                delta_p = np.zeros(self.transform.n_parameters)
                delta_p[fixed_p] = dp
            else:
                delta_p = dp
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() +
                                               delta_p)

            # update lists
            appearance_params.append(c)
            shape_params.append(self.transform.as_vector())
            residuals.append(self._residual(masked_i))
            costs.append(self._shape_cost(residuals[-1]))
            jacobians.append(j)

            # test convergence
            shape_norm = np.sum((target.as_vector() -
                                 self.transform.target.as_vector()
                                 )**2) / target.n_parameters
            if shape_norm < self.eps:
                break

        # define linearized cost functions
        def linearized_costs(k, x):
            lin_r = residuals[k] - jacobians[k].dot(x)
            return self._shape_cost(lin_r)

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_params, linearized_costs,
            appearance_params=appearance_params, gt_shape=gt_shape)


class SICN(Simultaneous):

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

        # compute U jacobian
        n_pixels = len(self.template.as_vector()[
            self.interface.i_mask])
        self._j_U = np.zeros((self.appearance_model.n_active_components,
                              n_pixels, self.transform.n_parameters))
        for k, u in enumerate(self._U.T):
            self.template2.from_vector_inplace(u)
            nabla_u = self.interface.gradient(self.template2)
            j_u = self.interface.steepest_descent_images(nabla_u, self._dw_dp)
            self._j_U[k, ...] = j_u

        # compute U inverse hessian
        self._inv_h_U = np.linalg.inv(self._masked_U.T.dot(self._masked_U))

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.i_mask]

        for _ in xrange(max_iters):

            # warp image
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.i_mask]

            if _ == 0:
                # project image onto the model bases
                c = self._pinv_U.T.dot(masked_i - masked_m)
            else:
                # compute gauss-newton appearance parameters updates
                masked_e = (masked_i -
                            self.template.as_vector()[
                                self.interface.i_mask])
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
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e))

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class SFC(Simultaneous):
    r"""
    Fast Simultaneous Inverse Compositional Gauss-Newton Algorithm
    """

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.i_mask]

        for _ in xrange(max_iters):

            # warp image
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.i_mask]

            if _ == 0:
                # project image onto the model bases
                c = self._pinv_U.T.dot(masked_i - masked_m)
            else:
                # compute gauss-newton appearance parameters updates
                masked_t = self.template.as_vector()[
                    self.interface.i_mask]
                dc = self._pinv_U.T.dot(masked_i - masked_t + j.dot(dp))
                c += dc

            # reconstruct appearance
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = masked_m - masked_i

            # compute model gradient
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
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e))

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class SFCN(Simultaneous):

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

        # compute U jacobian
        n_pixels = len(self.template.as_vector()[
            self.interface.i_mask])
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

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.i_mask]

        for _ in xrange(max_iters):

            # warp image
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.i_mask]

            if _ == 0:
                # project image onto the model bases
                c = self._pinv_U.T.dot(masked_i - masked_m)
            else:
                # compute gauss-newton appearance parameters updates
                masked_e = (masked_i -
                            self.template.as_vector()[
                                self.interface.i_mask])
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
            nabla_i = self.interface.gradient(i)
            # compute model second order gradient
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
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e))

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class SSC(Simultaneous):

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False, a=0.5):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.i_mask]

        for _ in xrange(max_iters):

            # warp image
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.i_mask]

            if _ == 0:
                # project image onto the model bases
                c = self._pinv_U.T.dot(masked_i - masked_m)
            else:
                # compute gauss-newton appearance parameters updates
                masked_t = self.template.as_vector()[
                    self.interface.i_mask]
                dc = self._pinv_U.T.dot(masked_i - masked_t + j.dot(dp))
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
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e))

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class SBC(Simultaneous):

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        n_shape_params = self.transform.n_parameters
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.i_mask]

        for _ in xrange(max_iters):

            # warp image
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.i_mask]

            if _ == 0:
                # project image onto the model bases
                c = self._pinv_U.T.dot(masked_i - masked_m)
            else:
                # compute gauss-newton appearance parameters updates
                masked_t = self.template.as_vector()[
                    self.interface.i_mask]
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
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e))

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


# Alternating Compositional Algorithms ----------------------------------------

class AIC(Alternating):
    r"""
    Alternating Inverse Compositional Gauss-Newton Algorithm
    """

    def _residual(self, masked_i, masked_t):
        return masked_i - masked_t

    def _cost(self, r):
        return r.T.dot(r) / r.shape[0]

    def _precompute(self):
        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False, fixed_p=None):
        # initialize transform
        self.transform.set_target(initial_shape)

        # initialize lists
        shape_params = [self.transform.as_vector()]
        appearance_params = []
        residuals = []
        costs = []
        jacobians = []

        for n_iters in xrange(max_iters):

            # warp image
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.i_mask]

            # reconstruct appearance
            c = self._pinv_U.T.dot(masked_i - self.masked_m)
            t = self._U.dot(c) + self.m
            self.template.from_vector_inplace(t)
            # mask reconstruction
            masked_t = self.template.as_vector()[self.interface.i_mask]
            # compute error image
            e = masked_t - masked_i

            # compute model gradient
            nabla_t = self.interface.gradient(self.template)
            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_t, self._dw_dp)
            if fixed_p:
                j = j[:, fixed_p]
            # compute hessian
            h = j.T.dot(j)
            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, j, e, prior, fixed_p)

            # update transform
            if fixed_p:
                delta_p = np.zeros(self.transform.n_parameters)
                delta_p[fixed_p] = dp
                dp = delta_p
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)

            # update lists
            appearance_params.append(c)
            shape_params.append(self.transform.as_vector())
            residuals.append(self._residual(masked_i, t))
            costs.append(self._cost(residuals[-1]))
            jacobians.append(j)

            # test convergence
            shape_norm = np.sum((target.as_vector() -
                                 self.transform.target.as_vector()
                                 )**2) / target.n_parameters
            if shape_norm < self.eps:
                break

        # define linearized cost functions
        def linearized_costs(k):
            def lin_f(x):
                lin_r = residuals[k] - jacobians[k].dot(x)
                return self._cost(lin_r)
            return lin_f

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_params, linearized_costs,
            appearance_params=appearance_params, gt_shape=gt_shape)


class AICN(Alternating):
    r"""
    Alternating Inverse Compositional Newton Algorithm
    """

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.i_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            i = i.as_vector()[self.interface.i_mask]
            c = self._pinv_U.T.dot(i - masked_m)
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = self.template.as_vector()[self.interface.i_mask] - i

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
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e))

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class AFC(Alternating):
    r"""
    Alternating Inverse Compositional Gauss-Newton Algorithm
    """

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.i_mask]

        for _ in xrange(max_iters):

            # warp image
            i = self.interface.warp(image)
            
            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.i_mask]
            c = self._pinv_U.T.dot(masked_i - masked_m)
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = (self.template.as_vector()[self.interface.i_mask] -
                 masked_i)

            # compute model gradient
            nabla_i = self.interface.gradient(i)

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_i, self._dw_dp)

            # compute hessian
            h = j.T.dot(j)

            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, j, e, prior)

            # update transform
            target = self.transform.target
            dt = self.transform.from_vector(dp)
            self.transform.from_vector_inplace(
                self.transform.as_vector() + dt.as_vector())
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e))

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class AFCN(Alternating):
    r"""
    Alternating Inverse Compositional Newton Algorithm
    """

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.i_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.i_mask]
            c = self._pinv_U.T.dot(masked_i - masked_m)
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = (self.template.as_vector()[self.interface.i_mask] -
                 masked_i)

            # compute model gradient
            nabla_i = self.interface.gradient(i)
            # compute model second order gradient
            nabla2_i = self.interface.gradient(Image(nabla_i))

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_i, self._dw_dp)

            # compute gauss-newton hessian
            h_gn = j.T.dot(j)
            # compute partial newton hessian
            h_pn = self.interface.partial_newton_hessian(nabla2_i, self._dw_dp)
            # compute full newton hessian
            h = e.dot(h_pn).reshape(h_gn.shape) + h_gn

            # compute newton parameter updates
            dp = self.interface.solve(h, j, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e))

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class ASC(Alternating):
    r"""
    Alternating Symmetric Compositional ESM Algorithm
    """

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False, a=0.5):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.i_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.i_mask]
            c = self._pinv_U.T.dot(masked_i - masked_m)
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = (self.template.as_vector()[self.interface.i_mask] -
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
            dt.from_vector_inplace(dt.as_vector() + (1-a) * dp)
            self.transform.from_vector_inplace(
                self.transform.as_vector() + dt.as_vector())
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e))

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class ABC(Alternating):
    r"""
    Alternating Bidirectional Compositional ESM Algorithm
    """

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        n_shape_params = self.transform.n_parameters
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.i_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.i_mask]
            c = self._pinv_U.T.dot(masked_i - masked_m)
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = (self.template.as_vector()[self.interface.i_mask] -
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
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e))

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


# Bayesian Compositional Algorithms -------------------------------------------

class BIC(Bayesian):
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

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # masked model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.i_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.i_mask]

            # compute error image
            e = masked_m - masked_i

            # compute gauss-newton parameter updates
            dp = self.interface.solve(self._h, self._j_po, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(self.project_out(e[..., None])[..., 0]))

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters, cost,
                                               gt_shape=gt_shape)


class BICN(Bayesian):
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

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # masked model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.i_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.i_mask]

            # compute error image
            e = masked_m - masked_i

            # project out appearance model from error
            e_po = self.project_out(e[..., None])[..., 0]
            # compute full newton hessian
            h = e_po.dot(self._h_pn).reshape(self._h_gn.shape) + self._h_gn

            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, self._j_po, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e_po))

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters, cost,
                                               gt_shape=gt_shape)


class BFC(Bayesian):
    r"""
    Bayesian Forward Compositional Gauss-Newton Algorithm
    """

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False, l=0.5):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # masked model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.i_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.i_mask]

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
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(self.project_out(e[..., None])[..., 0]))

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters, cost,
                                               gt_shape=gt_shape)


class BFCN(Bayesian):
    r"""
    Bayesian Forward Compositional Gauss-Newton Algorithm
    """

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False, l=0.5):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # mask model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.i_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.i_mask]

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
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e_po))

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters, cost,
                                               gt_shape=gt_shape)


class BSC(Bayesian):
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

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # masked model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.i_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.i_mask]

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
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(self.project_out(e[..., None])[..., 0]))

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters, cost,
                                               gt_shape=gt_shape)


class BBC(Bayesian):
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

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        n_shape_params = self.transform.n_parameters
        # masked model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.i_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.i_mask]

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
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(self.project_out(e[..., None])[..., 0]))

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters, cost,
                                               gt_shape=gt_shape)


# --------------------------------------------------------------------------

class Combined(AAMAlgorithm):

    def __init__(self, aam_interface, appearance_model, transform,
                 combined_model, eps=None, **kwargs):
        # call super constructor
        super(Combined, self).__init__(
            aam_interface, appearance_model, transform, eps, **kwargs)

        # set common state for all Fast Simultaneous AAM algorithms
        self._masked_U = self._U[self.interface.i_mask, :]

        self.combined_model = combined_model
        self.C = self.combined_model.components.T
        self.C_s = self.C[:self.transform.n_parameters - 4, :]
        self.C_a = self.C[self.transform.n_parameters - 4:, :]
        self.pinv_C_s = np.linalg.pinv(self.C_s).T
        self.pinv_C_a = np.linalg.pinv(self.C_a).T

        # pre-compute
        self._precompute()

    def project_out(self, j):
        return j - self._masked_U.dot(self._pinv_U.T.dot(j))
    

class CIC(Combined):
    r"""
    Combined Inverse Compositional Gauss-Newton Algorithm
    """

    def _residual(self, masked_i, masked_t):
        return masked_i - masked_t

    def _cost(self, r):
        return r.T.dot(r) / r.shape[0]

    def _precompute(self):
        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False, fixed_p=None):
        # initialize transform
        self.transform.set_target(initial_shape)

        # initialize lists
        shape_params = [self.transform.as_vector()]
        appearance_params = []
        residuals = []
        costs = []
        jacobians = []

        print max_iters

        for n_iter in xrange(max_iters):

            # warp image
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.i_mask]

            if n_iter == 0:
                c = 0
                # project image onto the model bases
                # c = self._pinv_U.T.dot(masked_i - self.masked_m)
                # t = self._U.dot(c) + self.m
                # self.template.from_vector_inplace(t)
                #
                # l = self.pinv_C_a.T.dot(c)
                # p = self.transform.as_vector().copy()
                # p[4:] = self.C_s.dot(l)
                # self.transform.from_vector_inplace(self.transform.as_vector() + p)
            
            # mask reconstruction
            masked_t = self.template.as_vector()[self.interface.i_mask]
            # compute error image
            e = masked_i - masked_t

            # compute model gradient
            nabla_t = self.interface.gradient(self.template)
            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_t, self._dw_dp)
            j = np.hstack((j[:, :4], j[:, 4:].dot(self.pinv_C_s) + self._U.dot(
                self.pinv_C_a)))
            if fixed_p:
                j = j[:, fixed_p]
            # compute hessian
            h = j.T.dot(j)
            # compute gauss-newton parameter updates
            dl = self.interface.solve(h, j, e, prior, fixed_p)
            print dl

            # update transform
            if fixed_p:
                delta_l = np.zeros(self.transform.n_parameters)
                delta_l[fixed_p] = dl
                dl = delta_l

            dp = -np.hstack((dl[:4], self.C_s.dot(dl[4:])))
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)

            dc = self.C_a.dot(dl[4:])
            c += dc
            t = self._U.dot(c) + self.m
            self.template.from_vector_inplace(t)

            # update lists
            appearance_params.append(c)
            shape_params.append(self.transform.as_vector())
            residuals.append(self._residual(masked_i, t))
            costs.append(self._cost(residuals[-1]))
            jacobians.append(j)

            # test convergence
            shape_norm = np.sum((target.as_vector() -
                                 self.transform.target.as_vector()
                                 )**2) / target.n_parameters
            if shape_norm < self.eps:
                #break
                1

            print 'hello'

        # define linearized cost functions
        def linearized_costs(k):
            def lin_f(x):
                lin_r = residuals[k] - jacobians[k].dot(x)
                return self._cost(lin_r)
            return lin_f

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_params, linearized_costs,
            appearance_params=appearance_params, gt_shape=gt_shape)