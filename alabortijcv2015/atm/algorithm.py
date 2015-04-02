from __future__ import division
from copy import copy

import numpy as np
import scipy
from scipy.sparse import block_diag

from menpo.feature import gradient as fast_gradient
from menpo.transform import Similarity

from .result import ATMAlgorithmResult, LinearATMAlgorithmResult


# Abstract Interfaces for ATM Algorithms --------------------------------------

class ATMInterface(object):

    def __init__(self, atm_algorithm):
        self.algorithm = atm_algorithm

    def dw_dp(self):
        pass

    def warp(self, image):
        pass

    def gradient(self, image):
        pass

    def steepest_descent_images(self, gradient, dw_dp):
        pass

    def partial_newton_hessian(self, gradient2, dw_dp):
        pass

    def solve(self, h, j, e, prior):
        pass


class StandardATMInterface(ATMInterface):

    def __init__(self, atm_algorithm, sampling_step=None):
        super(StandardATMInterface, self). __init__(atm_algorithm)

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

    def dw_dp(self):
        dw_dp = np.rollaxis(self.algorithm.transform.d_dp(
            self.algorithm.template.mask.true_indices()), -1)
        return dw_dp[self.dw_dp_mask].reshape((dw_dp.shape[0], -1,
                                               dw_dp.shape[2]))

    def warp(self, image):
        return image.warp_to_mask(self.algorithm.template.mask,
                                  self.algorithm.transform)

    def gradient(self, image):
        g = fast_gradient(image)
        g.set_boundary_pixels()
        return g.as_vector().reshape((2, image.n_channels, -1))

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
        jp = t.jp()[:h.shape[0], :h.shape[0]]

        if prior:
            inv_h = np.linalg.inv(h)
            dp = inv_h.dot(j.T.dot(e))
            dp = -np.linalg.solve(t.h_prior + jp.dot(inv_h.dot(jp.T)),
                                  t.j_prior * t.as_vector() - jp.dot(dp))
        else:
            dp = np.linalg.solve(h, j.T.dot(e))
            if jp.shape[0] is dp.shape[0]:
                dp = jp.dot(dp)
            else:
                dp = scipy.linalg.block_diag(jp, jp).dot(dp)

        return dp

    def seq_solve(self, h, j, es, jps, prior):
        jps = np.hstack([jp[:h.shape[0], :h.shape[0]] for jp in jps])

        if prior:
            raise NotImplementedError('Maybe we want different priors per '
                                      'frame?')
            # t = self.algorithm.transform
            # inv_h = np.linalg.inv(h)
            # dp = inv_h.dot(j.T.dot(e))
            # dp = -np.linalg.solve(t.h_prior + jp.dot(inv_h.dot(jp.T)),
            #                       t.j_prior * t.as_vector() - jp.dot(dp))
        else:
            block_es = block_diag([e for e in es.T]).T
            # Make sure the dot product is sparse
            res = block_es.T.dot(j).T
            dp = np.linalg.solve(h, res)
            if jps.shape[0] is dp.shape[0]:
                # Make sure the dot product is sparse
                block_dp = block_diag([i for i in dp.T])
                dp = block_dp.dot(jps.T).T
            else:
                raise NotImplementedError('Not sure what this case is...')
                # dp = scipy.linalg.block_diag(jp, jp).dot(dp)

        return dp

    def algorithm_result(self, image, shape_parameters, cost,
                         gt_shape=None):
        return ATMAlgorithmResult(
            image, self.algorithm, shape_parameters, cost,
            gt_shape=gt_shape)


class LinearATMInterface(StandardATMInterface):

    def solve(self, h, j, e, prior):
        t = self.algorithm.transform

        if prior:
            dp = -np.linalg.solve(t.h_prior + h,
                                  t.j_prior * t.as_vector() - j.T.dot(e))
        else:
            dp = np.linalg.solve(h, j.T.dot(e))

        return dp

    def seq_solve(self, h, j, es, jps, prior):
        if prior:
            raise NotImplementedError('Maybe we want different priors per '
                                      'frame?')
            # t = self.algorithm.transform
            # inv_h = np.linalg.inv(h)
            # dp = inv_h.dot(j.T.dot(e))
            # dp = -np.linalg.solve(t.h_prior + jp.dot(inv_h.dot(jp.T)),
            #                       t.j_prior * t.as_vector() - jp.dot(dp))
        else:
            # Block diagonal solution
            # block_es = block_diag([e for e in es.T]).T
            # res = block_es.T.dot(j).T
            # dp = np.linalg.solve(h, res)
            dp = np.zeros([h.shape[0], len(es)])
            for k, e in enumerate(es):
                dp[:, k] = np.linalg.solve(h, j.T.dot(e))

        return dp

    def algorithm_result(self, image, shape_parameters, cost,
                         gt_shape=None):
        return LinearATMAlgorithmResult(
            image, self.algorithm, shape_parameters, cost,
            gt_shape=gt_shape)


# Abstract Interfaces for ATM Algorithms  -------------------------------------


class ATMAlgorithm(object):

    def __init__(self, atm_interface, template, transform,
                 eps=10**-5, **kwargs):

        # set common state for all ATM algorithms
        self.template = template
        self.transform = transform
        self.eps = eps

        # set interface
        self.interface = atm_interface(self, **kwargs)

    def _precompute(self, **kwargs):
        pass

    def run(self, image, initial_shape, max_iters=20, gt_shape=None, **kwargs):
        pass


class ConstrainedTIC(ATMAlgorithm):
    r"""
    Template Inverse Compositional Gauss-Newton Algorithm
    """
    def __init__(self, atm_interface, template, transform,
                 eps=10**-5, **kwargs):

        self.landmark_weight = np.sqrt(kwargs.pop('landmark_weight', 1.0))
        self.data_weight = np.sqrt(kwargs.pop('data_weight', 1.0))

        # call super constructor
        super(ConstrainedTIC, self).__init__(
            atm_interface, template, transform, eps, **kwargs)

        # pre-compute
        self._precompute()

    def _precompute(self):
        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()
        self.nabla_t = self.interface.gradient(self.template)
        self.vec_template = self.template.as_vector()[self.interface.image_vec_mask]
        self.j = self.interface.steepest_descent_images(self.nabla_t,
                                                        self._dw_dp)

        self.j_nr = np.vstack([self.data_weight * self.j,
                               self.landmark_weight * self.transform.V.T])
        self.h_nr = self.j_nr.T.dot(self.j_nr)

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):
        if gt_shape is None:
            raise ValueError('The sparse GT is required for a constrained fit!')
        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]

        gt_s_v = gt_shape.as_vector()

        for _ in xrange(max_iters):

            # warp image
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # compute error image
            e = self.vec_template - masked_i

            # compute the landmark error
            gt_d = gt_s_v - self.transform.sparse_target.points.ravel()

            # Calculate delta_p
            e_tot = np.hstack([self.data_weight * e,
                               self.landmark_weight * gt_d])
            dp = self.interface.solve(self.h_nr, self.j_nr, e_tot, prior)

            # update transform
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # save cost
            cost.append(e.T.dot(e))

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            gt_shape=gt_shape)


class TIC(ATMAlgorithm):
    r"""
    Constrained Template Inverse Compositional Gauss-Newton Algorithm
    """
    def __init__(self, atm_interface, template, transform,
                 eps=10**-5, **kwargs):
        # call super constructor
        super(TIC, self).__init__(
            atm_interface, template, transform, eps, **kwargs)

        # pre-compute
        self._precompute()

    def _precompute(self):
        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()
        self.nabla_t = self.interface.gradient(self.template)
        self.vec_template = self.template.as_vector()[self.interface.image_vec_mask]
        self.j = self.interface.steepest_descent_images(self.nabla_t,
                                                        self._dw_dp)
        self.h = self.j.T.dot(self.j)

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]

        for _ in xrange(max_iters):
            # warp image
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # compute error image
            e = self.vec_template - masked_i

            # compute gauss-newton parameter updates
            dp = self.interface.solve(self.h, self.j, e, prior)

            # update transform
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # save cost
            cost.append(e.T.dot(e))

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            gt_shape=gt_shape)


class SequenceTIC(ATMAlgorithm):
    r"""
    Template Inverse Compositional Gauss-Newton Algorithm
    """
    def __init__(self, atm_interface, template, transform,
                 eps=10**-5, **kwargs):
        # call super constructor
        super(SequenceTIC, self).__init__(
            atm_interface, template, transform, eps, **kwargs)

        # pre-compute
        self._precompute()

    def _precompute(self):
        self._dw_dp = self.interface.dw_dp()
        self.nabla_t = self.interface.gradient(self.template)
        self.vec_template = self.template.as_vector()[self.interface.image_vec_mask]
        self.j = self.interface.steepest_descent_images(self.nabla_t,
                                                        self._dw_dp)
        self.h = self.j.T.dot(self.j)

    def run(self, images, initial_shapes, gt_shapes=None, max_iters=20,
            prior=False):
        # initialize cost
        costs = []
        # initialize transform

        shape_parameters = []
        for ish in initial_shapes:
            self.transform.set_target(ish)
            shape_parameters.append(self.transform.as_vector())

        seq_shape_parameters = [shape_parameters]
        im_vec_mask = self.interface.image_vec_mask

        for _ in xrange(max_iters):
            es = []
            jps = []
            for im, ps in zip(images, shape_parameters):
                self.transform.from_vector_inplace(ps)
                # warp the image
                i = self.interface.warp(im)
                # mask image
                masked_i = i.as_vector()[im_vec_mask]

                # compute error image
                es.append(self.vec_template - masked_i)
                if hasattr(self.transform, 'jp'):
                    jps.append(self.transform.jp())

            dp = self.interface.seq_solve(self.h, self.j, es, jps, prior)

            # update transform
            # make sure the REFERENCE to the shape parameters list changes,
            # so that we can update it for our list of lists
            new_shape_parameters = []
            for k, column in enumerate(dp.T):
                # copy
                new_shape_parameters.append(shape_parameters[k] + column)
            shape_parameters = new_shape_parameters

            seq_shape_parameters.append(shape_parameters)

            # save cost (sum of squared errors)
            costs.append([e.T.dot(e) for e in es])

        # return aam algorithm result
        costs = [c for c in zip(*costs)]
        seq_shape_parameters = [sp for sp in zip(*seq_shape_parameters)]
        algorithm_results = []
        for k, (im, s_params, cs) in enumerate(zip(images, seq_shape_parameters,
                                                   costs)):
            if gt_shapes:
                gt = gt_shapes[k]
            else:
                gt = None
            algorithm_results.append(self.interface.algorithm_result(
                im, s_params, cs, gt_shape=gt))
        return algorithm_results


class ConstrainedSequenceTIC(ATMAlgorithm):
    r"""
    Template Inverse Compositional Gauss-Newton Algorithm
    """
    def __init__(self, atm_interface, template, transform,
                 eps=10**-5, **kwargs):

        self.landmark_weight = np.sqrt(kwargs.pop('landmark_weight', 1.0))
        self.data_weight = np.sqrt(kwargs.pop('data_weight', 1.0))

        # call super constructor
        super(ConstrainedSequenceTIC, self).__init__(
            atm_interface, template, transform, eps, **kwargs)

        # pre-compute
        self._precompute()

    def _precompute(self):
        self._dw_dp = self.interface.dw_dp()
        self.nabla_t = self.interface.gradient(self.template)
        self.vec_template = self.template.as_vector()[self.interface.image_vec_mask]
        self.j = self.interface.steepest_descent_images(self.nabla_t,
                                                        self._dw_dp)
        self.h = self.j.T.dot(self.j)

        # B
        self.j_nr = np.vstack([self.data_weight * self.j,
                               self.landmark_weight * self.transform.V.T])
        self.h_nr = self.j_nr.T.dot(self.j_nr)
        # pinv(B)
        self.j_nr_inv = np.linalg.pinv(self.j_nr)

        U, _, _ = np.linalg.svd(self.j_nr, full_matrices=False)
        self.U = U[:, :10]

    def run(self, images, initial_shapes, gt_shapes=None, max_iters=20,
            prior=False):
        from research_utils.robust_pca import rpca_alm
        # initialize cost
        costs = []

        # initialize transform
        gt_s_vs = [gt.as_vector() for gt in gt_shapes]

        shape_parameters = []
        for ish in initial_shapes:
            self.transform.set_target(ish)
            shape_parameters.append(self.transform.as_vector())

        seq_shape_parameters = [shape_parameters]
        im_vec_mask = self.interface.image_vec_mask

        n_frames = len(images)
        dp = np.zeros([len(shape_parameters[0]), n_frames])

        for it in xrange(max_iters):
            es = []
            jps = []
            gt_ds = []
            for im, ps, gt_s_v in zip(images, shape_parameters, gt_s_vs):
                self.transform.from_vector_inplace(ps)
                # warp the image
                i = self.interface.warp(im)
                # mask image
                masked_i = i.as_vector()[im_vec_mask]

                # compute error image
                es.append(self.vec_template - masked_i)
                if hasattr(self.transform, 'jp'):
                    jps.append(self.transform.jp())
                # compute the landmark error
                gt_ds.append(gt_s_v - self.transform.sparse_target.points.ravel())

            # Calculate delta_p
            e_tots = [np.hstack([self.data_weight * e,
                                self.landmark_weight * gt_d])
                      for e, gt_d in zip(es, gt_ds)]
            dp = self.interface.seq_solve(self.h_nr, self.j_nr, e_tots, jps,
                                          prior)

            #for k, A in enumerate(e_tots):
                # U, S, V = np.linalg.svd(B, full_matrices=False)
                # tmp = np.linalg.inv(np.diag(S)).dot(U.T.dot(A))
                # U1, S1, V1 = np.linalg.svd(tmp, full_matrices=False)
                # diagS = np.diag(S1)
                # svp = sum(diagS > 1.5)
                # C_hat = U1[:, :svp].dot(np.diag(diagS[:svp] - 1.5)).dot(V1[:, :svp].T)
                # dp[:, k] = V.dot(C_hat)
                # C = pinv(B)*(U*U'*A);
                #dp[:, k] = self.j_nr_inv.dot(self.U.dot(self.U.T.dot(A)))


            # make sure the REFERENCE to the shape parameters list changes,
            # so that we can update it for our list of lists
            new_shape_parameters = []
            for k in range(n_frames):
                new_shape_parameters.append(shape_parameters[k] + dp[:, k])

            C = np.vstack(new_shape_parameters).T
            # U, S, V = np.linalg.svd(C[4:, :], full_matrices=False)
            # svdp = sum(S > 1.5)
            # C1 = U[:, :svdp].dot(np.diag(S)[:svdp, :svdp].dot(V[:, :svdp].T))

            if it % 5 == 0:
                C1, _ = rpca_alm(C[4:, :], verbose=False)
                C[4:, :] = C1

            for k in range(n_frames):
                new_shape_parameters[k] = C[:, k]

            shape_parameters = new_shape_parameters
            seq_shape_parameters.append(shape_parameters)

            # save cost (sum of squared errors)
            costs.append([e.T.dot(e) for e in es])

        # return aam algorithm result
        costs = [c for c in zip(*costs)]
        seq_shape_parameters = [sp for sp in zip(*seq_shape_parameters)]
        algorithm_results = []
        for k, (im, s_params, cs) in enumerate(zip(images, seq_shape_parameters,
                                                   costs)):
            if gt_shapes:
                gt = gt_shapes[k]
            else:
                gt = None
            algorithm_results.append(self.interface.algorithm_result(
                im, s_params, cs, gt_shape=gt))
        return algorithm_results
