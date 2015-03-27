from __future__ import division

from alabortijcv2015.fitter import Fitter
from alabortijcv2015.transform import OrthoMDTransform, OrthoLinearMDTransform

from .algorithm import StandardATMInterface, LinearATMInterface, TAIC
from menpo.transform import Scale, AlignmentAffine
from menpo.visualize import print_dynamic
from .result import ATMFitterResult


# Abstract Interface for ATM Fitters ------------------------------------------


class ATMFitter(Fitter):

    def fit_sequence(self, images, initial_shapes, max_iters=50, gt_shapes=None,
                     crop_image=None, **kwargs):
        r"""
        Fits the multilevel algorithm to an image.

        Parameters
        -----------
        images: :map:`Image` or subclass
            The images to be fitted.
        initial_shape: :map:`PointCloud`
            The initial shape estimate from which the fitting procedure
            will start.
        max_iters: `int` or `list` of `int`, optional
            The maximum number of iterations.
            If `int`, specifies the overall maximum number of iterations.
            If `list` of `int`, specifies the maximum number of iterations per
            level.
        gt_shape: :map:`PointCloud`
            The ground truth shape associated to the image.
        crop_image: `None` or float`, optional
            If `float`, it specifies the proportion of the border wrt the
            initial shape to which the image will be internally cropped around
            the initial shape range.
            If `None`, no cropping is performed.

            This will limit the fitting algorithm search region but is
            likely to speed up its running time, specially when the
            modeled object occupies a small portion of the image.
        **kwargs:
            Additional keyword arguments that can be passed to specific
            implementations of ``_fit`` method.

        Returns
        -------
        multi_fitting_result: :map:`MultilevelFittingResult`
            The multilevel fitting result containing the result of
            fitting procedure.
        """

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
        r"""
        Fits the algorithm to the multilevel pyramidal images.

        Parameters
        -----------
        images: :class:`menpo.image.masked.MaskedImage` list
            The images to be fitted.
        initial_shapes: :class:`menpo.shape.PointCloud`
            The initial shape from which the fitting will start.
        gt_shapes: :class:`menpo.shape.PointCloud` list, optional
            The original ground truth shapes associated to the multilevel
            images.
        max_iters: int or list, optional
            The maximum number of iterations.
            If int, then this will be the overall maximum number of iterations
            for all the pyramidal levels.
            If list, then a maximum number of iterations is specified for each
            pyramidal level.

            Default: 50

        Returns
        -------
        algorithm_results: :class:`menpo.fg2015.fittingresult.FittingResult` list
            The fitting object containing the state of the whole fitting
            procedure.
        """
        max_iters = self._prepare_max_iters(max_iters)

        seq_algorithm_results = []
        shapes = seq_initial_shapes
        for j, (ims, alg, it, s) in enumerate(zip(seq_images,
                                                  self._algorithms,
                                                  max_iters, self.scales)):
            print_dynamic('Iteration {}'.format(j))
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

        return [r for r in zip(*seq_algorithm_results)]

    def _fitter_result(self, image, algorithm_results, affine_correction,
                       gt_shape=None):
        return ATMFitterResult(image, self, algorithm_results,
                               affine_correction, gt_shape=gt_shape)


# Concrete Implementations of ATM Fitters -------------------------------------

class StandardATMFitter(ATMFitter):

    def __init__(self, global_atm, algorithm_cls=TAIC, n_shape=None, **kwargs):
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

    def __init__(self, global_atm, algorithm_cls=TAIC, n_shape=None, **kwargs):
        super(LinearATMFitter, self).__init__()

        self.dm = global_atm
        self._algorithms = []
        self._check_n_shape(n_shape)

        for j, (template, sm) in enumerate(zip(self.dm.appearance_models,
                                               self.dm.templates)):
            md_transform = OrthoLinearMDTransform(
                sm, self.dm.n_landmarks)

            algorithm = algorithm_cls(LinearATMInterface, template,
                                      md_transform, **kwargs)

            self._algorithms.append(algorithm)
