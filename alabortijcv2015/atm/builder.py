from __future__ import division

from menpo.transform import Translation, UniformScale
from menpo.visualize import print_dynamic, progress_bar_str
from menpo.feature import no_op
from menpofit.transform import DifferentiablePiecewiseAffine

from ..builder import \
    compute_features, scale_images, \
    build_shape_model, build_reference_frame, \
    compute_reference_shape_from_shapes

# Abstract Interface for ATM Builders -----------------------------------------


class ATMBuilder(object):

    def build(self, template, shapes, group=None, label=None,
              reference_shape=None,
              verbose=False):
        # compute reference shape
        if reference_shape is None:
            reference_shape = compute_reference_shape_from_shapes(
                shapes, diagonal=self.diagonal, verbose=verbose)

        shape_models = []
        templates = []

        if verbose:
            print_dynamic('Building models\n')

        data_prepare = self._prepare_data(template, shapes, reference_shape,
                                          group=group, label=label,
                                          verbose=verbose)
        for k, (level_template, scaled_shapes) in enumerate(data_prepare):
            if verbose:
                print_dynamic(' - Building shape model')
            s_shape_model = build_shape_model(
                scaled_shapes, self.max_shape_components)
            shape_models.append(s_shape_model)

            warped_template = self._warp_template(level_template,
                                                  s_shape_model.mean())
            templates.append(warped_template)

            if verbose:
                print_dynamic(' - Level {} - Done\n'.format(k))

        # reverse the list of shape and appearance models so that they are
        # ordered from lower to higher resolution
        shape_models.reverse()
        templates.reverse()

        return self._build_atm(shape_models, templates, reference_shape)

    def _prepare_data(self, template, shapes, reference_shape, group=None,
                      label=None, verbose=False):
        normalized_template = template.rescale_to_reference_shape(
            reference_shape, group=group, label=label)

        # build models at each scale
        # for each pyramid level (high --> low)
        for j, s in enumerate(self.scales):
            # obtain image representation
            if j == 0:
                # compute features at highest level
                feature_image = compute_features([normalized_template],
                                                 verbose=verbose,
                                                 features=self.features)[0]
                level_template = feature_image
            elif self.scale_features:
                # scale features at other levels
                level_template = scale_images([feature_image], s,
                                              verbose=verbose)[0]
            else:
                # scale images and compute features at other levels
                scaled_image = scale_images([normalized_template], s,
                                            verbose=verbose)[0]
                level_template = compute_features([scaled_image],
                                                  verbose=verbose,
                                                  features=self.features)[0]

            # Rescaled shapes
            ref_frame_scale = UniformScale(s, reference_shape.n_dims)
            level_shapes = [ref_frame_scale.apply(s) for s in shapes]

            if self.scale_shapes:
                shape_model_shapes = level_shapes
            else:
                shape_model_shapes = shapes

            yield level_template, shape_model_shapes

    def _build_atm(self, shape_models, templates, reference_shape):
        pass


# Concrete Implementations of ATM Builders ------------------------------------

class GlobalATMBuilder(ATMBuilder):

    def __init__(self, features=no_op, transform=DifferentiablePiecewiseAffine,
                 trilist=None, diagonal=None, sigma=None, scales=(1, .5),
                 scale_shapes=True, scale_features=True,
                 max_shape_components=None, boundary=3):

        self.features = features
        self.transform = transform
        self.trilist = trilist
        self.diagonal = diagonal
        self.sigma = sigma
        self.scales = list(scales)
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features
        self.max_shape_components = max_shape_components
        self.boundary = boundary

    def _build_reference_frame(self, mean_shape):
        return build_reference_frame(mean_shape, boundary=self.boundary,
                                     trilist=self.trilist)

    def _warp_template(self, template, ref_shape, group=None,
                       label=None, verbose=True):
        if verbose:
            print_dynamic(' - Warping template')

        ref_frame = self._build_reference_frame(ref_shape)
        transform = self.transform(ref_frame.landmarks['source'].lms,
                                   template.landmarks[group][label])

        # warp template to reference frame
        warped_template = template.warp_to_mask(ref_frame.mask, transform)
        warped_template.landmarks['source'] = ref_frame.landmarks['source']

        return warped_template

    def _build_atm(self, shape_models, templates, reference_shape):
        from alabortijcv2015.atm import GlobalATM

        return GlobalATM(shape_models, templates, reference_shape,
                         self.transform, self.features, self.sigma,
                         list(reversed(self.scales)),
                         self.scale_shapes, self.scale_features)


class LinearGlobalATMBuilder(GlobalATMBuilder):

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

        shape_model = GlobalATMBuilder._build_shape_model(
            shapes, max_components)

        return shape_model

    def _warp_images(self, images, _, ref_shape, verbose=True):
        from ..builder import _build_reference_frame

        ref_frame = GlobalATMBuilder._build_reference_frame(self, ref_shape)
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
                print_dynamic('Warping images - {}'.format(
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
        from alabortijcv2015.atm import LinearGlobalATM

        return LinearGlobalATM(shape_models, appearance_models,
                               self.reference_frame.landmarks['source'].lms,
                               self.transform,
                               self.features, self.sigma,
                               list(reversed(self.scales)),
                               self.scale_shapes, self.scale_features,
                               0)
