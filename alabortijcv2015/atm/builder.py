from __future__ import division
import numpy as np
from itertools import chain


from scipy.spatial.kdtree import KDTree

from menpo.transform import UniformScale, AlignmentSimilarity, PiecewiseAffine
from menpo.visualize import print_dynamic
from menpo.feature import no_op
from menpo.shape import PointCloud, TriMesh
from menpo.image import MaskedImage
from menpofit.transform import DifferentiablePiecewiseAffine

from ..builder import \
    compute_features, scale_images, \
    build_shape_model, build_reference_frame, \
    compute_reference_shape_from_shapes, scale_shape_diagonal

# Abstract Interface for ATM Builders -----------------------------------------


class ATMBuilder(object):

    def build(self, template, shapes, group=None, label=None,
              reference_shape=None, verbose=False):
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
                 max_shape_components=None):

        self.features = features
        self.transform = transform
        self.trilist = trilist
        self.diagonal = diagonal
        self.sigma = sigma
        self.scales = list(scales)
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features
        self.max_shape_components = max_shape_components
        self.boundary = 0

    def _build_reference_frame(self, mean_shape):
        return build_reference_frame(mean_shape, boundary=self.boundary)

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

    def __init__(self, features=no_op, trilist=None, diagonal=None, sigma=None,
                 scales=(1, .5), scale_features=True, max_shape_components=None,
                 boundary=0):

        self.features = features
        self.trilist = trilist
        self.diagonal = diagonal
        self.sigma = sigma
        self.scales = list(scales)
        self.scale_features = scale_features
        self.max_shape_components = max_shape_components
        self.boundary = boundary
        self.transform = None

    def build(self, template, uv_images, group=None, group_sparse=None,
              verbose=False):
        if group is None and group_sparse is not None:
            raise ValueError('Cannot use None for the group key because '
                             'two landmark groups are required on the '
                             'template for both the sparse and dense '
                             'landmarks.')
        # uv_images is a list of tuples, (u, v) where u and v are multichannel
        # images where each channel is a shape in the sequence
        # We assume that the UV flow images are already all in the same
        # reference frame, as they have been learnt from a given reference
        # frame. We also assume that all the flow images are masked.
        reference_shape = zero_flow_grid_pcloud(uv_images[0][0].shape,
                                                triangulated=False,
                                                mask=uv_images[0][0].mask)
        if self.diagonal:
            reference_shape = scale_shape_diagonal(reference_shape,
                                                   self.diagonal)

        shape_models = []
        templates = []
        reference_frames = []
        dense_indices = []
        sparse_masks = []

        if verbose:
            print_dynamic('Building models\n')

        data_prepare = self._prepare_data(template, uv_images, reference_shape,
                                          group=group,
                                          verbose=verbose)
        for k, (lvl_tmplt, lvl_shapes, lvl_refframe) in enumerate(data_prepare):
            if verbose:
                print_dynamic(' - Building shape model')
            s_shape_model = build_shape_model(
                lvl_shapes, self.max_shape_components)
            shape_models.append(s_shape_model)

            warped_template = self._sample_template(lvl_tmplt, lvl_refframe)
            templates.append(warped_template)
            reference_frames.append(lvl_refframe)

            if group_sparse is not None:
                zero_flow = PointCloud(lvl_refframe.as_vector(
                    keep_channels=True).T)
                dense_ind, sparse_mask = sparse_landmark_indices_from_dense(
                    zero_flow, lvl_tmplt.landmarks[group_sparse].lms)
                dense_indices.append(dense_ind)
                sparse_masks.append(sparse_mask)

            if verbose:
                print_dynamic(' - Level {} - Done\n'.format(k))

        # reverse the list of shape and appearance models so that they are
        # ordered from lower to higher resolution
        shape_models.reverse()
        templates.reverse()
        reference_frames.reverse()
        dense_indices.reverse()
        sparse_masks.reverse()

        return self._build_atm(shape_models, templates, reference_frames,
                               dense_indices=dense_indices,
                               sparse_masks=sparse_masks)

    def build_sparse_template(self, template, uv_images,
                              transform=PiecewiseAffine,
                              group_sparse=None, verbose=False):
        # uv_images is a list of tuples, (u, v) where u and v are multichannel
        # images where each channel is a shape in the sequence
        # We assume that the UV flow images are already all in the same
        # reference frame, as they have been learnt from a given reference
        # frame. We also assume that all the flow images are masked.
        self.transform = transform
        sparse_ref_shape = uv_images[0][0].landmarks[group_sparse].lms
        if self.diagonal:
            sparse_ref_shape = scale_shape_diagonal(sparse_ref_shape,
                                                    self.diagonal)

        shape_models = []
        templates = []
        reference_frames = []
        dense_indices = []
        sparse_masks = []

        if verbose:
            print_dynamic('Building models\n')

        data_prepare = self._prepare_data_sparse(template, uv_images,
                                                 sparse_ref_shape,
                                                 group=group_sparse,
                                                 verbose=verbose)
        for k, (lvl_tmplt, lvl_shapes, lvl_refframe) in enumerate(data_prepare):
            warped_template = self._warp_template(lvl_tmplt,
                                                  lvl_refframe,
                                                  group=group_sparse,
                                                  verbose=verbose)

            templates.append(warped_template)
            reference_frames.append(lvl_refframe)

            if verbose:
                print_dynamic(' - Building shape model')
            s_shape_model = build_shape_model(
                lvl_shapes, self.max_shape_components)
            shape_models.append(s_shape_model)

            zero_flow = PointCloud(lvl_refframe.as_vector(
                keep_channels=True).T)
            dense_ind, sparse_mask = sparse_landmark_indices_from_dense(
                zero_flow, lvl_tmplt.landmarks[group_sparse].lms)
            dense_indices.append(dense_ind)
            sparse_masks.append(sparse_mask)

            if verbose:
                print_dynamic(' - Level {} - Done\n'.format(k))

        # reverse the list of shape and appearance models so that they are
        # ordered from lower to higher resolution
        shape_models.reverse()
        templates.reverse()
        reference_frames.reverse()
        dense_indices.reverse()
        sparse_masks.reverse()

        return self._build_atm(shape_models, templates, reference_frames,
                               dense_indices=dense_indices,
                               sparse_masks=sparse_masks)

    def _prepare_data_sparse(self, template, uv_images, reference_shape,
                             group=None, verbose=False):
        normalized_template = template.rescale_to_reference_shape(
            reference_shape, group=group)

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

            # Flatten list of tuples (all us, all vs)
            # TODO UV is not currently scaled by diagonal
            uv_list = list(chain(*zip(*uv_images)))
            level_uvs = scale_images(uv_list, s, verbose=verbose)
            # TODO: Some really weird bug where I have re-constrain the mask
            # on the template image or the mask doesn't match the triangulation
            # which is really annoying, so here I just make sure that the
            # template actually matches my shapes. So I have to re-sample
            # the bloody shapes.
            level_uvs[0].constrain_mask_to_landmarks(group=group)
            for luv in level_uvs:
                luv.mask = level_uvs[0].mask

            half_list = len(uv_list) // 2
            # Scaled UV images to pointclouds
            level_shapes = []
            for u, v in zip(level_uvs[:half_list], level_uvs[half_list:]):
                level_shapes.append(pointclouds_from_uv(u, v))
            # Flatten list
            level_shapes = list(chain(*level_shapes))

            # Build a reference frame from the scaled mask
            level_tmplt = level_uvs[0]
            level_ref_frame = self._build_reference_frame(level_tmplt)
            level_ref_frame.landmarks = level_tmplt.landmarks
            zero_flow = PointCloud(level_ref_frame.as_vector(
                keep_channels=True).T)

            # Make sure the scale of the pointclouds is correct to the new frame
            for ls in level_shapes:
                AlignmentSimilarity(ls, zero_flow).apply_inplace(ls)

            yield level_template, level_shapes, level_ref_frame

    def _prepare_data(self, template, uv_images, reference_shape,
                      group=None, verbose=False):
        normalized_template = template.rescale_to_reference_shape(
            reference_shape, group=group)

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

            # Flatten list of tuples (all us, all vs)
            # TODO UV is not currently scaled by diagonal
            uv_list = list(chain(*zip(*uv_images)))
            level_uvs = scale_images(uv_list, s, verbose=verbose)
            half_list = len(uv_list) // 2
            # Scaled UV images to pointclouds
            level_shapes = []
            for u, v in zip(level_uvs[:half_list], level_uvs[half_list:]):
                level_shapes.append(pointclouds_from_uv(u, v))
            # Flatten list
            level_shapes = list(chain(*level_shapes))

            # Build a reference frame from the scaled mask
            level_tmplt = level_uvs[0]
            level_ref_frame = self._build_reference_frame(level_tmplt)
            level_ref_frame.landmarks = level_tmplt.landmarks
            zero_flow = PointCloud(level_ref_frame.as_vector(
                keep_channels=True).T)

            # Make sure the scale of the pointclouds is correct to the new frame
            for ls in level_shapes:
                AlignmentSimilarity(ls, zero_flow).apply_inplace(ls)

            yield level_template, level_shapes, level_ref_frame

    def _build_reference_frame(self, template_im):
        zero_flow = zero_flow_grid_pcloud(template_im.shape,
                                          mask=template_im.mask)
        ref_frame = MaskedImage.init_blank(template_im.shape,
                                           mask=template_im.mask,
                                           n_channels=2)
        ref_frame.from_vector_inplace(zero_flow.points.T.ravel())
        return ref_frame

    def _sample_template(self, template, ref_frame, verbose=True):
        if verbose:
            print_dynamic(' - Warping template')

        sample_points = ref_frame.as_vector(keep_channels=True).T
        warped_template = ref_frame.from_vector(
            template.sample(sample_points),
            n_channels=template.n_channels)
        warped_template.landmarks['source'] = PointCloud(sample_points)

        return warped_template

    def _warp_template(self, template, ref_frame, group=None,
                       label=None, verbose=True):
        if verbose:
            print_dynamic(' - Warping template')

        transform = self.transform(ref_frame.landmarks[group].lms,
                                   template.landmarks[group].lms)

        # warp template to reference frame
        warped_template = template.as_unmasked().warp_to_mask(ref_frame.mask,
                                                              transform)
        warped_template.landmarks['source'] = ref_frame.landmarks[group]

        return warped_template

    def _build_atm(self, shape_models, templates, reference_frames,
                   dense_indices=None, sparse_masks=None):
        from alabortijcv2015.atm import LinearGlobalATM

        return LinearGlobalATM(shape_models, templates,
                               reference_frames,
                               self.features, self.sigma,
                               list(reversed(self.scales)),
                               self.scale_features,
                               dense_indices=dense_indices,
                               sparse_masks=sparse_masks)


def pointclouds_from_uv(u, v):
    return [PointCloud(np.vstack([v1, u1]).T)
            for u1, v1 in zip(u.as_vector(keep_channels=True),
                              v.as_vector(keep_channels=True))]


def grid_triangulation(shape):
    height, width = shape
    row_to_index = lambda x: x * width
    top_triangles = lambda x: np.concatenate([np.arange(row_to_index(x), row_to_index(x) + width - 1)[..., None],
                                              np.arange(row_to_index(x) + 1, row_to_index(x) + width)[..., None],
                                              np.arange(row_to_index(x + 1), row_to_index(x + 1) + width - 1)[..., None]], axis=1)

    # Half edges are opposite directions
    bottom_triangles = lambda x: np.concatenate([np.arange(row_to_index(x + 1), row_to_index(x + 1) + width - 1)[..., None],
                                                 np.arange(row_to_index(x) + 1, row_to_index(x) + width)[..., None],
                                                 np.arange(row_to_index(x + 1) + 1, row_to_index(x + 1) + width)[..., None]], axis=1)

    trilist = []
    for k in xrange(height - 1):
        trilist.append(top_triangles(k))
        trilist.append(bottom_triangles(k))

    return np.concatenate(trilist)


def zero_flow_grid_pcloud(shape, triangulated=False, mask=None):
    point_grid = np.meshgrid(range(0, shape[0]),
                             range(0, shape[1]), indexing='ij')
    point_grid_vec = np.vstack([p.ravel() for p in point_grid]).T

    if triangulated:
        trilist = grid_triangulation(shape)
        pcloud = TriMesh(point_grid_vec, trilist=trilist)
    else:
        pcloud = PointCloud(point_grid_vec)

    if mask is not None:
        return pcloud.from_mask(mask.pixels.ravel())
    else:
        return pcloud


def sparse_landmark_indices_from_dense(dense_landmarks, sparse_lmarks):
    points = dense_landmarks.points

    tree = KDTree(points)

    indices = np.array([tree.query(p)[1] for p in sparse_lmarks.points])
    sparse_landmark_mask = np.zeros_like(indices, dtype=np.bool)
    uniq_indices_set = set()
    uniq_indices_list = []
    for k, i in enumerate(indices):
        if i not in uniq_indices_set:
            uniq_indices_set.add(i)
            uniq_indices_list.append(i)
            sparse_landmark_mask[k] = True

    return np.array(uniq_indices_list), sparse_landmark_mask
