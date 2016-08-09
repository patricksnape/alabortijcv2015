from menpo.transform import PiecewiseAffine, AlignmentSimilarity, Similarity
from menpo.visualize import print_dynamic, progress_bar_str
from menpo.feature import no_op
from menpo.image import MaskedImage
from menpo.shape import PointCloud, TriMesh
from menpo.feature import fast_dsift
import numpy as np
from scipy.spatial import KDTree


def pointclouds_from_uv(u, v, add_zero=False):
    if add_zero:
        zero = zero_flow_grid_pcloud(u.shape).points

    pclouds = []
    for u1, v1 in zip(u.as_vector(keep_channels=True),
                      v.as_vector(keep_channels=True)):
        if add_zero:
            u1 = u1 + zero[:, 1]
            v1 = v1 + zero[:, 0]
        pclouds.append(PointCloud(np.vstack([v1, u1]).T))
    return pclouds


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
    for k in range(height - 1):
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


def align_shape_with_bb(shape, bounding_box):
    r"""
    Returns the Similarity transform that aligns the provided shape with the
    provided bounding box.
    Parameters
    ----------
    shape: :class:`menpo.shape.PointCloud`
        The shape to be aligned.
    bounding_box: (2, 2) ndarray
        The bounding box specified as:
            np.array([[x_min, y_min], [x_max, y_max]])
    Returns
    -------
    transform : :class: `menpo.transform.Similarity`
        The align transform
    """
    shape_box = PointCloud(shape.bounds())
    bounding_box = PointCloud(bounding_box)
    return AlignmentSimilarity(shape_box, bounding_box, rotation=False)


def noisy_align(source, target, noise_std=0.04, rotation=False):
    r"""
    Constructs and perturbs the optimal similarity transform between source
    to the target by adding white noise to its weights.
    Parameters
    ----------
    source: :class:`menpo.shape.PointCloud`
        The source pointcloud instance used in the alignment
    target: :class:`menpo.shape.PointCloud`
        The target pointcloud instance used in the alignment
    noise_std: float
        The standard deviation of the white noise
        Default: 0.04
    rotation: boolean
        If False the second parameter of the Similarity,
        which captures captures inplane rotations, is set to 0.
        Default:False
    Returns
    -------
    noisy_transform : :class: `menpo.transform.Similarity`
        The noisy Similarity Transform
    """
    transform = AlignmentSimilarity(source, target, rotation=rotation)
    parameters = transform.as_vector()
    parameter_range = np.hstack((parameters[:2], target.range()))
    noise = (parameter_range * noise_std *
             np.random.randn(transform.n_parameters))
    return Similarity.init_identity(source.n_dims).from_vector(parameters + noise)


def build_atm(template_im, shape_models, reference_frames, scales, feature=no_op,
              sparse_group=None, scale_features=True, verbose=True):
    from alabortijcv2015.atm import LinearGlobalATM
    from alabortijcv2015.aam.builder import scale_images, compute_features

    # Normalise with the LARGEST scale
    normalized_template = template_im.rescale_to_pointcloud(
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
                                                      transform,
                                                      warp_landmarks=False)
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
    from menpo.model import PCAModel
    from alabortijcv2015.aam.builder import normalize_images, scale_images, compute_features
    from menpofit.transform import DifferentiablePiecewiseAffine
    from alabortijcv2015.aam import GlobalAAM

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
                                        transform,
                                        warp_landmarks=False)
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
    from alabortijcv2015.aam import LinearGlobalAAM
    from alabortijcv2015.aam.builder import normalize_images, scale_images, compute_features
    from menpo.model import PCAModel

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
                                        transform,
                                        warp_landmarks=False)
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
