"""Shared image preprocessing for local SfM."""

from pi3.utils import cropping


def crop_resize(image, K, size, mask=None):
    """Resize then center crop an image, intrinsics, and optional mask."""

    height, width = size
    image, mask, K, _, _ = cropping.rescale_image_depthmap(
        image, mask, K, (width, height)
    )
    resized_K = cropping.camera_matrix_of_crop(K, image.size, (width, height))
    bbox = cropping.bbox_from_intrinsics_in_out(
        K, resized_K, (width, height)
    )
    image = image.crop(bbox)
    if mask is not None:
        left, top, right, bottom = bbox
        mask = mask[top:bottom, left:right]
    return image, mask, resized_K
