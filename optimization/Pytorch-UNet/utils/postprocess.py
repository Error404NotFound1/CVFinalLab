import numpy as np
from scipy.ndimage import binary_opening, binary_closing


def postprocess_mask(mask, min_size=50, kernel_size=3):
    """
    Post-process segmentation mask with morphological operations.

    Args:
        mask: numpy array of shape (H, W) with class labels 0/1/2
        min_size: minimum object size to keep (pixels)
        kernel_size: size of morphological kernel

    Returns:
        Cleaned mask
    """
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    # Process each class separately
    cleaned_mask = np.zeros_like(mask)
    for class_id in [0, 1, 2]:
        binary_mask = (mask == class_id).astype(np.uint8)

        # Remove small noise with opening
        binary_mask = binary_opening(binary_mask, structure=kernel)

        # Fill small holes with closing
        binary_mask = binary_closing(binary_mask, structure=kernel)

        cleaned_mask[binary_mask > 0] = class_id

    return cleaned_mask


def remove_small_objects(mask, min_size=50):
    """Remove small disconnected regions."""
    from scipy.ndimage import label

    cleaned_mask = np.zeros_like(mask)
    for class_id in [0, 1, 2]:
        binary_mask = (mask == class_id)
        labeled, num_features = label(binary_mask)

        for region_id in range(1, num_features + 1):
            region = (labeled == region_id)
            if region.sum() >= min_size:
                cleaned_mask[region] = class_id

    return cleaned_mask
