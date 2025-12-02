import numpy as np
from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
from skimage import morphology
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.filters import threshold_otsu


def convert_images(image_file):
    """Reads an image from the given file path,
    converts it to grayscale and binary formats and returns both images as a tuple.

    Args:
        image_file (str)): Path to the image file.
    Returns:
        tuple: A tuple containing the binary image and the grayscale image.

    """

    try:
        car_image = imread(image_file, as_gray=True)

        # the next line is not compulsory however, a grey scale pixel
        # in skimage ranges between 0 & 1. multiplying it with 255
        # will make it range between 0 & 255 (something we can relate better with

        gray_car_image = car_image * 255
        fig, (ax1, ax2) = plt.subplots(1, 2)  # 1 row, 2 columns
        ax1.imshow(gray_car_image, cmap="gray")
        threshold_value = threshold_otsu(gray_car_image)
        binary_car_image = (
            gray_car_image > threshold_value
        )  # Evaluates True(white) or False(black)
        ax2.imshow(binary_car_image, cmap="gray")
        # plt.show()
        return binary_car_image, gray_car_image
    except Exception as e:
        print(f"Error in reading the image from {image_file}")
        raise (e)


def draw_bounding_box(binary_car_image, gray_car_image):
    """Perform connected component analysis on a binary image,
    identifies connected regions, and draws bounding boxes around them on the grayscale image.

    Args:
        binary_car_image (numpy.ndarray): The binary image to analyze.
        gray_car_image (numpy.ndarray): The grayscale image on which to draw bounding boxes.
    Returns:
        list: A list of plate-like objects extracted from the binary image.
    """
    try:
        # this gets all the connected regions and groups them together
        label_image = measure.label(binary_car_image)

        # Define the dimensions of a typical number plate
        plate_dimensions = (
            0.03 * label_image.shape[0],
            0.12 * label_image.shape[0],
            0.1 * label_image.shape[1],
            0.4 * label_image.shape[1],
        )
        min_height, max_height, min_width, max_width = plate_dimensions
        plate_objects_cordinates = []
        plate_like_objects = []

        fig, (ax1) = plt.subplots(1)
        ax1.imshow(gray_car_image, cmap="gray")

        # regionprops creates a list of properties of all the labelled regions
        for region in regionprops(label_image):
            if region.area < 50:
                # if the region is so small then it's likely not a number plate
                continue

            # the bounding box coordinates and its dimensions
            min_row, min_col, max_row, max_col = region.bbox
            region_height = max_row - min_row
            region_width = max_col - min_col

            # ensuring that the region identified satisfies the condition of a typical number plate
            if (
                region_height >= min_height
                and region_height <= max_height
                and region_width >= min_width
                and region_width <= max_width
                and region_width > region_height
            ):
                plate_like_objects.append(
                    binary_car_image[min_row:max_row, min_col:max_col]
                )
                plate_objects_cordinates.append((min_row, min_col, max_row, max_col))
                rectBorder = patches.Rectangle(
                    (min_col, min_row),
                    max_col - min_col,
                    max_row - min_row,
                    edgecolor="red",
                    linewidth=2,
                    fill=False,
                )
                ax1.add_patch(rectBorder)
            # Draws the bounding boxes on the grayscale image
        # plt.show()
        return plate_like_objects
    except Exception as e:
        print("Error in connected component analysis")
        raise (e)


def vertical_projection_score(candidate):
    """
    Compute a score for how likely this binary candidate is to be a plate
    by analysing the vertical projection profile. We look for multiple peaks
    (i.e. columns that contain characters) instead of a single large blob.

    Args:
        candidate (numpy.ndarray): The binary image candidate to evaluate.
    Returns:
        float: The computed score for the candidate.
        numpy.ndarray: The inverted binary image of the candidate.
    """
    inverted_candidate = np.invert(candidate)
    projection = inverted_candidate.sum(axis=0).astype(float)

    if projection.size == 0:
        return 0.0, inverted_candidate

    normalized = projection / max(1, candidate.shape[0])
    max_value = normalized.max()
    if max_value == 0:
        return 0.0, inverted_candidate

    threshold = 0.5 * max_value
    mask = normalized >= threshold
    mask_int = mask.astype(int)
    transitions = np.diff(np.concatenate(([0], mask_int, [0])))
    peak_count = np.count_nonzero(transitions == 1)

    coverage = normalized.mean()
    score = peak_count * (1.0 + coverage)
    return score, inverted_candidate


def select_plate_using_vertical_projection(plate_like_objects):
    """
    Use vertical projection (sum of pixels in each column) to choose the most likely
    license plate candidate. Characters introduce several distinct bright peaks when inverted,
    unlike headlights or other blobs that typically produce a single uniform region.

    Args:
        plate_like_objects (list): List of binary image candidates.
    Returns:
        numpy.ndarray: The binary image of the selected license plate candidate.
    """
    if plate_like_objects is None or len(plate_like_objects) == 0:
        raise ValueError("No plate-like objects found to segment.")

    best_plate = None
    best_score = -np.inf

    for candidate in plate_like_objects:
        score, inverted_candidate = vertical_projection_score(candidate)

        if score > best_score:
            best_score = score
            best_plate = inverted_candidate

    return enhance_plate_resolution(best_plate)


def enhance_plate_resolution(plate, scale_factor=3):
    """
    Upscale the binary plate image to improve character segmentation accuracy.
    """
    if plate.size == 0 or scale_factor <= 1:
        return plate

    upscaled = resize(
        plate.astype(float),
        (plate.shape[0] * scale_factor, plate.shape[1] * scale_factor),
        order=1,
        preserve_range=True,
    )
    return (upscaled > 0.5).astype(plate.dtype)


def split_wide_region(region, width_threshold):
    """
    If a region is too wide (likely around two characters), split it by finding
    local minima in the vertical projection profile and slicing at those valleys.
    """
    projection = region.sum(axis=0)
    normalized = projection / (projection.max() or 1)
    valley_mask = normalized < 0.2

    splits = []
    start = 0
    for i in range(1, len(valley_mask)):
        if valley_mask[i - 1] and not valley_mask[i]:
            splits.append(i)
    subregions = []
    prev = 0
    for split in splits:
        if split - prev > width_threshold:
            subregions.append(region[:, prev:split])
        prev = split
    if region.shape[1] - prev > width_threshold:
        subregions.append(region[:, prev:])

    return subregions if subregions else [region]


def segment_characters(license_plate):
    """
    Segment characters from the detected license plate. Draw bounding boxes around each character
    and resize them to a standard size for further processing.
    """

    # remove small bright artifacts (e.g. screws) by applying a morphological opening
    cleaned_plate = morphology.opening(license_plate, morphology.disk(2.2))
    labelled_plate = measure.label(cleaned_plate)
    # labelled_plate = measure.label(license_plate)

    fig, ax1 = plt.subplots(1)
    ax1.imshow(license_plate, cmap="gray")
    # the next two lines is based on the assumptions that the width of
    # a characters should be between 5% and 15% of the license plate,
    # and height should be between 35% and 75%
    # this will eliminate some
    character_dimensions = (
        0.35 * license_plate.shape[0],
        0.75 * license_plate.shape[0],
        0.02 * license_plate.shape[1],
        0.18 * license_plate.shape[1],
    )
    min_height, max_height, min_width, max_width = character_dimensions

    characters = []
    counter = 0
    column_list = []
    for regions in regionprops(labelled_plate):
        y0, x0, y1, x1 = regions.bbox
        region_height = y1 - y0
        region_width = x1 - x0

        # ensuring that the region identified satisfies the condition of a typical character
        if (
            region_height > min_height
            and region_height < max_height
            and region_width > min_width
            and region_width < max_width
        ):
            pad_y = max(1, int(0.15 * region_height)) # add padding to the character region
            pad_x = max(1, int(0.22 * region_width)) 
            y0_pad = max(0, y0 - pad_y) 
            y1_pad = min(license_plate.shape[0], y1 + pad_y)
            x0_pad = max(0, x0 - pad_x)
            x1_pad = min(license_plate.shape[1], x1 + pad_x)
            roi = license_plate[y0_pad:y1_pad, x0_pad:x1_pad] # region of interest

            # If the region is too wide (>12%), it may contain multiple characters, so split it
            sub_regions = [roi]
            if (x1_pad - x0_pad) > (0.12 * license_plate.shape[1]):
                sub_regions = split_wide_region(roi, width_threshold=int(0.03 * license_plate.shape[1]))

            for idx, sub_roi in enumerate(sub_regions):
                sub_width = sub_roi.shape[1]
                sub_offset = x0_pad if idx == 0 else x0_pad + sum(sr.shape[1] for sr in sub_regions[:idx])
                rect_border = patches.Rectangle(
                    (sub_offset, y0_pad),
                    sub_width,
                    y1_pad - y0_pad,
                    edgecolor="red",
                    linewidth=2,
                    fill=False,
                )
                ax1.add_patch(rect_border)

                resized_char = resize(sub_roi, (40, 40))
                characters.append(resized_char)
                column_list.append(sub_offset)
            
    return characters, column_list
    # plt.show()
