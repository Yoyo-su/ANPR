import numpy as np
from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import matplotlib.pyplot as plt


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

    return best_plate


def segment_characters(license_plate):
    """
    Segment characters from the detected license plate. The input can either be
    the list of plate-like objects (in which case the best candidate will be
    selected internally) or the already selected/inverted license plate array.
    """

    labelled_plate = measure.label(license_plate)

    fig, ax1 = plt.subplots(1)
    ax1.imshow(license_plate, cmap="gray")
    # the next two lines is based on the assumptions that the width of
    # a characters should be between 5% and 15% of the license plate,
    # and height should be between 35% and 60%
    # this will eliminate some
    character_dimensions = (
        0.35 * license_plate.shape[0],
        0.60 * license_plate.shape[0],
        0.05 * license_plate.shape[1],
        0.15 * license_plate.shape[1],
    )
    min_height, max_height, min_width, max_width = character_dimensions

    characters = []
    counter = 0
    column_list = []
    for regions in regionprops(labelled_plate):
        y0, x0, y1, x1 = regions.bbox
        region_height = y1 - y0
        region_width = x1 - x0

        if (
            region_height > min_height
            and region_height < max_height
            and region_width > min_width
            and region_width < max_width
        ):
            roi = license_plate[y0:y1, x0:x1]

            # draw a red bordered rectangle over the character.
            rect_border = patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0, edgecolor="red", linewidth=2, fill=False
            )
            ax1.add_patch(rect_border)

            # resize the characters to 20X20 and then append each character into the characters list
            resized_char = resize(roi, (20, 20))
            characters.append(resized_char)

            # this is just to keep track of the arrangement of the characters
            column_list.append(x0)

    # plt.show()
