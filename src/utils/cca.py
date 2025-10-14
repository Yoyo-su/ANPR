from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def connected_component_analysis(binary_car_image, gray_car_image):
    """This function performs connected component analysis on a binary image,
    identifies connected regions, and draws bounding boxes around them on the grayscale image.

    Args:
        binary_car_image (numpy.ndarray): The binary image to analyze.
        gray_car_image (numpy.ndarray): The grayscale image on which to draw bounding boxes.
    Returns:
        None
    """
    try:
        # this gets all the connected regions and groups them together
        label_image = measure.label(binary_car_image)
        fig, (ax1) = plt.subplots(1)
        ax1.imshow(gray_car_image, cmap="gray")

        # regionprops creates a list of properties of all the labelled regions
        for region in regionprops(label_image):
            if region.area < 50:
                # if the region is so small then it's likely not a license plate
                continue

            # the bounding box coordinates
            minRow, minCol, maxRow, maxCol = region.bbox
            rectBorder = patches.Rectangle(
                (minCol, minRow),
                maxCol - minCol,
                maxRow - minRow,
                edgecolor="red",
                linewidth=2,
                fill=False,
            )
            ax1.add_patch(rectBorder)
            # let's draw a red rectangle over those regions
        # plt.show()
    except Exception as e:
        print("Error in connected component analysis")
        raise (e)
