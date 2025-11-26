from skimage.io import imread
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt


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
