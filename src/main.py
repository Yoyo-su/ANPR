from src.utils.localisation import convert_images
from src.utils.cca import draw_bounding_box, select_plate_using_vertical_projection, segment_characters
import matplotlib.pyplot as plt

# Define the path to your image file here
image_file = "car.jpg"

# Call the function to convert the image into gray scale and binary and display them
binary_car_image, gray_car_image = convert_images(image_file)
# plt.show()

# Call the function to perform connected component analysis and draw bounding boxes
plate_like_objects = draw_bounding_box(binary_car_image, gray_car_image)
# plt.show()

# Call the function to select the most probable license plate using vertical projection
license_plate = select_plate_using_vertical_projection(plate_like_objects)

# Call the function to segment characters from the detected plate-like objects
segment_characters(license_plate)
plt.show()
