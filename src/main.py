from src.utils.localisation import convert_images
from src.utils.cca import connected_component_analysis
import matplotlib.pyplot as plt

# Define the path to your image file here
image_file = "car.jpg"

# Call the function to convert the image into gray scale and binary and display them
binary_car_image, gray_car_image = convert_images(image_file)
# plt.show()

# Call the function to perform connected component analysis and display the results
connected_component_analysis(binary_car_image, gray_car_image)
plt.show()
