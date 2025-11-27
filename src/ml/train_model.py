import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import joblib
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.transform import resize
from skimage.feature import hog


letters = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
            'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N','O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z'
        ]

target_size = (40, 40)  # each character image will be resized to 40x40 pixels


def read_training_data(training_directory):
    image_data = []
    target_data = []
    sub_dirs = os.listdir(training_directory)
    for sub_dir in sub_dirs:
        for each_letter in letters:
            for each in range(100):
                image_path = os.path.join(training_directory, sub_dir, each_letter, each_letter + '_' + str(each) + '.jpg')
                # read each image of each character
                img_details = imread(image_path, as_gray=True,)
                # resize each image to a standard size
                resized_image = resize(img_details, target_size, anti_aliasing=True, preserve_range=True)
                # converts each character image to binary image
                binary_image = resized_image < threshold_otsu(resized_image)
                # extract Histogram of Oriented Gradients features to capture stroke direction/shape of character
                hog_features = hog(
                    binary_image.astype("float32"),
                    orientations=9,
                    pixels_per_cell=(4, 4),
                    cells_per_block=(2, 2),
                    block_norm="L2-Hys",
                    visualize=False,
                    feature_vector=True,
                )
                image_data.append(hog_features) # append the hog features to image data
                target_data.append(each_letter) # append the corresponding label to target data

    return (np.array(image_data), np.array(target_data))

def cross_validation(model, num_of_fold, train_data, train_label):
    # this uses the concept of cross validation to measure the accuracy
    # of a model, the num_of_fold determines the type of validation
    # e.g if num_of_fold is 4, then we are performing a 4-fold cross validation
    # it will divide the dataset into 4 and use 1/4 of it for testing
    # and the remaining 3/4 for the training
    accuracy_result = cross_val_score(model, train_data, train_label,
                                      cv=num_of_fold)
    print("Cross Validation Result for ", str(num_of_fold), " -fold")

    print(accuracy_result * 100)


current_dir = os.path.dirname(os.path.realpath(__file__))

training_dataset_dir = os.path.join(current_dir, 'training_data')

image_data, target_data = read_training_data(training_dataset_dir)

# the kernel can be 'linear', 'poly' or 'rbf'
# the probability was set to True so as to show
# how sure the model is of it's prediction

# SVC model has been tuned to give better accuracy with the following parameters
svc_model = SVC(kernel='rbf', C=10, gamma=0.001, probability=True, class_weight='balanced')    
cross_validation(svc_model, 5, image_data, target_data)

# let's train the model with all the input data
svc_model.fit(image_data, target_data)

# we will use the joblib module to persist the model
# into files. This means that the next time we need to
# predict, we don't need to train the model again
save_directory = os.path.join(current_dir, 'models/svc/')
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
joblib.dump(svc_model, save_directory+'/svc.pkl')
