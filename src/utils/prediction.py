import os
import joblib
import numpy as np
from skimage.filters import threshold_otsu
from skimage.feature import hog


LETTER_TO_DIGIT = {
    "O": "0",
    "D": "0",
    "Q": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "T": "7",
    "B": "8",
}

DIGIT_TO_LETTER = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "3": "B",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
    "9": "P",
}

PLATE_PATTERN_IN = [
    "alpha",
    "alpha",
    "digit",
    "digit",
    "alpha",
    "alpha",
    "digit",
    "digit",
    "digit",
    "digit",
]

PLATE_PATTERN_UK = [
    "alpha",
    "alpha",
    "digit",
    "digit",
    "alpha",
    "alpha",
    "alpha",
]

def prepare_character_image(char_img):
    """
    Convert the segmented character into the HOG feature vector used during training.
    """
    normalized = 1.0 - char_img.astype(float)
    thresh = threshold_otsu(normalized)
    binary = (normalized < thresh).astype("float32")
    features = hog(
        binary,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True,
    )
    return features.reshape(1, -1)


def best_label_for_expected_type(probabilities, classes, expected):
    " Find the best label matching the expected type (digit or alpha) based on probabilities."
    if expected not in {"digit", "alpha"}:
        return None
    candidates = []
    for idx, label in enumerate(classes):
        if expected == "digit" and label.isdigit():
            candidates.append((probabilities[idx], label))
        elif expected == "alpha" and label.isalpha():
            candidates.append((probabilities[idx], label))
    if not candidates:
        return None
    return max(candidates)[1]


def enforce_plate_pattern(labels, probabilities, classes):
    " Adjust the predicted labels to conform to known license plate patterns. "
    if len(labels) == len(PLATE_PATTERN_IN):
        pattern = PLATE_PATTERN_IN
    elif len(labels) == len(PLATE_PATTERN_UK):
        pattern = PLATE_PATTERN_UK
    else:
        return "".join(labels)

    adjusted = list(labels)
    for idx, expected in enumerate(pattern):
        ch = adjusted[idx]
        if expected == "digit" and not ch.isdigit():
            replacement = best_label_for_expected_type(probabilities[idx], classes, "digit")
            adjusted[idx] = replacement or LETTER_TO_DIGIT.get(ch.upper(), ch)
        elif expected == "alpha" and not ch.isalpha():
            replacement = best_label_for_expected_type(probabilities[idx], classes, "alpha")
            adjusted[idx] = replacement or DIGIT_TO_LETTER.get(ch, ch)
    return "".join(adjusted)


def predict_characters(characters, column_list):
    # load the model
    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_dir = os.path.join(current_dir, '../ml/models/svc/svc.pkl')
    model = joblib.load(model_dir)

    classification_result = []
    probabilities = []
    for each_character in characters:
        feature_vector = prepare_character_image(each_character)
        probas = model.predict_proba(feature_vector)[0]
        best_index = int(np.argmax(probas))
        classification_result.append(model.classes_[best_index])
        probabilities.append(probas)

    print(classification_result)

    plate_string = ''.join(classification_result)
    print(plate_string)

    # it's possible the characters are wrongly arranged
    # since that's a possibility, the column_list will be
    # used to sort the letters in the right order

    combined = list(zip(column_list, classification_result, probabilities))
    combined.sort(key=lambda item: item[0])
    sorted_columns = [item[0] for item in combined]
    sorted_labels = [item[1] for item in combined]
    sorted_probabilities = [item[2] for item in combined]
    rightplate_string = ''.join(sorted_labels)

    print(rightplate_string)

    post_processed = enforce_plate_pattern(sorted_labels, sorted_probabilities, model.classes_)
    if post_processed != rightplate_string:
        print(f"Pattern-adjusted plate: {post_processed}")
