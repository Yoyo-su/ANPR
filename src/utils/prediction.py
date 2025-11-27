import os
import joblib
import numpy as np
from skimage.filters import threshold_otsu


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

PLATE_PATTERN = [
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


def _prepare_character_image(char_img):
    """
    Apply the same preprocessing used during training: binarize via Otsu
    thresholding and flatten to a 1x400 vector.
    """
    normalized = 1.0 - char_img.astype(float)
    thresh = threshold_otsu(normalized)
    binary = (normalized < thresh).astype(np.uint8)
    return binary.reshape(1, -1)


def _enforce_plate_pattern(plate_string):
    if len(plate_string) != len(PLATE_PATTERN):
        return plate_string

    chars = list(plate_string)
    for idx, expected in enumerate(PLATE_PATTERN):
        ch = chars[idx]
        if expected == "digit" and not ch.isdigit():
            chars[idx] = LETTER_TO_DIGIT.get(ch.upper(), ch)
        elif expected == "alpha" and not ch.isalpha():
            chars[idx] = DIGIT_TO_LETTER.get(ch, ch)
    return "".join(chars)


def predict_characters(characters, column_list):
    # load the model
    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_dir = os.path.join(current_dir, "../ml/models/svc/svc.pkl")
    model = joblib.load(model_dir)

    classification_result = []
    for each_character in characters:
        feature_vector = _prepare_character_image(each_character)
        result = model.predict(feature_vector)
        classification_result.append(result[0])

    print(classification_result)

    plate_string = "".join(classification_result)
    print(plate_string)

    # it's possible the characters are wrongly arranged
    # since that's a possibility, the column_list will be
    # used to sort the letters in the right order

    column_list_copy = column_list[:]
    sorted_columns = sorted(column_list)
    rightplate_string = ""
    for each in sorted_columns:
        rightplate_string += plate_string[column_list_copy.index(each)]

    print(rightplate_string)

    post_processed = _enforce_plate_pattern(rightplate_string)
    if post_processed != rightplate_string:
        print(f"Pattern-adjusted plate: {post_processed}")
