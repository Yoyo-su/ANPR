import pytest
import numpy
from src.utils.localisation import convert_images
from src.utils.cca import draw_bounding_box


@pytest.mark.describe("Tests for convert_images function")
class TestConvertImages:
    @pytest.mark.it("Test convert_images function with a valid image path")
    def test_display_image_valid_path(self):
        result = convert_images("tests/data/test_image.jpg")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(arr, numpy.ndarray) for arr in result)

    @pytest.mark.it("Test convert_images function with an invalid image path")
    def test_display_image_invalid_path(self):
        with pytest.raises(Exception):
            convert_images("invalid_path.jpg")

    @pytest.mark.it("Test convert_images function with an invalid file type")
    def test_display_image_invalid_file_type(self):
        with pytest.raises(Exception):
            convert_images("tests/data/test_image.txt")


@pytest.mark.describe("Tests for draw_bounding_box function")
class TestConnectedComponentAnalysis:
    @pytest.mark.it(
        "Test draw_bounding_box function with valid binary and grayscale images"
    )
    def test_draw_bounding_box_valid_images(self):
        binary_image, gray_image = convert_images("tests/data/test_image.jpg")
        result = draw_bounding_box(binary_image, gray_image)
        assert isinstance(result, list)
        assert all(isinstance(arr, numpy.ndarray) for arr in result)

    @pytest.mark.it("Test draw_bounding_box function with invalid binary image")
    def test_draw_bounding_box_invalid_binary_image(self):
        _, gray_image = convert_images("tests/data/test_image.jpg")
        with pytest.raises(Exception):
            draw_bounding_box(None, gray_image)

    @pytest.mark.it("Test draw_bounding_box function with invalid grayscale image")
    def test_draw_bounding_box_invalid_grayscale_image(self):
        binary_image, _ = convert_images("tests/data/test_image.jpg")
        with pytest.raises(Exception):
            draw_bounding_box(binary_image, None)
