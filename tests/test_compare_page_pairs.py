"""Checks for the experimental pair comparison, independent of model weights."""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
from compare_page_pairs import model_input


def test_model_input_color_and_normalization():
    with TemporaryDirectory() as directory:
        path = Path(directory) / 'red.png'
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        image[:, :] = (0, 0, 255)  # OpenCV BGR
        assert cv2.imwrite(str(path), image)
        tensor = model_input(path)
        assert tensor.shape == (1, 3, 224, 224)
        np.testing.assert_allclose(tensor[0, :, 0, 0], [(1-.485)/.229, -.456/.224, -.406/.225], rtol=1e-5)


def test_missing_image_is_an_error():
    try:
        model_input(Path('/no-such-scanstudio-image.png'))
    except ValueError:
        return
    raise AssertionError('missing image should raise ValueError')


if __name__ == '__main__':
    test_model_input_color_and_normalization()
    test_missing_image_is_an_error()
    print('pair comparison tests passed')
