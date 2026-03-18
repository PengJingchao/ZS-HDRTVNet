import math
from typing import Optional, Tuple

import cv2
import numpy as np


class AddNoise:
    def __init__(self, prob: float = 0.01, gamma_range: Tuple[float, float] = (0.5, 2.0)):
        self.prob = prob
        self.gamma_range = gamma_range

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.uniform(0.0, 1.0) > 0.5:
            return image

        noisy = np.copy(image)
        mask = np.random.rand(*image.shape[:2]) < self.prob
        if not np.any(mask):
            return noisy

        gamma_values = np.random.uniform(
            self.gamma_range[0],
            self.gamma_range[1],
            size=(noisy.shape[0], noisy.shape[1], 1),
        )
        noisy[mask] = np.clip(255.0 * ((noisy[mask] / 255.0) ** gamma_values[mask]), 0.0, 255.0)
        return noisy


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            image = np.fliplr(image)
        return image


class RandomRotation:
    def __init__(self, degrees):
        if isinstance(degrees, (int, float)):
            degrees = (-degrees, degrees)
        self.degrees = degrees

    def __call__(self, image: np.ndarray) -> np.ndarray:
        angle = np.random.uniform(self.degrees[0], self.degrees[1])
        height, width = image.shape[:2]
        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )


class RandomColorDisturbance:
    def __init__(
        self,
        probability: float = 0.3,
        brightness_range: Tuple[float, float] = (-25.0, 25.0),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        saturation_range: Tuple[float, float] = (0.8, 1.1),
        hue_range: Tuple[float, float] = (-10.0, 10.0),
    ):
        self.probability = probability
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range

    def __call__(self, ldr_img: np.ndarray, hdr_img: np.ndarray):
        if np.random.uniform(0.0, 1.0) > self.probability:
            return ldr_img, hdr_img

        disturbance_type = np.random.choice(["brightness_and_contrast", "saturation", "hue"])
        if disturbance_type == "brightness_and_contrast":
            beta = np.random.uniform(*self.brightness_range)
            ldr_img = cv2.convertScaleAbs(ldr_img, beta=beta)
            hdr_img = hdr_img + beta
            alpha = np.random.uniform(*self.contrast_range)
            ldr_img = cv2.convertScaleAbs(ldr_img, alpha=alpha)
            hdr_img = hdr_img * alpha
        elif disturbance_type == "saturation":
            factor = np.random.uniform(*self.saturation_range)
            ldr_img = self._adjust_saturation(ldr_img, factor)
            hdr_img = self._adjust_saturation(hdr_img, factor)
        else:
            shift = np.random.uniform(*self.hue_range)
            ldr_img = self._adjust_hue(ldr_img, shift)
            hdr_img = self._adjust_hue(hdr_img, shift)
        return ldr_img, hdr_img

    @staticmethod
    def _adjust_saturation(image: np.ndarray, factor: float) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], factor)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    @staticmethod
    def _adjust_hue(image: np.ndarray, shift: float) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[:, :, 0] = hsv[:, :, 0] + shift
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


class RandomErasing:
    def __init__(self, probability: float = 0.5, sl: float = 0.02, sh: float = 0.4, r1: float = 0.3, mean=None):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.mean = [0.0] if mean is None else mean

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.uniform(0.0, 1.0) > self.probability:
            return image

        height, width = image.shape[:2]
        area = height * width
        for _ in range(100):
            target_area = np.random.uniform(self.sl, self.sh) * area
            aspect_ratio = np.random.uniform(self.r1, 1.0 / self.r1)
            erase_h = int(round(math.sqrt(target_area * aspect_ratio)))
            erase_w = int(round(math.sqrt(target_area / aspect_ratio)))
            if erase_w >= width or erase_h >= height:
                continue
            top = np.random.randint(0, height - erase_h)
            left = np.random.randint(0, width - erase_w)
            image = np.copy(image)
            fill_value = 0.0 if np.issubdtype(image.dtype, np.floating) else 0
            if image.ndim == 3:
                image[top : top + erase_h, left : left + erase_w, :] = fill_value
            else:
                image[top : top + erase_h, left : left + erase_w] = fill_value
            return image
        return image


class RandomResizedCrop:
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img_shape, scale, ratio):
        height, width = img_shape
        area = height * width
        for _ in range(10):
            target_area = area * np.random.uniform(scale[0], scale[1])
            aspect_ratio = np.exp(np.random.uniform(math.log(ratio[0]), math.log(ratio[1])))
            crop_width = int(round(math.sqrt(target_area * aspect_ratio)))
            crop_height = int(round(math.sqrt(target_area / aspect_ratio)))
            if 0 < crop_width <= width and 0 < crop_height <= height:
                top = np.random.randint(0, height - crop_height + 1)
                left = np.random.randint(0, width - crop_width + 1)
                return top, left, crop_height, crop_width

        in_ratio = width / height
        if in_ratio < min(ratio):
            crop_width = width
            crop_height = round(crop_width / min(ratio))
        elif in_ratio > max(ratio):
            crop_height = height
            crop_width = round(crop_height * max(ratio))
        else:
            crop_width = width
            crop_height = height
        top = (height - crop_height) // 2
        left = (width - crop_width) // 2
        return top, left, crop_height, crop_width

    def __call__(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        top, left, crop_height, crop_width = self.get_params((height, width), self.scale, self.ratio)
        image = image[top : top + crop_height, left : left + crop_width]
        size = (self.size, self.size) if isinstance(self.size, int) else self.size
        return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


class RandomAffine:
    def __init__(self, degrees, translate=None, scale=None, shear=None):
        self.degrees = degrees if isinstance(degrees, tuple) else (-degrees, degrees)
        self.translate = translate
        self.scale = scale
        self.shear = shear

    @staticmethod
    def get_params(
        degrees: Tuple[float, float],
        translate: Optional[Tuple[float, float]],
        scale_ranges: Optional[Tuple[float, float]],
        shears: Optional[Tuple[float, float]],
        img_size: Tuple[int, int],
    ):
        angle = float(np.random.uniform(degrees[0], degrees[1]))
        if translate is not None:
            max_dx = float(translate[0] * img_size[0])
            max_dy = float(translate[1] * img_size[1])
            translations = (
                int(round(np.random.uniform(-max_dx, max_dx))),
                int(round(np.random.uniform(-max_dy, max_dy))),
            )
        else:
            translations = (0, 0)

        scale = float(np.random.uniform(scale_ranges[0], scale_ranges[1])) if scale_ranges is not None else 1.0
        shear_x = shear_y = 0.0
        if shears is not None:
            shear_x = float(np.random.uniform(shears[0], shears[1]))
            if len(shears) == 4:
                shear_y = float(np.random.uniform(shears[2], shears[3]))
        return angle, translations, scale, (shear_x, shear_y)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        angle, translations, scale, shear = self.get_params(
            self.degrees,
            self.translate,
            self.scale,
            self.shear,
            (width, height),
        )
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
        matrix[:, 2] += np.array(translations)
        if shear[0] != 0.0 or shear[1] != 0.0:
            shear_matrix = np.array([[1.0, shear[0], 0.0], [shear[1], 1.0, 0.0], [0.0, 0.0, 1.0]])
            matrix = np.dot(shear_matrix, np.vstack([matrix, [0.0, 0.0, 1.0]]))[:2]
        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if isinstance(self.size, int):
            size = (self.size, self.size)
        else:
            size = tuple(self.size)
        return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
