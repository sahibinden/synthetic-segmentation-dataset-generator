import os
import glob
from configparser import ConfigParser
import random
from tqdm import tqdm

from PIL import Image
from PIL import ImageStat
import numpy as np
import cv2 as cv


class SyntheticSegmentationDatasetGenerator:
    """
    A class for synthetic segmentation dataset generator.
    """

    def __init__(self):
        """
        Initializes config file and related object fields.
        """

        self.config = ConfigParser()
        self.config.read('dataset_generator_parameters.ini')

        self.background_images_folder = self.config['synthetic_segmentation_dataset_generator_params'].get(
            'background_images_folder')
        self.object_image_folder = self.config['synthetic_segmentation_dataset_generator_params'].get(
            'segmentation_object_image_folder')
        self.object_mask_folder = self.config['synthetic_segmentation_dataset_generator_params'].get(
            'segmentation_object_mask_folder')

        self.synthetic_dataset_folder = self.config['synthetic_segmentation_dataset_generator_params'].get(
            'synthetic_dataset_output_folder')

        self.mask_image_reuse_count = self.config['synthetic_segmentation_dataset_generator_params'].getint(
            'mask_image_reuse_count')

        self.use_color_correction = self.config['synthetic_segmentation_dataset_generator_params'].get(
            'use_color_correction')

        self.color_correction_type = self.config['synthetic_segmentation_dataset_generator_params'].get(
            'color_correction_type')

        self.max_object_count_per_image = self.config['synthetic_segmentation_dataset_generator_params'].getint(
            'max_object_count_per_image')

        self.object_min_height_threshold = self.config['synthetic_segmentation_dataset_generator_params'].getfloat(
            'object_min_height_threshold')

    def resize_background(self, background_image, object_image_height):
        """
        Resizes background image according to object image width and height.
        The height of the background image will be twice of the object image height.

        :param background_image     :background image to resize
        :param object_image_height  :object image height value
        :return                     :resized background image
        """

        org_width, org_height = background_image.size
        new_height = int(object_image_height * 2)
        resize_coefficient = org_height / new_height
        new_width = int(org_width / resize_coefficient)

        background_image = background_image.resize((new_width, new_height))

        return background_image

    def crop_object_and_mask(self, object_image, mask):
        """
        Crops both object image and mask image according to bound box of the white area in the mask image.

        :param object_image :object image
        :param mask         :mask image
        :return             :cropped object image and cropped mask image
        """

        object_image_array = np.asarray(object_image)
        mask_image_array = np.asarray(mask)

        thresh = cv.threshold(mask_image_array, 128, 255, cv.THRESH_BINARY)[1]

        contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)

        mask_crop = mask_image_array[y:y + h, x:x + w]
        object_image_crop = object_image_array[y:y + h, x:x + w]

        return Image.fromarray(object_image_crop), Image.fromarray(mask_crop)

    def make_object_background_transparent(self, image, mask):
        """
        Makes object image background transparent according to object mask.

        :param image    : object image
        :param mask     : mask image
        :return         : object image with transparent background
        """

        original_image_rgba = image.convert("RGBA")
        mask_image_rgba = mask.convert("RGBA")

        width_mask, height_mask = mask_image_rgba.size

        org_pixdata = original_image_rgba.load()
        mask_pixdata = mask_image_rgba.load()

        for y in range(height_mask):
            for x in range(width_mask):
                if mask_pixdata[x, y] == (0, 0, 0, 255):
                    org_pixdata[x, y] = (0, 0, 0, 0)

        return original_image_rgba, mask

    def adjust_object_contrast(self, background_image, image_mask, object_image, object_mask, paste_location, mode):
        """
        Change color of the object image based on the color of the area the object is being pasted on.

        :param background_image: background image
        :param image_mask: image mask
        :param object_image: object image
        :param object_mask: object mask
        :param paste_location: coordinates the object is going to be pasted on
        :param mode: mode of the color adjustment
        :return: color adjusted object image
        """
        background_image = np.array(background_image)
        mask_image = np.array(image_mask)
        mask_image = np.delete(mask_image, np.s_[3:], axis=2)
        background_image[mask_image == 0] = 0

        object_image = np.array(object_image)
        object_image_slice = object_image[..., 3].reshape(object_image.shape[0], object_image.shape[1], 1)
        object_image = cv.cvtColor(object_image, cv.COLOR_BGRA2RGB)
        object_mask = np.array(object_mask)
        object_image[object_mask == 0] = 0

        background_image_cropped = background_image[paste_location[1]:paste_location[1] + object_image.shape[0],
                                   paste_location[0]:paste_location[0] + object_image.shape[1]]

        mask_image_cropped = mask_image[paste_location[1]:paste_location[1] + object_image.shape[0],
                             paste_location[0]:paste_location[0] + object_image.shape[1]]

        if mode == "contrast":
            img_contrast = np.sum(background_image_cropped) / (np.count_nonzero(mask_image_cropped == 255))
            object_contrast = np.sum(object_image) / (np.count_nonzero(mask_image_cropped == 255))
            contrast_ratio = object_contrast / img_contrast
            object_image_adjusted = cv.convertScaleAbs(object_image, alpha=(1 / contrast_ratio), beta=0)

        elif mode == "luminance":
            hsv_background_image = cv.cvtColor(background_image_cropped, cv.COLOR_BGR2HSV)
            img_brightness = hsv_background_image[..., 2].mean()
            hsv_object_image = cv.cvtColor(object_image, cv.COLOR_BGR2HSV)
            object_brightness = hsv_object_image[..., 2].mean()
            brightness_dif = img_brightness - object_brightness
            object_image_adjusted = cv.convertScaleAbs(object_image, alpha=1, beta=brightness_dif)

        else:
            print("Contrast mode is not defined.")

        object_image_adjusted = np.c_[object_image_adjusted, object_image_slice]
        object_image_adjusted = cv.cvtColor(object_image_adjusted, cv.COLOR_BGRA2RGBA)

        return Image.fromarray(object_image_adjusted)

    def merge_object_and_background(self, background_image_name, object_image_names, mask_image_names,
                                    output_image_name):
        """
        Pre-processes the given object image and places it in a random position within the given background image.
        Also updates the mask image accordingly.
        Saves both the merged image and related mask image.

        :param background_image_name    :name of the background image
        :param object_image_names       :names of the object image
        :param mask_image_names         :names of the mask image
        :param output_image_name        :name of the output image
        """
        background_image = Image.open(self.background_images_folder + "/" + background_image_name)

        merged = False
        background_resized = False

        for object_index, object_image_name in enumerate(object_image_names):
            object_image = Image.open(self.object_image_folder + "/" + object_image_name)
            mask_image = Image.open(self.object_mask_folder + "/" + mask_image_names[object_index])

            # crop the object image according to object mask bounding box
            object_image, mask_image = self.crop_object_and_mask(object_image, mask_image)

            # make transparent the object image's background
            object_image, mask_image = self.make_object_background_transparent(object_image, mask_image)
            object_image_width, object_image_height = object_image.size

            if not background_resized:
                # resize selected background image according to the size of object image
                background_image = self.resize_background(background_image, object_image_height)
                background_width, background_height = background_image.size
                new_mask = Image.new('RGBA', (background_width, background_height))
                background_resized = True

            if (background_width >= object_image_width) & (background_height >= object_image_height) & (
                    object_image_height >= self.object_min_height_threshold):
                paste_location_width = random.randint(0, (background_width - object_image_width))

                # paste_location_height = random.randint((background_height - object_image_height - 40),
                #                                        (background_height - object_image_height))
                paste_location_height = (background_height - object_image_height)
                paste_location = (paste_location_width, paste_location_height)

                new_mask.paste(mask_image, paste_location)

                # generate a new mask related to merged background and object image
                if self.use_color_correction == "True":
                    object_image = self.adjust_object_contrast(background_image, new_mask, object_image, mask_image,
                                                               paste_location, self.color_correction_type)
                # recoloring the mask image
                mask_array = np.array(mask_image)
                mask_array[mask_array > 0] = 255 - 30 * object_index
                mask_image = Image.fromarray(mask_array)
                new_mask.paste(mask_image, paste_location)

                # place object image to the background image according to paste location
                background_image.paste(object_image, paste_location, object_image)

                merged = True

        if merged:
            # save mask image and merged image
            background_image.save(self.synthetic_dataset_folder + "/image/" + output_image_name + ".png")
            new_mask.convert("RGB").save(self.synthetic_dataset_folder + "/mask/" + output_image_name + "_mask.png")

    def get_image_related_to_mask(self, mask_name):
        """
        Gets related object image name using mask image name (suitable for the PennFudan dataset)

        :param mask_name    :mask name
        :return             :returns related object image name
        """

        # gets the related object image name
        object_name = mask_name.split("_")[0] + ".png"
        return object_name

    def generate_dataset(self):
        """
        Generates a synthetic segmentation dataset using background images and object images & masks.
        """

        # generate necessary folders
        if not os.path.isdir(self.synthetic_dataset_folder):
            os.mkdir(self.synthetic_dataset_folder)
        if not os.path.isdir(self.synthetic_dataset_folder + "/image"):
            os.mkdir(self.synthetic_dataset_folder + "/image")
        if not os.path.isdir(self.synthetic_dataset_folder + "/mask"):
            os.mkdir(self.synthetic_dataset_folder + "/mask")

        # get background images list
        backgrounds = os.listdir(self.background_images_folder)
        backgrounds = sorted(backgrounds)

        # get mask images list
        masks = os.listdir(self.object_mask_folder)
        masks = sorted(masks)

        for counter, mask_name in tqdm(enumerate(masks)):
            object_image_names = []
            mask_names = []

            # get original object image related to the mask image
            object_image_name = self.get_image_related_to_mask(mask_name)
            object_image_names.append(object_image_name)
            mask_names.append(mask_name)

            object_count = random.randint(1, self.max_object_count_per_image)

            for o_count in range(object_count - 1):
                additional_mask_index = random.randint(0, len(masks) - 1)
                additional_mask_name = masks[additional_mask_index]
                additional_object_image_name = self.get_image_related_to_mask(additional_mask_name)
                object_image_names.append(additional_object_image_name)
                mask_names.append(additional_mask_name)

            # for each mask, generate synthetic image as many reuse count
            for r_count in range(self.mask_image_reuse_count):
                # select a random background image to merge with related object image
                selected_background_index = random.randint(0, len(backgrounds) - 1)
                background_image_name = backgrounds[selected_background_index]

                output_image_name = str((counter * self.mask_image_reuse_count) + r_count)

                # merge the selected background image with the related object image
                self.merge_object_and_background(background_image_name, object_image_names, mask_names,
                                                 output_image_name)


SyntheticSegmentationDatasetGenerator().generate_dataset()
