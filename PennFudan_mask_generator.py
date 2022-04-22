from PIL import Image
import numpy as np
import glob
import os
import shutil
from configparser import ConfigParser

class PennFudanMaskGenerator:
    """
    A class for PennFudan mask generator.
    """

    def __init__(self):
        self.config = ConfigParser()
        self.config.read('dataset_generator_parameters.ini')

        self.mask_path = self.config['synthetic_segmentation_dataset_generator_params'].get(
            'PennFudan_mask_folder') + "*.png"
        self.mask_folder_name = self.config['synthetic_segmentation_dataset_generator_params'].get(
            'segmentation_object_mask_folder') +"/"#PennFudanPed/mask_folder/"

    def generate_masks(self):
        """
        Takes Penn Fudan image masks and generates seperate mask for each pedestrian in image.
        Saves generated masks under the PennFudanPed/mask_folder folder
        """
        if os.path.exists(self.mask_folder_name) and os.path.isdir(self.mask_folder_name):
            shutil.rmtree(self.mask_folder_name)
            print('old result folder removed:', self.mask_folder_name)
        os.mkdir(self.mask_folder_name)
        print('result folder created:', self.mask_folder_name)

        masks = glob.glob(self.mask_path)
        masks = sorted(masks)

        for msk in range(len(masks)):
            mask_name = masks[msk].split("/")[-1].split(".")[0]
            print(mask_name)

            mask = Image.open(masks[msk])
            mask = np.array(mask)
            unique_masks = np.unique(mask)
            unique_masks = unique_masks[1:]
            masks_in_image = mask == unique_masks[:, None, None]
            masks_in_image = masks_in_image.astype(int)

            for single_mask in range(len(masks_in_image)):
                masks_in_image[single_mask][masks_in_image[single_mask] > 0.5] = 255
                ped_mask = Image.fromarray(masks_in_image[single_mask].astype(np.uint8))
                ped_mask.save(self.mask_folder_name + mask_name +"_{}.png".format(single_mask))


PennFudanMaskGenerator().generate_masks()