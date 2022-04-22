import os
import glob
from configparser import ConfigParser
from tqdm import tqdm
import math

import cv2 as cv

class BackgroundImagesGenerator:
    """
    A class for background images generator.
    """

    def __init__(self):
        """
        Initializes config file and related object fields.
        """
        self.config = ConfigParser()
        self.config.read('dataset_generator_parameters.ini')
        self.video_folder = self.config['background_images_generator_params'].get('video_source_path')
        self.background_images_output_folder = self.config.get('background_images_generator_params', 'background_images_output_folder')
        self.background_img_name_prefix = self.config.get('background_images_generator_params', 'background_img_name_prefix')
        self.frame_delay = self.config['background_images_generator_params'].getint('frame_delay')
        self.max_image_count = self.config['background_images_generator_params'].getint('max_image_count')

    def max_label(self, name, folder):
        """
        Looks the given folder and check the files with pattern 'name_###.jpg' to extract the largest label present
        """

        path_pattern = os.path.join(folder, name + "_*.jpg")
        existing_files = glob.glob(path_pattern)
        if not existing_files:
            biggest_label = 0
        else:
            existing_labels = map(lambda s: int(s.split('_')[-1].split('.')[0]), existing_files)
            biggest_label = max(existing_labels)
        return biggest_label

    def extract_images_from_video(self, video, folder=None, delay=30, name="file", silent=True):
        """
        Read a downloaded video from its path and extract screenshots every few seconds, set by the delay parameter.
        Images are saved in the specified folder or the cwd if none is specified and a maximum number of
        screenshots can be specified. The files are named "name_##.jpg" and the labelling starts where it already stops
        in the folder.
        """

        video_capture = cv.VideoCapture(video)
        count = 0
        num_images = 0
        if not folder:
            folder = os.getcwd()
        label = self.max_label(name, folder)
        success = True
        fps = int(video_capture.get(cv.CAP_PROP_FPS))

        total_frame_count = video_capture.get(cv.CAP_PROP_FRAME_COUNT)
        max_image_count = math.floor(video_capture.get(cv.CAP_PROP_FRAME_COUNT)/(fps*delay))

        while success and num_images < max_image_count:
            success, image = video_capture.read()
            num_images += 1
            label += 1
            file_name = name + "_" + str(label) + ".jpg"
            path = os.path.join(folder, file_name)
            cv.imwrite(path, image)
            if cv.imread(path) is None:
                os.remove(path)
            else:
                if not silent:
                    print(f'Image successfully written at {path}')
            count += delay * fps
            video_capture.set(1, count)

    def generate_background_images_from_videos(self):
        """
        Takes each video from video source then generates images from them.
        """

        if not os.path.isdir(self.background_images_output_folder):
            os.mkdir(self.background_images_output_folder)

        print(os.listdir(self.video_folder))
        for video in tqdm(os.listdir(self.video_folder)):
            if video.endswith(".mp4"):
                self.extract_images_from_video(self.video_folder+video, self.background_images_output_folder, self.frame_delay,
                                               self.background_img_name_prefix)


BackgroundImagesGenerator().generate_background_images_from_videos()