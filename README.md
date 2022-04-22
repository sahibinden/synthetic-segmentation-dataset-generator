Synthetic Segmentation Dataset Generator Using Realistic City Backgrounds
---------

A project to generate synthetic dataset with realistic city backgrounds for image segmentation problems.

### If you use this code in your research project, please cite our SİU 2022 paper:
```bibtex
@INPROCEEDINGS{sahibindensiu2022,
  author={İşlek, İrem and Aksaylı, N. Deniz and Güngör, Onur and Karaman, Çağla Çığ},
  booktitle={2022 30th Signal Processing and Communications Applications Conference (SIU)},
  title={Generating Synthetic Image Segmentation Dataset Using Realistic City Backgrounds},
  year={2022}
  }
```

### Setup - Install requirements

```bash
pipenv install -r requirements.txt
```

### For Background Images Dataset Generation

**Note:** This step is needed if you want to generate background images from a mp4 video. \
Please define the source video path for background images and the folder name \
for output background images using the dataset_generator_parameters.ini file.

```bash
pipenv run python background_images_generator.py
```

### For Object/Mask Dataset

**Note:** This step is needed if you want to use the pedestrian objects from the PennFudan Dataset. \
Otherwise assign the related path of your objects/masks dataset to the "segmentation_object_image_folder" \
and "segmentation_object_mask_folder" variables in the dataset_generator_parameters.ini file.

```bash
pipenv run python PennFudan_mask_generator.py
```

### For Synthetic Dataset Generation (Merging Background Images and Objects/Masks)

**Note:** This step merges background images with the object images and related image masks to generate a synthetic segmentation dataset. \
Please check synthetic_segmentation_dataset_generator_params from the dataset_generator_parameters.ini file \
to customize your synthetic dataset.

```bash
pipenv run python synthetic_segmentation_dataset_generator.py
```

### Parameters for Customization
These parameters can be changed from dataset_generator_parameters.ini file

**frame_delay:** determines the delay of taking frames from the background video (in seconds) \
**mask_image_reuse_count:** determines how many synthetic samples are created for each mask \
**object_min_height_threshold:** determines a threshold for using objects which has min n pixel height \
**max_object_count_per_image:** determines the maximum object count per image
