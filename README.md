# Interpretation of Lesional Detection via Counterfactual Generation

In this page, interpretation of lesional detection via counterfactual generation framework is devised to provide visual interpretation for classifying chest X-ray images.

![pic](https://user-images.githubusercontent.com/44894722/131521106-2b0a5823-3068-4de5-8e9a-b19768652564.png)

## Dependencies
* python 3.7.11
* numpy 1.20.3
* h5py 3.4.0
* pytorch 1.6.0
* Pillow 8.3.1
* scipy 1.6.2
* torchray 1.0.0.2
* torchvision 0.7.0
* opencv-python 4.5.3.56

## Pretrained Models
Due to upload limit of Github, this page does not include checkpoint files. Please send request email to seongyeop@kaist.ac.kr if the checkpoint files are needed.
Once you have the parameters, create two folders './parameters/CheXNet_ChestX-ray14/' and 'parameters/GAN_ChestX-ray14/' then locate each parameter in it.

## How to use

Here we provide the inference model for Chest X-ray 14 dataset lesional detection.

We include two samples of the dataset in './dataset/image' folder, and a text file including directory of the samples, class and bounding box information in './dataset/list'.
The format of the dataloader text file is as follows, '00017544_003.png 0 0 0 1 0 1 0 0 0 1 0 0 0 0 585.386666666667 178.953489583333 288.995555555556 584.817777777778 0 0 0 1 0 0 0 0 0 0 0 0 0 0', where the first component is the file name of the image, the second is multi-label ground truth one-hot vector, the third is the bounding box coordinate in (x,y,w,h) order, and the last component is the ground truth one-hot vector for the included bounding box.

For further inference with the full-dataset, please infer to 'https://www.kaggle.com/nih-chest-xrays/data' to download the dataset. The images are expected to be included in './dataset/' folder, however, the users can modify the dataset directory in main.py file by modifying the argument parser parameter.

By running the main.py, the lesional detection results are to be created in './outputs' folder. The outputs will be three for one image, where 'Original_Image_#' is the original chest X-ray image, 'Generated_image_#' is the generated image by GAN model, and 'CheXGAN_Visualization_LesionName_#' is the difference map between the original image and the generated image.
