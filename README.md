This repository contains the code necessary to perform the training of a U-Net like model for the semantic segmentation task. Lightning is used to structure the training process.

For the model architecture defined in the `model.py` file I have used the code developed by 
Aladdin Persson (github.com/aladdinpersson) at https://www.youtube.com/watch?v=IHq1t7NxS8k&t=1412s.

Alladin's implementation:
https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet

The dataset used with this code is the same as the one proposed by the video 
https://www.kaggle.com/c/carvana-image-masking-challenge. 
Important: To use another dataset, changes in the CaravanaDataset class may be necessary.
