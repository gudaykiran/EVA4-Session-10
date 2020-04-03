# EVA4-Session-10
1. Importing all the packages.

2. Modules are stored in a library which is available in Session 8 repository folder named Library.

For Session 10 the new modules are Data Transforms which includes data albumentations and gradcam module.

Modules are 1. Loading Data 2. Transforming the data (including albumenatation transforms) 3. Building the Basic Model 4. ResNet-18 Model 5. Including LR finder to find the learning rate 6. Training and testing the model 7. Plotting the Training AND Test accuracies. 8. Plotting the Misclassified images 9. Gradcam module and results with misclassified images.

     Modules Description:      
(i) Loading CIFAR 10 Datasets.

(ii) Transformations done:

Albumentations :

Rotate,
Horizontal flip,
RGB Shift,
Normalize,
Cutout strategy
(iii) Loading and running Train dataset and Test dataset:

#Implementing L1 regularization if self.L1lambda > 0: reg_loss = 0. for param in self. model. parameters (): reg_loss += torch. sum (param. abs ()) loss += self. L1lambda * reg_loss

We have used SGD Optimizer with learning rate 0.01% and Momentum 0.9 and included LR Scheduler (max learning rate 0.5)

Model parameters are 11,173,962

We used LR finder here : Benefits of Learning rates:

Converge faster to local minima

To get higher accuracy. The training should start from a relatively large learning rate in the beginning, as random weights are far from optimal, and then the learning rate can be decreased during training to allow for more fine-grained weight updates. The trick is to train a network starting from a low learning rate and increase the learning rate exponentially for every batch. Record the learning rate and training loss for every batch. Then, plot the loss and the learning rate. Typically, it looks like this:

Ran the model with 50 epochs

Train set: Average loss: 0.0014, Accuracy: 93.95%; Test set: Average loss: 0.3081, Accuracy: 90.84%

Final Accuracy of the model is 90.84%

Albumentations:

We can improve the performance of the model by augmenting the data we already have.

Albumentations package

Albumentations package is written based on numpy, OpenCV, and imgaug. It is a very popular package written by Kaggle masters and used widely in Kaggle competitions. Moreover, this package is very efficient. You may find the benchmarking results here and the full documentation for this package here. Albumentations package is capable of: • Over 60 pixel-level and spatial-level transformations; • Transforming images with masks, bounding boxes, and keypoints; • Organizing augmentations into pipelines; • PyTorch integration.

PyTorch Integration

When using PyTorch you can effortlessly migrate from torchvision to Albumentatios because this package provides specialized utilities to use with PyTorch. Migrating to Albumentations helps to speed up the data generation part and train deep learning models faster.

One of the Pytorch integration example:

Import pytorch utilities from albumentations
from albumentations.pytorch import ToTensor

Define the augmentation pipeline
augmentation_pipeline = A.Compose( [ A.HorizontalFlip(p = 0.5), # apply horizontal flip to 50% of images A.OneOf( [ # apply one of transforms to 50% of images A.RandomContrast(), # apply random contrast A.RandomGamma(), # apply random gamma A.RandomBrightness(), # apply random brightness ], p = 0.5 ),

    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    
    ToTensor() # convert the image to PyTorch tensor
],
p = 1
)

Load the augmented data
Define the demo dataset
class DogDataset2(Dataset): ''' Sample dataset for Albumentations demonstration. The dataset will consist of just one sample image. '''

def __init__(self, image, augmentations = None):
    self.image = image
    self.augmentations = augmentations # save the augmentations

def __len__(self):
    return 1 # return 1 as we have only one image

def __getitem__(self, idx):
    # return the augmented image
    # no need to convert to tensor, because image is converted to tensor already by the pipeline
    augmented = self.augmentations(image = self.image)
    return augmented['image']
Initialize the dataset, pass the augmentation pipeline as an argument to init function
train_ds = DogDataset2(image, augmentations = augmentation_pipeline)

Initilize the dataloader
trainloader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)

Grad – CAM: Gradient-weighted Class Activation Mapping (Grad-CAM), uses the class-specific gradient information flowing into the final convolutional layer of a CNN to produce a coarse localization map of the important regions in the image. Grad-CAM is a strict generalization of the Class Activation Mapping. Unlike CAM, Grad-CAM requires no re-training and is broadly applicable to any CNN-based architectures. We take the final convolutional feature map, and then we weight every channel in that feature with the gradient of the class with respect to the channel. It tells us how intensely the input image activates different channels by how important each channel is with regard to the class. It does not require any re-training or change in the existing architecture.

Steps:

Load a pre-trained model
Load an image which can be processed by this model
Infer the image and get the topmost class index
Take the output of the final convolutional layer
Compute the gradient of the class output value w.r.t to L feature maps
Pool the gradients over all the axes leaving out the channel dimension
Weigh the output feature map with the computed gradients (+ve)
Average the weighted feature maps along channels
Normalize the heat map to make the values between 0 and 1
