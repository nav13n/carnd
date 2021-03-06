### **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[InitialDataDistribution]: ./examples/InitialDataDistribution.png "Initial Data Distribution"
[FinalDataDistribution]: ./examples/FinalDataDistribution.png" "Final Data Distribution"
[Model]: ./examples/Model.png "Model"
[Augmentation1]: ./examples/Augmentation1.png "Augmentation Image"
[Augmentation2]: ./examples/Augmentation2.png "Augmentation Image"
[LossVisualisation]: ./examples/LossVisualisation.png "Loss Visualisation"
[Video]: ./output.mp4 "Track1"

### 1. Project Setup

#### a. Setup

The project directory contains the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md  summarizing the results

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

#### b. Run
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

### 2. Model Architecture and Training Strategy

#### a. Model Architecture
I chose to use [comma.ai](https://github.com/commaai/research/blob/master/train_steering_model.py) model (Line 319-340) due to its simplicity. The network structure can be summarized as follows:

- A normalization layer on the top of the network to normalize the input images.
- Convolutional layer, 16 feature maps with a size of 8×8, an Exponential Linear Unit(elu) activation function.
- Convolutional layer, 32 feature maps with a size of 5×5, an elu activation function.
- Convolutional layer, 64 feature maps with a size of 5×5, an elu activation function.
- Flatten layer.
- Dropout set to 20%.
- ELU non-linearity layer
- Fully connected layer with 512 units and a elu activation function.
- Dropout set to 50%.
- ELU non-linearity layer
- Fully connected output layer with 1 unit and no activation function as this is a regression problem, not classification.


#### b. Reducing Overfitting

The model contains dropout layers in order to reduce overfitting (model.py (model.py lines 333 and 336).  
In addition to that, the model was trained and validated on different data sets and each epochs was fed a randomly augmented data set of 20000 images using generators to avoid overfitting. 
(model.py line 354-358)

#### c. Model Parameter Tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 340).

#### d. Training data

Training data was chosen to keep the vehicle driving on the road. 
I used a combination of center lane driving, recovering from the left and right sides of the road to simulate proper driving. 

Instead of trying to gather my own driving data, I decided to use the data provided by Udacity as a personal challenge. 
The dataset contains around 8036 frames of car driving on track 1
Each frame contains images from left, right, center camera, and corresponding steering, brake and throttle values. 

Using the 8036 samples of driving_log.csv file, I created an image_frames list and corresponding steering_angles list. Each element of image_frames list contains a list of center, left, right camera image files name in the same order. (line 41-63, model.py)
This dataset was randomly shuffled and further split into train and validation data set with 90-10 split ratio.

To efficiently train the network without loading all the images in one go, I made use of keras generators to generate data on the fly for each batch of epoch.
For each batch of the training epoch, the generator function randomly creates augmented training data for each slot of the batch size
,by picking a random input image frame and steering angle, loading one of the images among left, centre right with adjusted angle,
and and randomly applying augmentation functions on the loaded image frame and adjusted angle. (line 297-313, model.py)
The preprocessing of cropping and resizing the data was also performed in the generator function itself.

To make sure model never sees the validation set data, two different generator functions were created for training and validation set
using the training and validation set data.

The train and validation generator were finally used with keras fit_generator function train the model for 20000 epoch of batch size 32.

Visualising the generated training set shows pretty balanced dataset that allowed model to generalise well.
![FinalDataDistribution]. 

### 3. Solution Design Approach

As a first step, I created a bare-bones regression model with just one flattening and  dense layer with two epoch to get a feel of things. It barely worked but at least the car was moving.
Next, I tried with  LeNet](http://yann.lecun.com/exdb/lenet/) model with three epochs and the training data provided by Udacity.
On the first track, the car went straight to the lake. I needed to do some pre-processing.  
Apparently, a large part of the image contained unnecessary details about the environment, like top containing sky and bottom of the image containing hood of the car, thereby confusing rather helping the training, I decided to crop the images before passing in to the network. Also,
I added a lambda layer to normalise the input data to zero means. This step allowed the car to move a bit further, but it didn't get past the first turn. 

Visualising the dataset, it was apparent the data was highly biased towards 0 angle predictions.
![InitialDataDistribution]. 
To add more training data and also to balance the dataset, I decided to use image augmentation techniques as suggested [here]. (lines 102 -175)
More specifically, I have used brightness augmentation, shadow augmentation, flipping, random horizon and vertical shift of the image to generate augmented data.
Some example of the generated augmented data on random images from the dataset are as follows:
![Augmentation1]
![Augmentation2]
In addition to that, to help the car recover from sharp turn, recovery data was generated by adding a correction factor to the left camera images and subtracting a correction factor fom right camera images.
The rationale was that left camera would have to move slight right and right camera would have to move left to get back to lane.                        

As a next step, I experimented training with more powerful models like [nVidia Autonomous Car Group](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)  and comma.ai model with this augmented dataset.
Both worked pretty well but I decided to finally stick with [comma.ai](https://github.com/commaai/research/blob/master/train_steering_model.py) model as contrary to NVIDIA model, I was able to experiment with it on my humble macbook without the need of a GPU and ended up working pretty well.
Initially I trained for more than 10 epochs but ended up overfitting the dataset,  To combat the overfitting, I trained with less number of epochs and passed the randomly augmented data per epoch using keras fit_generator function. 
It ensured high variance while keeping the memory footprint low. I also reduced the number of epochs
raining for 6 epochs with 20000 images per epoch just took around 15 min on an average MacbookPro.
![LossVisualisation]


### 4. Final Model Architecture and Results

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

![Model]

The final step was to run the simulator to see how well the car was driving around track one. A video of the car driving on the track one is [here](./output.mp4)
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

