#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)


[image0]: ./examples/input.png "Input"
[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/grayscale.png "NormalisedGrayscale"
[image3]: ./examples/lenet.png "Lenet"
[image4]: ./test_examples/P1.png "Traffic Sign 1"
[image5]: ./test_examples/P2.png "Traffic Sign 2"
[image6]: ./test_examples/P3.png "Traffic Sign 3"
[image7]: ./test_examples/P4.png "Traffic Sign 4"
[image8]: ./test_examples/P5.png "Traffic Sign 5"
[image9]: ./test_examples/P6.png "Traffic Sign 6"
[image10]: ./test_examples/P7.png "Traffic Sign 7"
[image11]: ./test_examples/P8.png "Traffic Sign 8"
[image12]: ./test_examples/P9.png "Traffic Sign 9"
[image13]: ./test_examples/P10.png "Traffic Sign 10"
[image14]: ./examples/collected_test_data.png "Collected Test Data"
[image15]: ./examples/collected_test_data_labeled.png "Collected Test Data Labeled"
[image16]: ./examples/prediction.png "Prediction"
[image17]: ./examples/top5_softmax1.png "Top5 Softmax 1"
[image18]: ./examples/top5_softmax2.png "Top5 Softmax 2"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/nav13n/carnd/tree/master/CarND-LaneLines-P1/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 samples
* The size of the validation set is 4410 samples
* The size of test set is 12630 samples
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

Below is an example of a unique sample of each class present in the training set:

![UniqueSignClasses][image0]


#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed between different classes.

![Training Data Distribution Among Classes][image1]

Apparently, the training data set is skewed with respect to particular classes. Some of the classes contain more than 
2000 samples while some has just 180 samples.This imbalance of classes in training set would introduce bias in our 
training. To eliminate this bias and train better, we can augment the data set with additional data generation through 
flipping, rotating, scaling, shearing, affine transformations  and  perspective transformation of images classes which are underrepresented.


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I have applied following pre-processing steps to the image data:

1. Normalisation: I have first applied normalization to take away the effect of brightness/contrast variations and 
make features of the images more apparent. Normalising the image values also helps in converging the gradients faster. 
After experimenting with basic mean and standard deviation based techniques, I settled for  
Contrast Limited Adaptive Histogram Equalization which gave me the best result for feature set.

2. Grayscale Conversion: To simplify and make training process efficient, I have converted the normalised images from RGB to grayscale. 
The rationale behind the operation was that If we had three color channels in the input image, we would need to perform 
any operation on all the channels and then combine the results. Hence converting to grayscale would reduces the amount of
data to be processed by about 1/3 and allows the algorithm to run faster. However it comes at the cost of throwing away 
data (color data) that may be very helpful or required for many image processing applications. In my exploration, 
I didn't get any better results by preserving color information. 

Here is an example of applying pre-processing to all the input images.

![NormalisedGrayscale][image2]

3. One hot encoding the labels: Since the label just contains class identifiers while cross entropy computation expects 
one hot encoded value, I have applied on hot encoding to the label of training data

I have refrained from using additional data at first to better understand effect of other parameters like 
learning rate, dropouts, regularization, batch size and tune the network with these constraints to perform best. 
The rationale is a fine tuned model would anyway get better with addition of data.   
In the interest of time, I would explore this in a future revisit of this project. 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

#### LeNet

The neural network model is based on [LeNet](http://yann.lecun.com/exdb/lenet/) by Yann LeCun.  It is a convolutional neural network designed to recognize visual patterns directly from pixel images with minimal preprocessing.  It can handle hand-written characters very well. 

![LeNet][image3]

Source: http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

- The inputs are 32x32 (Black-White 1 channel) images
- LeNet uses 2x2 sub-sampling (valid padding, max pooling) in convolutional layers (no dropout)
- LeNet uses a sigmoid squashing function - a scaled hyperbolic tangent: $Atanh(Sa)$ where A=1.7159 and S=2/3
- LeNet has one 7x12 bitmap for each class, which is a label.  There are 10 classes (the digits '0' - '9')
- 7x12 = 84 which is why the output is 84
- The output is compared with all the labels (bitmaps) to calculate the error
- The class with the smallest error is an estimated digit value

#### Modified Model 

Our model is adapted from the LeNet as follows.  

- The input to the model are 32x32 grayscale images
- The activation function is ReLU except for the output layer which uses Softmax
- The output has 43 classes

|Layer                       | Ouput    |
|----------------------------|:--------:|
|Input                       | 32x32x1  |
|Convolution (valid, 5x5x6)  | 28x28x6  |
|Activation  (ReLU)          | 28x28x6  |
|Max Pooling (valid, 2x2)    | 14x14x6  |
|Convolution (valid, 5x5x16) | 10x10x16 |
|Activation  (ReLU)          | 10x10x16 |
|Max Pooling (valid, 2x2)    | 5x5x16   |
|Flatten                     | 400      |
|Dense                       | 120      |
|Activation  (ReLU)          | 120      |
|Dropout                     | 120      |
|Dense                       | 84       |
|Activation  (ReLU)          | 84       |
|Dropout                     | 84       |
|Dense                       | 43       |
|Activation  (Softmax)       | 43       |


#### Regularization

In addition to the changes above, I have used following regularization techniques to minimize overfitting of training data:

Dropout. A dropout of value 0.5 is applied to fully connected layers 3 and 4 while training. It's not applied to convolution layers as shared weights in convolutional layers are good regularizers themselves. 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I started with basic LeNet model with 10 Epochs and 128 Batch size and learning rate of 0.001.

I have used AdamOptimizer as the cost function optimiser. The AdamOptimizer improves on traditional gradient descent by
using momentum (moving averages of the parameters), facilitating efficient dynamic adjustment of hyperparameters.

I got around 80% accuracy with this bare bones models without any kind of initial image pre processing. 

Adding training data pre-processing increased validation accuracy to around 90%. 

To improve the accuracy, I played with increasing epochs and reducing the batch size so that more granular features can 
be learned in different iterations. It increased the accuracy but the validation accuracy started swaying between high and low after certain epochs. This was a sign that 
learning rate was high. 

Decreasing the learning rate to 0.0001 and tested with 100 epochs and 64 batch size improved the
validation accuracy. However looking at the training vs validation accuracy ratio at higher epochs, It became apparent that 
model is overfitting the training data. 

To handle overfitting of the training data , I have added two dropout layers with 50% keep probability between fully connected layers 3-4 and 4-5.
The idea was to randomly drop components of neural network (outputs) from a layer of neural network and force more neurons
in each layer to learn the multiple characteristics of the neural network thereby generalising the learning process.


In the end, I got best results for 
1. Epoch Size : 100
2. Batch Size : 64
3. Learning Rate : 0.0001
4. Dropout : 0.5 while training

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.953
* test set accuracy of 0.938

Dicussed above.
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are a number of German traffic signs that I found on the web:

![collected test data][image14] 

Since collected examples contains images of similar geometrical pattern and varying clarity, they would be difficult to classify in prediction.
For example: Image 1 can be mistaken for 4, 5 and 7 for similar looking and unclear features
Likewise Image 2 can be mistaken for 8

A label annotated version of the image is as below:
![collected test data_labeled][image15] 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:
![prediction][image16] 

The model was able to correctly guess 8 of the 9 traffic signs, which gives an accuracy of approximately 90%. This compares favorably to the accuracy on the test set of 93.8%
It failed for speed limit sign of 60 kmph and inaccurately identified it speed limit sign of 50 kmph. The accurate version 
was its second best guess.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 20th cell of the Ipython notebook.
A detailed view of top 5 softmax probabilities is as below:
![top5 softmax][image17] 
![top5 softmax][image18] 


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


