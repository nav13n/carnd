##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG.png
[image3]: ./examples/sliding_window.png
[image4]: ./examples/sliding_window_1x.png
[image5]: ./examples/sliding_window_1.5x.png
[image6]: ./examples/sliding_window_2x.png
[image7]: ./examples/sliding_window_combined.png
[image8]: ./examples/heatmap.png
[image9]: ./examples/thresholded_heatmap.png
[image10]: ./examples/label.png
[image11]: ./examples/final.png
[image12]: ./examples/heatmap_all.png
[image13]: ./examples/labels_all.png
[image14]: ./examples/final_all_frames.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.
The code used in this project is inside a jupyter notebook[here.](./CarND-Vehicle-Detection.ipnyb)

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it! 


####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

####Exploring Dataset

I started by reading in all the `vehicle` and `non-vehicle` images in given udacity dataset.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Example Of Car and Non-Car Images][image1]


#### Computing Histogram Of Gradients
I then explored different color spaces and different HOG parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`) using `get_hog_features` method.  I grabbed a random images from each of the two classes and displayed them to get a feel for what the output looks like.

Here is an example of `get_hog_features` output on gray scale images of both classes with HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![HOG Feature Example][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I experimented with a number of different combinations of color spaces and HOG parameters and trained a linear SVM using different combinations of HOG features extracted from the color channels. I settled on my final choice of HOG parameters based upon the performance of the SVM classifier produced using them. 
I considered both accuracy and speed of the feature extraction as well as classification for the combinations. I discarded RGB color space, for its undesirable properties under changing light conditions. There was relatively little variation in the final accuracy when running the SVM with some of the individual channels of HSV,HLS and LUV. 
YUV and YCrCb provided very high accuracy but were at times unstable. I finally settled with YCrCb space and a low value of pixels_per_cell=(8,8). Using larger values of than orient=10 did not have a striking effect and only increased the feature vector. Similarly, using values larger than cells_per_block=(2,2) did not improve results, which is why these values were chosen.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

First under section titled 'Explore feature extraction parameters',  I first used `extract features` method on each image to extract HOG, Spatial and Color Histogram features and create a feature vector of length 6696. 
Following parameters were used to create the feature vector:
* colorspace = 'YCrCb' 
* orient = 10
* pix_per_cell = 8
* cell_per_block = 2
* hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
* spatial_size= (16,16)
* hist_bins=16
Next, I performed normalisation on feature vector and split the data into train and test set with 80-20 ratio.
Next, I trained a linear SVM classifier on training set as mentioned under section titled 'Training a linear SVM classifier with extracted image features'
Accuracy of the trained classifier on test set of SVC =  0.9868 and it took around 0.00346 Seconds to predict 10 labels with SVC

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
Under section titled 'Detect cars in an image using trained classifier', I adapted the `find_cars` method from classroom materials. It combines feature extraction with a sliding window search, but rather than perform feature extraction on each window individually which can be time consuming, the features are extracted for the entire image (or a selected portion of it) and then these full-image features are subsampled according to the size window and then fed to the classifier. 
For subsampling, the size of the window was dynamically computed scaling a 64x64 fixed window with a scale factor.
The method performs the classifier prediction on the extracted features for each window region and returns a list of rectangle objects corresponding to the windows that generated a positive prediction. 
![Sliding Window Example][image3]

I explored several combination of window sizes and positions and window overlap in X and Y directions to identify cars at various zoom levels in image frame.
Some of the search results with window scale factor 1x, 1.5x and 2x are as below:
![Sliding Window 1x] [image4]
![Sliding Window 1.5x][image5]
![Sliding Window 2x][image6]



####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I combined feature extraction and sliding window search under `combined_detection` method with following parameters   
* sliding window scale and size : [[400,464,1.0],[420,484,1.0],[400,496,1.5],[436,532,1.5],[400,528,2.0],[432,560,2.0],[400,600,3.5],[460,668,3.5]]
* colorspace = 'YCrCb' 
* orient = 10
* pix_per_cell = 8
* cell_per_block = 2
* hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
* spatial_size=(16, 16)
* hist_bins=16  
The result of the pipeline on a test image is as follows.
     
![Sliding Window Combined][image7] 
I didn't do anything specific to optimise the classifier at this stage as parameters itself were chosen based on accuracy and speed of the classifier in earlier stages. 

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Under the section titled 'Filter False Positives Using Heatmap Thresholding', I used heatmaps to filter out false positives. From the positive detections, I created a heatmap using `add_heatmap` method and used `apply_thresholding` method on the heat map to identify vehicle positions. 
I then used the `scipy.ndimage.measurements.label()` inside the `label` method to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected using `draw_boxes` method.

Here's an example result showing the heatmap of frame of video, the result of label and the bounding boxes overlaid.
![Heatmap Example][image8]
![Thresholded Heatmap Example][image9]
![Label Example][image10]
![Final Bounding Box Overlay Example][image11]

### Here are six frames and their corresponding heatmaps:

![Heatmap Example On Six Test Frames][image12]

### Here is the output of `scipy.ndimage.measurements.label()` on the heatmap from all six frames:
![Label Example on Six Test Frames][image13]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![Resultant Images on Six Test Frames][image14]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The problems that I faced while implementing this project were mainly concerned with detection accuracy while balancing it with classification speed.
Experimentation with training the classifier with different color spaces has given pretty good detections per frame but still the pipeline is likely to fail cases where vehicles (or the HOG features) don't resemble those in the training dataset.  Lighting and environmental conditions also seem to have impact on detections (e.g. a white car against a white background) failures.
Applying strategies to remove false positives has given a pretty good accuracy but they are also introducing another problem: vehicles that significantly change position from one frame to the next (e.g. oncoming traffic) will tend to escape being labeled. Similarly thresholding heatmaps in subsequent frames also seem be maximising the overlap on incoming vehicles and false detections.

Given plenty of time to pursue it, I would like to combine a very high accuracy classifier preferably on a more diverse image dataset and work on maximising high overlap of the search windows with more intelligent tracking like
- determine vehicle location and speed to predict its location in subsequent frames
- begin with expected vehicle locations and nearest (largest scale) search areas, and preclude overlap and redundant detections from smaller scale search areas to speed up execution while retaining accuracy of classification.

Another totally different way to improve the detections in pipeline could be to use convolutional neural networks approach that generalises well under different environmental conditions.
