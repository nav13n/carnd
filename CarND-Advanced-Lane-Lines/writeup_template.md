**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistorted_checkerboard.png "Undistorted"
[image2]: ./examples/undistorted.png "Undistorted Example Image"
[image3]: ./examples/warped.png "Warped Image"
[image4]: ./examples/polyfit.png "Polyfit"
[image5]: ./examples/detected_lane.png "Detected Lane"
[image6]: ./examples/final_output.png "Final Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

The code used for performing these operations are inside CarND-Advanced-Lane-Laines jupyter notebook.

#### 1. Camera Calibration and Image Undistortion

I used the provided 20 camera calibration to compute camera calibration and distortion matrix inside `calibrate_camera(images_grob_string)`   method.
I iterated on each of the provided images, finding the chessboard corners and corresponding coordinates on the board using `cv2.findChessboardCorners()`. 
I stored all of these results in a single array that I passed to `cv2.calibrateCamera()` to calibrate to camera which returns the calibration matrix and distortion coefficients.
I then used `cv2.undistort()` inside `undistort()` function to undistort a given image using the computed calibration matrix and distortion coefficients.

Example: Distortion correction of checkerboard image

![alt text][image1]

Example: Distortion correction of test image

![alt text][image2]

#### 2. Filtering lane pixels using color and gradient threshold

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at methods `dir_thresh`, `mag_thresh`,`color_thresh` ).
I first applied a color filtering, keeping only white or yellow-ish pixels of the image. I then only kept pixels that either match a sufficient magnitude or direction threshold.

Example: Binary Thresholded Mask:

![alt text][image3]

#### 3. Apply a perspective transform to get a  birds-eye view

I then warped the image using`cv2.getPerspectiveTransform()` inside the ImageWarper class with following source and destination points
to get a bird's eye view of the image 
| Source        | Destination   |
|:-------------:|:-------------:|
| 260, 680      | 260, 720      |
| 580, 460      | 260, 0        |
| 700, 460      | 1020, 0       |
| 1020, 680     | 260, 720      |

An example of perspective transformation aplied to one of the images is as follows:
![alt text][image4]


#### 4. Lane detection

I first computed the histogram of the lower half of the picture to find the rough position of each lane line.
I then ran a sliding window vertically to detect the position of the center of each lane line on each part of the image. 
I then used these positions to compute polylines describing each lane line using the `np.polyfit()` method. (`sliding_window_ployfit` method  in `LaneFinder` class)

As an optimisation step, instead of using sliding window and histogram to do an blind search and compute polyline fit in each frame,
I used the last frame's lane fit values to just search in a margin around the previous line position. (`ployfit` method  in `LaneFinder` class)
An example of detected lanes in images using sliding window polyfit and prev fit respectively are as follows:
![alt text][image5]

#### 5. Measuring lane curvature

I used the computed polyline along with the estimated lane width (~3.7m) to compute the real-world lane curvature.
I also measured the position of the car in respect to the lane by computing the difference between the center of the lane and the center of the image.
(`measure curvature` method  in `LaneFinder` class)


#### 6. Displaying the detected lane

I then created a polygon using the curves of each computed polyline and warped back the result using the reversed source and destination of step #3.
I finally drew this polygon on the undistorted test image.(`draw_lane` method)


![alt text][image6]

---

### Pipeline (video)

Here's a [link to my video result](./project_video_done.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The implementation overall went pretty smoothly for the project video but It didn't do a lot well on the challenge videos. A lot of those failures are due to the extreme variations in lightening and colors of the roads in different environment.
Given some luxury of time, I would like to  go back and make it more robust by handling more lightning conditions and lane line colors exploring different color spaces.  
Another possible optimisation  would be keep track of previous few frames detections and use it to provide recovery options in no line is detected in current frame or if they differ too much from the previously detected lines.
This project has compelled me to appreciate the previous deep learning approaches we used in classification and regression problems of this program much more as they are able to generalise better with varying environmental conditions.
I wonder if I can make it more robust taking a neural network route.
