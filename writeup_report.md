#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup-data/layer_diagram.jpg =800x "layer diagram"
[image2]: ./writeup-data/center_2017_03_23_16_08_32_270.jpg "center lane drving"
[image3]: ./writeup-data/flipped_center_2017_03_23_16_08_32_270.jpg "filpped image"
[image4]: ./writeup-data/random_brightness_center_2017_03_23_16_08_32_270.jpg "random brightness"
[image5]: ./writeup-data/random_gamma_center_2017_03_23_16_08_32_270.jpg "random gamma"
[image6_0]: ./writeup-data/hsv0_center_2017_03_23_16_08_32_270.jpg "hsv0"
[image6_1]: ./writeup-data/hsv1_center_2017_03_23_16_08_32_270.jpg "hsv0"
[image6_2]: ./writeup-data/hsv2_center_2017_03_23_16_08_32_270.jpg "hsv0"
[image7]: ./writeup-data/center_2017_03_24_09_30_44_683.jpg "reverse drving"
[image8]: ./writeup-data/center_2017_03_24_23_33_11_617.jpg "recovery drving"
[image9]: ./writeup-data/center_2017_03_24_10_21_27_366.jpg "track 2 drving"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* data.py containing the generator that data load and preprocess
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64.
The convolution neural network includes RELU layers to introduce nonlinearity (model.py lines 15-19) 

The data is normalized in the model using a Keras lambda layer (code line 13), and cropped top 70 pixels and bottom 25 pixels (line 14)

The fully connected layers with dropouts and l2 regularizers are reduce to output angle data (lines 21 - 27)

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21, 23, 25). 

The model was trained, validated and tested on different data sets to ensure that the model was not overfitting (data.py lines 135 - 136). The model was tested test data sets and tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

* optimizer : adam (not tuned)
* l2 regularizer : 0.001
* dropout p : 0.5 ~ 0.7

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.

* track 1
  * 3 center lane drivings
  * 1 recovery driving
  * 1 reverse way driving
* track 2
  * 1 center lane drving
  * 1 recovery driving

###Model Architecture and Training Strategy

####1. Solution Design Approach
My first step was to use a convolution neural network model similar to the nvidia archetecture. (https://arxiv.org/pdf/1604.07316.pdf)

I thought this model might be appropriate because simple and easy to modify

To train better
* normalize data
* crop data to eliminate unnecessary parts 
* flip image to eliminate the left bias

In order to gauge how well the model was working, I split my image and steering angle data into a training, validation and test set.

To combat the overfitting
* add dropout layers
* add l2 regularizers

The model was less lossy and could prevent overfitting, but it went out of the way.
I tried various things and fixed it with traning data improvement
* more data
  * well driven data on center lane
  * recovery data to avoid out of the way
  * reverse way driving data to avoid side bias
  * track 2 data to generalize

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 13 - 27) consisted of a convolution neural network with the following layers

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x6 RGB & HSV image | 
| Preprocess            | nomalization & crop to 65x320x3  | 
| Convolution 5x5     	| 5x5 stride, 2x2 subsample, l2 regularizer, valid padding |
| RELU					|				|
| Convolution 5x5     	| 5x5 stride, 2x2 subsample, l2 regularizer, valid padding |
| RELU					|				|
| Convolution 5x5     	| 5x5 stride, 2x2 subsample, l2 regularizer, valid padding |
| RELU					|				|
| Convolution 3x3     	| 3x3 stride, l2 regularizer, valid padding |
| RELU					|				|
| Convolution 3x3     	| 3x3 stride, l2 regularizer, valid padding |
| RELU					|				|
| Flatten				|              |
| dropout				| keep probability 0.5	|
| Fully connected		| outputs 100, l2 regularizer |
| RELU					|		|
| dropout				| keep probability 0.7 |
| Fully connected		| outputs 50, l2 regularizer |
| RELU					|		|
| dropout				| keep probability 0.7	|
| Fully connected		| outputs 10, l2 regularizer |
| RELU					|		|
| Fully connected		| outputs 1    |

Here is a visualization of the architecture

![layer diagram][image1]

####3. Creation of the Training Set & Training Process

* good driving behavior recording
  * 2 laps on center lane driving

![center lane][image2]

* flip

![flip][image3]

  * random brightness

![random brightness][image4]

  * random alpha

![random alpha][image5]

* concat hsv color scale image
  
![hsv0][image6_0]
![hsv1][image6_1]
![hsv2][image6_2]

* reverse tarck recording
  * 1 lap on reverse way
  
![reverse][image7]

* recovery drive recording
  
![recovery][image8]

* track 2
  * train on track 2

![track2][image9]

* split data set
  * first split 20% test data set, 80% training & validation data set
  * then split 80% training & validation data set into 80% training, 20% validation data set