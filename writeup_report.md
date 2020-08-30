# **Traffic Sign Recognition Project** 

## Writeup

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

[image1]: ./examples/train_distribution.png "Training Set Distribution"
[image2]: ./examples/valid_distribution.png "Validation Set Distribution"
[image3]: ./examples/test_distribution.png "Test Set Distribution"
[image4]: ./examples/after_generating.png "Training Set Distribution After Generating Data"
[image5]: ./NewSigns/01-SpeedLimit.png "Traffic Sign 1"
[image6]: ./NewSigns/12-PriorityRoad.png "Traffic Sign 2"
[image7]: ./NewSigns/18-GeneralCaution.png "Traffic Sign 3"
[image8]: ./NewSigns/26-TrafficSignals.png "Traffic Sign 4"
[image9]: ./NewSigns/31-WildAnimals.png "Traffic Sign 5"

### Rubric Points
#### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
## Writeup / README

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used numpy to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed. It is obvious that there exists huge bias in the distribution of the training set, which affects my training results at first.

![alt text][image1]

![alt text][image2]

![alt text][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to solve the bias in the distribution of the training set. According to the distribution of the training set, I chose to simply generate data by replicate images from the old images. So how many times should I replicate the images? I used 900 to divide their number of the images for each class and plus 1 as the time to replicate the images in order to control the bias in the distribution. This step is definitely the most significant one in my case.

Here is the distribution after I generated additional data.

![alt text][image4]

If I didn't generate data and used all the following steps, the best validation accuracy was 81% with 100 epoches. I also chose to augment my new data by rotating the new images randomly, but the best validation accuracy was 84-85%. I just wondered how they can achieved 93% or above in the Knowledge...

Second, I decided to convert the images to grayscale because I was instructed to build a model based on what I learned from LeNet, and LeNet used grayscale images.

As a last step, I nomalised the images using their mean.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 15x15x6				    |
| Convolution 4x4	    | 1x1 stride, valid padding, outputs 12x12x16   |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 6x6x16                   |
| Convolution 3x3       | 1x1 stride, valid padding, outputs 4x4x32     |
| RELU                  |                                               |
| Flatten               | Outputs 512                                   |
| Fully connected		| Outputs 256                                   |
| RELU                  |                                               |
| Dropout               |                                               |
| Fully connected       | Outputs 128                                   |
| RELU                  |                                               |
| Dropout               |                                               |
| Fully connected       | Outputs 43                                    |
| Softmax				|             									|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Batch size : 128

Epoch : 20

Learning rate : 0.0008

Optimizer : AdamOptimizer 

Loss : Softmax_crossentropy 

Drop_out : 0.5

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* training set accuracy of 99%
* validation set accuracy of 96.3% 
* test set accuracy of 94.4%

I did following steps to get a good result:

* The first architecture that was tried was exactly the same as the LeNet.
* The problem of the initial architecture is that the validation accuracy was only 80-81% and wouldn't improve despite I changed the learning rate or epochs.
* I added one convolutional layer and generated more data. I don't know which one has more influence. When I added one convolutinal layer, the accuracy improved but was still below the specifications. But after the additional data, the accruacy just jump above the specifications.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]

Half of the images are not like those in the training set, which maybe difficult to classify.
Half of the images are similar to those in the training set, so they should be easy to my architecture.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)  | Speed limit (30km/h)  						| 
| Wild animals crossing     | Wild animals crossing 	     			|
| Keep right	| Keep right											|
| Traffic signals	     | Traffic signals				 				|
| Bumpy road(wrong)			| End of no passing by vehicles   				|
| Bicycles crossing | Bicycles crossing |
|Right-of-way at the next intersection | Right-of-way at the next intersection|
|General caution | General caution |
|Road work|Road work|
|Priority road|Priority road|



The model was able to correctly guess 9 of the 10 traffic signs, which gives an accuracy of 90%. The test set accuracy is 94.4%. I think the difference comes from that the number of the new traffic signs is small, which I think is doing a good job. Especially, the architecture can generalize to those images that are not similar to those in the training set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For almost all the images, the prediction probability is close to 100%

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100%  | Speed limit (30km/h)                          | 

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| 99.9%     | Wild animals crossing                     |

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| 100%    | Keep right                                            |

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| 99.9%        | Traffic signals                              |

This image is totally wrong, and the only wrong prediction.

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| 100%         | Bumpy road(wrong)                 |

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| 99.9% | Bicycles crossing |

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
|100%  | Right-of-way at the next intersection|

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
|100%  | General caution |

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
|100% |Road work|

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
|100% |Priority road|



