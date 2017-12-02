# Vehicle-Detection-and-Tracking
Vehicle Detection Tracking using Traditional as well as Deep Learning Approach 

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I have used two different approaches for the project , The deep learning approach using YOLO (You look only once) and the traditional approach using SVC. 
The HOG for the traditional approach can found on the cell 10 of the traditional.ipynb. 
I used the dataset consisting of Car Images and Non Car Images , The heatmap for the Car Images are shown below. 
![Car](https://github.com/Shreyas3108/Vehicle-Detection-and-Tracking/blob/master/for_readme/car.png?raw=true)
![Non Car](https://github.com/Shreyas3108/Vehicle-Detection-and-Tracking/blob/master/for_readme/nocar.png?raw=true)
I then explored different color spaces and different skimage.hog() parameters (orientations, pixels_per_cell, and cells_per_block)
Here is an example using the YCrCb color space and HOG parameters of orientations=8, pixels_per_cell=(8, 8) and cells_per_block=(2, 2):

![HOG](https://github.com/Shreyas3108/Vehicle-Detection-and-Tracking/blob/master/for_readme/heatmap.png?raw=true)

The DeepLearning YOLO approach doesn't require HOG

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and finally settled color_space = 'YCrCb'
spatial_size = (32, 32)
hist_bins = 32
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
spatial_feat = True
hist_feat = True
hog_feat = True 
I had crashes while iterating different combination then after few research i went with this combination. 
I also noted towards the beggining of iteration before the crash that YCrCb performs better than RGB , HSV. 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM which is on the cell 16 of the traditional.ipynb. 
For the deeplearning it can be found on cell 13 of the vehicleyolo.ipynb 
Thw weights were extracted from https://drive.google.com/uc?id=0B1tW_VtY7onibmdQWE1zVERxcjQ&export=download
The DeepLearning approach doesn't use HOG but the architecture of the model is as follows :- 
![Tiny YOLO](https://github.com/Shreyas3108/Vehicle-Detection-and-Tracking/blob/master/for_readme/architecture.png?raw=true)

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window approach can be found on the cell 12 of the traditional.ipynb which uses the features extracted from cell 11 , The HOG features extracted are used to plot the box. 
The multi boxes can be found here 
![Boxes](https://github.com/Shreyas3108/Vehicle-Detection-and-Tracking/blob/master/for_readme/boxes.png?raw=true) 

The Deep Learning approach for the same can be found on the cell 10. The class consists of finding box , combining two boxes into one and plotting of box. 

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![Traditional Approach](https://github.com/Shreyas3108/Vehicle-Detection-and-Tracking/blob/master/for_readme/heatmapthres.png?raw=true)

![Deep Learning Approach](https://github.com/Shreyas3108/Vehicle-Detection-and-Tracking/blob/master/for_readme/plot.png?raw=true)

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://github.com/Shreyas3108/Vehicle-Detection-and-Tracking/blob/master/project_video_output_svc.mp4)

The DeepLearning approach can be found here 

Here's a [link to my video result](https://github.com/Shreyas3108/Vehicle-Detection-and-Tracking/blob/master/project_video_output.mp4) 


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` (can be found on cell 31 of traditional.ipynb) and the bounding boxes then overlaid on the last frame of video:

![Final](https://github.com/Shreyas3108/Vehicle-Detection-and-Tracking/blob/master/for_readme/heatmapthres.png?raw=true)

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust? 

The major issue i faced was to identify the correct parameters , My laptop kept crashing while trying to find accuracy in the SVC. On the other hand , I feel the YOLO approach as ideal since it doesn't require vehicle/non vehicle set images and trained itself on the test_images provided by Udacity. I personally prefer YOLO approach over traditional approach and that maybe due to the fact that traditional approach requires a lot of processing (Which is weird). After ending up crashing and redoing 7-8 times i had to fix the parameters and train it. But same wasn't the case in terms of YOLO. 

### Reference 

https://arxiv.org/abs/1506.02640 
https://medium.com/@xslittlegrass/almost-real-time-vehicle-detection-using-yolo-da0f016b43de


