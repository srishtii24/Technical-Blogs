## Object Detection App using YOLOv3, OpenCV and Streamlit

***Computer Vision*** has numerous interesting applications and *Object Detection* is one of them. Using this application, we can can identify the objects present in the image with great accuracy. <br>

This blog article highlights the implementation of Object Detection app with YOLO (You Only Look Once) using the pre-trained model. YOLO is a state-of-the-art algorithm trained to identify thousands of objects types with great accuracy. This app extracts objects from images and identifies them using OpenCV and YOLOv3. 

Below is the snapshot of how our app looks like:


![objDetection.gif](https://cdn.hashnode.com/res/hashnode/image/upload/v1603974717391/c4EG6ibe7.gif)

### Table of Contents
>1. Introduction of the tools and frameworks
2. Pre-Requisites
3. Structure and workflow of the app
4. Code (Python Implementation)
5. Testing the app

### Introduction of the tools and frameworks
#### YOLO: 
The YOLO framework (You Only Look Once) takes the entire image in a single instance and then predicts the bounding box coordinates and class probabilities for these boxes. The biggest advantage of using YOLO is its speed – it’s extremely fast and accurate. With a GPU, it can process 45 frames per second  and with CPU, a frae per second.

YOLOv3 is the YOLO version 3, which uses some tricks to improve training and increase performance. This approach involves a single deep convolutional neural network (originally a version of GoogLeNet, later updated and called DarkNet) that splits the input into a grid of cells and each cell directly predicts a bounding box, confidence for these boxes, class probabilities and classifies the object. The result is a large number of candidate bounding boxes that are consolidated into a final prediction by a post-processing step.

Given below is the illustration of the YOLO Object Detection process flow ([source](https://arxiv.org/abs/1506.02640)): 
![yolo_design.jpg](https://cdn.hashnode.com/res/hashnode/image/upload/v1603865925681/K9DIxgZaf.jpeg)

#### How to install YOLO?
YOLO is a deep learning algorithm, so it doesn’t need any installation of itself. Instead we need a deep learning framework to run this algorithm.

Below are the 3 most used and known frameworks compatible with YOLO and the advantages and disadvantages of each:

1. ***Darknet*** : This framework was built by the developer of YOLO and made specifically for YOLO.
*Advantage*: It’s fast and can work with GPU or CPU.
*Disadvantage*: it works with Linux Operating System only.

2. ***Darkflow***: It’s the adaptation of darknet to TensorFlow (another deep learning framework).
*Advantage*: It’s fast and can work with GPU or CPU. Also it is compatible with Linux, Windows and Mac.
*Disadvantage*: The installation is really complex, especially on windows.

3. ***OpenCV***: OpenCV has a deep learning framework that works with YOLO. (Make sure to have OpenCV 3.4.2 or greater)
*Advantage*: it works without needing to install anything except opencv.
*Disadvantage*: it only works with CPU, so you can’t get really high speed to process videos in real time.

#### Streamlit: 
Streamlit is an open-source Python library that has been used in our app for the user interface. 
1. Make sure you have Python 3.6 or greater installed.
2. Install Streamlit using the command (in terminal):
```
pip install streamlit
``` 

#### OpenCV
OpenCV is a huge open-source library for *Computer Vision*. In this blog, our main focus is on how to use YOLO with OpenCV. It is the best approach for beginners, to get quickly the algorithm working without doing complex installations.


### Pre-requisites
- Download and Install Python 3.6 or greater.
- Install OpenCV, Streamlit, Pillow, NumPy and Matplotlib from the terminal using the command-
```
pip install streamlit opencv-python Pillow numpy matplotlib
``` 
- To run the algorithm we need three files:


1. **Weight file**: It is the pre-trained model, the core of the algorithm to detect the objects. Download it from  [here (237 MB)](https://drive.google.com/file/d/1CW49XDO5Y0urNKOCvg2bDKaPXkHbzZlZ/view?usp=sharing).

2. **Cfg file**: It is the configuration file which contains all the settings of the algorithm. Download it from  [here](https://drive.google.com/file/d/1Vcng6-DWIiTL6Kh8qzJTfXE-PjB1OsEH/view?usp=sharing).

3. **Name files**: It contains the name of the objects that the algorithm can detect. Download it from  [here](https://drive.google.com/file/d/1Vcng6-DWIiTL6Kh8qzJTfXE-PjB1OsEH/view?usp=sharing).


- Download few images to test the code. You can download sample images from  [here](https://drive.google.com/drive/folders/11kFbik7dkZcgC2GnIKdK6Uv71_0HQCdt?usp=sharing).

### Structure and workflow of the app
The app highlights two main functions- 
1. Show Demo.
2. Browse an Image. 

Using the *Browse an Image* option, you can upload (by browsing or simply dragging and dropping) an image to detect the objects in it. Using the *Show Demo* option, you can preview the demo of detecting objects in the image already provided. Playing with sliders on the sidebar, you can adjust various factors.

### The code
Let's dive straight into the code:
```
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def detect_objects(our_image):
	st.set_option('deprecation.showPyplotGlobalUse', False)

	col1, col2 = st.beta_columns(2)

	col1.subheader("Original Image")
	st.text("")
	plt.figure(figsize = (15,15))
	plt.imshow(our_image)
	col1.pyplot(use_column_width=True)

	# YOLO ALGORITHM
	net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
	
	classes = []
	with open("coco.names", "r") as f:
		classes = [line.strip() for line in f.readlines()]
	layer_names = net.getLayerNames()
	output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
	
	colors = np.random.uniform(0,255,size=(len(classes), 3))   


	# LOAD THE IMAGE
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	height,width,channels = img.shape


	# DETECTING OBJECTS (CONVERTING INTO BLOB)
	blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, crop = False)   #(image, scalefactor, size, mean(mean subtraction from each layer), swapRB(Blue to red), crop)
	
	net.setInput(blob)
	outs = net.forward(output_layers)

	class_ids = []
	confidences = []
	boxes =[]

	# SHOWING INFORMATION CONTAINED IN 'outs' VARIABLE ON THE SCREEN
	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)  
			confidence = scores[class_id] 
			if confidence > 0.5:   
				# OBJECT DETECTED
				#Get the coordinates of object: center,width,height  
				center_x = int(detection[0] * width)
				center_y = int(detection[1] * height)
				w = int(detection[2] * width)  #width is the original width of image
				h = int(detection[3] * height) #height is the original height of the image

				# RECTANGLE COORDINATES
				x = int(center_x - w /2)   #Top-Left x
				y = int(center_y - h/2)   #Top-left y
				
				#To organize the objects in array so that we can extract them later
				boxes.append([x,y,w,h])
				confidences.append(float(confidence))
				class_ids.append(class_id)

	score_threshold = st.sidebar.slider("Confidence Threshold", 0.00,1.00,0.5,0.01)
	nms_threshold = st.sidebar.slider("NMS Threshold", 0.00, 1.00, 0.4, 0.01)

	indexes = cv2.dnn.NMSBoxes(boxes, confidences,score_threshold,nms_threshold)      
	print(indexes)

	font = cv2.FONT_HERSHEY_SIMPLEX
	items = []
	for i in range(len(boxes)):
		if i in indexes:
			x,y,w,h = boxes[i]
			#To get the name of object
			label = str.upper((classes[class_ids[i]]))   
			color = colors[i]
			cv2.rectangle(img,(x,y),(x+w,y+h),color,3)     
			items.append(label)


	st.text("")
	col2.subheader("Object-Detected Image")
	st.text("")
	plt.figure(figsize = (15,15))
	plt.imshow(img)
	col2.pyplot(use_column_width=True)

	if len(indexes)>1:
		st.success("Found {} Objects - {}".format(len(indexes),[item for item in set(items)]))
	else:
		st.success("Found {} Object - {}".format(len(indexes),[item for item in set(items)]))


def object_main():
	"""OBJECT DETECTION APP"""

	st.title("Object Detection")
	st.write("Object detection is a central algorithm in computer vision. The algorithm implemented below is YOLO (You Only Look Once), a state-of-the-art algorithm trained to identify thousands of objects types. It extracts objects from images and identifies them using OpenCV and Yolo. This task involves Deep Neural Networks(DNN), yolo trained model, yolo configuration and a dataset to detect objects.")

	choice = st.radio("", ("Show Demo", "Browse an Image"))
	st.write()

	if choice == "Browse an Image":
		st.set_option('deprecation.showfileUploaderEncoding', False)
		image_file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])

		if image_file is not None:
			our_image = Image.open(image_file)  
			detect_objects(our_image)
			
	elif choice == "Show Demo":
		our_image = Image.open("images/person.jpg")
		detect_objects(our_image)

if __name__ == '__main__':
	object_main()
```

Breaking it down,
```
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
```
The first step is to import various packages, libraries and frameworks to be used in our code.

```
def detect_objects(our_image):
	st.set_option('deprecation.showPyplotGlobalUse', False)

	col1, col2 = st.beta_columns(2)

	col1.subheader("Original Image")
	st.text("")
	plt.figure(figsize = (15,15))
	plt.imshow(our_image)
	col1.pyplot(use_column_width=True)
```
Here we've created a function `detect_objects()` and given the `our_image` as the parameter. <br><br>
`st.set_option('deprecation.showPyplotGlobalUse', False)` suppress the deprecation warning of the pyplot. <br><br>
Next, we've declared two columns with two variables `col1` and `col2`. These columns can be placed horizontally using `st.beta_columns(2)` where `2` is the number of columns for the horizontal layout.<br><br>
In the column 1, we give a heading *Original Image* by using `col1.subheader("Original Image")` and then using the `plt.figure(figsize = (15,15))`, `plt.imshow(our_image)`, `col1.pyplot(use_column_width=True)`, we load and plot the original image in the column 1.

```
# YOLO ALGORITHM
	net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
	
	classes = []
	with open("coco.names", "r") as f:
		classes = [line.strip() for line in f.readlines()]
	layer_names = net.getLayerNames()
	output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
	
	colors = np.random.uniform(0,255,size=(len(classes), 3))   
```
So, here we loaded our YOLO algorithm by reading the files 
`yolov3.weights`, `yolov3.cfg` and `coco.names`. <br><br>
Using `classes = [line.strip() for line in f.readlines()]`, we load the classes from file `coco.names` into the list `classes=[]`. <br><br>
You can create different coloured rectangles for each detected objects using `colors = np.random.uniform(0,255,size=(len(classes), 3))`.

```
# LOAD THE IMAGE
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	height,width,channels = img.shape
```
Here, we load the image in which we want to detect the objects. Now that we have the algorithm ready to work and also we have the image, so next we have to pass the image into the network and do the detection. This is done below

```
# DETECTING OBJECTS (CONVERTING INTO BLOB)
	blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, crop = False)   
	
	net.setInput(blob)
	outs = net.forward(output_layers)
```
*Note*: 
1. We can’t pass right away the full image onto the network! Instead we first need to convert it to blob. Blob is used to extract features from the image and to resize them. 

2. YOLO accepts three sizes:


- 320×320 : It is small ,so, less accuracy but better speed.
- 609×609 It is bigger ,so, high accuracy and slow speed.
- 416×416 It is intermediate of the above two and you get a bit of both.


`net.setInput(blob)` passes the blob image into algorithm (network).
The `outs` is the result of the detection. `outs` is an array that contains all the informations about objects detected, their position and the confidence about the detection.


```
    class_ids = []
	confidences = []
	boxes =[]

	# SHOWING INFORMATION CONTAINED IN 'outs' VARIABLE ON THE SCREEN
	for out in outs:
		for detection in out:	
			scores = detection[5:]
			class_id = np.argmax(scores)  
			confidence = scores[class_id] 
```
Here, we declare three lists `class_ids`, `confidences` and `boxes` to store the information contained in `outs` variable in these lists. <br><br>
In the above code, we detect confidence: How confident is the algo that detection is correct? We have looped through the `outs` list to calculate the confidence and we choose a confidence threshold.<br><br>
 `class_id` is the number associated with `classes=[]` which tell us what object that is!

```
         if confidence > 0.5:   
				# OBJECT DETECTED
				#Get the coordinates of object: center,width,height  
				center_x = int(detection[0] * width)
				center_y = int(detection[1] * height)
				w = int(detection[2] * width)  #width is the original width of image
				h = int(detection[3] * height) #height is the original height of the image

				# RECTANGLE COORDINATES
				x = int(center_x - w /2)   #Top-Left x
				y = int(center_y - h/2)   #Top-left y

                             	boxes.append([x,y,w,h])
				confidences.append(float(confidence))
				class_ids.append(class_id)
```
Here, we set a threshold confidence of 0.5. If the calculated confidence is greater than the threshold, we consider the object correctly detected, otherwise we skip it.
The threshold goes from 0 to 1. Threshold closer to 1 implies greater accuracy of the detection, while closer to 0 implies less accuracy.

```
    score_threshold = st.sidebar.slider("Confidence Threshold", 0.00,1.00,0.5,0.01)
	nms_threshold = st.sidebar.slider("NMS Threshold", 0.00, 1.00, 0.4, 0.01)

	indexes = cv2.dnn.NMSBoxes(boxes, confidences,score_threshold,nms_threshold)   
	print(indexes)

	font = cv2.FONT_HERSHEY_SIMPLEX
	items = []
	for i in range(len(boxes)):
		if i in indexes:
			x,y,w,h = boxes[i]
			#To get the name of object
			label = str.upper((classes[class_ids[i]]))   #i is the specific object we are looping through.
			color = colors[i]
			cv2.rectangle(img,(x,y),(x+w,y+h),color,3)     
			items.append(label)
```
Function called ***Non-Mark Suppression*** remove double boxes or the NOISE(box inside a box) using some threshold (here, `nms_threshold` and `score_threshold`). <br><br>

We extract all the information:

- ***Box:*** It contains the coordinates of the rectangle surrounding the object detected.
- ***Label:*** It is the name of the object detected.
- ***Confidence:*** The confidence about the detection from 0 to 1.

```
    st.text("")
	col2.subheader("Object-Detected Image")
	st.text("")
	plt.figure(figsize = (15,15))
	plt.imshow(img)
	col2.pyplot(use_column_width=True)
```
Earlier, we declared two columns with two variables `col1` and `col2`. These columns were to be placed horizontally using `st.beta_columns(2)` where `2` is the number of columns for the horizontal layout.<br><br>
So here, In the column 2, we give a heading *Object-Detected Image* by using `col2.subheader("Object-Detected Image")` and then using the `plt.figure(figsize = (15,15))`, `plt.imshow(img)`, `col2.pyplot(use_column_width=True)`, we load and plot the image with detected objects in the column 2.

```
if len(indexes)>1:
		st.success("Found {} Objects - {}".format(len(indexes),[item for item in set(items)]))
	else:
		st.success("Found {} Object - {}".format(len(indexes),[item for item in set(items)]))
```
After the detection, we print the count and the labels of the objects detected in the image using the above code snippet.

```
def object_main():
	"""OBJECT DETECTION APP"""

	st.title("Object Detection")
	st.write("Object detection is a central algorithm in computer vision. The algorithm implemented below is YOLO (You Only Look Once), a state-of-the-art algorithm trained to identify thousands of objects types. It extracts objects from images and identifies them using OpenCV and Yolo. This task involves Deep Neural Networks(DNN), yolo trained model, yolo configuration and a dataset to detect objects.")
```
Next, we have our main function `object_main()`. `st.title("Object Detection")` gives the heading in the UI as *Object Detection*. Next, `st.write()` displays the brief description about the app.

```
	choice = st.radio("", ("Show Demo", "Browse an Image"))
	st.write()

	if choice == "Browse an Image":
		st.set_option('deprecation.showfileUploaderEncoding', False)
		image_file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])

		if image_file is not None:
			our_image = Image.open(image_file)  
			detect_objects(our_image)
			
	elif choice == "Show Demo":
		our_image = Image.open("images/person.jpg")
		detect_objects(our_image)

if __name__ == '__main__':
	object_main()
```

Here, we've asked for the choice between ***Show Demo*** and ***Browse an Image*** from the user using the radio button. <br><br>

If the choice is *Browse an Image*, the user is prompted to select the Browse button and upload an image with the extensions *.jpg,.png,.jpeg*. After the image is uploaded, we've used the `Image.open(image_file)` to load the image using the method Image from the Pillow package and then we call the `detect_objects(our_image)` function.<br><br>

Else if the choice is *Show Demo*, then we have provided an image stored in our local device with the name "girl_image.jpg" and then we've called the function `detect_objects(our_image)` which detect the objects in the image.<br><br>

At last, we've called the `object_main()` function.

#### Test the app
To test the app, save the above python code with the name, say, `app.py`. Then, in the terminal, write-

```
streamlit run app.py
```

Thankyou for reading, I would love to connect with you at  [LinkedIn](https://www.linkedin.com/in/srishtii24/). <br><br>