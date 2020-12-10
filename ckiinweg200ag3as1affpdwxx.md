## Eyes Detection App using Streamlit and OpenCV

In this blog article, you'll learn how to build a eye-detection app using ***Streamlit*** and ***OpenCV***. Below is the snapshot of how this app looks like:


![eye.gif](https://cdn.hashnode.com/res/hashnode/image/upload/v1603997342255/a6c0ofZAG.gif)

#### Introduction
Eye detection is an algorithm in computer vision. It detects eyes using Open-CV. The algorithm implemented below is a Haar-Cascade Classifier. 
Haar Cascade classifiers are an effective way for eye detection. Haar Cascade is a machine learning-based approach where a lot of positive and negative images are used to train the classifier.
- Positive images – These images contain the images which we want our classifier to identify.
- Negative Images – Images of everything else, which do not contain the object we want to detect.

#### Pre-requisites
- Download and Install Python.
- Install OpenCV, Streamlit, Pillow, NumPy and Matplotlib from the command prompt or the terminal using the command-

```
pip install streamlit opencv-python Pillow numpy matplotlib
``` 
- Download the haar cascade xml files from the [OpenCV Github Repository](https://github.com/opencv/opencv/tree/master/data/haarcascades). 

#### Structure and work-flow of the app
The app highlights two main functions- 
1. Show Demo.
2. Browse an Image. 

Using the *Browse an Image* option, you can upload (by browsing or simply dragging and dropping) an image to detect the eyes in it. Using the *Show Demo* option, you can preview the demo of detecting eyes in the image already provided. Playing with sliders on the sidebar, you can adjust various factors.

#### The code
Let's dive straight into the code-
```
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

eye_cascade = cv2.CascadeClassifier('frecog/haarcascade_eye.xml')

def detect_eyes(our_image):
	st.set_option('deprecation.showPyplotGlobalUse', False)
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	col1, col2 = st.beta_columns(2)

	col1.markdown("#### Original Image")
	#st.image(our_image, width = 800)
	plt.figure(figsize = (12,8))
	plt.imshow(our_image)
	col1.pyplot()


	scaleFactor = st.sidebar.slider("Scale Factor", 1.02,1.15,1.1,0.01)
	minNeighbors = st.sidebar.slider("Number of neighbors", 1, 15, 5, 1)
	minSize = st.sidebar.slider("Minimum Size", 10,50,20,1)



	#Detect Eyes
	eyes = eye_cascade.detectMultiScale(gray,scaleFactor=scaleFactor,minNeighbors=minNeighbors,flags = cv2.CASCADE_SCALE_IMAGE)

	#Draw Rectangle
	for (ex,ey,ew,eh) in eyes:
		#if ew > minSize:
		cv2.rectangle(gray, (ex,ey), (ex+ew,ey+eh), (255,255,255), 5)

	col2.markdown("#### Detected Eyes")
	#st.image(result_img, width = 800)
	plt.figure(figsize = (12,8))
	plt.imshow(gray, cmap = 'gray')
	col2.pyplot()


def eyes_main():
	"""EYES DETECTION APP"""

	st.title("Eyes Detection")
	st.write("Eye detection is a central algorithm in computer vision used to evaluate the eye location using OpenCV . The algorithm implemented below is a Haar-Cascade Classifier.")

	choice = st.radio("", ("Show Demo", "Browse an Image"))
	st.write("")

	if choice == "Browse an Image":
		st.set_option('deprecation.showfileUploaderEncoding', False)
		image_file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])

		if image_file is not None:
			our_image = Image.open(image_file)  
			detect_eyes(our_image)
			
	elif choice == "Show Demo":
		our_image = Image.open("images/eye.jpg")
		detect_eyes(our_image)


if __name__ == '__main__':
	eyes_main()
```
Breaking it down,
```
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
```
The first step is to import various packages, libraries and frameworks to be used in our code. Streamlit is an open-source app framework used to create the UI of this app. cv2 is the OpenCV module which is an open-source library that includes several hundreds of computer vision algorithms. Numpy, Matplotlib and Pillow are used to load the images.

```
eye_cascade = cv2.CascadeClassifier('frecog/haarcascade_eye.xml')
```
This line creates an eye cascade and initialises it. It then loads the eye cascade into memory so it’s ready for use. The cascade is just an XML file that contains the data to detect eyes.

```
def detect_eyes(our_image):
	st.set_option('deprecation.showPyplotGlobalUse', False)
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	col1, col2 = st.beta_columns(2)

	col1.markdown("#### Original Image")
	#st.image(our_image, width = 800)
	plt.figure(figsize = (12,8))
	plt.imshow(our_image)
	col1.pyplot()
```
Here we've created a function `detect_eyes()` and given the `our_image` as the parameter. <br><br>
`st.set_option('deprecation.showPyplotGlobalUse', False)` suppress the deprecation warning of the pyplot. <br><br>
Then, `new_img = np.array(our_image.convert('RGB'))` converts `our_image` into the numpy array with 3 channels and stores the data in the `new_img` variable. <br><br>
Next, `img = cv2.cvtColor(new_img,1)`  stores in `img` variable, the `new_img` with the default colours of channels provided with the image  (i.e., loads the full colour). <br><br>
To use the eye cascade and detect the eyes in the image, we need to first convert the image into gray. This is done using `gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)`<br><br>
Next, we've declared two columns with two variables `col1` and `col2`. These columns can be placed horizontally using `st.beta_columns(2)` where `2` is the number of columns for the horizontal layout.<br><br>
In the column 1, we give a heading *Original Image* by using `col1.markdown("#### Original Image")` and then using the `plt.figure(figsize = (12,8))`, `plt.imshow(our_image)`, `col1.pyplot(use_column_width=True)`, we load and plot the original image in the column 1.

```
	scaleFactor = st.sidebar.slider("Scale Factor", 1.02,1.15,1.1,0.01)
	minNeighbors = st.sidebar.slider("Number of neighbors", 1, 15, 5, 1)
	minSize = st.sidebar.slider("Minimum Size", 10,50,20,1)
#Detect Eyes
	eyes = eye_cascade.detectMultiScale(gray,scaleFactor=scaleFactor,minNeighbors=minNeighbors,flags = cv2.CASCADE_SCALE_IMAGE)
```
To determine the factor by which the detection window of the haar cascade classifier is scaled down per detection pass, we've used `scaleFactor = st.sidebar.slider("Scale Factor", 1.02,1.15,1.1,0.01)`. We've used slider to adjust the scaleFactor. A factor of 1.1 corresponds to an increase of 10%. Therefore, increase in the scale factor leads to  increase in performance, as the number of detection passes is reduced. However, as a consequence the reliability by which a face is detected is reduced. <br><br>

The argument `minNeighbors` determines the minimum number of neighbouring facial features that are needed to be present to indicate the detection of a eye by the haar cascade classifier. Decrease in this factor increases the amount of false-positive detections. Increase in the factor might lead to missing eyes in the image. This argument seems to have no influence on performance of the algorithm.<br><br>

The detection algorithm uses a moving window to detect objects.The argument `minSize` determines the minimum size of the detection window in pixels. Increase in the minimum detection window increases performance.<br><br>

Next, the haar cascade classfier can be used to detect the eyes in the image using its `detectMultiScale` method. The method takes the arguments `scaleFactor`, `minNeighbors` and `minSize` as an input. <br><br>`eyes = eye_cascade.detectMultiScale(gray,scaleFactor=scaleFactor,minNeighbors=minNeighbors,flags = cv2.CASCADE_SCALE_IMAGE)`detects the eyes and is the integral part of our code.<br><br> In this, the parameter`flags = cv2.CASCADE_SCALE_IMAGE` tells the cascade classifier that haar-like features have been applied on image data, which is what we are actually doing for detecting eyes.

```
	#Draw Rectangle
	for (ex,ey,ew,eh) in eyes:
		#if ew > minSize:
		cv2.rectangle(gray, (ex,ey), (ex+ew,ey+eh), (255,255,255), 5)
```
To iterate through the faces and draw a bounding box wherever an eye is detected,  the above code snippet is used. It returns 4 values: the ex and ey location of the bounding box, and the bounding box's width and height (ew , eh).<br><br>
These values are used to draw a bounding box using the built-in rectangle() function. Here, `gray` is the grayscaled image, `(ex,ey)` are the starting coordinates, `(ex+ew,ey+eh)` are the end coordinates, `(255, 255,255)` is the colour of the bounding box and `5` is the stroke or the thickness of the bounding box.
```
	col2.markdown("#### Detected Eyes")
	#st.image(result_img, width = 800)
	plt.figure(figsize = (12,8))
	plt.imshow(gray, cmap = 'gray')
	col2.pyplot()
```

Earlier, we declared two columns with two variables `col1` and `col2`. These columns were to be placed horizontally using `st.beta_columns(2)` where `2` is the number of columns for the horizontal layout.<br><br>
So here, In the column 2, we give a heading *Detected Faces* by using `col2.markdown("#### Detected Eyes")` and then using the `plt.figure(figsize = (12,8))`, `plt.imshow(gray, cmap='gray')`, `col2.pyplot(use_column_width=True)`, we load and plot the image with detected eyes in the column 2.


```
def eyes_main():
	"""EYES DETECTION APP"""

	st.title("Eyes Detection")
	st.write("Eye detection is a central algorithm in computer vision used to evaluate the eye location using OpenCV . The algorithm implemented below is a Haar-Cascade Classifier.")
```
Next, we have our main function `eyes_main()`. `st.title("Eyes Detection")` gives the heading in the UI as *Eyes Detection*. Next, `st.write()` displays the brief description about the app.

```
	choice = st.radio("", ("Show Demo", "Browse an Image"))
	st.write("")

	if choice == "Browse an Image":
		st.set_option('deprecation.showfileUploaderEncoding', False)
		image_file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])

		if image_file is not None:
			our_image = Image.open(image_file)  
			detect_eyes(our_image)
			
	elif choice == "Show Demo":
		our_image = Image.open("images/eye.jpg")
		detect_eyes(our_image)


if __name__ == '__main__':
	eyes_main()
```
Here, we've asked for the choice between ***Show Demo*** and ***Browse an Image*** from the user using the radio button. <br><br>

If the choice is *Browse an Image*, the user is prompted to select the Browse button and upload an image with the extensions *.jpg,.png,.jpeg*. After the image is uploaded, we've used the `Image.open(image_file)` to load the image using the method Image from the Pillow package and then we call the `detect_eyes(our_image)` function.<br><br>

Else if the choice is *Show Demo*, then we have provided an image stored in our local device with the name "eye.jpg" and then we've called the function `detect_eyes(our_image)` which detect the eyes in the image.<br><br>

At last, we've called the `eyes_main()` function.

#### Test the app
To test the app, save the above python code with the name, say, `app.py`. Then, in the terminal, write-

```
streamlit run app.py
``` 

Thankyou for reading, I would love to connect with you at  [LinkedIn](https://www.linkedin.com/in/srishtii24/). <br><br>






