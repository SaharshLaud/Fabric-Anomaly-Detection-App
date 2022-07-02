# import packages
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import streamlit as st
from PIL import Image



# Title and subtitle
st.markdown("<h1 style='text-align: center;'>Fabric Defect Detection using ML</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;color: darkblue'>Detect if a given fabric image has defects in it using OpenCV and Machine Learning!<br></h2>", unsafe_allow_html=True)

# Display background image
image = Image.open("background.png")
st.image(image, use_column_width=True)

 
# Function to display the images after anomaly detection
def print_images(image,hsv,v,blr,dst,binary,dilation,img,detection):
  st.write(detection)
  fig, ax = plt.subplots(2,4,figsize=(15,10))
  ax[0,0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  ax[0,0].set_title("Original")
  ax[0,1].imshow(cv2.cvtColor(hsv, cv2.COLOR_BGR2RGB))
  ax[0,1].set_title("HSV")
  ax[0,2].imshow(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))
  ax[0,2].set_title("Value")
  ax[0,3].imshow(cv2.cvtColor(blr, cv2.COLOR_BGR2RGB))
  ax[0,3].set_title("Blur")
  ax[1,0].imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
  ax[1,0].set_title("Filter")
  ax[1,1].imshow(binary,cmap='gray')
  ax[1,1].set_title("Binary")
  ax[1,2].imshow(dilation,cmap='gray')
  ax[1,2].set_title("Dilation")
  ax[1,3].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  ax[1,3].set_title("Output")
  st.pyplot(fig)
  fig.tight_layout()


# Function to detect anomaly using image processing and segmentation approach
def defect_detect(path):
        
    image = cv2.imread(path)
    if (type(image) is np.ndarray):

        img = image.copy()

        # Convert image to hsv format
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        h = hsv [:,:,0]
        s = hsv[:,:,1]
        v = hsv[:,:,2]
        detection = ""
        
        # Blur the image
        blr = cv2.blur(v,(15,15))
        
        # Noise filtering from the image
        dst = cv2.fastNlMeansDenoising(blr,None,10,7,21)
        
        # Applying image segmentation using cv2.threshold()
        _,binary = cv2.threshold(dst,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        # Defining kernel for adjusting the size of convolutions
        kernel = np.ones((5,5),np.uint8)
        
        # Applying morphological image processing operations. 
        erosion = cv2.erode(binary,kernel,iterations = 1)
        dilation = cv2.dilate(binary,kernel,iterations = 1)
        
        # Detecting defect in fabric based on dialation value and drawing contour over the defects
        if (dilation==0).sum() >1:
                detection = "Defective Fabric"
                contours,_ = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                for i in contours:
                    if cv2.contourArea(i) < 261121.0:
                        cv2.drawContours(img,i,-1,(0,255,0),3)
        else:
                        detection = "Good Fabric"
        
          
        # Returning the processed images and output
        print_images(image,hsv,v,blr,dst,binary,dilation,img,detection)
    else:
      st.write("#### The path to the image is invalid or doesn't exists!")
      st.write('#### Please provide a valid path to the image in the format --> (path/to/image.jpg)')


# Taking user input for image path
user_input = st.text_input('Please input Image Path')
if(st.button('Submit')):
    path = user_input.title().strip('"')
    defect_detect(path)