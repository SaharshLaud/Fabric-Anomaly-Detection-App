# Fabric Anomaly Detection App

## Detecting defects in fabric images using segmentation approach

>   Here we are going to detect the anomalies in given fabric images using segmentation approach and localize the defect and mask the defected region.

>  Segmentation is used to divide the image into subregions, and then using computer vision algorithms to localise and identify the defected regions.

   -  To detect defects in plain woven fabrics, we segment the regions of interest (ROI) from defected images.

   -  The processing is done by using images in grayscale mode, and an image enhancement technique is applied to highlight the regions with defects. 

   - To further enhance the accuracy and to reduce the algorithm complexity, the noise is removed by applying the low-pass filtering which highlights the defected regions. 

   - The defected regions are then segmented with edge detection based on first-order derivatives. 
   
   
## Streamlit App for detecting defected fabrics
> This repository also contains the source code for streamlit app that deploys the fabric anomaly detection model using OpenCV.
> The application takes the image path as input from the user and then returns the results for the particular fabric
