# Identifying and Tracking Solar Magnetic Flux Elements with Deep Learning

Haodi Jiang, Jiasheng Wang, Chang Liu, Ju Jing, Hao Liu, Jason T. L. Wang and Haimin Wang

Institute for Space Weather Sciences, New Jersey Institute of Technology

## Abstract

Deep learning has drawn significant interest in recent years due to its effectiveness in processing 
big and complex observational data gathered from diverse instruments. 
Here we propose a new deep learning method, called SolarUnet, 
to identify and track solar magnetic flux elements or features in observed vector
magnetograms based on the Southwest Automatic Magnetic Identification Suite (SWAMIS).
Our method consists of a data pre-processing component that prepares 
training data from the SWAMIS tool, a deep learning model implemented 
as a U-shaped convolutional neural network for fast and accurate image segmentation, 
and a post-processing component that prepares tracking results. 
SolarUnet is applied to data from the 1.6 meter Goode Solar 
Telescope at the Big Bear Solar Observatory. 
When compared to the widely used SWAMIS tool, 
SolarUnet is faster while agreeing mostly with SWAMIS on feature size and flux distributions, 
and complementing SWAMIS in tracking long-lifetime features. 
Thus, the proposed physics-guided deep learning-based tool 
can be considered as an alternative method for solar magnetic tracking.

----

Prerequisites:  
Note: SolarUnet was test on Python version 3.6.8  

Python Packages:  
The following python packages and modules are required to run SolarUnet  
astropy==4.0.1  
keras==2.2.4  
matplotlib==3.1.0  
numpy==1.19.5  
opencv-python==3.4.2.16  
scipy==1.2.2  
scikit-image==0.15.0  
scikit-learn==0.20.3  
tensorflow-gpu==1.12.3  

To install the required packages, you may use Python package manager "pip" as follow:
1.	Copy the above packages into a text file,  ie "requirements.txt"
2.	Execute the command:
pip install -r requirements.txt
Note: There is a requirements file already created for you to use that includes all packages with their versions. 
       The files are located in the root directory of the SolarUnet-magnetic-tracking
       
References:

Identifying and Tracking Solar Magnetic Flux Elements with Deep Learning. H. Jiang, J. Wang, C. Liu, J. Jing, H. Liu, J. T. L. Wang, H. Wang, ApJS, 250:5, 2020.

https://iopscience.iop.org/article/10.3847/1538-4365/aba4aa

http://arxiv.org/abs/2008.12080

https://web.njit.edu/~wangj/SolarUnet/
