# ML_Group_97
Introduction
Our team is working on using machine learning and computer vision to analyze chest X-rays, in order to help radiologists diagnose lung diseases in patients. AI models have been created with this goal, and are becoming commercially available. However, many of these models lack rigorous, peer-reviewed testing.[1] When studies have been conducted, results have been mixed. Only last year, the University of Copenhagen Department of Radiology evaluated four commercially available AI tools for their performance; compared to a trained radiologist, the models were found to have higher false-positive rates and difficulty finding smaller targets.[2] Clearly, the industry has room for improvement.
The dataset we are using, which can be found here, consists of anonymized chest x-rays taken from 32,717 different patients, totaling over 108,000 total x-rays images.[3] Some patients are healthy, while others are in different stages of eight possible lung diseases. Each image is labeled with the corresponding diagnosis from a radiologist.
Problem Definition
Currently, lung disease is a leading cause of death in the United States, claiming the lives of over 150,000 Americans a year. In fact, 35 million Americans suffer from a chronic lung disease such as asthma [4]. With this in mind, it is more important now than ever to be able to identify lung diseases accurately to ensure an early and prompt treatment. To help detect such disease, we propose a solution that takes a chest x-ray and identifies if it belongs to one of eight common lung diseases: Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, or Pneumothorax. 
Methods
Data Pre-Processing Methods:
For the tabular data:
Data Cleaning - handling missing values, outlier detection
Dimensionality Reduction - Employ principal component analysis to reduce number of variables to select while preserving the integrity of the data
Feature Selection - Create correlation matrix or deploy statistical techniques to determine which features are significant with respect to the predictor
For CT images:
Segmentation - helps isolate regions from the X-ray scans and allow the algorithm to focus on a specific area 
Image-normalization - X-ray scans with varying level of intensity should be normalized and scaled to reduce variability
Dimensionality reduction - Employ principal component analysis to reduce number of variables to select while preserving the integrity of the data

Machine Learning Methods:
CNN - Design architecture for everything for image classification and detection.
CNNs are convolutional neural networks, which are primarily used for image classification, because through convolutional layers they are able to automatically identify specific patterns. We can use pytorch to train the model.
SVM - Multi-class image classification
Support vector machines are traditional machine learning algorithms for classification. We can use VGG16, a pre-built CNN, to extract the features and pass the features into the SVM. We can also use scikit-learn to use SVMs for multi-classification.
Random Forest - Image Classification
Random forests is an ensemble learning method that is able to use tabular data, along with extracted features from CT scans to build models to accurately classify a specific lung disease type. Using scikit learn, we can plug the information into a random forest and train it.

Results and Discussion
The quantitative metrics are precision, recall, and F1. Ideally, it is intended that the methods will overall classify the X-ray images with 60% precision. Furthermore, the recall score will ideally be 0.7, which can be calculated by comparing our results to the labeled data. Based on the goals for precision and recall, the F1 will be approximately 0.6462. 


Citation
[1]	K. G. van Leeuwen, S. Schalekamp, M. J. C. M. Rutten, B. van Ginneken, and M. de Rooij, “Artificial intelligence in radiology: 100 commercially available products and their scientific evidence,” European Radiology, vol. 31, no. 6, pp. 3797–3804, Apr. 2021, doi: htt
ps://doi.org/10.1007/s00330-021-07892-z
[2]	Louis Lind Plesner et al., “Commercially Available Chest Radiograph AI Tools for Detecting Airspace Disease, Pneumothorax, and Pleural Effusion,” Radiology, vol. 308, no. 3, Sep. 2023, doi: https://doi.org/10.1148/radiol.231236.
[3]	“NIH Clinical Center provides one of the largest publicly available chest x-ray datasets to scientific community,” National Institutes of Health (NIH), Sep. 27, 2017. Available: https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community
[4]	“About Us: Our Impact,” www.lung.org, Sep. 16, 2024. Available: https://www.lung.org/about-us/our-impact
