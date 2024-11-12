# Introduction
To assist radiologists in diagnosing lung diseases in patients, our team is working on using machine learning algorithms to analyze chest X-rays. AI models have been created with this goal, and are becoming commercially available. However, many of these models lack rigorous, peer-reviewed testing. When studies have been conducted, results have been mixed. Only last year, the University of Copenhagen Department of Radiology evaluated four commercially available AI tools for their performance; compared to a trained radiologist, the models were found to have higher false-positive rates and difficulty finding smaller targets. Clearly, the industry has room for improvement.

The dataset we are using consists of anonymized chest x-rays taken from 32,717 different patients, totaling over 108,000 total x-rays images. Some patients are healthy, while others are in different stages of eight possible lung diseases. Each image is labeled with the corresponding diagnosis from a radiologist.

# Problem Definition
Currently, lung disease is a leading cause of death in the United States, claiming the lives of over 150,000 Americans a year. In fact, 35 million Americans suffer from a chronic lung disease such as asthma. With this in mind, it is more important now than ever to be able to identify lung diseases accurately to ensure an early and prompt treatment. To help detect such disease, we propose a solution that takes a chest X-ray and identifies if it belongs to one of eight common lung diseases: Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, or Pneumothorax.

# Methods
### Data Pre-Processing Methods:
**For the tabular data:**
- **Data Cleaning** - Handling missing values, outlier detection.
- **Data Visualization** - Initial exploratory data analysis to understand the relationship between different lung disease classifications and number of follow-ups.
- **Dimensionality Reduction** - Employ principal component analysis to reduce the number of variables while preserving data integrity.
- **Feature Selection** - Create a correlation matrix or use statistical techniques to determine which features are significant predictors.

**For CT images:**
- **Segmentation** - Isolates regions from X-ray scans, allowing the algorithm to focus on specific areas.
- **Image Normalization** - Scales X-ray scans with varying intensity levels to reduce variability.
- **Dimensionality Reduction** - Uses principal component analysis to reduce the number of variables while preserving data integrity.

![Chart](SS.png)

### Machine Learning Methods:
- **CNN** - Design architecture for image classification and detection. CNNs (convolutional neural networks) are ideal for image classification as they can automatically identify patterns through convolutional layers. We plan to use PyTorch to train the model.
- **SVM** - Multi-class image classification. Support vector machines are traditional machine learning algorithms for classification. We can use VGG16, a pre-built CNN, to extract features and feed them into the SVM. Scikit-learn will help implement SVMs for multi-classification.
- **Random Forest** - Image Classification. Random forests, an ensemble learning method, can use tabular data and extracted CT scan features to accurately classify lung disease types. Using Scikit-learn, we can plug the information into a random forest and train it.

# Results and Discussion
For this midterm, an accuracy of 9% was achieved in the end. Initially, a problem of ‘No Finding’ being significantly overrepresented had occurred (with a 70% accuracy for this), in that most patients don’t have any disease. Consequently, SMOTE was used as a resolution. A lot of features were present, and PCA was used to maintain 95% of variance while eliminating the need to use features that may be unnecessary. To use the pre-trained CNN (ResNet18), the black and white images needed to be converted to RGB. Furthermore, the possibility of some patients having multiple diagnoses may result in a lower accuracy due to the significant increase in the magnitude of the problem. In the second iteration, the result is fortunately no longer resulting in a convergence; however, the accuracy significantly decreased to 9%. This is because of the general nature of random forests not being ideal for image classification, as random forests consider each feature/pixel independently and thereby do not effectively determine pixels’ spatial relationships. Determining spatial relationships effectively is important because pixels and their neighbors are often related to each other. Furthermore, the images also vary in terms of scale and position, which could have also led to random forests not effectively handling these images well. However, we want to implement the CheXNet CNN into our model since it was trained on x-ray images to detect pneumonia. We believe using CheXNet will greatly improve our accuracy as the CNN can likely detect diseases with similar x-ray visuals.

Looking forward, we continue to improve our pre-processing methods, and by the time of our final project submission, we expect this to have improved the accuracy of our random forest method as well. 

![Chart](SS1.png)

# Contribution Table
![Gantt Chart](SS2.png)

| Name           | Midterm Contributions                                                                                                   |
|----------------|--------------------------------------------------------------------------------------------------------------------------|
| **Kevin Park** | Wrote the starting backbone code for developing the Random Forest algorithm. Collaborated closely with team members.    |
| **Srithan Nalluri** | Worked on the written portion, ensuring components are descriptive and up-to-date, and helped normalize images.    |
| **Sritan**     | Primarily worked on coding, testing, and debugging. Created the README file and uploaded all content to GitHub.         |
| **Maxi**       | Directed group work, organized work sessions, managed medical-related information, and found a CNN pre-trained on x-rays. |
| **Yiyang**     | Analyzed imbalance classification significance and conducted exploratory data analysis on meta-features and labels.      |

# References #
K. G. van Leeuwen, S. Schalekamp, M. J. C. M. Rutten, B. van Ginneken, and M. de Rooij, “Artificial intelligence in radiology: 100       
commercially available products and their scientific evidence,” European Radiology, vol. 31, no. 6, pp. 3797–3804, Apr. 2021, doi: https://doi.org/10.1007/s00330-021-07892-z

Louis Lind Plesner et al., “Commercially Available Chest Radiograph AI Tools for Detecting Airspace Disease, Pneumothorax, and Pleural Effusion,” Radiology, vol. 308, no. 3, Sep. 2023, doi: https://doi.org/10.1148/radiol.231236.

“NIH Clinical Center provides one of the largest publicly available chest x-ray datasets to scientific community,” National Institutes of Health (NIH), Sep. 27, 2017. Available: https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community

“About Us: Our Impact,” www.lung.org, Sep. 16, 2024. Available: https://www.lung.org/about-us/our-impact
