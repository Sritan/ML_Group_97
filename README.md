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
For the final project, an accuracy of 9% was achieved using Random Forest. Initially, the class "No Finding" was significantly overrepresented (resulting in a 70% accuracy for this class), as most patients did not have any disease. To address this issue, SMOTE was applied. Given the large number of features, PCA was used to retain 95% of the variance while reducing unnecessary features. To use the pre-trained CNN (ResNet18), black-and-white images were converted to RGB. However, the possibility of patients having multiple diagnoses increased the complexity of the classification problem and contributed to the low accuracy.

The poor performance can be attributed to the general limitations of Random Forest for image classification. Random Forest treats each feature (pixel) independently, failing to capture spatial relationships, which are critical in image data. Additionally, variations in image scale and position further reduced the effectiveness of Random Forest.

**Image 1:** Random Forest Analytics

An accuracy of 26% was achieved using SVM, which outperformed both Random Forest and random guessing. Random guessing would achieve an accuracy of 11.11% (1/9), meaning SVM achieved 134% better accuracy. Despite this improvement, the accuracy was still below expectations due to issues such as sampling errors caused by multi-class classification.

The SVM model was trained on features extracted by the pre-trained VGG model, reduced to two dimensions using PCA. Although the features appeared messy, SVM was able to classify labels better than Random Forest. The linear kernel's performance highlights the potential of SVM but also points to its limitations due to computational constraints preventing its application to high-dimensional input data.

**Images 2 and 3:** SVM Analytics

To improve SVM performance, future iterations should explore multi-class SVM models with higher-dimensional feature inputs and leverage advanced computational resources.

The CNN implementation achieved an accuracy of 81%, a commendable result. However, this high accuracy is misleading as it likely stems from dataset imbalance. Specifically, 81% of the images were labeled as "No Finding," leading the model to develop a bias toward predicting this class. While the use of `BCEWithLogitsLoss()` with `pos_weight` attempted to address class imbalance, the model still converged to predicting "No Finding." Despite additional efforts, such as SMOTE and balanced training sets, the issue persisted.

**Image 4:** CNN Analytics

Overall, in terms of raw accuracy, CNN performed the best. However, the bias towards "No Finding" limited its generalizability to unseen data. Improvements such as longer training epochs, better sampling strategies, and masking preprocessing systems could help address these shortcomings.

Based solely on accuracy, CNN appears to be the best algorithm. However, SVM demonstrated superior performance in avoiding sampling biases and handled the dataset's challenges better than Random Forest and CNN. Random Forest was the worst-performing algorithm due to its inability to handle high-dimensional image data and capture spatial relationships.

Future improvements include developing a masking system to improve feature extraction, similar to methods discussed in class. Additional CNN training epochs may help with minority class identification but would require significant computational resources. For all models, implementing a more effective sampling algorithm could help mitigate class imbalances.

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
