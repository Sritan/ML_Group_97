import os
import pandas as pd
import numpy as np
from keras.applications import VGG16
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Model
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import defaultdict
import random
from sklearn.decomposition import PCA

csv_path = 'Data_Entry_2017_v2020.csv'
df = pd.read_csv(csv_path)

# Initialize the VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=base_model.output)

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# Use vgg16 to extract the features of the data
def extract_features(image_paths):
    features = []
    for image_path in image_paths:
        img = preprocess_image(image_path)
        feature = model.predict(img)
        features.append(feature.flatten())
    return np.array(features)

def sample_evenly_individual_labels(df, image_folder, max_samples_per_label=100, total_limit=10000):

    image_paths = []
    labels = []
    label_count = defaultdict(int)
    counter = 0

    # Loop the data
    for img, label in zip(df['Image Index'], df['Finding Labels']):
        image_path = os.path.join(image_folder, img)
        if os.path.exists(image_path):
            counter += 1
            individual_labels = label.split('|')

            if any(label_count[l] >= max_samples_per_label for l in individual_labels):
                continue


            image_paths.append(image_path)
            labels.append(label)

            for l in individual_labels:
                label_count[l] += 1

    print(label_count)
    print(counter)
    # Randomize the data
    combined = list(zip(image_paths, labels))
    random.shuffle(combined)
    image_paths, labels = zip(*combined)
    return list(image_paths), list(labels)

image_folder = 'images/'
image_paths, labels = sample_evenly_individual_labels(df, image_folder, max_samples_per_label=100, total_limit=1000)

le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
import matplotlib.pyplot as plt

features = extract_features(image_paths)

pca = PCA(n_components=2)
labels_reduced = pca.fit_transform(features)
print(labels_encoded)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(labels_reduced[:, 0], labels_reduced[:, 1], c=labels_encoded, cmap='viridis', alpha=0.5)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA of Image Features')

handles, _ = scatter.legend_elements(prop="colors")
legend_labels = [le.classes_[i] for i in range(len(le.classes_))]
plt.legend(handles, legend_labels, title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()

# Split up the training and testing data
X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

# Train SVM
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Evaluate the model predictions
y_pred = svm.predict(X_test)

unique_labels = np.unique(y_test)

print(classification_report(y_test, y_pred, labels=unique_labels, target_names=le.classes_[unique_labels]))
