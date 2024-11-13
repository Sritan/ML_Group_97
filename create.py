import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from torchvision import models, transforms
from PIL import Image
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
 
df = pd.read_csv('Data_Entry_2017_v2020.csv')
df['Finding Labels'] = df['Finding Labels'].fillna('No Finding')

label_mapping = {
    'Cardiomegaly': 0,
    'Emphysema': 1,
    'Effusion': 2,
    'Hernia': 3,
    'Infiltration': 4,
    'Mass': 5,
    'Nodule': 6,
    'Pneumonia': 7,
    'No Finding': 8
}
 
def encode_label(label):
    primary_label = label.split('|')[0]
    return label_mapping.get(primary_label, 8)
 
df['label_encoded'] = df['Finding Labels'].apply(encode_label)
 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
 
model = models.resnet18(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])
model.eval()

def extract_features(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0)
        with torch.no_grad():
            features = model(image)
        return features.flatten().numpy()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return np.zeros(512)
 
image_folder = 'images/'
df['image_path'] = df['Image Index'].apply(lambda x: os.path.join(image_folder, x))
print("Extracting image features...")
image_features = df['image_path'].apply(extract_features)
image_features = np.stack(image_features.values)
 
gender_encoder = LabelEncoder()
df['Patient Gender Encoded'] = gender_encoder.fit_transform(df['Patient Gender'])
metadata_features = df[['Patient Age', 'Patient Gender Encoded']].values

X = np.hstack((image_features, metadata_features))
y = df['label_encoded'].values
 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
 
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)
 

smote = SMOTE(random_state=42)
 
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
 
rf_classifier = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
 
print("Training the Random Forest model...")
rf_classifier.fit(X_train_resampled, y_train_resampled)
  
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
 
y_pred = rf_classifier.predict(X_test)
 
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=label_mapping.keys())
 
print(f"Accuracy: {accuracy:.4f}\n")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
 
print("Random Forest Model Completed")
