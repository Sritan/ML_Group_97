import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv
from torchvision import transforms
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

print("Running!")
# Load the CSV file
df = pd.read_csv('xray/Data_Entry_2017_v2020.csv')
 
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
 
image_folder = 'images/'
 
available_images = set(os.listdir(image_folder))
 
df = df[df['Image Index'].isin(available_images)].reset_index(drop=True)
df['image_path'] = df['Image Index'].apply(lambda x: os.path.join(image_folder, x))
 
present_labels = set()
for labels in df['Finding Labels']:
    label_list = labels.split('|')
    present_labels.update(label_list)
 
present_label_mapping = {label: idx for idx, label in enumerate(sorted(present_labels))}
 
def encode_labels(labels):
    label_list = labels.split('|')
    encoded = [present_label_mapping[label] for label in label_list if label in present_label_mapping]
    return encoded
 
df['labels_encoded'] = df['Finding Labels'].apply(encode_labels)
 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
 
def load_image(image_path):
    try:
        image = Image.open(image_path).convert('L')
        image = transform(image)
        #print(f"Image shape: {image.shape}")
        return image
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None
 
print("Loading and preprocessing images...")
df['image_data'] = df['image_path'].apply(load_image)
 
df = df.dropna(subset=['image_data'])

gender_encoder = LabelEncoder()
df['Patient Gender Encoded'] = gender_encoder.fit_transform(df['Patient Gender'])
 
df['Patient Age'] = df['Patient Age'].astype(float)
age_mean = df['Patient Age'].mean()
age_std = df['Patient Age'].std()
df['Patient Age Normalized'] = (df['Patient Age'] - age_mean) / age_std
 
metadata_features = df[['Patient Age Normalized', 'Patient Gender Encoded']].values
metadata_features = torch.tensor(metadata_features, dtype=torch.float32)
 
 images = torch.stack(df['image_data'].tolist())
 
metadata = metadata_features[:len(images)]
 
mlb = MultiLabelBinarizer(classes=list(present_label_mapping.values()))
labels = mlb.fit_transform(df['labels_encoded'])
labels = torch.tensor(labels, dtype=torch.float32)
 
label_counts = labels.sum(axis=0)
 
single_sample_classes = np.where(label_counts == 1)[0]
 
inverse_label_mapping = {v: k for k, v in present_label_mapping.items()}
single_sample_class_names = [inverse_label_mapping[idx] for idx in single_sample_classes]
 
print("Classes with only one sample:", single_sample_class_names)
 
def remove_rare_classes(encoded_labels):
    return [idx for idx in encoded_labels if idx not in single_sample_classes]
 
df['labels_encoded'] = df['labels_encoded'].apply(remove_rare_classes)
 
df = df[df['labels_encoded'].map(len) > 0].reset_index(drop=True)
 
labels = mlb.fit_transform(df['labels_encoded'])
labels = torch.tensor(labels, dtype=torch.float32)
 
images = torch.stack(df['image_data'].tolist())
 
metadata_features = df[['Patient Age Normalized', 'Patient Gender Encoded']].values
metadata = torch.tensor(metadata_features, dtype=torch.float32)
 
mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
splits = mskf.split(np.zeros(len(labels)), labels.numpy())
 
train_indices, test_indices = next(splits)
 
train_images = images[train_indices]
train_metadata = metadata[train_indices]
train_labels = labels[train_indices]
 
test_images = images[test_indices]
test_metadata = metadata[test_indices]
test_labels = labels[test_indices]
  
label_counts = labels.sum(axis=0)
total_counts = label_counts.sum()
class_weights = total_counts / (len(present_label_mapping) * label_counts)
class_weights = class_weights.numpy()
 
class_weights = torch.tensor(class_weights, dtype=torch.float32)
 
 
class CustomDenseNet(nn.Module):
    def __init__(self, num_classes, metadata_features):
        super(CustomDenseNet, self).__init__()
        # Load pre-trained DenseNet121
        self.features = xrv.models.DenseNet(weights="densenet121-res224-all").features
        # Metadata fully connected layer
        self.metadata_fc = nn.Linear(metadata_features, 128)
        # Classification layer
        self.classifier = nn.Linear(1024 + 128, num_classes)
    
    def forward(self, x, metadata):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        # Process metadata
        metadata = F.relu(self.metadata_fc(metadata))
        # Concatenate image features and metadata
        x = torch.cat((x, metadata), dim=1)
        x = self.classifier(x)
        return x
 
# Instantiate the model
num_classes = len(present_label_mapping)
model = CustomDenseNet(num_classes=num_classes, metadata_features=metadata.shape[1])
 
# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
 
batch_size = 32
 
train_dataset = TensorDataset(train_images, train_metadata, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
 
test_dataset = TensorDataset(test_images, test_metadata, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
 
# Training parameters
num_epochs = 10  # Adjust as needed
 
# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images_batch, metadata_batch, labels_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(images_batch, metadata_batch)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images_batch.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
  
from sklearn.metrics import classification_report
 
model.eval()
all_preds = []
all_labels = []
 
with torch.no_grad():
    for images_batch, metadata_batch, labels_batch in test_loader:
        outputs = model(images_batch, metadata_batch)
        preds = torch.sigmoid(outputs)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels_batch.cpu().numpy())
 
all_preds = np.vstack(all_preds)
all_labels = np.vstack(all_labels)
 
# Make predictions binary
threshold = 0.5
binary_preds = (all_preds >= threshold).astype(int)
 
# Classification report for each class
for i, class_name in enumerate([k for k, v in sorted(present_label_mapping.items(), key=lambda item: item[1])]):
    print(f"Class: {class_name}")
    print(classification_report(all_labels[:, i], binary_preds[:, i], zero_division=0))
