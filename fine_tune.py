import json
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import Dataset

import clip
from transformers import CLIPProcessor, CLIPModel

annotation_path = "./cropped_images/cropped_annotations.json"
input_data = []
image_dir = "./cropped_images/"

categories = ["tower","insulator","spacer","damper","plate"]

def split_dataset(image_list, label_list, ratio=0.8):
    #Assert that lists are equal length
    if len(image_list) != len(label_list):
        print("Labe count and image count does not match! Aborting.")
        exit(1)
    indices = list(range(len(label_list)))

    train_size = int(ratio * len(indices))
    val_size = len(indices) - train_size

    train_indices, val_indices = random_split(indices, [train_size,val_size])
    print(image_list)
    image_list_train = [image_list[i] for i in train_indices]
    image_list_val = [image_list[i] for i in val_indices]
    label_list_train = [label_list[i] for i in train_indices]
    label_list_val = [label_list[i] for i in val_indices]

    return image_list_train, image_list_val, label_list_train, label_list_val

#Custom dataset class
class plad_label_dataset():
    def __init__(self, list_image_path,list_labels):
        self.image_path = list_image_path
        #self.label = list_labels
        self.label = clip.tokenize(list_labels)

    def __len__(self):
        return len(self.label)

    def __getitem__(self,idx):
        image = preprocess(Image.open(self.image_path[idx]))
        label = self.label[idx]
        return image, label


class CLIPFineTuner(nn.Module):
    def __init__(self, model, num_classes):
        super(CLIPFineTuner, self).__init__()
        self.model = model
        self.classifier = nn.Linear(model.visual.output_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.model.encode_image(x).float()
        return self.classifier(features)


#open annotations
with open(annotation_path, 'r') as f:
    annotations = json.load(f)

label_list = []
image_path_list = []

#print(annotations[0])
for annotation in annotations:
    img_path = image_dir + annotation["file_name"]
    label = annotation["category_name"]
    image_path_list.append(img_path)
    label_list.append(label)


image_list_train, image_list_val, label_list_train, label_list_val = split_dataset(image_path_list, image_path_list, ratio=0.8)

print(len(image_list_train),len(image_list_val),len(label_list_train), len(label_list_val))

#print(image_list_train)
train_loader = DataLoader(plad_label_dataset(image_list_train, label_list_train), batch_size=32, shuffle=True)
val_loader = DataLoader(plad_label_dataset(image_list_val, label_list_val), batch_size=32, shuffle=True)

#load base model and test
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
#model.to(device)

#Instantiate fine tuning model
num_classes = len(categories)
model_ft = CLIPFineTuner(model, num_classes).to(device)

#Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_ft.classifier.parameters(), lr=1e-4)


num_epochs = 5

for epoch in range(num_epochs):
    model_ft.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}, :Loss: 0.0000")

    for images, labels in pbar:
        print(images)
        print(labels)
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model_ft(images)
        print(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):/4f}')

    model_ft.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_ft(images)
            _, predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Validation accuracy: {100 * correct / total}%')


torch.save(model_ft.state_dict(), 'clip_finetuned.pth')
