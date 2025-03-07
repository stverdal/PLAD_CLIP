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

indices = [0,2,10,12,14,21,23,25,32]

def split_dataset(image_list, label_list, ratio=0.8):
    #Assert that lists are equal length
    if len(image_list) != len(label_list):
        print("Labe count and image count does not match! Aborting.")
        exit(1)
    indices = list(range(len(label_list)))

    train_size = int(ratio * len(indices))
    val_size = len(indices) - train_size

    train_indices, val_indices = random_split(indices, [train_size,val_size])
    #print(image_list)
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

        #print(list_labels)
        self.label = clip.tokenize(list_labels)

    def __len__(self):
        return len(self.label)

    def __getitem__(self,idx):
        image = preprocess(Image.open(self.image_path[idx]))
        print(idx)
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


image_list_train, image_list_val, label_list_train, label_list_val = split_dataset(image_path_list, label_list, ratio=0.8)

#print(image_list_train)
train_loader = DataLoader(plad_label_dataset(image_list_train, label_list_train), batch_size=32, shuffle=True)
val_loader = DataLoader(plad_label_dataset(image_list_val, label_list_val), batch_size=32, shuffle=True)

#load base model and test
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
#model.to(device)


text_inputs = torch.cat([clip.tokenize(f'a picture of a powerline {c}') for c in categories]).to(device)

#Instantiate fine tuning model
model.to(device)
#Define loss function and optimizer


for i,idx in enumerate(indices):
    #print(idx)

    #print(image_list_val[idx])

    #SQUEEZE?
    pil_image = Image.open(image_list_val[idx]
                           )
    image = preprocess(pil_image).unsqueeze(0).to(device)

    #print(image)
    label = label_list_val[idx]
    #print(label)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)


    #print(image_features)
    #print(text_features)


    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values,indices = similarity[0].topk(1)

    print("Similarity", similarity)
    print(f"Predicted {categories[indices[0]]}, ACtual {label}")
