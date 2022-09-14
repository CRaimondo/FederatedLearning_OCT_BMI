import torch
import cv2
import numpy as np 
import pandas as pd
import torchvision.transforms as transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
import PIL
from PIL import Image
import os
import matplotlib.pyplot as plt
from skimage import io
from medmnist import INFO
import warnings
warnings.simplefilter("ignore", UserWarning)


class MedMNIST(Dataset):
  def __init__(self, csv_file, image_path, transform, as_rgb = True):
    self.data_frame = pd.read_csv(csv_file)

    if self.data_frame.iloc[2,0] == 'TRAIN':
      print('\n+++++++++++++++++++++ Welcome to FL +++++++++++++++++++++\n')
      print(f"Number of training images: {len(self.data_frame)}")

    if self.data_frame.iloc[2,0] == 'VALIDATION':
      print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
      print(f"Number of validation images: {len(self.data_frame)}")


    self.image_path = image_path
    self.transform = transform
    self.as_rgb = as_rgb
    #

  def __len__(self):
    return len(self.data_frame)

  def __getitem__(self, idx):
    img_name = os.path.join(self.image_path, self.data_frame.iloc[idx,1]+ '.png')

    if self.as_rgb == True:
      image = Image.open(img_name).convert('RGB') 
          
    else:
      image = Image.open(img_name)

    # Define Transforms
    data_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                ])

    resize_transform = transforms.Compose([
                              transforms.ToPILImage(),
                              transforms.Resize((224, 224), 
                              interpolation=PIL.Image.NEAREST), 
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[.5], std=[.5])
                              ])


    if self.transform == resize_transform:
      image = resize_transform(image)
    else:
      image = data_transform(image)
    

    image_class = self.data_frame.iloc[idx, -1]

    sample_id = self.data_frame.iloc[idx,1]

    return {'sample_id': sample_id,
              'image': image.float(), 
              'label': image_class} 
    