import segmentation_models_pytorch as smp
import tensorflow as tf
import numpy as np
from PIL import Image
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

model = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2,                      # model output channels (number of classes in your dataset)
)

class CropRowDataset(Dataset):

    def __init__(self, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform

    def __len__(self):
        return 200 # save 10 for validation, total 210
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        mask = np.load('training_data/crop_row_'+str(idx)+'.npy')
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        image = Image.open('training_data/crop_row_'+str(idx)+'.jpg')
        imageArr = tf.keras.utils.img_to_array(image)
        sample = {'image': imageArr, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample

class CropRowValidatorSet(Dataset):
    def __init__(self, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform

    def __len__(self):
        return 10 # save 10 for validation, total 210
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        mask = np.load('training_data/crop_row_'+str(200+idx)+'.npy')
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        image = Image.open('training_data/crop_row_'+str(200+idx)+'.jpg')
        imageArr = tf.keras.utils.img_to_array(image)
        sample = {'image': imageArr, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample
data = CropRowDataset()
valData = CropRowValidatorSet()

for i in range(len(data)):
    sample = data[i]
    print(i, sample['image'].shape, sample['mask'].shape)

trainer = pl.Trainer(
    max_epochs=5,
)

trainer.fit(
    model, 
    train_dataloaders=data, 
    val_dataloaders=valData,
)

     
