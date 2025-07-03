import SimpleITK as sitk
import os
import cv2
import torch as tc
import numpy as np

class Dataset(tc.utils.data.Dataset):
    def __init__(self, data_names):
        self.images = []
        self.masks = []
        self.images_names = []
        self.borders = []

        slice_index = 0

        for name in data_names:
            self.borders.append("First slice of " + os.path.basename(name) + ": " + str(slice_index))

            image = sitk.ReadImage(name)
            image = sitk.GetArrayFromImage(image).T
            if image.shape[1] == 666:   #Normalize shape of data
                image = image[:, 77:589, :]
            for i in range(image.shape[2]):
                self.images.append(image[:, :, i])   #Adding every slice separately
                self.images_names.append(os.path.basename(name) + "Slice" + str(i))
                slice_index += 1

            mask_path = name.replace(".nrrd", ".seg.nrrd")   #The masks have the same name as the files but their format is different
            if not os.path.exists(mask_path):
                raise ValueError(f"There is no mask for file {os.path.basename(name)}")

            mask = sitk.ReadImage(mask_path)
            mask = sitk.GetArrayFromImage(mask).T
            if mask.shape[1] == 666:   #Normalize shape of mask
                mask = mask[:, 77:589, :]
            for i in range(mask.shape[2]):
                self.masks.append(mask[:, :, i])   #To match slices of data we also slice masks

    def Preprocess_Data(self):
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        for i in range(len(self.images)):
            image = self.images[i]
            if "R" in self.images_names[i]:   #Rider files have NaN values that are read as -2000
                image[image == -2000] = 0
            image = (image - np.min(image)) / (np.max(image) - np.min(image))   #Normalize to 0-1 range
            image = np.uint8(255*(image))   #Scale to uint8 in 0-255 range for CV2 functions
            image = cv2.resize(image, (224, 224))
            image = clahe.apply(image)   #Clahe increase the contrast
            self.images[i] = tc.from_numpy(np.float32(image)/255) #Preparing data so that the model can process it effectively

        for i in range(len(self.masks)):
            mask = self.masks[i]
            mask = cv2.resize(mask, (224, 224))
            mask = (mask > 0)
            self.masks[i] = tc.from_numpy(np.float32(mask))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.masks[index]

    def Get_name(self, index):
        return self.images_names[index]

    def Get_borders(self):
        print(self.borders)