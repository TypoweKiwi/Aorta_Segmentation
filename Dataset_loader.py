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
            image = self.load_image(name=name)
            mask = self.load_image(name=name, load_mask=True)
            for i in range(image.shape[2]):
                self.images.append(cv2.resize(image[:, :, i], (224, 224)))   #Adding every slice separately
                self.images_names.append(os.path.basename(name) + "Slice" + str(i))

                resized_mask = cv2.resize((mask[:, :, i] > 0).astype(np.float32), (224, 224)) #We apply > 0 condition to assure that mask has True or False value (1 or 0)
                self.masks.append(tc.from_numpy(resized_mask )) 
                slice_index += 1

    def preprocess_data(self):
        for i in range(len(self.images)):
            image = self.apply_preprocesing_functions(self.images[i])
            self.images[i] = tc.from_numpy(np.float32(image)/255) #Preparing data so that the model can process it effectively
    
    def load_image(self, name, load_mask=False):
        if load_mask:
            name = name.replace(".nrrd", ".seg.nrrd")
        image = sitk.ReadImage(name)
        image = sitk.GetArrayFromImage(image).T.astype(np.float32)
        if image.shape[1] == 666:   #Normalize shape of data
            image = image[:, 77:589, :]
        if "Dongyang" in name and not load_mask: #Normalization of background - to see more, look into data_overview.ipynb
            image = image + 1024
        else:
            image[image < 0] = 0
        return image
    
    def apply_preprocesing_functions(self, image):
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        image = (image - np.min(image)) / (np.max(image) - np.min(image))   
        image = np.uint8(255*(image))   
        image = clahe.apply(image)   
        image = cv2.medianBlur(image, 3)
        return image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.masks[index]

    def get_name(self, index):
        return self.images_names[index]

    def get_borders(self):
        print(self.borders)