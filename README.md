## Overview

This project is an extended version of a university coursework project.  
It focuses on the task of **aorta segmentation from CT images**, comparing the performance of three different deep learning architectures:
* **U-Net**  
* **FCN with ResNet-50 backbone** (with pretrained and non-pretrained weights)  
* **DeepLabv3 with MobileNetV3-Large backbone** (with pretrained and non-pretrained weights)  

The models are trained using **supervised learning** implemented in **PyTorch**. Architectures are imported from **MONAI** and **torchvision** libraries.  

The project includes:
* Dataset analysis and preprocessing
* Custom dataset loader preparation
* Model training and evaluation
* Comparison of model performance  

Used evaluation criteria:
* Training time  
* **Dice coefficient**  
* **Hausdorff Distance (HD95)**
* 
## Dataset

The CT dataset used in this project originates from the article:  
["AVT: Multicenter aortic vessel tree CTA dataset collection with ground truth segmentation masks"](https://www.sciencedirect.com/science/article/pii/S2352340922000130).  
It is publicly available and can be downloaded from Figshare: [https://doi.org/10.6084/m9.figshare.14806362](https://doi.org/10.6084/m9.figshare.14806362).

A more detailed description and analysis of the dataset can be found in `data_overview.ipynb`.

## File structure

The repository contains:
* `data_overview.ipynb` - notebook dedicated to dataset analysis and exploration.
* `preprocessing_testing.ipynb` - notebook for visualization and testing of preprocessing methods applied to the dataset.
* `Dataset_loader.py` - custom dataset loader module, built based on insights from `data_overview.ipynb` and `preprocessing_testing.ipynb`.
* `model_training.ipynb` - notebook focused on training the segmentation models.
* `model_eval.ipynb` - notebook dedicated to evaluating model performance and comparing results.
* `training_info/` - directory containing training information such as loss curves and training times.
* `data_names/` - directory with dictionary files containing filenames belonging to train, validation, and test subsets (train/validation/test split).

Although each notebook contains detailed Markdown documentation and can be explored independently,  
the recommended order for running the project is:
1. **`data_overview.ipynb`** 
2. **`preprocessing_testing.ipynb`** 
3. **`Dataset_loader.py`** 
4. **`model_training.ipynb`** 
5. **`model_eval.ipynb`** 

## Usage

Each notebook in this repository (.ipynb files) was developed and tested in cloud environments such as Kaggle and Google Colab.
Running the project locally is also possible, but requires manual adjustments of file paths.
*Kaggle
    * Create a private dataset with your data and mount it in the notebook.
    * Ensure correct paths are set for dataset and loader modules.
*Colab
    * Upload data to Google Drive and mount the drive in the notebook.
    * Ensure correct paths are set for dataset and loader modules.
    * Note: The free version of Colab may not provide enough memory to fully load the dataset through the custom loader. In that case, consider using Colab Pro or running on Kaggle instead.
* Hardware requirements:
    * GPU acceleration is strongly recommended.
    * The experiments in this project were run on NVIDIA T4 GPUs.
    * Running only on CPU is possible, but training will be significantly slower.

Note: To run **`model_eval.ipynb`**, the following files are required:
* Training logs from the `training_info/` directory (loss curves, training times).  
* The split dictionary from the `data_names/` directory (train/validation/test subsets).  
* Pretrained models:  
    * [U-Net model](https://github.com/TypoweKiwi/Aorta_Segmentation/releases/download/Models/UNet_trained.pth)
    * [FCN ResNet-50](https://github.com/TypoweKiwi/Aorta_Segmentation/releases/download/Models/FCN_trained.pth)
    * [FCN ResNet-50 (pretrained)](https://github.com/TypoweKiwi/Aorta_Segmentation/releases/download/Models/FCN_pretrained_trained.pth)
    * [DeepLabv3 MobileNetV3](https://github.com/TypoweKiwi/Aorta_Segmentation/releases/download/Models/DeepLabv3_trained.pth)
    * [DeepLabv3 MobileNetV3 (pretrained)](https://github.com/TypoweKiwi/Aorta_Segmentation/releases/download/Models/DeepLabv3_pretrained_trained.pth)

## Evaluation Metrics
* **Dice coefficient** – measures the overlap between predicted and ground-truth segmentations. Higher values indicate better performance.  
* **Hausdorff Distance (HD95)** – measures boundary similarity between segmentations. Lower values indicate better performance.  
* **Training time** – used as an additional practical metric for model comparison. (focus on relative differences rather than the absolute numbers)

## Results

|Model                                  | Avg. Dice | Avg. HD95 | Training Time [m] |
|---------------------------------------|-----------|-----------|-------------------|
|**U-Net**                              |**0.780**  |15.03      |2.52               |
|**FCN (ResNet-50)**                    |0.741      |14.39      |58.63              |
|**FCN (ResNet-50, pretrained)**        |0.746      |**12.26**  |46.59              |
|**DeepLabv3 (MobileNetV3)**            |0.545      |18.74      |11.48              |
|**DeepLabv3 (MobileNetV3, pretrained)**|0.579      |13.36      |14.64              |

**U-Net** achieved the best overall performance: highest Dice score, very competitive HD95, and by far the fastest training time.  
**FCN pretrained** delivered the best boundary accuracy (lowest HD95), but required the longest training.  
**DeepLabv3 models** showed unstable training behavior and underperformed, even when pretrained weights were used (unsuitable for this task).  
**Transfer learning** improved results for both FCN and DeepLabv3, with a stronger impact on boundary precision (HD95).  
Despite U-Net’s strong results, **overall performance is still below expectations**, indicating the need for further optimization.

**For detailed metrics, training curves, and extended discussion of results, please refer to** `model_eval.ipynb`.

## Future Improvements
To improve segmentation quality, potential strategies include:  
1. Removal of empty slices.  
2. Experimentation with different input image sizes (e.g., resizing or cropping) to find a better trade-off between spatial detail and computational efficiency.  
3. Extended hyperparameter optimization using Optuna (learning rate schedules, batch sizes, more trials).
4. Different backbones testing. 
