# **PhaseFormer: Plug and Play Image Restoration**  
**Author: Dingjie Peng $^*$**     
**$*$ Waseda University**
* * *
This is an extension work of the [DRUNet](https://github.com/cszn/DPIR).  

We use transformer based method to train a deep denoiser model for image denoising, 
then plug it into the image deblurring, image demosaicing by Half Quadratic Splitting method.  

The test code `test_deblur.py`, `test_demosaic.py`, `utils` are borrow from [cszn](https://github.com/cszn/DPIR), 
then modified by Dingjie Peng.  
* * *
**Requirements**  
The installation of pytorch should use CUDA.
- pytorch >= 1.8.0  
- numpy
- hdf5storage  
- cv2  
- scipy  
* * *
## 1. Download free-available dataset for testing
- CBSD60, McM, Kodak, Set14, set3c, Urban100
- Download the dataset you like from [Google drive](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u), 
it's recommended to download the above-mentioned datasets. The uploaded code contains the McM and Kodak dataset.
- Unzip and put dataset in the folder `data/testset`
* * *
## 2. Download the model weights
- Download from Google drive [model_phaseformer_S_d4_1100.pth](https://drive.google.com/file/d/1eSZGyVFkW5ZYb9xVWYXIgECOkABVaIEv/view?usp=sharing)
- Put weights in the folder `model_zoo`
* * *
## 3. Image Denoising
`python test_denoise.py --dataset CBSD68 --sigma 25`  
use `--dataset {dataset name}` to specify the test dataset.  
use `--sigma {noise level}` to specify the level of Gaussian noise.   
use `--save_results` to save the denoise results.  
results will be saved in `./results/{dataset}_denoise/`
* * *
## 4. Image Deblurring
`python test_deblur.py --dataset set3c --sigma 7.65`   
use `--dataset {dataset name}` to specify the test dataset.  
use `--sigma {noise level}` to specify the level of Gaussian noise in [0, 50].   
use `--iter {#********iterations}` to specify the number of iterations for convergence. Default 16.   
use `--augment` to use the data augmentation.  
use `--save_results` to save the deblur results.  
results will be saved in `./results/{dataset}_deblur/`

## 5. Image Demosaicing
`python test_demosaic.py --dataset set3c --sigma 7.65`  
use `--dataset {dataset name}` to specify the test dataset.  
use `--sigma {noise level}` to specify the level of Gaussian noise.   
use `--iter {#iterations}` to specify the number of iterations for convergence. Default 40.  
use `--augment` to use the data augmentation.  
use `--save_results` to save the demosaic results.  
results will be saved in `./results/{dataset}_demosaic/` 

### Declaration
The implementation of the model and modification of the train/test code 
is all done by Dingjie Peng from Waseda University.  
email: [kefipher9013@asagi.waseda.jp](kefipher9013@asagi.waseda.jp)
