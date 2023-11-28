import matplotlib.pylab as plt 
import SimpleITK as sitk
import numpy as np
from pathlib import Path
import os 
from Envs.biopsy_env import *

""" #This block of code is used to open multiple nifti files from a folder 
def load_file(file_path):
    img = sitk.ReadImage(file_path)
    data = sitk.GetArrayFromImage(img)
    return data

def display(file_path):
    files = [f for f in os.listdir(file_path) if f.endswith('.nii')]
    print(files)

    for file in files:
        file_path = os.path.join(file_path,file)
        mri_data= load_file(file_path)
        # Display slices using matplotlib
        num_slices = mri_data.shape[-1]

        fig, axes = plt.subplots(1, num_slices, figsize=(15, 5))

        for i in range(num_slices):
            axes[i].imshow(data[:, :, i], cmap='gray', vmin=0, vmax=255)
            axes[i].set_title(f"Slice {i + 1}")
            axes[i].axis('off')

        plt.show()
        input("Press Enter to view the next MRI scan...") """

def load_and_display_nifti(file_path):
    # Load NIfTI file
    mri_img = sitk.ReadImage(str(file_path))
    mri_data = sitk.GetArrayFromImage(mri_img)
    mri_data = np.squeeze(mri_data)

    img_size = np.shape(mri_data)
    print(img_size)
    # Display slices using matplotlib
    # num_slices = mri_data.shape[-1]
    transposed_image= np.transpose(mri_data,[1,2,0])
    plt.imshow(transposed_image[:,:,45], cmap='gray')
    plt.text(20, 20, 'matplotlib EXAMPLE',color = 'green', horizontalalignment='center',verticalalignment='center')
    plt.show()

if __name__ == "__main__":
    # code for single file
    file_path = r"BiopsyGame\Data\ProstateDataset\lesion\Patient005876472_study_0.nii.gz"
    load_and_display_nifti(file_path)


    # #code for viewing the entire folder 
    # folder_path = r"BiopsyGame\Data\ProstateDataset\t2w"
    # display(folder_path)

    # code for viewing all of the files in the path
    # for f in os.listdir(r".\BiopsyGame\Data\ProstateDataset\t2w"):
    #     print(f)
