from matplotlib import pyplot as plt 
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
    # transposed_image= np.transpose(mri_data)
    vox_dims = [0.5, 0.5, 1]
    transposed_image= (mri_data)
    sliceno=45
    print (img_size)
    plt.title("This is slice 45")
    plt.figure(1)
    plt.imshow(mri_data[:,:,sliceno], cmap='gray') 
    plt.figure(2)
    plt.title("This is slice on the first one ")
    plt.imshow(mri_data[int(sliceno),:,:], cmap='gray') 

    # plt.imshow(transposed_image[:,45,:], cmap='gray',aspect=vox_dims[2]/vox_dims[0])
    plt.show()
    return mri_img
    # ax = plt.gca().set_aspect(1/np.array(vox_dims[1],vox_dims[2],vox_dims[3]))

def multiple_display(file_path):
    # Load NIfTI file
    mri_img = sitk.ReadImage(str(file_path))
    mri_data = sitk.GetArrayFromImage(mri_img)
    mri_data = np.squeeze(mri_data)

    #CODE TO SHOW MULTIPLE SLICES AT ONCE
    # Determine the number of slices to display (e.g., 20 slices)
    num_slices_to_display = 20

    # Calculate the spacing between the slices
    num_slices_total = mri_data.shape[0]
    slice_spacing = max(1, num_slices_total // num_slices_to_display)

    # Create a figure with subplots for displaying the slices
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))

    # Iterate through and display the selected slices
    for i in range(num_slices_to_display):
        slice_index = i * slice_spacing
        ax = axes[i // 5, i % 5]
        ax.imshow(mri_data[:, slice_index, :], cmap='gray')
        ax.set_title(f"Slice {slice_index + 1}")
        ax.axis('off')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()
    plt.text(20, 20, 'matplotlib EXAMPLE',color = 'green', horizontalalignment='center',verticalalignment='center')
    plt.show()
    return mri_img


if __name__ == "__main__":
    # code for single file
    file_path = r"Data\ProstateDataset\t2w\Patient688976372_study_0.nii.gz"
    # mri_image=load_and_display_nifti(file_path)
    # shape=np.shape(mri_image)
    # slices=mri_image.GetSpacing()
    # dimensions=mri_image.GetDimension()
    # print (f'The shape of this image is {shape}')
    # print(f"the voxel size is {slices}")
    # print(f'the pixel dimension of the mri image is {dimensions}')
    load_and_display_nifti(file_path)


    # #code for viewing the entire folder 
    # folder_path = r"BiopsyGame\Data\ProstateDataset\t2w"
    # display(folder_path)

    # code for viewing all of the files in the path
    # for f in os.listdir(r".\BiopsyGame\Data\ProstateDataset\t2w"):
    #     print(f)
