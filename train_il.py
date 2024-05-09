import torch 
import numpy as np
import pandas as pd 
from torch.utils.data import DataLoader, Dataset
from Envs.biopsy_env import TemplateGuidedBiopsy
from torchvision.models import resnet18, resnet50, vgg16
from torch.utils.tensorboard import SummaryWriter 
import os 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils.data_utils import ImageReader, LabelLesions

def load_data(filename):
    df = pd.read_csv(filename)  # Read the Excel file into a DataFrame
    return df

def get_blank_row_indices(data):
    """
    Returns idx of data separation 
    """
    blank_row_indices = data[data.isnull().all(axis=1)].index.tolist() 
    #blank_row_indices += [1]*len(blank_row_indices) # returns idx of each starting position 
    
    start_row_indices = [idx+1 for idx in blank_row_indices] #
    #start_row_indices = blank_row_indices
    start_row_indices = [0] + start_row_indices # start from 1 
    
    return start_row_indices

def split_data(data):
    # Find the indices of blank rows
    blank_row_indices = get_blank_row_indices(data)
    
    # Split the data indices into training and validation indices with 80:20 ratio
    train_indices, val_indices = train_test_split(blank_row_indices, test_size=0.2, random_state=42)
    
    return train_indices, val_indices

def separate_data(filename):
    data = load_data(filename)
    train_indices, val_indices = split_data(data)
    
    # Example of how to obtain one data 
    idx = 0
    sampled_idx = train_indices[idx] 
    patient_data = data.iloc[sampled_idx : sampled_idx+5]
    
    # Split the data based on the indices
    train_data = pd.concat([data.iloc[start:end] for start, end in zip([0] + val_indices, train_indices + [None])])
    val_data = pd.concat([data.iloc[start:end] for start, end in zip(val_indices, train_indices)])

    return train_data, val_data

read_img = ImageReader()

class ImageLoader():

    def __init__(self, folder_name):
        
        self.folder_name = folder_name
        
        # Folder names
        self.lesion_folder = os.path.join(folder_name, 'lesion')
        self.mri_folder = os.path.join(folder_name, 't2w')
        self.prostate_folder = os.path.join(folder_name, 'prostate_mask')

    def _get_patient_list(self, folder_name):
        """
        A function that lists all the patient names
        """
        all_file_names = [f for f in os.listdir(folder_name) if not f.startswith('.')]
        #all_file_paths = [os.path.join(folder_name, file_name) for file_name in self.all_file_names]

        return all_file_names

    def _normalise(self, img):
        """
        A function that converts an image volume from np.float64 to uint8 
        between 0 to 255

        """

        max_img = np.max(img)
        min_img = np.min(img)

        #Normalise values between 0 to 1
        normalised_img =  ((img - min_img)/(max_img - min_img)) 

        return normalised_img.astype(np.float)

    def get_obs(self, patient_name, lesion_idx, slice_array):
        """
        
        slice_array = [y,x,z] used for sagittal / axial plotting 
        
        """

        # Read prostate mask, lesion mask, prostate mask separately using ImageReader   
        mri_vol = torch.tensor(np.transpose(self._normalise(read_img(os.path.join(self.mri_folder, patient_name))), [1, 2, 0]))
        lesion_mask = torch.tensor(np.transpose(self._normalise(read_img(os.path.join(self.lesion_folder, patient_name))), [1, 2, 0]))
        prostate_mask = torch.tensor(np.transpose(self._normalise(read_img(os.path.join(self.prostate_folder, patient_name))), [1, 2, 0]))
        
        # Obtain single lesion mask 
        lesion_labeller = LabelLesions()
        lesion_centroids, num_lesions, lesion_statistics, multiple_lesions = lesion_labeller((os.path.join(self.lesion_folder, patient_name),))
        single_lesion_mask = torch.tensor([multiple_lesions == lesion_idx]).squeeze()*1.0

        # Upsample each volume 
        ax_mr = torch.nn.functional.interpolate((mri_vol[:,:,slice_array[2]*4]).unsqueeze(0).unsqueeze(0),(96,96)).squeeze()
        sag_mr = torch.nn.functional.interpolate((mri_vol[:, slice_array[0], :]).unsqueeze(0).unsqueeze(0), (96,96)).squeeze()
        ax_p = torch.nn.functional.interpolate((prostate_mask[:,:,slice_array[2]*4]).unsqueeze(0).unsqueeze(0),(96,96)).squeeze()
        sag_p = torch.nn.functional.interpolate((prostate_mask[:, slice_array[0], :]).unsqueeze(0).unsqueeze(0),(96,96)).squeeze()
        ax_l = torch.nn.functional.interpolate((single_lesion_mask[:,:,slice_array[2]*4]).unsqueeze(0).unsqueeze(0),(96,96)).squeeze()
        sag_l = torch.nn.functional.interpolate((single_lesion_mask[:, slice_array[0], :]).unsqueeze(0).unsqueeze(0),(96,96)).squeeze()
        
        # # for debugging
        # fig, axs = plt.subplots(3,2)
        # axs[0,0].imshow(ax_mr)
        # axs[0,1].imshow(sag_mr)
        # axs[1,0].imshow(ax_p)
        # axs[1,1].imshow(sag_p)
        # axs[2,0].imshow(ax_l)
        # axs[2,1].imshow(sag_l)
        
        # Stack each into array of 6 channels 
        obs = torch.stack([ax_mr, sag_mr, ax_p, sag_p, ax_l, sag_l]) # 96 x 96 -> 100 X 100 X 24 OBS change dto 6 x 96 x 96 
        
        return obs 

class GameDataset(Dataset):
    """
    TODO: Implement a dataset which loads in CSV files, obtains an environment to sample images from 
    """
    
    def __init__(self, folder_name, csv_path, mode = 'train'):
        """
        Input : csv path for loading results from dataset
        
        # TODO : split data into 'train', 'val' and 'test' sets
            Important : 
            When splitting data -> ensure lesions froßm the same patient are in the same group 
            Ie if a patient has 4 lesions, all 4 of them must be in the train set; not 2 in train and not 2 in val
        
        """  
    
        #self.biopsy_env = TemplateGuidedBiopsy()
        self.mode = mode 
        self.csv_path = csv_path
    
        # Load data  
        self.data = pd.read_csv(csv_path)
        self.image_loader = ImageLoader(folder_name)
        
        # Split up into train / val indices
        if self.mode != 'test':
            self.train_indices, self.val_indices = split_data(self.data)
        else:
            self.test_indices = get_blank_row_indices(self.data)
            
        # indix from action=1 to remove first one
        if self.mode == 'train':
            # Uncorrected csv file : actions/obs are not paired (action corresponds to time = t+1, whereas obs is at time=t)
            self.combined_data = pd.concat([self.data.iloc[idx+1:idx+5] for idx in self.train_indices])
            
            # corrected csv file : paired obs and actions at time = t 
            self.paired_data = pd.concat([self.pair_obs_actions(self.data.iloc[idx:idx+5]) for idx in self.train_indices])
            
        elif self.mode == 'val':
            self.combined_data = pd.concat([self.data.iloc[idx+1:idx+5] for idx in self.val_indices])
            self.paired_data = pd.concat([self.pair_obs_actions(self.data.iloc[idx:idx+5]) for idx in self.val_indices])
        else:
            self.combined_data = pd.concat([self.data.iloc[idx+1:idx+5] for idx in self.test_indices])
            self.paired_data = pd.concat([self.pair_obs_actions(self.data.iloc[idx:idx+5]) for idx in self.test_indices])
        
        # Drop values with less than specific range as they could be bug! ie double clicking 
        self.paired_data.drop(self.paired_data[self.paired_data['x'] > 30].index, inplace = True)
        self.paired_data.drop(self.paired_data[self.paired_data['x'] < -30].index, inplace = True)
        self.paired_data.drop(self.paired_data[self.paired_data['y'] > 30].index, inplace = True)
        self.paired_data.drop(self.paired_data[self.paired_data['y'] < -30].index, inplace = True)
        
        # For unnormalising later for actions!
        self.x_range = {'max' : self.paired_data['x'].max(), 'min' : self.paired_data['x'].min() }
        self.y_range = {'max' : self.paired_data['y'].max(), 'min' : self.paired_data['y'].min() }
        
        # Normalise data between -1 and 1
        scaler = MinMaxScaler(feature_range = (-1,1))
        self.paired_data['x'] = scaler.fit_transform(self.paired_data['x'].values.reshape(-1,1))
        self.paired_data['y'] = scaler.fit_transform(self.paired_data['y'].values.reshape(-1,1))
        self.paired_data['z'] = scaler.fit_transform(self.paired_data['z'].values.reshape(-1,1))
        
        self.data_len = len(self.paired_data)

    def __getitem__(self, idx):
        """
        Given an index, sample paired input / label data
        
        TODO: Given an index, sample a random action and observation pair 
    
        Returns:
        
        Input image : (6 x width x height of image)
        where 6 channels are the following:
            Axial, Sagittal MR
            Axial, Sagittal Gland
            Axial, Sagittal Target
        
        Output labels :  (3 x 1) or (1 x 3)
            3 actions : x,y,z 
        """
        
        # 1. Subsample idx pair (shuffle data = True) 
        patient_data = self.paired_data.iloc[idx]
        
        # 2. Get actions
        actions = torch.tensor([patient_data['x'].item(), patient_data['y'].item(), patient_data['z'].item()])
        
        # 3. Obtain observation from environment
            # put patient name
            # put lesion name 
            # slice axial, sagittal views for t2w, prostate, lesion (6 slices)
            # concat slices 
        
        patient_name = patient_data['patient_id'] + '_study_0.nii.gz'
        lesion_idx = int(patient_data['lesion_idx'])
        slice_array = np.array([int(patient_data['img_x']), int(patient_data['img_y']), int(patient_data['img_z'])])
        obs = self.image_loader.get_obs(patient_name, lesion_idx, slice_array)

        return obs, actions 
    
    def __len__(self):
        """
        Returns length of dataset : num_patients x num_lesions x num_action_idx (how many steps there are)
        """
        
        return self.data_len
    
    def pair_obs_actions(self,group_df):
        """
        Fixes csv file and obtains paired-obs action pairs
        ie observed obs at time = t, and corresponding action at time = t
        """
        # shift x,y actions 
        shifted_xy = group_df.iloc[0:, 6:8] - group_df.iloc[0:, 6:8].shift(1) # current action is current_pos - previous_pos
        shifted_z = group_df.z[1:] # curreent depth is action at time =t 
        
        # turn z into apex (0) centroid (1) or base (2)
        centroid = group_df.z.iloc[0]
        depths = []
        for val in shifted_z:
            if val < centroid: 
                depths.append(0)
            elif val == centroid:
                depths.append(1)
            else:
                depths.append(2)
        
        new_df = pd.DataFrame({'patient_id' : group_df.patient_id[0:-1],
                               'lesion_idx' : group_df.lesion_idx[0:-1],
                                'img_x' : group_df.img_x[0:-1], 
                            'img_y' : group_df.img_y[0:-1], 
                            'img_z' : group_df.img_z[0:-1],
                            'x' : shifted_xy.x[1:].values,
                            'y' : shifted_xy.y[1:].values,
                            'z' : depths})
        return new_df 

class RegressionNetwork(torch.nn.Module):
    """
    TODO: A network to perform regression to learn three actions 
    """

    def __init__(self, input_channels = 6, num_actions = 3, feature_extractor = 'resnet'):
        
        super(RegressionNetwork, self).__init__()
        
        if feature_extractor == 'resnet':
            #base_model = resnet18(pretrained = True)
            self.base_model = resnet18(pretrained = True)
            
            self.base_model.conv1 = torch.nn.Conv2d(input_channels, 64, kernel_size = (7,7), stride = (2,2), padding = (3,3), bias = False)
            #self.features = torch.nn.Sequential(*list(base_model.modules())[0:-1])[1:]
        
            # Change first dimension to num_input channels (ie 6 channels)
            #self.features[0] = torch.nn.Conv2d(input_channels, 64, kernel_size = (7,7), stride = (2,2), padding = (3,3), bias = False)
            #self.features[0] = torch.nn.Conv2d(input_channels, 64, kernel_size = (7,7), stride = (2,2), padding = (3,3), bias = False)
            
            # Change final  layer 
            num_features = self.base_model.fc.in_features # find number of features in second to last layer 
            self.base_model.fc = torch.nn.Linear(num_features, out_features = num_actions)
            
        else:
            base_model = vgg16(pretrained = True)
            self.features = torch.nn.Sequential(base_model.features, base_model.avgpool)
            
            # Change first dimension to num_Input channels 
            self.features[0][0] = torch.nn.Conv2d(input_channels, kernel_size = (3,3), stride = (1,1), padding = (1,1))
            
            # Change final layer
            num_features = base_model.classifier[-1].in_features
            self.final_layer = torch.nn.Linear(num_features, out_features = num_actions)


    def forward(self, x):
        #x = self.features(x) # extract features from input image 
        #x = self.final_layer(x) # obtain estimated actions 
        x = self.base_model(x)
        
        return x 

def count_groups(filename):
    df = pd.read_csv(filename, header=None)  # Read the CSV file into a DataFrame
    num_groups = sum(df.isnull().all(axis=1)) + 1  # Count the number of all-null rows and add 1

    return num_groups

if __name__ == '__main__':
    
    # Initialise path names
    SAVE_FOLDER = 'results'
    #CSV_PATH = '/Users/ianijirahmae/Documents/DATASETS/ImitationLearning/EntireData.csv'
    #DATA_PATH = '/Users/ianijirahmae/Documents/DATASETS/Data_by_modality'
    
    CSV_PATH = os.path.join('.', 'Data', 'ProstateDataset', 'EntireData.csv')
    DATA_PATH = os.path.join('.', 'Data', 'ProstateDataset')
    
    
    #num_dataset = count_groups(CSV_PATH)
    #train_data, val_data = separate_data(CSV_PATH)
    # print("Training data:")
    # print(train_data)
    # print("\nValidation data:")
    # print(val_data)
    
    # game_ds = GameDataset(DATA_PATH, CSV_PATH, 'train')
    # data = game_ds[0]
    
    os.makedirs(SAVE_FOLDER, exist_ok = True)
    writer = SummaryWriter(os.path.join(SAVE_FOLDER, 'runs'))
    
    # Load paths / datasets
    train_data = GameDataset(DATA_PATH, CSV_PATH, 'train')
    train_loader = DataLoader(train_data, shuffle = True)
    val_data = GameDataset(DATA_PATH, CSV_PATH, 'val')
    val_loader = DataLoader(val_data, shuffle = True)

    # Load model and device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    model = RegressionNetwork(input_channels= 6,num_actions = 3)
    model = model.to(device)
    
    # Load objective function, optimiser 
    loss_fn = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr = 0.0001)
    
    # Start training implementation 
    NUM_EPOCHS = 1000
    EVAL_FREQ = 5 
    BEST_VAL_LOSS = np.inf
    
    for epoch_num in range(NUM_EPOCHS):
        
        epoch_loss = [] # keep track of loss
        
        model.train()
        for idx, (imgs, labels) in enumerate(train_loader):

            # Move images to device
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Train model 
            
            optimiser.zero_grad()   # clear gradients 
            pred_actions = model(imgs.float())
            loss = loss_fn(pred_actions.float(), labels.float())    # Compute loss function
            loss.backward() # Backpropagate loss with respect to gradients 
            optimiser.step() # Update weights with respect to loss function 
            
            # Save epoch loss 
            epoch_loss.append(loss.item())
        
        # Compute mean epoch loss and save to tensorboard for visualisation of results 
        mean_loss = np.mean(epoch_loss)
        writer.add_scalar('Loss/train', mean_loss, epoch_num)
        print(f"\n Epoch {epoch_num} Loss : {mean_loss}")
        
        # Save updated model
        train_model_path = os.path.join(SAVE_FOLDER, 'train_model.pth')
        torch.save(model.state_dict(), train_model_path)
        
        #### evaluate every eval_freq num of epochs 
        if (epoch_num % EVAL_FREQ) == 0:
        
            epoch_loss_val = [] 
            
            model.eval()
            with torch.no_grad():
                
                for idx, (imgs, labels) in enumerate(val_loader):
                    
                    imgs, labels = imgs.to(device), labels.to(device)
                    
                    # Comptue loss on validation dataset 
                    pred_actions = model(imgs.float())
                    loss = loss_fn(pred_actions.float(), labels.float())  
                    epoch_loss_val.append(loss.item())
                    
                mean_val_loss = np.mean(epoch_loss_val)
                writer.add_scalar('Loss/val', mean_val_loss, epoch_num)
                print(f"Epoch {epoch_num} Val Loss : {mean_val_loss}")
               
                # Save best val model 
                val_model_path = os.path.join(SAVE_FOLDER, 'best_val_model.pth')
                if mean_val_loss < BEST_VAL_LOSS:
                
                    # SAVE MODEL 
                    torch.save(model.state_dict(), val_model_path)
                    BEST_VAL_LOSS = mean_val_loss 
                    print(f"New best val loss, model saving")
                
    print(f"Finished training")
         
                

        
