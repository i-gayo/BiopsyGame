import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt 
from utils.data_utils import * 
from utils.environment_utils import * 
from Envs.biopsy_env import *
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import CnnPolicy

def generate_grid(prostate_centroid):
    """
    Generates 2D grid of grid point coords on image coordinates
    
    Arguments:
    :prostate_centroid (ndarray) : centroid in x,y,z convention of prostate gland 
    
    Returns:
    :grid_coords (ndarray) : 2 x 169 grid coords x,y convention 
    """
    
    x_grid = (np.arange(-30,35,5))*2 + prostate_centroid[0]
    y_grid = (np.arange(-30,35,5))*2 + prostate_centroid[1]

    grid = np.zeros((200,200))
    for i in range(-60, 65, 10):
        for j in range(-60, 65, 10):
            x_val = int(prostate_centroid[1])+j
            y_val = int(prostate_centroid[0]) +i
            
            grid[x_val - 1:x_val+2 , y_val -1 : y_val+2] = 1

    grid_coords = np.array(np.where(grid == 1))  # given in y, x 
    
    # change to x,y convention instead of y,x 
    grid_coords[[0,1],:] = grid_coords[[1,0],:]
    
    return grid, grid_coords
    
if __name__ == '__main__':
    
    ps_path = '/Users/ianijirahmae/Documents/DATASETS/Data_by_modality'
    csv_path = '/Users/ianijirahmae/Documents/PhD_project/MRes_project/Reinforcement Learning/patient_data_multiple_lesions.csv'
    log_dir = 'game'
    os.makedirs(log_dir, exist_ok=True)

    #DATASETS 
    PS_dataset = Image_dataloader(ps_path, csv_path, use_all = True, mode  = 'test')
    Data_sampler = DataSampler(PS_dataset)
    
    # HYPERPARAMETERS
    RATE = 0.1
    SCALE = 0.25
    NUM_EPISODES = 2
    
    #### 1. Load biopsy env ####
    biopsy_env = TemplateGuidedBiopsy(Data_sampler,results_dir = 'game', reward_fn = 'reward', \
        max_num_steps = 20, deform = True, deform_rate = RATE, deform_scale = SCALE, start_centre= True)
    
    ### 2. Load RL model for inference :for now, a random policy     ####
    policy_kwargs = dict(features_extractor_class = NewFeatureExtractor, features_extractor_kwargs=dict(multiple_frames = True, num_channels = 5))
    agent = PPO(CnnPolicy, env = biopsy_env, policy_kwargs = policy_kwargs)
    
    ### 3. User interface     ####
    for i in range(NUM_EPISODES):
        
        obs = biopsy_env.reset()
        vols = biopsy_env.get_img_data()
        done = False 
        num_steps = 0 
        
        while ((num_steps <= 4)):
            # Obtain lesion and mri vols from data 
            lesion_vol = biopsy_env.get_lesion_mask() # get individual lesion mask 
            
            mri_vol = vols['mri_vol']
            prostate_vol = vols['prostate_mask']
            prostate_centroid = np.mean(np.where(prostate_vol), axis = 1)
            SLICE_NUM = int(prostate_centroid[-1])
            
            # Define grid coords 
            grid, grid_coords = generate_grid(prostate_centroid)

            # Obtain agents predicted actions 
            actions,_ = agent.predict(obs)
            #TODO : convert this to action / grid pos for agents!!! 
            
            plt.figure()
            mask_l = np.ma.array(obs[0,:,:,:].numpy(), mask=(obs[0,:,:,:].numpy()==0.0))
            mask_p = np.ma.array(obs[1,:,:,:].numpy(), mask=(obs[1,:,:,:].numpy()==0.0))
            mask_n= np.ma.array(obs[-1,:,:,:].numpy(), mask=(obs[-1,:,:,:].numpy()==0.0))
            mask_n_1= np.ma.array(obs[-2,:,:,:].numpy(), mask=(obs[-2,:,:,:].numpy()==0.0))
            mask_n_2= np.ma.array(obs[-3,:,:,:].numpy(), mask=(obs[-3,:,:,:].numpy()==0.0))
            mri_ds = mri_vol[::2,::2,::4]
            needle = np.ma.array(grid, mask = (grid == 0.0))
            needle_ds = needle[::2,::2]
            x_cent = int(prostate_centroid[1]/2)
            y_cent = int(prostate_centroid[0]/2)
            
            # crop between y_cent-35:y_cent+30, x_cent-30:x_cent+40; but user input neext to select grid positions within [100,100]
            plt.imshow(mri_ds[:,:, int(SLICE_NUM/4)], cmap ='gray')
            plt.imshow(50*needle_ds[:,:], cmap='jet', alpha = 0.5)
            plt.imshow(np.max(mask_p[:,:,:], axis =2),cmap='coolwarm_r', alpha=0.5)
            plt.imshow(np.max(mask_n_1[:,:,:], axis =2),cmap='Wistia', alpha=0.4)
            plt.imshow(np.max(mask_n_2[:,:,:], axis =2),cmap='Wistia', alpha=0.4)
            plt.imshow(50*needle_ds[:,:], cmap='jet', alpha = 0.3)
            plt.imshow(np.max(mask_l[:,:,:], axis =2),cmap='summer', alpha=0.6)
            plt.imshow(np.max(mask_n[:,:,:], axis =2),cmap='Wistia', alpha=0.5)
            plt.axis('off')
            ### 4. Take in user actions to implement strategy ###
            grid_pos = plt.ginput(0,0)     
            grid_pos = np.array(grid_pos[0])
            grid_pos -= (prostate_centroid/2)[0:-1]
            
            # Swap x and y 
            #grid_pos[0], grid_pos[1] = grid_pos[1], grid_pos[0]
            
            # Take input action (ie clicked position - original position), then scale
            if num_steps == 0:
                current_pos = np.array([0,0])
            else:
                current_pos = biopsy_env.get_current_pos()
            
            #grid_pos = np.swapaxes(grid_pos, 1, 0)
            taken_actions = round_to_05(grid_pos - current_pos) 
            taken_actions = taken_actions / 10 # normalise between (-1,1)       
            taken_actions = np.append(taken_actions, ([1]))                                                                                         
            
            # Take step in environment 
            obs, reward, done, info = biopsy_env.step(taken_actions)
            print(f"Current pos : {current_pos}Agent suggested actions : {actions} our actions : {taken_actions} \n Done :{done} num_steps {num_steps}")
            num_steps += 1
            
            # BUG: observaiton of needle not alligned with actions!!! 
    print('chicken')
     
    
    
    
    