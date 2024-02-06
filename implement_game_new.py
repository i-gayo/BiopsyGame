import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt 
from utils.data_utils import * 
from utils.environment_utils import * 
from Envs.biopsy_env import *
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import CnnPolicy

def slice_select(slice_num):
    """
    Prompts the user to slect a slice to view 
    
    Arguements:
    slice_num (int): The total number of slices available in the targeted view
    
    Returns:
    selected_slice: (int) : The selected slice number."""
    while True:
        try:
            selected_slice= int(input(f"Enter the slice number to view from the range (0-{slice_num-1}) : "))
            if 0 <= selected_slice < slice_num:
                return selected_slice
            else:
                print(f"Please enter a number between 0 and {slice_num-1}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def sagittal_plot(mri_ds,mri_vol_shape):
        """
        Generates a plot of the sagittal view continously generating plots when a new input is 
        """
        mri_ds_sviewt1= mri_ds
        mri_ds_sviewt1 = np.transpose(mri_ds,[1,2,0])
        mri_vol_shape = np.shape(mri_ds_sviewt1)
        prev_slice_num = None
        print(f"the number of slices is {mri_vol_shape}")

        while True:
            try:
                user_input = input(f"Please enter a number between 0 and {mri_vol_shape[2] - 1} ('exit' to quit): ")
                if user_input.lower() == 'exit':
                    break

                selected_slice = int(user_input)
                if selected_slice < 0 or selected_slice >= mri_vol_shape[2]:
                    continue

                if selected_slice == prev_slice_num:
                    print("You've selected the same slice. Displaying again.")
                else:
                    plt.figure(1)
                    plt.title(f"Slice No: {selected_slice} sagittal view ")
                    plt.imshow(mri_ds_sviewt1[:,int(selected_slice)//2,:], cmap ='gray')
                    plt.show()

                prev_slice_num = selected_slice

            except ValueError:
                print("Invalid input. Please enter a valid number.")
        
def view_test(mri_ds,mri_vol_shape,dimensions):
    #setting the spatial coordinates of the voxels
    x_axis = np.arange(mri_vol_shape[0]) * dimensions[0]
    y_axis = np.arange(mri_vol_shape[1]) * dimensions[1]
    z_axis = np.arange(mri_vol_shape[2]) * dimensions[2]
    aspect_ratio = 1
    print(f"the shape of mri_vol is {np.shape(mri_ds)}")
    
    # showing multiple plots   
    plt.figure(1)
    plt.title("Slice on 1st column")
    plt.imshow(mri_ds[15,:,:], cmap ='gray')
    plt.figure(2)
    plt.title("Slice on 2nd column")
    plt.imshow(mri_ds[:,15,:], cmap ='gray')
    plt.figure(3)
    plt.title("Slice on 3rd column")
    plt.imshow(mri_ds[:,:,15], cmap ='gray') 

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

        grid = np.zeros((100,100))
        for i in range(-30, 35, 5):
            for j in range(-30, 35, 5):
                x_val = round(prostate_centroid[1]/2)+j
                y_val = round(prostate_centroid[0]/2) +i
                
                grid[y_val - 1:y_val+2 , x_val -1 : x_val+2] = 1

        grid_coords = np.array(np.where(grid == 1))  # given in y, x 
        
        # change to x,y convention instead of y,x 
        grid_coords[[0,1],:] = grid_coords[[1,0],:]
        
        return grid, grid_coords

def generate_grid_old(prostate_centroid):
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

def multiple_display(mri_data):
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
        ax.imshow(mri_data[: :, slice_index], cmap='gray')
        ax.set_title(f"Slice {slice_index + 1}")
        ax.axis('off')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()
    plt.text(20, 20, 'matplotlib EXAMPLE',color = 'green', horizontalalignment='center',verticalalignment='center')
    plt.show()

def plotter(current_max_needle,reward,totalreward,obs,vols,done,num_steps,hit,biopsy_env,agent,data):
    """
    Responsible for generating the graph but also taking into account the maximum number of needles required

    """
    while ((num_steps <= current_max_needle)):
        # Obtain lesion and mri vols from data 
        lesion_vol = biopsy_env.get_lesion_mask() # get individual lesion mask 
        totalreward = totalreward + reward 

        mri_vol = vols['mri_vol']
        prostate_vol = vols['prostate_mask']
        prostate_centroid = np.mean(np.where(prostate_vol), axis = 1)
        #print(f"Game rostate centroid : {prostate_centroid}")
        SLICE_NUM = int(prostate_centroid[-1])

        # Define grid coords 
        grid, grid_coords = generate_grid(prostate_centroid)

        # Obtain agents predicted actions 
        actions,_ = agent.predict(obs)
        #TODO : convert this to action / grid pos for agents!!! 
    
        #creating another subplot 
        fig, axs = plt.subplots(1)

        #plotting for the axial view 
        mask_l = np.ma.array(obs[0,:,:,:].numpy(), mask=(obs[0,:,:,:].numpy()==0.0))
        mask_p = np.ma.array(obs[1,:,:,:].numpy(), mask=(obs[1,:,:,:].numpy()==0.0))
        mask_n= np.ma.array(obs[-1,:,:,:].numpy(), mask=(obs[-1,:,:,:].numpy()==0.0))
        mask_n_1= np.ma.array(obs[-2,:,:,:].numpy(), mask=(obs[-2,:,:,:].numpy()==0.0))
        mask_n_2= np.ma.array(obs[-3,:,:,:].numpy(), mask=(obs[-3,:,:,:].numpy()==0.0))
        mri_ds = mri_vol[::2,::2,::4]
        needle = np.ma.array(grid, mask = (grid == 0.0))
        #needle_ds = needle[::2,::2]
        x_cent = int(prostate_centroid[1]/2)
        y_cent = int(prostate_centroid[0]/2)

        #plotting for the sagittal view 
        # mri_ds_sviewt1 = mri_ds
        mri_ds_sviewt1 = np.transpose(mri_ds,[2,1,0])
        mri_vol_shape= np.shape(mri_vol)
        dimensions=[0.5,0.5,1]
        # view_test(mri_ds_sviewt1,mri_vol_shape,dimensions)
        multiple_display(mri_ds_sviewt1)
        #sagittal_plot(mri_ds,mri_vol)


        #the axial view 
        # crop between y_cent-35:y_cent+30, x_cent-30:x_cent+40; but user input neext to select grid positions within [100,100]
        # plt.figure(1)
        # plt.imshow(mri_ds[:,:, int(SLICE_NUM/4)], cmap ='gray')
        # plt.imshow(50*needle[:,:], cmap='jet', alpha = 0.5)
        # plt.imshow(np.max(mask_p[:,:,:], axis =2),cmap='coolwarm_r', alpha=0.5)
        # plt.imshow(np.max(mask_n_1[:,:,:], axis =2),cmap='Wistia', alpha=0.4)
        # plt.imshow(np.max(mask_n_2[:,:,:], axis =2),cmap='Wistia', alpha=0.4)
        # plt.imshow(50*needle[:,:], cmap='jet', alpha = 0.3)
        # plt.imshow(np.max(mask_l[:,:,:], axis =2),cmap='summer', alpha=0.6)
        # plt.imshow(np.max(mask_n[:,:,:], axis =2),cmap='Wistia', alpha=0.5)
        # print(f"the shape of mri_vol is {np.shape(mri_ds)}")
        
        # ADDING labels to grid positions!!!
        first_x = np.min(np.where(grid == 1)[1])
        first_y = np.min(np.where(grid == 1)[0])
        last_x = np.max(np.where(grid == 1)[1])
        last_y = np.max(np.where(grid == 1)[0])
        s = 'A  a  B  b  C  c  D  d  E  e  F  f  G' # fontsize 10.5 
        #s = '-30 -25 -20 -15 -10 -5 0 5 10 15 20 25 30' #font size 8
        plt.text(first_x, first_y - 5, s, fontsize = 10.5, color = 'aqua', bbox=dict(fill=False, edgecolor='green', linewidth=1))#, transform= axs.transAxes)
        grid_labels = np.arange(7, 0.5, -0.5)
        #grid_labels = np.arange(-30, 35, 5)
        for idx, label in enumerate(grid_labels):
            plt.text(first_x-10, first_y + (idx*5.15), label, fontsize = 10.5, color = 'aqua')
            plt.text(last_x+5, first_y + (idx*5.15), label, fontsize = 10.5, color = 'aqua')

        #Displays the rewards metrics within the game 
        #The data comes from the library from the info dictionary within game dev
        #checking for needle hit 
        if data["needle_hit"] == True: hit='HIT'
        else: hit='MISS'
        plt.text(first_x - 18,first_y-14.5, f'Total Result: {totalreward} ' ,fontsize = 12.5, color = 'yellow')
        plt.text((last_x*0.5), first_y-14.5, f'Previous Result: {reward} ({hit})',fontsize = 12.5, color = 'greenyellow')
        plt.text(first_x-15,last_y+16,f"CCL:{data['norm_ccl']} ",fontsize= 10.5,color = 'salmon')
        #print (f"The current lesion size is : {data['lesion_size']}")

        plt.axis('off')

        # Take input action (ie clicked position - original position), then scale
        if num_steps == 0:
            current_pos = np.array([0,0])
        else:
            current_pos = biopsy_env.get_current_pos()


        # Convert agent actions -> positions to choose 
        suggested_pos = round_to_05((actions[0:-1] * 10) + current_pos)
        #my_pos = round_to_05((taken_actions[0:-1]*10) + current_pos)

        # Convert predicted actions to grid pos (A, E)
        x_dict = ['A', 'a', 'B', 'b', 'C', 'c', 'D', 'd', 'E', 'e', 'F', 'f', 'G']
        grid_vals = np.arange(-30,35, 5)
        x_idx = x_dict[(np.where(grid_vals == suggested_pos[0]))[0][0]]

        y_dict = [str(num) for num in np.arange(7, 0.5, -0.5)]
        y_idx = y_dict[(np.where(grid_vals == suggested_pos[1]))[0][0]]

        suggested_str = 'Suggested GRID POSITION: [' + x_idx + ',' + y_idx + ']'
        plt.text(first_x-10, last_y + 10, suggested_str, fontsize = 12, color = 'magenta')
        # End of all the plots and figures so show them here 
        plt.show()

        ### 4. Take in user actions to implement strategy ###
        grid_pos = plt.ginput(1,0) #0,0)     
        #grid_pos = round_to_05(np.array(grid_pos[0]))
        prostate_cent_xy = np.array([prostate_centroid[1], prostate_centroid[0]])/2
        grid_pos -= ((prostate_cent_xy))

        raw_actions = np.round(grid_pos - current_pos)
        taken_actions = round_to_05(np.round(grid_pos - current_pos))

        taken_actions = taken_actions / 10 # normalise between (-1,1)  
        #taken_actions[0], taken_actions[1] = taken_actions[1], taken_actions[0]     
        taken_actions = np.append(taken_actions, ([1]))                                                                                         

        # Take step in environment 
        obs, reward, done, data = biopsy_env.step(taken_actions)

        #print(f"Current pos : {current_pos}Agent suggested actions : {actions} our actions : {taken_actions} \n Done :{done} num_steps {num_steps}")
        num_steps += 1

        plt.close()

    return obs,reward,data,totalreward

def run_game(NUM_EPISODES=5, log_dir = 'game'):
    
    data_path = '.\Data\ProstateDataset'
    csv_path = '.\Data\ProstateDataset\patient_data_multiple_lesions.csv'
    log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)
    
    # Load biopsy envs and datasets
    PS_dataset = Image_dataloader(data_path, csv_path, use_all = True, mode  = 'test')
    Data_sampler = DataSampler(PS_dataset)
    biopsy_env = TemplateGuidedBiopsy(Data_sampler,results_dir = log_dir, reward_fn = 'reward', \
        max_num_steps = 20, deform = True, start_centre= True)
    data=biopsy_env.get_info()
    reward=0
    totalreward=0
    
    # Load agent
    policy_kwargs = dict(features_extractor_class = NewFeatureExtractor, features_extractor_kwargs=dict(multiple_frames = True, num_channels = 5))
    agent = PPO(CnnPolicy, env = biopsy_env, policy_kwargs = policy_kwargs)
    
    # Run game for num episodes
    #princh t(f"Loading game script. Running for {NUM_EPISODES} episodes")

    for i in range(NUM_EPISODES):
        obs = biopsy_env.reset()
        vols = biopsy_env.get_img_data()
        done = False 
        num_steps = 0 
        hit=""
        current_patient=biopsy_env.get_info()
        #checking for lesion size of the next patient

        print(current_patient['lesion_size'])
        
        if current_patient['lesion_size']<=750:
            obs,reward,data,totalreward=plotter(3,reward,totalreward,obs,vols,done,num_steps,hit,biopsy_env,agent,data)
            print("Threshold a ")
        elif current_patient['lesion_size']>=751 and current_patient['lesion_size']<=1000 :
            obs,reward,data,totalreward=plotter(4,reward,totalreward,obs,vols,done,num_steps,hit,biopsy_env,agent,data)
            print("Threshold b ")
        elif current_patient['lesion_size']>=1001 and current_patient['lesion_size']<=2000 :
            obs,reward,data,totalreward=plotter(5,reward,totalreward,obs,vols,done,num_steps,hit,biopsy_env,agent,data)
            print("Threshold c ")
        elif current_patient['lesion_size']>=2001 and current_patient['lesion_size']<=20000:
            obs,reward,data,totalreward=plotter(6,reward,totalreward,obs,vols,done,num_steps,hit,biopsy_env,agent,data)
            print("Threshold d ")
        else:
            print("Check again for lesion size (ERROR) everything should have been accounted for ")

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
    NUM_EPISODES = 5
    
    #### 1. Load biopsy env ####
    biopsy_env = TemplateGuidedBiopsy(Data_sampler,results_dir = 'game', reward_fn = 'reward', \
        max_num_steps = 20, deform = True, deform_rate = RATE, deform_scale = SCALE, start_centre= True)
    
    ### 2. Load RL model for inference :for now, a random policy     ####
    policy_kwargs = dict(features_extractor_class = NewFeatureExtractor, features_extractor_kwargs=dict(multiple_frames = True, num_channels = 5))
    agent = PPO(CnnPolicy, env = biopsy_env, policy_kwargs = policy_kwargs)
    
    ### 3. User interface     ####
    # obs = biopsy_env.reset()
    # obs = biopsy_env.reset()
    # obs = biopsy_env.reset()
    # obs = biopsy_env.reset()
    # obs = biopsy_env.reset()
    # obs = biopsy_env.reset()
    # obs = biopsy_env.reset()
    # obs = biopsy_env.reset()
    # obs = biopsy_env.reset()
    # obs = biopsy_env.reset()
    
    #TODO: fix bug in maybe round_to_05 unction? 
    for i in range(NUM_EPISODES):
        
        # reseset twice
        #if i == 0:
        #    obs = biopsy_env.reset()

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
            #print(f"Game rostate centroid : {prostate_centroid}")
            SLICE_NUM = int(prostate_centroid[-1])
            
            # Define grid coords 
            grid, grid_coords = generate_grid(prostate_centroid)

            # Obtain agents predicted actions 
            actions,_ = agent.predict(obs)
            #TODO : convert this to action / grid pos for agents!!! 
            
            #adding an additional subplot 
            fig, axs = plt.subplots(1)
            mask_l = np.ma.array(obs[0,:,:,:].numpy(), mask=(obs[0,:,:,:].numpy()==0.0))
            mask_p = np.ma.array(obs[1,:,:,:].numpy(), mask=(obs[1,:,:,:].numpy()==0.0))
            mask_n= np.ma.array(obs[-1,:,:,:].numpy(), mask=(obs[-1,:,:,:].numpy()==0.0))
            mask_n_1= np.ma.array(obs[-2,:,:,:].numpy(), mask=(obs[-2,:,:,:].numpy()==0.0))
            mask_n_2= np.ma.array(obs[-3,:,:,:].numpy(), mask=(obs[-3,:,:,:].numpy()==0.0))
            mri_ds = mri_vol[::2,::2,::4]
            needle = np.ma.array(grid, mask = (grid == 0.0))
            #needle_ds = needle[::2,::2]
            x_cent = int(prostate_centroid[1]/2)
            y_cent = int(prostate_centroid[0]/2)
            

            #plot for the axial view 
            # crop between y_cent-35:y_cent+30, x_cent-30:x_cent+40; but user input neext to select grid positions within [100,100]
            plt.imshow(mri_ds[:,:, int(SLICE_NUM/4)], cmap ='gray')
            plt.imshow(50*needle[:,:], cmap='jet', alpha = 0.5)
            plt.imshow(np.max(mask_p[:,:,:], axis =2),cmap='coolwarm_r', alpha=0.5)
            plt.imshow(np.max(mask_n_1[:,:,:], axis =2),cmap='Wistia', alpha=0.4)
            plt.imshow(np.max(mask_n_2[:,:,:], axis =2),cmap='Wistia', alpha=0.4)
            plt.imshow(50*needle[:,:], cmap='jet', alpha = 0.3)
            plt.imshow(np.max(mask_l[:,:,:], axis =2),cmap='summer', alpha=0.6)
            plt.imshow(np.max(mask_n[:,:,:], axis =2),cmap='Wistia', alpha=0.5)


            # ADDING labels to grid positions!!!
            first_x = np.min(np.where(grid == 1)[1])
            first_y = np.min(np.where(grid == 1)[0])
            last_x = np.max(np.where(grid == 1)[1])
            last_y = np.max(np.where(grid == 1)[0])
            s = 'A  a  B  b  C  c  D  d  E  e  F  f  G' # fontsize 10.5 
            #s = '-30 -25 -20 -15 -10 -5 0 5 10 15 20 25 30' #font size 8
            plt.text(first_x, first_y - 5, s, fontsize = 10.5, color = 'aqua', bbox=dict(fill=False, edgecolor='green', linewidth=1))#, transform= axs.transAxes)
            grid_labels = np.arange(7, 0.5, -0.5)
            #grid_labels = np.arange(-30, 35, 5)
            for idx, label in enumerate(grid_labels):
                plt.text(first_x-10, first_y + (idx*5.15), label, fontsize = 10.5, color = 'aqua')
                plt.text(last_x+5, first_y + (idx*5.15), label, fontsize = 10.5, color = 'aqua')
                
            plt.axis('off')
            
            # Convert agent actions -> 
            # Take input action (ie clicked position - original position), then scale
            if num_steps == 0:
                current_pos = np.array([0,0])
            else:
                current_pos = biopsy_env.get_current_pos()
            #positions to choose 
            suggested_pos = round_to_05((actions[0:-1] * 10) + current_pos)
            #my_pos = round_to_05((taken_actions[0:-1]*10) + current_pos)
            
            # Convert predicted actions to grid pos (A, E)
            x_dict = ['A', 'a', 'B', 'b', 'C', 'c', 'D', 'd', 'E', 'e', 'F', 'f', 'G']
            grid_vals = np.arange(-30,35, 5)
            x_idx = x_dict[(np.where(grid_vals == suggested_pos[0]))[0][0]]
            
            y_dict = [str(num) for num in np.arange(7, 0.5, -0.5)]
            y_idx = y_dict[(np.where(grid_vals == suggested_pos[1]))[0][0]]
            
            suggested_str = 'Suggested GRID POSITION: [' + x_idx + ',' + y_idx + ']'
            plt.text(first_x-10, last_y + 10, suggested_str, fontsize = 12, color = 'magenta')
            
            
            
            ### 4. Take in user actions to implement strategy ###
            grid_pos = plt.ginput(1,0) #0,0)     
            #grid_pos = round_to_05(np.array(grid_pos[0]))
            prostate_cent_xy = np.array([prostate_centroid[1], prostate_centroid[0]])/2
            grid_pos -= ((prostate_cent_xy))

        
            # Swap x and y 
            #grid_pos[0], grid_pos[1] = grid_pos[1], grid_pos[0]
            
            
            #grid_pos = np.swapaxes(grid_pos, 1, 0)

            raw_actions = np.round(grid_pos - current_pos)
            taken_actions = round_to_05(np.round(grid_pos - current_pos))

            taken_actions = taken_actions / 10 # normalise between (-1,1)  
            #taken_actions[0], taken_actions[1] = taken_actions[1], taken_actions[0]     
            taken_actions = np.append(taken_actions, ([1]))                                                                                         
            
            # Take step in environment 
            obs, reward, done, info = biopsy_env.step(taken_actions)
            
            #print(f"Current pos : {current_pos}Agent suggested actions : {actions} our actions : {taken_actions} \n Done :{done} num_steps {num_steps}")
            num_steps += 1
            
            # BUG: observaiton of needle not alligned with actions!!! 
    print('chicken')
     
    
    
    
    