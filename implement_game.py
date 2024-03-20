from colorama import Fore, Back, Style
import csv
import numpy as np
import datetime
import SimpleITK as sitk
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from utils.data_utils import *
from utils.environment_utils import *
from Envs.biopsy_env import *
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import CnnPolicy
from PIL import Image


def generate_grid(prostate_centroid):
    """
    Generates 2D grid of grid point coords on image coordinates

    Arguments:
    :prostate_centroid (ndarray) : centroid in x,y,z convention of prostate gland

    Returns:
    :grid_coords (ndarray) : 2 x 169 grid coords x,y convention
    """

    x_grid = (np.arange(-30, 35, 5)) * 2 + prostate_centroid[0]
    y_grid = (np.arange(-30, 35, 5)) * 2 + prostate_centroid[1]

    grid = np.zeros((100, 100))
    for i in range(-30, 35, 5):
        for j in range(-30, 35, 5):
            x_val = round(prostate_centroid[1] / 2) + j
            y_val = round(prostate_centroid[0] / 2) + i

            grid[y_val - 1 : y_val + 2, x_val - 1 : x_val + 2] = 1

    grid_coords = np.array(np.where(grid == 1))  # given in y, x

    # change to x,y convention instead of y,x
    grid_coords[[0, 1], :] = grid_coords[[1, 0], :]

    return grid, grid_coords


def generate_grid_old(prostate_centroid):
    """
    Generates 2D grid of grid point coords on image coordinates

    Arguments:
    :prostate_centroid (ndarray) : centroid in x,y,z convention of prostate gland

    Returns:
    :grid_coords (ndarray) : 2 x 169 grid coords x,y convention
    """

    x_grid = (np.arange(-30, 35, 5)) * 2 + prostate_centroid[0]
    y_grid = (np.arange(-30, 35, 5)) * 2 + prostate_centroid[1]

    grid = np.zeros((200, 200))
    for i in range(-60, 65, 10):
        for j in range(-60, 65, 10):
            x_val = int(prostate_centroid[1]) + j
            y_val = int(prostate_centroid[0]) + i

            grid[x_val - 1 : x_val + 2, y_val - 1 : y_val + 2] = 1

    grid_coords = np.array(np.where(grid == 1))  # given in y, x

    # change to x,y convention instead of y,x
    grid_coords[[0, 1], :] = grid_coords[[1, 0], :]

    return grid, grid_coords


def select_depth_interactive(ax, first_y):
    depth_options = ["0: Apex", "1: Centroid", "2: Base"]
    x_positions = [0.5, 1, 1.5]  # Preset x positions for the text

    for option, x in zip(depth_options, x_positions):
        ax.text(
            x,
            0.75,
            option,
            ha="center",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )

    ax.set_xlim(-1, 3)
    ax.set_ylim(0, 1)
    ax.set_title(" FIRST Select the depth of the prostate :", color="white")
    ax.axis("off")
    plt.draw()

    print("Please click on the plot to select the depth.")
    click_depth = plt.ginput(1, 0)  # Wait for one click
    x_click = click_depth[0][0]
    print(f"X CLICK value {x_click}")

    # Determine the depth based on the x-coordinate of the click
    if x_click < 0.7:
        action = 0  # Apex
    elif x_click < 1.25:
        action = 1  # Centroid
    else:
        action = 2  # Base
    print(Fore.MAGENTA + f"action selected {action}" + Fore.RESET)
    return action


def convert_depth(action, prostate_mask, prostate_centroid):
    """
    Convert the action into a depth value based on the given prostate mask and centroid.

    Parameters:
    action (int): The action to convert into a depth value.
    prostate_mask (numpy.ndarray): The mask representing the prostate.
    prostate_centroid (tuple): The centroid coordinates of the prostate.

    Returns:
    int: The converted depth value.

    Raises:
    ValueError: If an invalid action is selected.
    """
    if action == 0:
        depth = np.min(np.where(prostate_mask == 1)[-1])
    elif action == 1:
        depth = prostate_centroid[2]
    elif action == 2:
        depth = np.max(np.where(prostate_mask == 1)[-1])
    else:
        raise ValueError("Invalid action selected.")

    print(Fore.MAGENTA + f"DEPTH SELECTED {depth}" + Fore.RESET)
    return round(depth)


def coord_converter(coordinates, prostate_centroid):
    """
    Converts the coordinates from the grid to the image coordinates

    Arguments:
    :coordinates (ndarray) : 3 x 1 array of grid coordinates

    Returns:
    :image_coords (ndarray) : 3 x 1 array of image coordinates
    """
    # Convert to image coordinates
    x_idx = coordinates[0]
    y_idx = coordinates[1]

    # Converts range from (-30,30) to image grid array

    x_idx = (x_idx) * 2 + round(prostate_centroid[0])
    y_idx = (y_idx) * 2 + round(prostate_centroid[1])

    x_grid_pos = round(x_idx)
    y_grid_pos = round(y_idx)

    # print(Fore.BLUE + f"X GRID POS {x_grid_pos} Y GRID POS {y_grid_pos}" + Fore.RESET)

    return x_grid_pos, y_grid_pos


# Global variable to keep track of the last logged patient ID
last_patient_id = None
last_lesion_idx = None


# debug this
def log_user_input(
    file_path, patient_id, lesion_idx, action_idx, img_x, img_y, img_z, x, y, z
):
    global last_patient_id, last_lesion_idx

    # Check if the file already exists to decide on writing headers
    file_exists = os.path.isfile(file_path)

    # Open the CSV file
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)

        # If the file does not exist, write the column headers
        if not file_exists:
            writer.writerow(
                [
                    "patient_id",
                    "lesion_idx",
                    "action_idx",
                    "img_x",
                    "img_y",
                    "img_z",
                    "x",
                    "y",
                    "z",
                ]
            )

        # Check if the patient ID has changed (and it's not the first entry) to add a blank line
        if (patient_id != last_patient_id or lesion_idx != last_lesion_idx) and (
            last_patient_id is not None or last_lesion_idx is not None
        ):
            writer.writerow([])  # Insert a blank line for a new patient

        # Update the last_patient_id with the current patient_id
        last_patient_id = patient_id
        last_lesion_idx = lesion_idx

        # Data to be written to the CSV
        row_data = [patient_id, lesion_idx, action_idx, img_x, img_y, img_z, x, y, z]
        writer.writerow(row_data)

    print("Logged data for patient ID:", patient_id)


def plotter(
    reward,
    totalreward,
    obs,
    vols,
    done,
    num_steps,
    hit,
    biopsy_env,
    agent,
    data,
    sag_index,
    depth,
    file_path,
    depth_action,
    current_episode,
):
    """
    This function plots the game environment and updates the views at each iteration.
    It takes in various parameters including the reward, total reward, observations, volumes, game status, number of steps,
    hit status, biopsy environment, agent, data, sagittal index, and depth.
    The function iterates until the number of steps is less than or equal to 4.
    It obtains lesion and MRI volumes from the data and calculates the prostate centroid.
    The function generates grid coordinates based on the prostate centroid.
    It predicts the agent's actions and plots the sagittal and axial views of the game environment.
    The function also displays reward metrics and allows the user to select grid positions.
    User actions are taken to implement the strategy, and the environment is updated accordingly.
    The function returns the updated observations, reward, data, total reward, sagittal index, and depth.
    """
    while num_steps <= 4:
        # Obtain lesion and mri vols from data
        lesion_vol = biopsy_env.get_lesion_mask()  # get individual lesion mask
        totalreward = totalreward + reward

        mri_vol = vols["mri_vol"]
        prostate_vol = vols["prostate_mask"]
        tumour_vol = vols["tumour_mask"]
        prostate_centroid = np.mean(np.where(prostate_vol), axis=1)
        # print(Fore.YELLOW + f" PROSTATE CENTROIDS {prostate_centroid}" + Fore.RESET)
        SLICE_NUM = int(prostate_centroid[-1])

        # Define grid coords
        grid, grid_coords = generate_grid(prostate_centroid)

        print(np.unique(grid))

        # Obtain agents predicted actions
        actions, _ = agent.predict(obs)
        # TODO : convert this to action / grid pos for agents!!!

        fig = plt.figure(figsize=(15, 6), facecolor="#2c3e50")
        gs = gridspec.GridSpec(2, 3, height_ratios=[3, 1])

        # sizing first row
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[0, 2])

        # sizing second row
        ax4 = plt.subplot(gs[1, :])  # This will span all columns

        mask_l = np.ma.array(
            obs[0, :, :, :].numpy(), mask=(obs[0, :, :, :].numpy() == 0.0)
        )
        mask_p = np.ma.array(
            obs[1, :, :, :].numpy(), mask=(obs[1, :, :, :].numpy() == 0.0)
        )
        mask_n = np.ma.array(
            obs[-1, :, :, :].numpy(), mask=(obs[-1, :, :, :].numpy() == 0.0)
        )
        mask_n_1 = np.ma.array(
            obs[-2, :, :, :].numpy(), mask=(obs[-2, :, :, :].numpy() == 0.0)
        )
        mask_n_2 = np.ma.array(
            obs[-3, :, :, :].numpy(), mask=(obs[-3, :, :, :].numpy() == 0.0)
        )
        mri_ds = mri_vol[::2, ::2, ::4]
        needle = np.ma.array(grid, mask=(grid == 0.0))
        # needle_ds = needle[::2,::2]
        x_cent = int(prostate_centroid[1] / 2)
        y_cent = int(prostate_centroid[0] / 2)

        # Plot of the sagittal view which updates at each iteration
        # Finding the new sagittal slice_number
        grid_index = data["current_pos"]
        # taking only the x value of sag_index as that is what is being plotted
        sag_index = coord_converter(grid_index, prostate_centroid)
        print(Fore.BLUE + f"THE grid INDEX IS {grid_index}" + Fore.RESET)
        print(Fore.BLUE + f"THE SAG INDEX IS {sag_index}" + Fore.RESET)
        sag_index_plt = sag_index[0]

        # attempting to flip the view
        index = mri_vol[:, sag_index_plt, :]
        flipped_index = np.fliplr(index)
        ax1.imshow(flipped_index, cmap="grey", aspect=0.5)
        ax1.set_title(f"sagittal view for previous needle placement", color="white")
        ax1.axis("off")

        # Plotting for the axial view
        # crop between y_cent-35:y_cent+30, x_cent-30:x_cent+40; but user input neext to select grid positions within [100,100]
        ax2.imshow(mri_ds[:, :, int(SLICE_NUM / 4)], cmap="gray")
        ax2.imshow(50 * needle[:, :], cmap="jet", alpha=0.5)
        ax2.imshow(np.max(mask_p[:, :, :], axis=2), cmap="coolwarm_r", alpha=0.5)
        ax2.imshow(np.max(mask_n_1[:, :, :], axis=2), cmap="Wistia", alpha=0.4)
        ax2.imshow(np.max(mask_n_2[:, :, :], axis=2), cmap="Wistia", alpha=0.4)
        ax2.imshow(50 * needle[:, :], cmap="jet", alpha=0.3)
        ax2.imshow(np.max(mask_l[:, :, :], axis=2), cmap="summer", alpha=0.6)
        ax2.imshow(np.max(mask_n[:, :, :], axis=2), cmap="Wistia", alpha=0.5)
        ax2.axis("off")

        # ADDING labels to grid positions!!!
        first_x = np.min(np.where(grid == 1)[1])
        first_y = np.min(np.where(grid == 1)[0])
        last_x = np.max(np.where(grid == 1)[1])
        last_y = np.max(np.where(grid == 1)[0])
        s = "A a B b C c D d E e F f G"  # fontsize 10.5
        # s = '-30 -25 -20 -15 -10 -5 0 5 10 15 20 25 30' #font size 8
        ax2.text(
            first_x,
            first_y - 5,
            s,
            fontsize=11,
            color="aqua",
            bbox=dict(fill=False, edgecolor="green", linewidth=1),
        )  # , transform= axs.transAxes)
        grid_labels = np.arange(7, 0.5, -0.5)
        # grid_labels = np.arange(-30, 35, 5)
        for idx, label in enumerate(grid_labels):
            ax2.text(
                first_x - 10, first_y + (idx * 5.15), label, fontsize=10.5, color="aqua"
            )
            ax2.text(
                last_x + 5, first_y + (idx * 5.15), label, fontsize=10.5, color="aqua"
            )

        # Displays the rewards metrics within the game
        # The data comes from the library from the info dictionary within game dev
        # checking for needle hit

        # change the posititons of the text to be more visible
        if data["needle_hit"] == True:
            hit = "HIT"
        else:
            hit = "MISS"
        ax2.set_title(
            f"Episode counter : {current_episode+1} \n Total Reward: {totalreward} \n Previous Reward: {reward} ({hit})",
            color="#FFDB58",
        )
        ax2.text(
            first_x - 15,
            last_y + 16,
            f"CCL:{data['norm_ccl']} ",
            fontsize=12.5,
            color="#FFDB58",
        )

        # plotting for the axial view
        # Allow user to select the depth of the prostate
        depth = convert_depth(depth_action, prostate_vol, prostate_centroid) / 4

        ax3.imshow(mri_ds[:, :, int(depth)], cmap="gray")

        # Adding the additional mask layers
        mask_p_ax = mask_p[:, :, int(depth)]
        mask_l_ax = mask_l[:, :, int(depth)]
        ax3.imshow(mask_p_ax, cmap="coolwarm_r", alpha=0.5)
        ax3.imshow(mask_l_ax, cmap="summer", alpha=0.6)

        # additonal text
        if depth_action == 0:
            depth_str = "Apex"
        elif depth_action == 1:
            depth_str = "Midgland"
        else:
            depth_str = "Base"
        ax3.set_title(f" Axial view for depth selection {depth_str}", color="white")
        ax3.axis("off")

        # Interactive plot for depth
        depth_action = select_depth_interactive(ax4, first_y)

        # Take input action (ie clicked position - original position), then scale
        if num_steps == 0:
            current_pos = np.array([0, 0])
        else:
            current_pos = biopsy_env.get_current_pos()

        # Convert agent actions -> positions to choose
        suggested_pos = round_to_05((actions[0:-1] * 10) + current_pos)

        # Change to boundary poisiotns (ie bertween -30,30 instead of -35,30)
        for idx, pos in enumerate(suggested_pos):
            # if greater than 30, change to grid 30
            if pos > 30:
                suggested_pos[idx] = 30
            # if less than -30, change to grid 30
            elif pos < -30:
                suggested_pos[idx] = -30

        # my_pos = round_to_05((taken_actions[0:-1]*10) + current_pos)

        # Convert predicted actions to grid pos (A, E)
        x_dict = ["A", "a", "B", "b", "C", "c", "D", "d", "E", "e", "F", "f", "G"]
        grid_vals = np.arange(-30, 35, 5)
        x_idx = x_dict[(np.where(grid_vals == suggested_pos[0]))[0][0]]

        y_dict = [str(num) for num in np.arange(7, 0.5, -0.5)]
        y_idx = y_dict[(np.where(grid_vals == suggested_pos[1]))[0][0]]

        suggested_str = "Suggested GRID POSITION: [" + x_idx + "," + y_idx + "]"
        ax2.text(first_x - 10, last_y + 10, suggested_str, fontsize=12, color="magenta")

        ### 4. Take in user actions to implement strategy ###
        grid_pos = plt.ginput(1, 0)  # 0,0)
        # grid_pos = round_to_05(np.array(grid_pos[0]))
        prostate_cent_xy = np.array([prostate_centroid[1], prostate_centroid[0]]) / 2
        grid_pos -= prostate_cent_xy

        raw_actions = np.round(grid_pos - current_pos)
        taken_actions = round_to_05(np.round(grid_pos - current_pos))

        taken_actions = taken_actions / 10  # normalise between (-1,1)
        # taken_actions[0], taken_actions[1] = taken_actions[1], taken_actions[0]
        #taken_actions = np.append(taken_actions, ([1]))

        taken_actions = np.append(taken_actions, [depth_action])
        
        # Take step in environment
        obs, reward, done, data = biopsy_env.step(taken_actions)

        # print(f"Current pos : {current_pos}Agent suggested actions : {actions} our actions : {taken_actions} \n Done :{done} num_steps {num_steps}")
        num_steps += 1

        plt.tight_layout()

        plt.close()

        # Inputting all of the data in into the csv file
        # checking for all of the data being written into the csv file
        # print(Fore.LIGHTYELLOW_EX + f"num step is {num_steps}" + Fore.RESET)
        current_patient = data["patient_name"]
        patient_id = current_patient.split("\\")[-1].split("_")[0]
        print(Fore.LIGHTBLUE_EX + f" the current patient is {patient_id} " + Fore.RESET)
        log_user_input(
            file_path,
            patient_id,
            data["lesion_idx"],
            num_steps,
            sag_index[0],
            sag_index[1],
            depth,
            grid_index[0],
            grid_index[1],
            depth,
        )

    return obs, reward, data, totalreward, sag_index, depth


def intro():
    # fig, ax = plt.subplots(figsize=(12, 6), facecolor="#2c3e50")
    fig, ax = plt.subplot_mosaic(
        [["left", "upper right"], ["left", "lower right"]],
        figsize=(12, 7),
        layout="constrained",
        facecolor="#2c3e50",
    )
    # for k, ax in axd.items():
    #     annotate_axes(ax, f'axd[{k!r}]', fontsize=14)
    fig.suptitle(
        "Welcome to the Prostate Biopsy Game!",
        fontsize=16,
        fontweight="bold",
        color="white",
    )

    # Set the title of the plot
    # ax.set_title("Welcome to the Prostate Biopsy Game!", fontsize=16, fontweight="bold")

    # Add text with instructions on how to play the game
    blurb = """
    This game aims to simulate the process of performing a prostate biopsy.
    The game screen is divided into 3 plots:
    From left to right 
    1. Sagittal view of the prostate
    2. Game screen - where you can click to select the target location for the needle
    3. Axial view of the prostate

    You will take 2 actions, first select the depth of the prostate and then select the target location for the needle.
    Based on the actions, the game will update the views and display reward metrics.
    Your goal is to accurately target lesions within the prostate.
    The prostate,lesion and needle masks are displayed in the game screen.
    With the color red representing the prostate,green representing the lesion and yellow for the needle.
    The game progresses through multiple rounds - aim for the highest score!
    """

    instructions = """
    How to Play:
    1. Select the depth of the biopsy needle using your mouse.
    The options for depth are (from sagittal/side view)
        0 - Apex (front of prostate)
        1 - Centroid (middle of prostate)
        2 - Base (back of prostate)
    2. Click on the middle plot to choose the target location for the needle.
    3. Your goal is to accurately target lesions within the prostate.
    4. The game progresses through multiple rounds - aim for the highest score!

    Good luck and have fun!

    CLOSE THIS WINDOW (CLICK X ON THE WINDOW) TO START PLAYING
    """

    # Display the instructions
    blurb_position = 0.75
    ax["left"].text(
        0.01,
        blurb_position,
        blurb,
        color="#FFDB58",
        ha="left",
        va="center",
        fontsize=10,
    )
    ax["left"].text(
        0.01,
        (blurb_position - 0.5),
        instructions,
        color="white",
        ha="left",
        va="center",
        fontsize=10,
    )

    # Hide the axes
    ax["left"].axis("off")

    # displaying images
    img_folder = os.path.join(".", "Figures")
    img1 = Image.open(os.path.join(img_folder, "planes.png"))
    img2 = Image.open(os.path.join(img_folder, "game_labelled.png"))
    ax["upper right"].imshow(img1)
    ax["upper right"].axis("off")
    ax["lower right"].imshow(img2)
    ax["lower right"].axis("off")
    # Show the plot
    plt.imshow
    plt.show()


def episode_counter(csv_path):
    # Read the CSV file using pandas
    df = pd.read_csv(csv_path)

    # Count the number of patients
    num_patients = len(df)

    # Calculate the total number of lesions
    num_lesions = df["num_lesions"].sum()

    # Calculate the number of episodes
    num_episode = num_lesions

    return num_episode


def run_game(NUM_EPISODES, log_dir="game"):

    data_path = os.path.join(".", "Data", "ProstateDataset")
    csv_path = os.path.join(
        ".", "Data", "ProstateDataset", "patient_data_multiple_lesions.csv"
    )
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_path = f"user_input_data_{datetime_str}.csv"
    log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)

    # Load biopsy envs and datasets
    PS_dataset = Image_dataloader(data_path, csv_path, use_all=True, mode="train")
    Data_sampler = DataSampler(PS_dataset)
    biopsy_env = TemplateGuidedBiopsy(
        Data_sampler,
        results_dir=log_dir,
        reward_fn="reward",
        max_num_steps=20,
        deform=True,
        start_centre=True,
    )
    # NUM_EPISODES = episode_counter(csv_path)
    # print (f"Number of episodes: {NUM_EPISODES}")
    data = biopsy_env.get_info()
    reward = 0
    totalreward = 0
    # initialising variable
    sag_index = 0
    depth = 0
    depth_action = 1
    image_data = biopsy_env.get_img_data()
    step = biopsy_env.get_step_count()

    # Load agent1
    policy_kwargs = dict(
        features_extractor_class=NewFeatureExtractor,
        features_extractor_kwargs=dict(multiple_frames=True, num_channels=5),
    )
    agent = PPO(CnnPolicy, env=biopsy_env, policy_kwargs=policy_kwargs)

    # Run game for num episodes
    print(f"Loading game script. Running for {NUM_EPISODES} episodes")
    intro()
    for i in range(NUM_EPISODES):
        obs = biopsy_env.reset()
        vols = biopsy_env.get_img_data()
        done = False
        num_steps = 0
        hit = ""

        plotter(
            reward,
            totalreward,
            obs,
            vols,
            done,
            num_steps,
            hit,
            biopsy_env,
            agent,
            data,
            sag_index,
            depth,
            file_path,
            depth_action,
            i,
        )


if __name__ == "__main__":

    ps_path = "/Users/ianijirahmae/Documents/DATASETS/Data_by_modality"
    csv_path = "/Users/ianijirahmae/Documents/PhD_project/MRes_project/Reinforcement Learning/patient_data_multiple_lesions.csv"
    log_dir = "game"
    os.makedirs(log_dir, exist_ok=True)

    # DATASETS
    PS_dataset = Image_dataloader(ps_path, csv_path, use_all=True, mode="train")
    Data_sampler = DataSampler(PS_dataset)

    # HYPERPARAMETERS
    RATE = 0.1
    SCALE = 0.25
    NUM_EPISODES = 20

    #### 1. Load biopsy env ####
    biopsy_env = TemplateGuidedBiopsy(
        Data_sampler,
        results_dir="game",
        reward_fn="reward",
        max_num_steps=20,
        deform=True,
        deform_rate=RATE,
        deform_scale=SCALE,
        start_centre=True,
    )

    ### 2. Load RL model for inference :for now, a random policy     ####
    policy_kwargs = dict(
        features_extractor_class=NewFeatureExtractor,
        features_extractor_kwargs=dict(multiple_frames=True, num_channels=5),
    )
    agent = PPO(CnnPolicy, env=biopsy_env, policy_kwargs=policy_kwargs)

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

    # TODO: fix bug in maybe round_to_05 unction?
    for i in range(NUM_EPISODES):

        # reseset twice
        # if i == 0:
        #    obs = biopsy_env.reset()

        obs = biopsy_env.reset()
        vols = biopsy_env.get_img_data()
        done = False
        num_steps = 0

        while num_steps <= 4:
            # Obtain lesion and mri vols from data
            lesion_vol = biopsy_env.get_lesion_mask()  # get individual lesion mask

            mri_vol = vols["mri_vol"]
            prostate_vol = vols["prostate_mask"]
            prostate_centroid = np.mean(np.where(prostate_vol), axis=1)
            # print(f"Game rostate centroid : {prostate_centroid}")
            SLICE_NUM = int(prostate_centroid[-1])

            # Define grid coords
            grid, grid_coords = generate_grid(prostate_centroid)

            # Obtain agents predicted actions
            actions, _ = agent.predict(obs)
            # TODO : convert this to action / grid pos for agents!!!

            # adding an additional subplot
            fig, axs = plt.subplots(1)
            mask_l = np.ma.array(
                obs[0, :, :, :].numpy(), mask=(obs[0, :, :, :].numpy() == 0.0)
            )
            mask_p = np.ma.array(
                obs[1, :, :, :].numpy(), mask=(obs[1, :, :, :].numpy() == 0.0)
            )
            mask_n = np.ma.array(
                obs[-1, :, :, :].numpy(), mask=(obs[-1, :, :, :].numpy() == 0.0)
            )
            mask_n_1 = np.ma.array(
                obs[-2, :, :, :].numpy(), mask=(obs[-2, :, :, :].numpy() == 0.0)
            )
            mask_n_2 = np.ma.array(
                obs[-3, :, :, :].numpy(), mask=(obs[-3, :, :, :].numpy() == 0.0)
            )
            mri_ds = mri_vol[::2, ::2, ::4]
            needle = np.ma.array(grid, mask=(grid == 0.0))
            # needle_ds = needle[::2,::2]
            x_cent = int(prostate_centroid[1] / 2)
            y_cent = int(prostate_centroid[0] / 2)

            # plot for the axial view
            # crop between y_cent-35:y_cent+30, x_cent-30:x_cent+40; but user input neext to select grid positions within [100,100]
            plt.imshow(mri_ds[:, :, int(SLICE_NUM / 4)], cmap="gray")
            plt.imshow(50 * needle[:, :], cmap="jet", alpha=0.5)
            plt.imshow(np.max(mask_p[:, :, :], axis=2), cmap="coolwarm_r", alpha=0.5)
            plt.imshow(np.max(mask_n_1[:, :, :], axis=2), cmap="Wistia", alpha=0.4)
            plt.imshow(np.max(mask_n_2[:, :, :], axis=2), cmap="Wistia", alpha=0.4)
            plt.imshow(50 * needle[:, :], cmap="jet", alpha=0.3)
            plt.imshow(np.max(mask_l[:, :, :], axis=2), cmap="summer", alpha=0.6)
            plt.imshow(np.max(mask_n[:, :, :], axis=2), cmap="Wistia", alpha=0.5)

            # ADDING labels to grid positions!!!
            first_x = np.min(np.where(grid == 1)[1])
            first_y = np.min(np.where(grid == 1)[0])
            last_x = np.max(np.where(grid == 1)[1])
            last_y = np.max(np.where(grid == 1)[0])
            s = "A  a  B  b  C  c  D  d  E  e  F  f  G"  # fontsize 10.5
            # s = '-30 -25 -20 -15 -10 -5 0 5 10 15 20 25 30' #font size 8
            plt.text(
                first_x,
                first_y - 5,
                s,
                fontsize=10.5,
                color="aqua",
                bbox=dict(fill=False, edgecolor="green", linewidth=1),
            )  # , transform= axs.transAxes)
            grid_labels = np.arange(7, 0.5, -0.5)
            # grid_labels = np.arange(-30, 35, 5)
            for idx, label in enumerate(grid_labels):
                plt.text(
                    first_x - 10,
                    first_y + (idx * 5.15),
                    label,
                    fontsize=10.5,
                    color="aqua",
                )
                plt.text(
                    last_x + 5,
                    first_y + (idx * 5.15),
                    label,
                    fontsize=10.5,
                    color="aqua",
                )

            plt.axis("off")

            # Convert agent actions ->
            # Take input action (ie clicked position - original position), then scale
            if num_steps == 0:
                current_pos = np.array([0, 0])
            else:
                current_pos = biopsy_env.get_current_pos()
            # positions to choose
            suggested_pos = round_to_05((actions[0:-1] * 10) + current_pos)
            # my_pos = round_to_05((taken_actions[0:-1]*10) + current_pos)

            # Convert predicted actions to grid pos (A, E)
            x_dict = ["A", "a", "B", "b", "C", "c", "D", "d", "E", "e", "F", "f", "G"]
            grid_vals = np.arange(-30, 35, 5)
            x_idx = x_dict[(np.where(grid_vals == suggested_pos[0]))[0][0]]

            y_dict = [str(num) for num in np.arange(7, 0.5, -0.5)]
            y_idx = y_dict[(np.where(grid_vals == suggested_pos[1]))[0][0]]

            suggested_str = "Suggested GRID POSITION: [" + x_idx + "," + y_idx + "]"
            plt.text(
                first_x - 10, last_y + 10, suggested_str, fontsize=12, color="magenta"
            )

            ### 4. Take in user actions to implement strategy ###
            grid_pos = plt.ginput(1, 0)  # 0,0)
            # grid_pos = round_to_05(np.array(grid_pos[0]))
            prostate_cent_xy = (
                np.array([prostate_centroid[1], prostate_centroid[0]]) / 2
            )
            grid_pos -= prostate_cent_xy

            # Swap x and y
            # grid_pos[0], grid_pos[1] = grid_pos[1], grid_pos[0]

            # grid_pos = np.swapaxes(grid_pos, 1, 0)

            raw_actions = np.round(grid_pos - current_pos)
            taken_actions = round_to_05(np.round(grid_pos - current_pos))

            taken_actions = taken_actions / 10  # normalise between (-1,1)
            # taken_actions[0], taken_actions[1] = taken_actions[1], taken_actions[0]
            taken_actions = np.append(taken_actions, ([1]))

            # Take step in environment
            obs, reward, done, info = biopsy_env.step(taken_actions)

            # print(f"Current pos : {current_pos}Agent suggested actions : {actions} our actions : {taken_actions} \n Done :{done} num_steps {num_steps}")
            num_steps += 1

            # BUG: observaiton of needle not alligned with actions!!!
    print("chicken")
