from colorama import Fore, Back, Style
import csv
import numpy as np
import datetime
import SimpleITK as sitk
from matplotlib import pyplot as plt
from utils.data_utils import *
from utils.environment_utils import *
from Envs.biopsy_env import *
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import CnnPolicy


def view_test(mri_ds, mri_vol_shape, dimensions):
    # setting the spatial coordinates of the voxels
    x_axis = np.arange(mri_vol_shape[0]) * dimensions[0]
    y_axis = np.arange(mri_vol_shape[1]) * dimensions[1]
    z_axis = np.arange(mri_vol_shape[2]) * dimensions[2]
    aspect_ratio = 1
    print(f"the shape of mri_vol is {np.shape(mri_ds)}")

    # showing multiple plots
    plt.figure(1)
    plt.title("Slice on 1st column")
    plt.imshow(mri_ds[15, :, :], cmap="gray")
    plt.figure(2)
    plt.title("Slice on 2nd column")
    plt.imshow(mri_ds[:, 15, :], cmap="gray")
    plt.figure(3)
    plt.title("Slice on 3rd column")
    plt.imshow(mri_ds[:, :, 15], cmap="gray")


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


def multiple_display(mri_data):
    # CODE TO SHOW MULTIPLE SLICES AT ONCE
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
        index = mri_data[:, slice_index, :]
        flipped_index = np.fliplr(index)
        ax.imshow(flipped_index, cmap="gray", aspect=0.5)
        ax.set_title(f"Slice {slice_index + 1}")
        ax.axis("off")

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()
    plt.text(
        20,
        20,
        "matplotlib EXAMPLE",
        color="green",
        horizontalalignment="center",
        verticalalignment="center",
    )
    plt.show()


def select_depth_action_interactive(ax):
    depth_options = ["0: Apex", "1: Centroid", "2: Base"]
    x_positions = [0, 1, 2]  # Preset x positions for the text

    for option, x in zip(depth_options, x_positions):
        ax.text(
            x,
            0.5,
            option,
            ha="center",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )

    ax.set_xlim(-1, 3)
    ax.set_ylim(0, 1)
    ax.axis("off")
    plt.draw()

    print("Please click on the plot to select the depth.")
    click_depth = plt.ginput(1)  # Wait for one click
    x_click = click_depth[0][0]

    # Determine the depth based on the x-coordinate of the click
    if x_click < 1:
        action = 0  # Apex
    elif x_click < 2:
        action = 1  # Centroid
    else:
        action = 2  # Base

    return action


def select_depth_action(prostate_mask, prostate_centroid):
    """
    Prompts the user to select a depth action and returns the z-coordinate for the selected action.

    Arguments:
    :prostate_centroid: The centroid of the prostate volume
    :prostate_mask: The prostate mask volume

    Returns:
    :z_coordinate: The z-coordinate for the apex, centroid, or base of the prostate based on the user's choice.
    """
    # Prompt the user for a depth action
    while True:
        action = int(input("Select a depth action (0: Apex, 1: Centroid, 2: Base): "))
        if action in [0, 1, 2]:
            break
        else:
            print("Invalid action selected. Please enter 1, 2, or 3.")

    # Return the appropriate z-coordinate
    if action == 0:
        depth = np.min(np.where(prostate_mask == 1)[-1])
    elif action == 1:
        depth = prostate_centroid[2]
        # return coordinates['centroid']
    elif action == 2:
        depth = np.max(np.where(prostate_mask == 1)[-1])
        # return coordinates['base']
    else:
        raise ValueError("Invalid action selected.")
    print(Fore.MAGENTA + f"DEPTH SELECTED {depth}" + Fore.RESET)
    return round(depth)


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
    # Converts range from (-30,30) to image grid array
    print(Fore.RED + "WITHIN THE FUNCTION" + Fore.RESET)
    print(coordinates)
    x_idx = coordinates[0]
    y_idx = coordinates[1]

    # Converts range from (-30,30) to image grid array

    x_idx = (x_idx) * 2 + round(prostate_centroid[0])
    y_idx = (y_idx) * 2 + round(prostate_centroid[1])

    x_grid_pos = round(x_idx)
    y_grid_pos = round(y_idx)

    # print(Fore.BLUE + f"X GRID POS {x_grid_pos} Y GRID POS {y_grid_pos}" + Fore.RESET)

    return x_grid_pos, y_grid_pos


def coord_converter_alt(coordinates, prostate_centroid, mri_shape):
    # Initialise coordinates to be 0,0,0 at top left corner of img volume
    y_vals = np.asarray(range(mri_shape[0])).astype(float)
    x_vals = np.asarray(range(mri_shape[1])).astype(float)
    z_vals = np.asarray(range(mri_shape[2])).astype(float)

    x, y, z = np.meshgrid(x_vals, y_vals, z_vals)

    # Centre coordinates at rectum position
    x -= prostate_centroid[0]
    y -= prostate_centroid[1]
    z -= prostate_centroid[2]

    # Convert to 0.5 x 0.5 x 1mm dimensions
    img_coords = [x * 0.5, y * 0.5, z]

    return img_coords


def depth_check(mri_ds, mask_p, mask_l, prostate_centroid, prostate_mask):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    dep1 = np.min(np.where(prostate_mask == 1)[-1])
    dep2 = prostate_centroid[2]
    dep3 = np.max(np.where(prostate_mask == 1)[-1])
    # apex
    mask_p_1 = mask_p[:, :, int(dep1 / 4)]
    mask_l_1 = mask_l[:, :, int(dep1 / 4)]
    axs[0].imshow(mri_ds[:, :, int(dep1 / 4)], cmap="gray")
    axs[0].imshow(mask_p_1, cmap="coolwarm_r", alpha=0.5)
    axs[0].imshow(mask_l_1, cmap="summer", alpha=0.6)
    axs[0].axis("off")
    axs[0].set_title("Apex")
    # midgland
    mask_p_2 = mask_p[:, :, int(dep2 / 4)]
    mask_l_2 = mask_l[:, :, int(dep2 / 4)]
    axs[1].imshow(mri_ds[:, :, int(dep2 / 4)], cmap="gray")
    axs[1].imshow(mask_p_2, cmap="coolwarm_r", alpha=0.5)
    axs[1].imshow(mask_l_2, cmap="summer", alpha=0.6)
    axs[1].axis("off")
    axs[1].set_title("Midgland")
    # base
    mask_p_3 = mask_p[:, :, int(dep3 / 4)]
    mask_l_3 = mask_l[:, :, int(dep3 / 4)]
    axs[2].imshow(mri_ds[:, :, int(dep3 / 4)], cmap="gray")
    axs[2].imshow(mask_p_3, cmap="coolwarm_r", alpha=0.5)
    axs[2].imshow(mask_l_3, cmap="summer", alpha=0.6)
    axs[2].axis("off")
    axs[2].set_title("Base")


# Global variable to keep track of the last logged patient ID
last_patient_id = None
last_lesion_idx = None


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

        # Obtain agents predicted actions
        actions, _ = agent.predict(obs)
        # TODO : convert this to action / grid pos for agents!!!

        fig, axs = plt.subplots(2, 3, figsize=(15, 5))
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

        # MASK TESTING
        # depth_check(mri_ds, mask_p, mask_l, prostate_centroid, prostate_vol)
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
        index_ds = mri_ds[:, int(sag_index_plt / 4), :]
        flipped_index = np.fliplr(index)
        # flipped_index_ds = np.fliplr(index_ds)
        axs[0, 0].imshow(flipped_index, cmap="grey", aspect=0.5)
        # axs[0].imshow(prostate_vol[:,50,:], cmap = 'coolwarm_r', alpha = 0.5)
        # axs[0].imshow(np.max(mask_p[:,:,:], axis =2),cmap='coolwarm_r', alpha=0.5)
        # axs[0].imshow(np.max(mask_l[:,:,:], axis =2),cmap='summer', alpha=0.6)
        # plt.show(multiple_display(mri_vol))
        axs[0, 0].set_title(f"sagittal view for {sag_index}")
        axs[0, 0].axis("off")

        # Plotting for the axial view
        # crop between y_cent-35:y_cent+30, x_cent-30:x_cent+40; but user input neext to select grid positions within [100,100]
        axs[0, 1].imshow(mri_ds[:, :, int(SLICE_NUM / 4)], cmap="gray")
        axs[0, 1].imshow(50 * needle[:, :], cmap="jet", alpha=0.5)
        axs[0, 1].imshow(np.max(mask_p[:, :, :], axis=2), cmap="coolwarm_r", alpha=0.5)
        axs[0, 1].imshow(np.max(mask_n_1[:, :, :], axis=2), cmap="Wistia", alpha=0.4)
        axs[0, 1].imshow(np.max(mask_n_2[:, :, :], axis=2), cmap="Wistia", alpha=0.4)
        axs[0, 1].imshow(50 * needle[:, :], cmap="jet", alpha=0.3)
        axs[0, 1].imshow(np.max(mask_l[:, :, :], axis=2), cmap="summer", alpha=0.6)
        axs[0, 1].imshow(np.max(mask_n[:, :, :], axis=2), cmap="Wistia", alpha=0.5)
        axs[0, 1].axis("off")

        # ADDING labels to grid positions!!!
        first_x = np.min(np.where(grid == 1)[1])
        first_y = np.min(np.where(grid == 1)[0])
        last_x = np.max(np.where(grid == 1)[1])
        last_y = np.max(np.where(grid == 1)[0])
        s = "A  a  B  b  C  c  D  d  E  e  F  f  G"  # fontsize 10.5
        # s = '-30 -25 -20 -15 -10 -5 0 5 10 15 20 25 30' #font size 8
        axs[0, 1].text(
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
            axs[0, 1].text(
                first_x - 10, first_y + (idx * 5.15), label, fontsize=10.5, color="aqua"
            )
            axs[1, 1].text(
                last_x + 5, first_y + (idx * 5.15), label, fontsize=10.5, color="aqua"
            )

        # Displays the rewards metrics within the game
        # The data comes from the library from the info dictionary within game dev
        # checking for needle hit
        if data["needle_hit"] == True:
            hit = "HIT"
        else:
            hit = "MISS"
        axs[0, 1].text(
            first_x - 15,
            first_y - 14.5,
            f"Total Result: {totalreward} ",
            fontsize=12.5,
            color="yellow",
        )
        axs[0, 1].text(
            (last_x * 0.48),
            first_y - 14.5,
            f"Previous Result: {reward} ({hit})",
            fontsize=12.5,
            color="greenyellow",
        )
        axs[0, 1].text(
            first_x - 15,
            last_y + 16,
            f"CCL:{data['norm_ccl']} ",
            fontsize=12.5,
            color="cyan",
        )

        # plotting for the axial view
        # Allow user to select the depth of the prostate
        # depth = select_depth_action(prostate_vol, prostate_centroid)
        # seperate the selection and the conversion of the depth
        # convert depth first
        # plot then select for the next step
        depth = convert_depth(depth_action, prostate_vol, prostate_centroid)
        axs[1, 0].remove()
        axs[1, 2].remove()
        depth = depth / 4

        axs[0, 2].imshow(mri_ds[:, :, int(depth)], cmap="gray")
        # Adding the additional mask layers
        mask_p_ax = mask_p[:, :, int(depth)]
        mask_l_ax = mask_l[:, :, int(depth)]
        axs[0, 2].imshow(mask_p_ax, cmap="coolwarm_r", alpha=0.5)
        axs[0, 2].imshow(mask_l_ax, cmap="summer", alpha=0.6)
        axs[0, 2].set_title(f" Depth showing axial view ")
        axs[0, 2].axis("off")

        depth_action = select_depth_interactive(
            axs[1, 1], prostate_vol, prostate_centroid
        )

        # Take input action (ie clicked position - original position), then scale
        if num_steps == 0:
            current_pos = np.array([0, 0])
        else:
            current_pos = biopsy_env.get_current_pos()

        # Convert agent actions -> positions to choose
        suggested_pos = round_to_05((actions[0:-1] * 10) + current_pos)
        # my_pos = round_to_05((taken_actions[0:-1]*10) + current_pos)

        # Convert predicted actions to grid pos (A, E)
        x_dict = ["A", "a", "B", "b", "C", "c", "D", "d", "E", "e", "F", "f", "G"]
        grid_vals = np.arange(-30, 35, 5)
        x_idx = x_dict[(np.where(grid_vals == suggested_pos[0]))[0][0]]

        y_dict = [str(num) for num in np.arange(7, 0.5, -0.5)]
        y_idx = y_dict[(np.where(grid_vals == suggested_pos[1]))[0][0]]

        suggested_str = "Suggested GRID POSITION: [" + x_idx + "," + y_idx + "]"
        axs[0, 1].text(
            first_x - 10, last_y + 10, suggested_str, fontsize=12, color="magenta"
        )

        ### 4. Take in user actions to implement strategy ###
        grid_pos = plt.ginput(1, 0)  # 0,0)
        # grid_pos = round_to_05(np.array(grid_pos[0]))
        prostate_cent_xy = np.array([prostate_centroid[1], prostate_centroid[0]]) / 2
        grid_pos -= prostate_cent_xy

        raw_actions = np.round(grid_pos - current_pos)
        taken_actions = round_to_05(np.round(grid_pos - current_pos))

        taken_actions = taken_actions / 10  # normalise between (-1,1)
        # taken_actions[0], taken_actions[1] = taken_actions[1], taken_actions[0]
        taken_actions = np.append(taken_actions, ([1]))

        # Take step in environment
        obs, reward, done, data = biopsy_env.step(taken_actions)

        # print(f"Current pos : {current_pos}Agent suggested actions : {actions} our actions : {taken_actions} \n Done :{done} num_steps {num_steps}")
        num_steps += 1

        plt.close()

        # Inputting all of the data in into the csv file
        # checking for all of the data being written into the csv file
        # CURRENT STEP WONT WORK STUCK AT 0
        print(Fore.LIGHTYELLOW_EX + f"num step is {num_steps}" + Fore.RESET)
        current_patient = data["patient_name"]
        patient_id = current_patient.split("\\")[-1].split("_")[0]
        print(data["lesion_idx"])
        print(Fore.LIGHTBLUE_EX + f" the current patient is {patient_id} " + Fore.RESET)
        # log_user_input(
        #     file_path,
        #     patient_id,
        #     data["lesion_idx"],
        #     num_steps,
        #     sag_index[0],
        #     sag_index[1],
        #     depth,
        #     grid_index[0],
        #     grid_index[1],
        #     depth,
        # )

    return obs, reward, data, totalreward, sag_index, depth


def intro():
    fig, ax = plt.subplots(figsize=(12, 6))

    # Set the title of the plot
    ax.set_title("Welcome to the Prostate Biopsy Game!", fontsize=16, fontweight="bold")

    # Add text with instructions on how to play the game
    blurb = """
    This game simulates the process of a template guided prostate biopsy 
    The game has two views: sagittal and axial. The sagittal view shows a slice of the prostate from the side,
    while the axial view shows a slice from the top. 
    The game progresses through multiple rounds, and your goal is to
    accurately target lesions within the prostate using a biopsy needle.
    """
    instructions = """
    How to Play:
    1. Select the depth of the biopsy needle using your mouse. (type it out on the terminal for now)
    The options for depth are (from sagittal/side view)
        0 - Apex (front of prostate)
        1 - Centroid (middle of prostate)
        2 - Base (back of prostate)
    2. Click on the screen to choose the target location for the biopsy.
    3. Your goal is to accurately target lesions within the prostate.
    4. The game progresses through multiple rounds - aim for the highest score!

    Good luck and have fun!

    CLOSE THIS WINDOW (CLICK X ON THE WINDOW) TO START PLAYING
    """

    # Display the instructions
    ax.text(0.01, 0.8, blurb, color="red", ha="left", va="center", fontsize=10)
    ax.text(0.01, 0.4, instructions, ha="left", va="center", fontsize=10)

    # Hide the axes
    ax.axis("off")

    # Show the plot
    plt.show()


def run_game(NUM_EPISODES=5, log_dir="game"):

    data_path = ".\Data\ProstateDataset"
    csv_path = ".\Data\ProstateDataset\patient_data_multiple_lesions.csv"
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_path = f"user_input_data_{datetime_str}.csv"
    log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)

    # Load biopsy envs and datasets
    PS_dataset = Image_dataloader(data_path, csv_path, use_all=True, mode="test")
    Data_sampler = DataSampler(PS_dataset)
    biopsy_env = TemplateGuidedBiopsy(
        Data_sampler,
        results_dir=log_dir,
        reward_fn="reward",
        max_num_steps=20,
        deform=True,
        start_centre=True,
    )
    data = biopsy_env.get_info()
    reward = 0
    totalreward = 0
    # initialising variable
    sag_index = 0
    depth = 1
    depth_action = 0
    image_data = biopsy_env.get_img_data()
    step = biopsy_env.get_step_count()

    # Load agent1
    policy_kwargs = dict(
        features_extractor_class=NewFeatureExtractor,
        features_extractor_kwargs=dict(multiple_frames=True, num_channels=5),
    )
    agent = PPO(CnnPolicy, env=biopsy_env, policy_kwargs=policy_kwargs)

    # Run game for num episodes
    # princh t(f"Loading game script. Running for {NUM_EPISODES} episodes")
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
        )


if __name__ == "__main__":

    ps_path = "/Users/ianijirahmae/Documents/DATASETS/Data_by_modality"
    csv_path = "/Users/ianijirahmae/Documents/PhD_project/MRes_project/Reinforcement Learning/patient_data_multiple_lesions.csv"
    log_dir = "game"
    os.makedirs(log_dir, exist_ok=True)

    # DATASETS
    PS_dataset = Image_dataloader(ps_path, csv_path, use_all=True, mode="test")
    Data_sampler = DataSampler(PS_dataset)

    # HYPERPARAMETERS
    RATE = 0.1
    SCALE = 0.25
    NUM_EPISODES = 5

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
