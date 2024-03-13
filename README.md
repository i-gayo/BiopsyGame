# BiopsyGame
An interactive user game that simulates a prostate biopsy procedure 

## Game screen 
<div style="display: flex;">
  <img src="Figures/new_intro_screen.png" alt="Image 1" style="width: 40%;">
  <img src="Figures/new_3.png" alt="Image 2" style="width: 40%;">
</div>

Examples of the game screen can be visualised here. Grid points are shown in blue, whilst lesions and prostate gland masks are displayed in green and red respectively. Previously visited grid positions can also be seen, for up to two timesteps behind, and displayed as yellow box points. 

## Game instructions
Users are shown images of the MR volume, prostate and lesion masks, along with a brachytherapy grid. 

To select a grid position, simply click on a desired grid point, then close the image. A new image will pop up with an updated screen : a yellow dot signifies the previously visited grid positions. 
The user repeats the same process until all 5 biopsy needles have been fired.

## Comparison with RL predictions 

The user can also choose to fire grid positions suggested by trained RL agents.
TODO: Display RL suggestions in "info bar" on game screen

# Download instructions  

# Setting up 

## Clone repo to workspace 
Clone this repository 

## Setup conda environment 

Download Anaconda or miniconda using the following following links 

Anaconda: https://docs.anaconda.com/free/anaconda/install/#installation

Miniconda :https://docs.anaconda.com/free/miniconda/

Once anaconda/miniconda is installed, create a conda environment with the necessary packages in requirements.txt

replacing <env> with a suitable name for the environment
```bash
conda create --name <env> --file requirements.txt
```
## Activating environment 
To activate the environment enter the following line into terminal 

```bash
activate <env>
```
# Running application 

To run the game enter the following line into terminal 

```bash
python play_game.py
```
# Once you are finished playing the game 

After you are done playing the game please answer the following quesionnaire 

https://forms.gle/djSBxRjfsRDb2oi88