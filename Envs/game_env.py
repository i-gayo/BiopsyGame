import numpy as np 
import torch 
from matplotlib import pyplot as plt 

class BiopsyGame():
    """
    A game environment that samples a patient with lesion, and allows users 
    to chosoe grid positions to sample lesion    
    """
    def __init__(self):
        pass
    
    def sample_new_data(self):
        """
        Sample new data 
        """
        pass 
    
    def step(self, actions):
        """
        Updates environment based on current actions 
        """
    
    def reset(self):
        """
        Resets environment, sample new patient 
        """
        
    def add_reg_noise(self):
        """
        Adds reg noise 
        """
        raise NotImplementedError
    
    def deform(self, scale = 0.5, rate = 0.25):
        """
        Deforms current volume 
        """
        raise NotImplementedError
        
        
    def play_game(self):
        """
        Game interface
        """
        
        # Load data volume (2D projection)
        
        # Load grid positions on top (centred on prostate gland)
        
        # Allow user to interact through clicks / input 
        
        # Take action -> update state 
        
        
    def update_score(self):
        """
        Updates score based on current hit rate
        """ 
    