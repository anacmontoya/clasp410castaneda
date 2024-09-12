#!/usr/bin/env python3

'''
This file contains tools and scripts for completing Lab 1 for CLaSP 410.
To reproduce the plots shown in the lab report, do this...
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.style.use('fivethirtyeight')
#plt.ion()

nx, ny = 3,3 # Number of cells in X (from top to bottom) and Y (left to right) direction
prob_spread = 1.0 # Chance to spread to adjacent cells.
max_iter = 4 # Maximum number of iterations

def fire_spread(nx=3, ny=3, maxiter=4, pspread=1.0):
    '''
    This function performs a fire/disease simulation
    '''

    # Create forest and set initial conditions
    forest = np.zeros([max_iter, nx, ny], dtype=int) + 2

    # Set bare spots

    # Set Fire
    istart, jstart = nx//2, ny//2
    forest[0,istart,jstart] = 3

    # Set colormap:
    forest_cmap = ListedColormap(['tan', 'darkgreen', 'crimson'])
    # Plot initial condition:
    fig, ax = plt.subplots(1,1)
    contour = ax.matshow(forest[0, :, :], vmin=1, vmax=3, cmap=forest_cmap)
    ax.set_title(f'Iteration = {0:03d}')
    plt.colorbar(contour, ax=ax)
    plt.pause(0.5)  # Show the first plot briefly without blocking

    # Propagate solution
    for k in range(max_iter-1): # Time loop
        # Set chance to burn
        ignite = np.random.rand(nx,ny)

        # Use current step to set up next step:
        forest[k+1,:,:] = forest[k,:,:]

        #burn north to south
        doburn = (forest[k, :-1, :] == 3) & (forest[k,1:,:] == 2)  & \
            (ignite[1:,:] <= pspread)
        forest[k+1, 1:,:][doburn] = 3

        # Burn in each cardinal direction
        # # Burn north to south
        # for i in range(nx-1):
        #     for j in range(ny):
        #         # Is current patch burning and adjacent forested?
        #         if (forest[k,i,j] == 3) & (forest[k,i+1,j] == 2): 
        #             # Spread fire to new cell:
        #             forest[k+1,i+1,j] = 3
        # Burn south to north
        for i in range(1,nx):
            for j in range(ny):
                # Is current patch burning and adjacent forested?
                if (forest[k,i,j] == 3) & (forest[k,i-1,j] == 2): 
                    # Spread fire to new cell:
                    forest[k+1,i-1,j] = 3
        # Burn west to east
        for i in range(nx):
            for j in range(ny-1):
                # Is current patch burning and adjacent forested?
                if (forest[k,i,j] == 3) & (forest[k,i,j+1] == 2): 
                    # Spread fire to new cell:
                    forest[k+1,i,j+1] = 3
        # Burn east to west
        for i in range(nx):
            for j in range(1,ny):
                # Is current patch burning and adjacent forested?
                if (forest[k,i,j] == 3) & (forest[k,i,j-1] == 2): 
                    # Spread fire to new cell:
                    forest[k+1,i,j-1] = 3

        # Set currently burning to bare:
        wasburn = forest[k,:,:] == 3 # Find cells that were burning
        forest[k+1, wasburn] = 1       # ... they are now bare

        fig, ax = plt.subplots(1,1)
        contour = ax.matshow(forest[k+1, :, :], vmin=1, vmax=3, cmap=forest_cmap)
        ax.set_title(f'Iteration = {k+1:03d}')
        plt.colorbar(contour, ax=ax)
        plt.pause(0.5)  # Show the first plot briefly without blocking

    plt.ioff()  # Turn off interactive mode
    plt.show()  # Keep the last plot open

fire_spread(ny=5, maxiter=7)