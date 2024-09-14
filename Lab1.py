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

def fire_spread(nx=3, ny=3, maxiter=4, pspread=1.0, pbare=0.0, pstart=0.1, graph=True, bare_seed=False, fire_seed=True):
    '''
    This function performs a fire/disease simulation and optionally plots
    the 2D grid after every step. Returns number of steps for simulation to
    end, as well as percentage of forest left.

    Parameters
    ==========
    nx, ny : integer, defaults to 3
        Set the x (i, top to botton) and y (j, left to right) size of grid.
        Default is 3 squares in each direction.
    maxiter : int, defaults to 4
        Set the maximum number of iterations including initial condition
    pspread : float, defaults to 1
        Chance fire spreads from 0 to 1 (0 to 100%).
    pbare : float, defaults to 0
        Chance of area starting as bare ground instead of forest from 0 to 1
    pstart : float, defaults to 0.1
         Chance of cell catching fire at start of simulation
    graph : bool, defaults to True
        Determines whether to graph 2D forest grid after every step
    bare_seed: bool, defaults to False
        Determines whether to use a random seed for reproducible initial bare conditions
    fire_seed: bool, defaults to True
        Determines whether to use a random seed for reproducible initial burning conditions
    '''

    # Create forest and set initial conditions
    forest = np.zeros([maxiter, nx, ny], dtype=int) + 2

    # Set bare spots using pbare
    if bare_seed == True:
        np.random.seed(13) #Set seed for reproducible bare conditions
    bare = np.random.rand(nx,ny) <= pbare
    forest[0,:,:][bare] = 1

    # Set fire using pstart
    if fire_seed == True:
        np.random.seed(42) #Set seed for reproducible starting fire
    start_fire = (np.random.rand(nx,ny) <= pstart) & (forest[0,:,:] != 1) #bare ground cannot catch fire
    forest[0,:,:][start_fire] = 3

    #Plot initial conditions
    if graph == True: 
        # Set colormap:
        forest_cmap = ListedColormap(['tan', 'darkgreen', 'crimson'])
        # Plot initial condition:
        fig, ax = plt.subplots(1,1)
        contour = ax.matshow(forest[0, :, :], vmin=1, vmax=3, cmap=forest_cmap)
        ax.set_title(f'Forest Status (Step = {0:03d})')
        # Add colorbar and set its ticks
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_ticks([1, 2, 3])
        cbar.set_ticklabels(['1 - Bare', '2 - Forest', '3 - Fire'])
        # Add x and y axis labels
        ax.set_xlabel('Y-axis [m]')
        ax.set_ylabel('X-axis [m]')
        #fig.savefig(f'fig0000.png')
        plt.pause(0.5)  # Show the first plot briefly without blocking

    # Propagate solution
    for k in range(maxiter-1): # Time loop
        # Set chance to burn
        ignite = np.random.rand(nx,ny)

        # Use current step to set up next step:
        forest[k+1,:,:] = forest[k,:,:]

        # Burn in each cardinal direction
        # Burn north to south
        doburn_n2s = (forest[k, :-1, :] == 3) & (forest[k,1:,:] == 2)  & \
            (ignite[1:,:] <= pspread)
        forest[k+1, 1:,:][doburn_n2s] = 3

        # Burn south to north
        doburn_s2n = (forest[k, 1:, :] == 3) & (forest[k,:-1,:] == 2)  & \
            (ignite[:-1,:] <= pspread)
        forest[k+1, :-1,:][doburn_s2n] = 3

        # Burn west to east
        doburn_w2e = (forest[k, :, :-1] == 3) & (forest[k,:,1:] == 2)  & \
            (ignite[:,1:] <= pspread)
        forest[k+1, :,1:][doburn_w2e] = 3

        # Burn east to west
        doburn_e2w = (forest[k, :, 1:] == 3) & (forest[k,:,:-1] == 2)  & \
            (ignite[:,:-1] <= pspread)
        forest[k+1, :,:-1][doburn_e2w] = 3

        # Set currently burning to bare:
        wasburn = forest[k,:,:] == 3 # Find cells that were burning
        forest[k+1, wasburn] = 1       # ... they are now bare

        # Plot next steps:
        if graph == True:
            # Plot
            fig, ax = plt.subplots(1,1)
            contour = ax.matshow(forest[k+1, :, :], vmin=1, vmax=3, cmap=forest_cmap)
            ax.set_title(f'Forest Status (Step = {k+1:03d})')
            # Add colorbar and set its ticks
            cbar = plt.colorbar(contour, ax=ax)
            cbar.set_ticks([1, 2, 3])
            cbar.set_ticklabels(['1 - Bare', '2 - Forest', '3 - Fire'])
            # Add x and y axis labels
            ax.set_xlabel('Y-axis [m]')
            ax.set_ylabel('X-axis [m]')

            #fig.savefig(f'fig{k+1:04d}.png')
            plt.pause(0.5)  # Show the first plot briefly without blocking

        # Quit if no spots are on fire.
        nBurn = (forest[k+1, :, :] == 3).sum()
        if nBurn == 0:
            nForest = (forest[k+1, :, :] == 2).sum()
            print(f"Burn completed in {k+1} steps")
            forest_left = (nForest / (nx * ny)) * 100 # Percentage of forest left
            print(f'With {forest_left}% forest left')
            break
    if graph == True:
        plt.ioff()  # Turn off interactive mode
        plt.show()  # Keep the last plot open
    return k+1, forest_left


#fire_spread(ny=10, nx=10, pstart=0.1, pspread=0.5, pbare=0.1, maxiter=20, graph=True)


def explore_burnrate():
    '''
    Varies the probability of fire spreading between adjacent cells.
    Keeps track of number of steps required for every simulation to end.
    Returns plot of steps/forest percentage vs probability of fire
    propagating
    '''

    prob = np.arange(0, 1, .1) # array of probabilities to test and plot
    nsteps = np.zeros(prob.size) # empty array, will be populated with steps in each simulation
    forest_left = np.zeros(prob.size) # empty array, will be populated with forest percentage

    # Loop through ith step with corresponding probability p:
    for i, p in enumerate(prob):
        print(f"Burning for pspread = {p}") # Print probability to burn
        # Since we are exploring the burn rate, we will use a seed for reproducible bare ground:
        nsteps[i], forest_left[i] = fire_spread(ny=5, pspread=p, maxiter=100, 
                                                graph=False, bare_seed=True)
            
    # Plot Steps/Forest Percentage vs Probability of Fire Propagating
    fig, ax1 = plt.subplots()
    # Plot Steps vs Prob of fire propagating
    color = 'tab:blue'
    ax1.set_xlabel('Probability of fire propagating')
    ax1.set_ylabel('Number of steps until fire stops', color=color)
    ax1.plot(prob, nsteps, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Plot Forest Percentage left after burning vs Prob. of propagation
    # Second plot shares X-axis with first plot
    ax2 = ax1.twinx()

    color = 'tab:green'
    ax2.set_ylabel('Percentage of Forest left after burning', color=color)
    ax2.plot(prob, forest_left, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

explore_burnrate()

def explore_barerate():
    '''
    Varies the probability of a cell starting as bare ground, ie. varies
    percentage of initial forest in grid.
    Keeps track of number of steps required for every simulation to end, as
    well as percentage of forest left.
    Returns plot of steps/forest percentage vs probability.
    '''
    prob = np.arange(0, 1, 0.1) # array of probabilities to test and plot
    nsteps = np.zeros(prob.size) # empty array, will be populated with steps in each simulation
    forest_left = np.zeros(prob.size) # empty array, will be populated with forest percentage

    # Loop through ith step with corresponding probability p:
    for i, p in enumerate(prob):
        print(f"Burning for pbare = {p}") # Print probability to burn
        nsteps[i], forest_left[i] = fire_spread(ny=5, pbare=p, maxiter=100, graph=False)
            
    # Plot Steps/Forest Percentage vs Probability of Starting Bare
    fig, ax1 = plt.subplots()
    # Plot Steps vs Prob of starting bare
    color = 'tab:blue'
    ax1.set_xlabel('Probability of cell starting bare')
    ax1.set_ylabel('Number of steps until fire stops', color=color)
    ax1.plot(prob, nsteps, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Plot Forest Percentage left after burning vs Prob. of starting bare
    # Second plot shares X-axis with first plot
    ax2 = ax1.twinx()

    color = 'tab:green'
    ax2.set_ylabel('Percentage of Forest left after burning', color=color)
    ax2.plot(prob, forest_left, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

#explore_barerate()