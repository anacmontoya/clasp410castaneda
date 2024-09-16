#!/usr/bin/env python3

'''
This file contains tools and scripts for completing Lab 1 for CLaSP 410.
To reproduce the plots shown in the lab report, do this...
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score

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

    # Independent Random State for spreading:
    # Ensures random spreding conditions even if bare/fire initial
    # Conditions are reproducible
    spread_rdm = np.random.RandomState() 
    # Propagate solution
    for k in range(maxiter-1): # Time loop
        # Set chance to burn with independent random state
        ignite = spread_rdm.rand(nx,ny)
        # print(f"Ignite array at step {k}:\n{ignite}")

        # Use current step to set up next step:
        forest[k+1,:,:] = forest[k,:,:]

        # Spread in each cardinal direction
        # Spread north to south
        doburn_n2s = (forest[k, :-1, :] == 3) & (forest[k,1:,:] == 2)  & \
            (ignite[1:,:] <= pspread)
        forest[k+1, 1:,:][doburn_n2s] = 3

        # Spread south to north
        getsick_s2n = (forest[k, 1:, :] == 3) & (forest[k,:-1,:] == 2)  & \
            (ignite[:-1,:] <= pspread)
        forest[k+1, :-1,:][getsick_s2n] = 3

        # Spread west to east
        doburn_w2e = (forest[k, :, :-1] == 3) & (forest[k,:,1:] == 2)  & \
            (ignite[:,1:] <= pspread)
        forest[k+1, :,1:][doburn_w2e] = 3

        # Spread east to west
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
            print(f"Spread completed in {k+1} steps")
            forest_left = (nForest / (nx * ny)) * 100 # Percentage of forest left
            print(f'With {forest_left}% forest left')
            break
    if graph == True:
        plt.ioff()  # Turn off interactive mode
        plt.show()  # Keep the last plot open
    return k+1, forest_left


#fire_spread(ny=5, nx=5, pspread=0.5, pbare=0.1, maxiter=20, graph=False)

''
def explore_burnrate(ntrials=5):
    '''
    Varies the probability of fire spreading between adjacent cells.
    Keeps track of number of steps required for every simulation to end.
    Returns plot of steps/forest percentage vs probability of fire
    propagating

    Parameters
    ==========
    ntrials : int, defaults to 5
        Number of trials to do. E.g. if set to 5, the simulations with probabilities
        between 0 and 1 will be run five times each. Values are then averaged for plotting
    '''

    prob = np.arange(0, 1, .05) # array of probabilities to test and plot
    nsteps = np.zeros((ntrials, prob.size)) # will be populated with steps in each simulation
    # every row is one silumation running from lowest to highest probability
    # every column is a different run with the same prob.
    forest_left = np.zeros((ntrials, prob.size)) # will be populated with forest percentage

    for j in range(ntrials): # loop through number of trials
        # Loop through ith step with corresponding probability p:
        for i, p in enumerate(prob):
            print(f"Burning for pspread = {p}") # Print probability to burn
            # Since we are exploring the burn rate, we will use a seed for reproducible bare ground:
            nsteps[j,i], forest_left[j,i] = fire_spread(ny=100,nx=100, pspread=p, maxiter=400, 
                                                    graph=False, bare_seed=True, pbare=0.1)
            
    # Average values
    nsteps_avg = np.zeros(prob.size)
    nsteps_avg = np.mean(nsteps, axis=0)
    forest_left_avg = np.zeros(prob.size)
    forest_left_avg =  np.mean(forest_left, axis=0)

    # print('nsteps:'+str(nsteps))
    # print('forest left:'+str(forest_left))  
    
    # Perform polynomial fits
    a, b, c = np.polyfit(prob, nsteps_avg, 2) #degree 2
    equation_text_1 = f'$y = {a:.2f}x^2 + {b:.2f}x + {c:.2f}$' #degree 2
    prob_fit = np.linspace(min(prob), max(prob),100)
    nsteps_avg_fit = a * prob_fit**2 + b * prob_fit + c #degree 2
    nsteps_avg_pred = a * prob**2 + b * prob + c #degree 2
    r2 = r2_score(nsteps_avg, nsteps_avg_pred)
    print('R2 value for steps:'+str(r2))

    d, e, f = np.polyfit(prob, forest_left_avg, 2) #degree 2
    equation_text_2 = f'$y = {d:.2f}x^2 + {e:.2f}x + {f:.2f}$' #degree 2
    forest_left_avg_fit = d * prob_fit**2 + e * prob_fit + f #degree 2
    forest_left_avg_pred = d * prob**2 + e * prob + f #degree 2
    r2 = r2_score(forest_left_avg, forest_left_avg_pred)
    print('R2 value for forest:'+str(r2))

    # Plot Steps/Forest Percentage vs Probability of Fire Propagating
    fig, ax1 = plt.subplots()
    # Plot Steps vs Prob of fire propagating
    color = 'tab:blue'
    ax1.set_xlabel('Probability of fire propagating')
    ax1.set_ylabel('Number of steps until fire stops', color=color)
    ax1.scatter(prob, nsteps_avg, color=color) # Collected data
    ax1.plot(prob_fit, nsteps_avg_fit, color=color, alpha=0.7,
             label=equation_text_1) # Best-fit line
    ax1.tick_params(axis='y', labelcolor=color)

    # Plot Forest Percentage left after burning vs Prob. of propagation
    # Second plot shares X-axis with first plot
    ax2 = ax1.twinx()

    color = 'tab:green'
    ax2.set_ylabel('Forest left after burning [%]', color=color)
    ax2.scatter(prob, forest_left_avg, color=color) # Collected data
    ax2.plot(prob_fit, forest_left_avg_fit, color=color, alpha=0.7,
             label=equation_text_2) # Best-fit line
    ax2.tick_params(axis='y', labelcolor=color)

    fig.legend(bbox_to_anchor=(0.65,0.3),fontsize='x-small')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

#explore_burnrate(ntrials=50) #run as is for:
# graph where steps increases until 0.5 prob then sudden decrease. while forest left decreases

def explore_barerate(ntrials=2):
    '''
    Varies the probability of a cell starting as bare ground, ie. varies
    percentage of initial forest in grid.
    Keeps track of number of steps required for every simulation to end, as
    well as percentage of forest left.
    Returns plot of steps/forest percentage vs probability.
    '''

    prob = np.arange(0, 1, .05) # array of probabilities to test and plot
    nsteps = np.zeros((ntrials, prob.size)) # empty array, will be populated with
    # steps in each simulation every row is one silumation running from lowest to 
    # highest probability every column is a different run with the same prob.
    forest_left = np.zeros((ntrials, prob.size)) # will be populated with forest percentage

    for j in range(ntrials): # loop through number of trials
        # Loop through ith step with corresponding probability p:
        for i, p in enumerate(prob):
            print(f"Burning for pbare = {p}") # Print probability to be bare
            nsteps[j,i], forest_left[j,i] = fire_spread(nx=100,ny=100, pbare=p, maxiter=100,
                                                        graph=False, bare_seed=False, pspread=0.5, 
                                                        pstart=0.3)

    # Average values
    nsteps_avg = np.zeros(prob.size)
    nsteps_avg = np.mean(nsteps, axis=0)
    forest_left_avg = np.zeros(prob.size)
    forest_left_avg =  np.mean(forest_left, axis=0)

    # Perform polynomial fits
    a, b, c = np.polyfit(prob, nsteps_avg, 2) #deg 2
    equation_text_1 = f'$y = {a:.2f}x^2 + {b:.2f}x + {c:.2f}$' #deg 2
    prob_fit = np.linspace(min(prob), max(prob),100)
    nsteps_avg_fit = a * prob_fit**2 + b * prob_fit + c #deg 2
    nsteps_avg_pred = a * prob**2 + b * prob + c #deg 2
    r2 = r2_score(nsteps_avg, nsteps_avg_pred)
    print('R2 value for steps:'+str(r2))

    d, e, f = np.polyfit(prob, forest_left_avg, 2) #deg 2
    equation_text_2 = f'$y = {d:.2f}x^2 + {e:.2f}x + {f:.2f}$' #deg 2
    forest_left_avg_fit = d * prob_fit**2 + e * prob_fit + f #deg 2
    forest_left_avg_pred = d * prob**2 + e * prob + f #deg 2
    r2 = r2_score(forest_left_avg, forest_left_avg_pred)
    print('R2 value for forest:'+str(r2))

    # print('nsteps:'+str(nsteps))
    # print('forest left:'+str(forest_left))   

    # Plot Steps/Forest Percentage vs Probability of Starting Bare
    fig, ax1 = plt.subplots()
    # Plot Steps vs Prob of starting bare
    color = 'tab:blue'
    ax1.set_xlabel('Probability of cell starting bare')
    ax1.set_ylabel('Number of steps until fire stops', color=color)
    ax1.scatter(prob, nsteps_avg, color=color) # Collected data
    ax1.plot(prob_fit, nsteps_avg_fit, color=color, alpha=0.7,
             label=equation_text_1) # Best-fit line
    ax1.tick_params(axis='y', labelcolor=color)

    # Plot Forest Percentage left after burning vs Prob. of starting bare
    # Second plot shares X-axis with first plot
    ax2 = ax1.twinx()

    color = 'tab:green'
    ax2.set_ylabel('Forest left after burning [%]', color=color)
    ax2.scatter(prob, forest_left_avg, color=color) # Collected data
    ax2.plot(prob_fit, forest_left_avg_fit, color=color, alpha=0.7,
             label=equation_text_2) # Best-fit line
    ax2.tick_params(axis='y', labelcolor=color)

    fig.legend(bbox_to_anchor=(0.7,0.4),fontsize='small')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

#explore_barerate(ntrials=50) #more trials make behaviour more obvious

# =======================================================================================

# Disease simulation section 

#want to explore pfatal and pimmune

def disease_spread(nx=3, ny=3, maxiter=4, pspread=1.0, pimmune=0.0, pstart=0.1, pfatal=0.5, graph=True, immune_seed=False, start_seed=True):
    '''
    This function performs a disease simulation and optionally plots
    the 2D grid after every step. Returns number of steps for simulation to
    end, as well as percentage of healthy/alive people.

    Parameters
    ==========
    nx, ny : integer, defaults to 3
        Set the x (i, top to botton) and y (j, left to right) size of grid.
        Default is 3 squares in each direction.
    maxiter : int, defaults to 4
        Set the maximum number of iterations including initial condition
    pspread : float, defaults to 1
        Chance fire spreads from 0 to 1 (0 to 100%).
    pbare -immune : float, defaults to 0
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
    people = np.zeros([maxiter, nx, ny], dtype=int) + 2

    # Set bare spots using pbare
    if immune_seed == True:
        np.random.seed(13) #Set seed for reproducible immune conditions
    immune = np.random.rand(nx,ny) <= pimmune
    people[0,:,:][immune] = 1

    # Set fire using pstart
    if start_seed == True:
        np.random.seed(42) #Set seed for reproducible starting sick people
    start_sick = (np.random.rand(nx,ny) <= pstart) & (people[0,:,:] != 1) #immune people cannot get sick
    people[0,:,:][start_sick] = 3

    #Plot initial conditions
    if graph == True: 
        # Set colormap:
        people_cmap = ListedColormap(['black','tan', 'darkgreen', 'crimson'])
        # Plot initial condition:
        fig, ax = plt.subplots(1,1)
        contour = ax.matshow(people[0, :, :], vmin=0, vmax=3, cmap=people_cmap)
        ax.set_title(f'People Status (Step = {0:03d})')
        # Add colorbar and set its ticks
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_ticks([0, 1, 2, 3])
        cbar.set_ticklabels(['0 - Dead', '1 - Immune', '2 - Healthy', '3 - Sick'])
        # Add x and y axis labels
        ax.set_xlabel('Y-axis')
        ax.set_ylabel('X-axis')
        #fig.savefig(f'fig0000.png')
        plt.pause(0.5)  # Show the first plot briefly without blocking

    # Independent Random State for spreading:
    # Ensures random spreding conditions even if bare/fire initial
    # Conditions are reproducible
    spread_rdm = np.random.RandomState() 
    # Propagate solution
    for k in range(maxiter-1): # Time loop
        # Set chance to burn with independent random state
        getsick = spread_rdm.rand(nx,ny)
        # print(f"Ignite array at step {k}:\n{ignite}")

        # Use current step to set up next step:
        people[k+1,:,:] = people[k,:,:]

        # Spread in each cardinal direction
        # Spread north to south
        getsick_n2s = (people[k, :-1, :] == 3) & (people[k,1:,:] == 2)  & \
            (getsick[1:,:] <= pspread)
        people[k+1, 1:,:][getsick_n2s] = 3

        # Spread south to north
        getsick_s2n = (people[k, 1:, :] == 3) & (people[k,:-1,:] == 2)  & \
            (getsick[:-1,:] <= pspread)
        people[k+1, :-1,:][getsick_s2n] = 3

        # Spread west to east
        getsick_w2e = (people[k, :, :-1] == 3) & (people[k,:,1:] == 2)  & \
            (getsick[:,1:] <= pspread)
        people[k+1, :,1:][getsick_w2e] = 3

        # Spread east to west
        getsick_e2w = (people[k, :, 1:] == 3) & (people[k,:,:-1] == 2)  & \
            (getsick[:,:-1] <= pspread)
        people[k+1, :,:-1][getsick_e2w] = 3

        # set survivors and deceased
        survival_rdm = np.random.RandomState() #new random state for survival grid
        survival = survival_rdm.rand(nx,ny) # individual survival chance for each person

        deceased_msk = (people[k,:,:] == 3) & (survival[:,:]<=pfatal) #Find sick people & check if they will die
        survivor_msk = (people[k,:,:] == 3) & (survival[:,:]>pfatal) # Find sick people & check if they will live

        people[k+1, deceased_msk] = 0
        people[k+1, survivor_msk] = 1



        # Plot next steps:
        if graph == True:
            # Plot
            fig, ax = plt.subplots(1,1)
            contour = ax.matshow(people[k+1, :, :], vmin=0, vmax=3, cmap=people_cmap)
            ax.set_title(f'People Status (Step = {k+1:03d})')
            # Add colorbar and set its ticks
            cbar = plt.colorbar(contour, ax=ax)
            cbar.set_ticks([0, 1, 2, 3])
            cbar.set_ticklabels(['0 - Dead', '1 - Immune', '2 - Healthy', '3 - Sick'])
            # Add x and y axis labels
            ax.set_xlabel('Y-axis')
            ax.set_ylabel('X-axis')

            #fig.savefig(f'fig{k+1:04d}.png')
            plt.pause(0.5)  # Show the first plot briefly without blocking

        # Quit if no spots are on fire.
        nSick = (people[k+1, :, :] == 3).sum()
        if nSick == 0:
            print(f"Spread completed in {k+1} steps")
            nHealthy = (people[k+1, :, :] == 2).sum()
            healthy_left = (nHealthy / (nx * ny)) * 100 # Percentage of healthy left
            print(f'With {healthy_left}% healthy ones left')
            nImmune = (people[k+1, :, :] == 1).sum()
            Immune_left = (nImmune / (nx * ny)) * 100 # Percentage of immune left
            print(f'With {Immune_left}% immune ones left')
            break
    if graph == True:
        plt.ioff()  # Turn off interactive mode
        plt.show()  # Keep the last plot open
    return k+1, healthy_left, Immune_left

#disease_spread(maxiter=400, nx=100, ny=100, graph=False, pimmune=0.1, pspread=0.5, pfatal=0.3)

def explore_spreadrate(ntrials=5):
    '''
    Varies the probability of fire spreading between adjacent cells.
    Keeps track of number of steps required for every simulation to end.
    Returns plot of steps/forest percentage vs probability of fire
    propagating

    Parameters
    ==========
    ntrials : int, defaults to 5
        Number of trials to do. E.g. if set to 5, the simulations with probabilities
        between 0 and 1 will be run five times each. Values are then averaged for plotting
    '''

    prob = np.arange(0, 1, .05) # array of probabilities to test and plot
    nsteps = np.zeros((ntrials, prob.size)) # will be populated with steps in each simulation
    # every row is one silumation running from lowest to highest probability
    # every column is a different run with the same prob.
    healthy_left = np.zeros((ntrials, prob.size)) # will be populated with forest percentage
    immune_left = np.zeros((ntrials, prob.size))

    for j in range(ntrials): # loop through number of trials
        # Loop through ith step with corresponding probability p:
        for i, p in enumerate(prob):
            print(f"Spreading for pspread = {p}") # Print probability to burn
            # Since we are exploring the burn rate, we will use a seed for reproducible bare ground:
            nsteps[j,i], healthy_left[j,i], immune_left[j,i] = disease_spread(ny=100,nx=100, pspread=p, maxiter=400, 
                                                    graph=False, immune_seed=True, pimmune=0.1, pfatal=0.3)
            
    # Average values
    nsteps_avg = np.zeros(prob.size)
    nsteps_avg = np.mean(nsteps, axis=0)
    healthy_left_avg = np.zeros(prob.size)
    healthy_left_avg =  np.mean(healthy_left, axis=0)
    immune_left_avg = np.zeros(prob.size)
    immune_left_avg =  np.mean(immune_left, axis=0)

    print('nsteps:'+str(nsteps))
    print('healthy left:'+str(healthy_left))  
    print('immune left:'+str(immune_left))  
    
    # Perform polynomial fits
    a, b, c = np.polyfit(prob, nsteps_avg, 2) #degree 2
    equation_text_1 = f'$y = {a:.1f}x^2 + {b:.1f}x + {c:.1f}$' #degree 2
    prob_fit = np.linspace(min(prob), max(prob),100)
    nsteps_avg_fit = a * prob_fit**2 + b * prob_fit + c #degree 2
    nsteps_avg_pred = a * prob**2 + b * prob + c #degree 2
    r2 = r2_score(nsteps_avg, nsteps_avg_pred)
    print('R2 value for steps:'+str(r2))

    d, e, f = np.polyfit(prob, healthy_left_avg, 2) #degree 2
    equation_text_2 = f'$y = {d:.1f}x^2 + {e:.1f}x + {f:.1f}$' #degree 2
    healthy_left_avg_fit = d * prob_fit**2 + e * prob_fit + f #degree 2
    healthy_left_avg_pred = d * prob**2 + e * prob + f #degree 2
    r2 = r2_score(healthy_left_avg, healthy_left_avg_pred)
    print('R2 value for healthy:'+str(r2))

    g, h, i = np.polyfit(prob, immune_left_avg, 2) #degree 2
    equation_text_3 = f'$y = {g:.1f}x^2 + {h:.1f}x + {i:.1f}$' #degree 2
    immune_left_avg_fit = g * prob_fit**2 + h * prob_fit + i #degree 2
    immune_left_avg_pred = g * prob**2 + h * prob + i #degree 2
    r2 = r2_score(immune_left_avg, immune_left_avg_pred)
    print('R2 value for immune:'+str(r2))

    fit_lines = [equation_text_1, equation_text_2, equation_text_3]

    # Plot Steps/Healthy Percentage vs Probability of Fire Propagating
    fig, ax1 = plt.subplots()
    # Plot Steps vs Prob of disease propagating
    color = 'tab:blue'
    alpha = 0.5
    ax1.set_xlabel('Probability of disease propagating', fontsize='medium')
    ax1.set_ylabel('Number of steps until disease stops', color=color, fontsize='medium')
    ax1.scatter(prob, nsteps_avg, color=color) # Collected data
    ax1.plot(prob_fit, nsteps_avg_fit, color=color, alpha=alpha)#,
    #          label=equation_text_1) # Best-fit line
    ax1.tick_params(axis='y', labelcolor=color)

    # Plot Healthy Percentage left vs Prob. of propagation
    # Second plot shares X-axis with first plot
    ax2 = ax1.twinx()

    color_healthy = 'tab:green'
    color_immune = 'tab:orange'  # Use a different color for immune people

    ax2.set_ylabel('Healthy/Immune after simulation [%]', color=color_healthy, fontsize='medium')
    ax2.scatter(prob, healthy_left_avg, color=color_healthy, label='Healthy People') # Collected data
    ax2.plot(prob_fit, healthy_left_avg_fit, color=color_healthy, alpha=alpha)#,
    #          label=equation_text_2) # Best-fit line
    ax2.plot(prob_fit, immune_left_avg_fit, color=color_immune, alpha=alpha)
    ax2.tick_params(axis='y', labelcolor=color_immune)

    ax2.scatter(prob, immune_left_avg, color=color_immune, label='Immune People')

    fig.legend(bbox_to_anchor=(0.6,0.3),fontsize='x-small')

    # Create proxy artists for the second legend (best-fit lines)
    proxy_lines = [
        Line2D([0], [0], color='tab:blue', lw=2),
        Line2D([0], [0], color='tab:green', lw=2),
        Line2D([0], [0], color='tab:orange', lw=2)
    ]

    # Second legend: Best-fit line equations
    second_legend = ax2.legend(proxy_lines, fit_lines, loc='upper right',
                               bbox_to_anchor=(1.05, 1.2), fontsize='x-small', ncol=2)
                            #title='Fit Equations', ncol=3)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # fig.subplots_adjust(right=2)
    plt.show()

# explore_spreadrate(ntrials=50)

def explore_immunerate(ntrials=2):
    '''
    Varies the probability of a cell starting as bare ground, ie. varies
    percentage of initial forest in grid.
    Keeps track of number of steps required for every simulation to end, as
    well as percentage of forest left.
    Returns plot of steps/forest percentage vs probability.
    '''

    prob = np.arange(0, 1, .05) # array of probabilities to test and plot
    nsteps = np.zeros((ntrials, prob.size)) # empty array, will be populated with
    # steps in each simulation every row is one silumation running from lowest to 
    # highest probability every column is a different run with the same prob.
    healthy_left = np.zeros((ntrials, prob.size)) # will be populated with forest percentage

    for j in range(ntrials): # loop through number of trials
        # Loop through ith step with corresponding probability p:
        for i, p in enumerate(prob):
            print(f"Burning for pimmune = {p}") # Print probability to be bare
            nsteps[j,i], healthy_left[j,i] = fire_spread(nx=100,ny=100, pimmune=p, maxiter=100,
                                                        graph=False, immune_seed=False, pspread=0.5, 
                                                        pstart=0.3)

    # Average values
    nsteps_avg = np.zeros(prob.size)
    nsteps_avg = np.mean(nsteps, axis=0)
    healthy_left_avg = np.zeros(prob.size)
    healthy_left_avg =  np.mean(healthy_left, axis=0)

    # Perform polynomial fits
    a, b, c = np.polyfit(prob, nsteps_avg, 2) #deg 2
    equation_text_1 = f'$y = {a:.2f}x^2 + {b:.2f}x + {c:.2f}$' #deg 2
    prob_fit = np.linspace(min(prob), max(prob),100)
    nsteps_avg_fit = a * prob_fit**2 + b * prob_fit + c #deg 2
    nsteps_avg_pred = a * prob**2 + b * prob + c #deg 2
    r2 = r2_score(nsteps_avg, nsteps_avg_pred)
    print('R2 value for steps:'+str(r2))

    d, e, f = np.polyfit(prob, healthy_left_avg, 2) #deg 2
    equation_text_2 = f'$y = {d:.2f}x^2 + {e:.2f}x + {f:.2f}$' #deg 2
    healthy_left_avg_fit = d * prob_fit**2 + e * prob_fit + f #deg 2
    healthy_left_avg_pred = d * prob**2 + e * prob + f #deg 2
    r2 = r2_score(healthy_left_avg, healthy_left_avg_pred)
    print('R2 value for healthy:'+str(r2))

    # print('nsteps:'+str(nsteps))
    # print('forest left:'+str(forest_left))   

    # Plot Steps/Forest Percentage vs Probability of Starting Bare
    fig, ax1 = plt.subplots()
    # Plot Steps vs Prob of starting bare
    color = 'tab:blue'
    ax1.set_xlabel('Rate of immune people')
    ax1.set_ylabel('Number of steps until fire stops', color=color)
    ax1.scatter(prob, nsteps_avg, color=color) # Collected data
    ax1.plot(prob_fit, nsteps_avg_fit, color=color, alpha=0.7,
             label=equation_text_1) # Best-fit line
    ax1.tick_params(axis='y', labelcolor=color)

    # Plot Forest Percentage left after burning vs Prob. of starting bare
    # Second plot shares X-axis with first plot
    ax2 = ax1.twinx()

    color = 'tab:green'
    ax2.set_ylabel('Forest left after burning [%]', color=color)
    ax2.scatter(prob, healthy_left_avg, color=color) # Collected data
    ax2.plot(prob_fit, healthy_left_avg_fit, color=color, alpha=0.7,
             label=equation_text_2) # Best-fit line
    ax2.tick_params(axis='y', labelcolor=color)

    fig.legend(bbox_to_anchor=(0.7,0.4),fontsize='small')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()