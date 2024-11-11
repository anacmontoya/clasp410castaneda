#1/usr/bin/env python3
'''
Tools and methods for solving our heat equation/diffusion. Solves permafrost
and wire problem from Lab 04. Uncomment lines to get figures from report.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker

plt.style.use('bmh') # use a style sheet
plt.ion() # interactive mode to eliminnte plt.show later on

def temp_kanger(t, warming_shift = 0): 
    '''
    For an array of times in days, return timeseries of temperature for Kangerlussuaq, Greenland.

    Parameters
    ----------
    t: time (array)
        Numpy array of times in days

    warming_shift: float (defaults to 0.5)
        value to be added to mean temperatures in degrees Celsius

    Returns
    -------
    '''
    # Kangerlussuaq average temperature:
    t_kanger = np.array([-19.7, -21.0, -17., -8.4, 2.3, 8.4,
                        10.7, 8.5, 3.1, -6.0, -12.0, -16.9])

    t_amp = (t_kanger - t_kanger.mean()).max() # amplitude of seasonal variation

    curve = t_amp*np.sin(np.pi/180 * t - np.pi/2) + t_kanger.mean() # Calculate daily changing temperatures

    curve = curve + warming_shift # add the warming

    return curve # return the array of temperature for boundary conditions


def heatdiff(xmax, tmax, dx, dt, c2=1, conditions = 'permafrost', warming_shift = 0.5, debug = True):
    '''
    This function solves the heat equation and returns solution plus time and
    space arrays. Can be used to solve permafrost problem or wire problem
    from class.

    Parameters:
    -----------
    xmax: float
        Maximum value in spatial space
    tmax: float
        Maximum Value in time space
    dx: float
        Size of spatial step
    dt: float
        Size of time step
    c2 : float, defaults to 1.0
        thermal diffusivity constant
    conditions: str, defaults to 'permafrost'
        Determines initial/boundary conditions to be used. Must be selected
        'permafrost' or 'wire'
    warming_shift: float, defaults to 0.5
        value to be added to mean temperatures in degrees Celsius if Warming == True
    debug: bool, defaults to True
        prints variables and arryas at different points to help debug

    Returns:
    --------
    xgrid: array
        numpy array with spatial grid used
    tgrid: array
        numpy array of time grid used
    U: 2D array
        matrix representing temperature solutions

    '''
    # Check solution is stable:
    if dt > dx**2 / (2*c2):
        raise ValueError('dt is too large. Must be less than dx**2 / (2*c2) for stability')

    # Start by calculating size of array: MxN
    M = int(np.round(xmax / dx + 1))
    N = int(np.round(tmax / dt + 1))

    # create time and space arrays:
    xgrid, tgrid = np.arange(0, xmax+dx, dx), np.arange(0, tmax+dt, dt)

    if debug:
        print(f'Our grid goes from 0 to {xmax}m and 0 to {tmax}s')
        print(f'Our spatial step is {dx} and time step is {dt}')
        print(f'There are {M} points in space and {N} points in time.')
        print('Here is our spatial grid:')
        print(xgrid)
        print('Here is our time grid:')
        print(tgrid)

    # Initialize our data array:
    U = np.zeros((M,N))

    # Set initial/boundary conditions depending on problem:
    if conditions == 'permafrost':
        # Set boundary conditions:
        U[0, :] = temp_kanger(tgrid, warming_shift = warming_shift)
        U[-1, :] = 5

    elif conditions == 'wire':
        # Set initial conditions:
        U[:, 0] = 4*xgrid - 4*xgrid**2

        # Set boundary conditions:
        U[0, :] = 0
        U[-1, :] = 0

    # Set our "r" constant.
    r = c2 * dt / dx**2

    # Solve! Forward differnce ahoy.
    for j in range(N-1):
        U[1:-1, j+1] = (1-2*r) * U[1:-1, j] + \
            r*(U[2:, j] + U[:-2, j])

    if conditions == 'permafrost':
        # print change of temp in last year
        change_U = U[21:51,-1] - U[21:51,-2] #look at depths between 20 and 50 meters
        max_change = np.max(np.abs(change_U))
        index = np.where((change_U == max_change) | (change_U == -max_change))[0][0]
        print(f'the maximum change is {max_change}\u00B0C and it occurs at depth {(index+21)*dx}m')

    if debug == True:
        print(f'surface temperatures: {U[0,:]}')

    # Return grid and result:
    return xgrid, tgrid, U


def plot_wire():
    '''
    This function plots color map solution for wire problem (figure 1 in report)
    '''
    x, t, heat = heatdiff(xmax=1, tmax=0.2, dx=0.2, dt=0.02, debug=False, conditions='wire')
    print(heat)

    # Create a figure/axes object
    fig, ax = plt.subplots(1, 1)

    # Create a color map and add a color bar.
    map = ax.pcolor(t, x, heat, cmap='seismic', vmin=0, vmax=1)
    plt.colorbar(map, ax=ax, label='Temperature ($C$)')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Length [m]')
    ax.set_title('Temperature Distribution of Wire in Time')

# Uncomment for figure 1:
# plot_wire()

def plot_greenland(y, tick_interval, warming_shift=0.5):
    '''
    Plots heat map for Greenland as well as temperature profile of permafrost.
    (figure 2-5 in report)

    Parameters
    ----------
    y: float
        number of years for which to plot
    tick_interval: integer
        defines where to put ticks in the time axis. Must be in days and a multiple of 365
    warming: bool, defaults to False:
        Decides if a positive shift is added to the mean temperatures
    warming_shift: float, defaults to 0.5
        value to be added to mean temperatures in degrees Celsius if Warming == True
    '''
    xmax = 100 # 100 meters
    y = y
    #tick_interval = tick_interval # 1 year
    tmax = 365*y
    dx=1
    dt=0.5
    c2 = 0.25 * (1/1000)**2 * (86400) # 0.25 mm^2/s covert to m^2/days

    x, t, heat = heatdiff(xmax=xmax, tmax=tmax, dx=dx, dt=dt, c2=c2, conditions='permafrost', 
                              debug=False, warming_shift=warming_shift)

    # Create a figure/axes object
    fig, ax = plt.subplots(1, 1)

    # Create a color map and add a color bar
    map = ax.pcolor(t, x, heat, cmap='seismic', vmin=-25, vmax=25)
    plt.colorbar(map, ax=ax, label = 'Temperature [\u00B0C]')

    # Sets locations and labels of ticks so that they do not appear every year \
    # and looks cluttered
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_interval))
    position_xticks = [tick_interval * i for i in range(y // (tick_interval // 365) + 1)]
    ax.set_xticks(position_xticks)
    ax.set_xticklabels([str(i * (tick_interval // 365)) for i in range(y // (tick_interval // 365) + 1)])
    
    ax.set_xlabel('Years')
    ax.set_ylabel('Depth [m]')
    ax.set_title("Ground Temperature: Kangerlussuaq")
    ax.invert_yaxis()

    # Set indexing for the final year of results:
    loc = int(-365/dt) # Final 365 days of the result

    # Extract the min values over final year
    winter = heat[:,loc:].min(axis=1) # axis=1 is horizontak axis
    summer = heat[:,loc:].max(axis=1)
    change = summer - winter
    print(np.where(np.min(change[:-1]) == change[:-1]))

    fig2, ax2 = plt.subplots(1,1)
    ax2.plot(winter, x, label='winter')
    ax2.plot(summer,x, label='summer')
    ax2.set_xlabel('Temperature [\u00B0C]')
    ax2.set_ylabel('Depth [m]')
    ax2.set_title('Ground Temperature: Kangerlussuaq')
    ax2.invert_yaxis()
    fig2.legend()
    plt.show()

# Uncomment for figure 2-5:
# NOTE: each line below returns 2 plots but only the color map was used in final report

# plot_greenland(y=5, tick_interval=365)
# plot_greenland(y=20, tick_interval=365*5)
# plot_greenland(y=50, tick_interval=365*5)
# plot_greenland(y=100, tick_interval=365*10)


def warming(y, warming_shift=0.5):
    '''
    Plots 2 temperature profiles side by side. Left side is permafrost
    problem under normal conditions and conditions where temperature is
    shifted by warming_shift.

    Parameters
    ----------
    y: float
        number of years for which to plot
    warming_shift: float, defaults to 0.5
        value to be added to mean temperatures in degrees Celsius if Warming == True
    '''
    xmax = 100 # 100 meters

    #tick_interval = tick_interval # 1 year
    tmax = 365*y
    dx=0.2
    dt=0.5
    c2 = 0.25 * (1/1000)**2 * (86400) # 0.25 mm^2/s covert to m^2/days

    # Call the diffusion function:
    x, t, heat = heatdiff(xmax=xmax, tmax=tmax, dx=dx, dt=dt, c2=c2, conditions='permafrost', 
                              debug=False, warming_shift = 0)
    
    x_w, t_w, heat_w = heatdiff(xmax=xmax, tmax=tmax, dx=dx, dt=dt, c2=c2, conditions='permafrost', 
                              debug=False, warming_shift = warming_shift)

    # Set indexing for the final year of results:
    loc = int(-365/dt) # Final 365 days of the result

    # Extract the min values over final year in absence of climate change
    winter = heat[:,loc:].min(axis=1) # axis = 1 is horizontal axis
    summer = heat[:,loc:].max(axis=1)

    # Extract the min values over final year for warming scenario
    winter_w = heat_w[:,loc:].min(axis=1) # axis = 1 is horizontal axis
    summer_w = heat_w[:,loc:].max(axis=1)

    # find at what depth does ground starts to freeze in summer (end of active layer)
    # and where permafrost layer ends (temp goes above 0 again deep in ground)

    # normal conditions:
    # find end of active layer:
    frozen = summer < 0
    perm_top = x[np.where(frozen==True)[0][0]]

    # find end of permafrost layer:
    thawed = winter > 0
    perm_base = x[np.where(thawed == True)[0][0]]
    perm_depth = perm_base - perm_top

    # warming conditions:
    # find end of active layer (start of permafrost):
    frozen_w = summer_w < 0
    perm_top_w = x[np.where(frozen_w==True)[0][0]]

    # find end of permafrost layer:
    thawed_w = winter_w > 0
    perm_base_w = x[np.where(thawed_w == True)[0][0]]
    perm_depth_w = perm_base_w - perm_top_w
    
    # change of permafrost after warming: negative value represents a decrease with warming
    print(f'Surface warming by {warming_shift}\u00B0C changed permafrost extent by: {perm_depth_w - perm_depth}m')

    # temp profiles for summer and winter
    # plot normal conditions on first subplot
    fig, ax = plt.subplots(1,2, figsize = (16, 10))

    #Plot the summer and winter temperature curves under a non-warming scenario
    ax[0].plot(winter, x, label = 'winter')
    ax[0].plot(summer, x, label = 'summer')

    #Plot horizonal lines at the top and base of the permafrost
    ax[0].axhline(y = perm_top, color='black', linestyle='dashed')
    ax[0].axhline(y = perm_base, linestyle='dashed', color='black')
    limits = ax[0].set_xlim([winter.min() - 2, summer.max() + 2]) # set limits based on temperature range

    # Fill the area between the top and bottom to shade the permafrost layer
    ax[0].fill_between(limits, perm_top, perm_base, color = 'deepskyblue', alpha = 0.5, # fill between the lines
                    label = f"Permafrost: top = {round(perm_top, 1)}m, " 
                    f"base = {round(perm_base, 1)}m, thickness = {round(perm_depth, 1)}m")
    
    # labels, title, legend, invert y-axis
    ax[0].set_xlabel('Temperature [\u00B0C]')
    ax[0].set_ylabel('Depth [m]')
    ax[0].set_title('Ground Temperature (Normal Conditions)')
    ax[0].legend(loc = 'lower left', frameon = True, fontsize = 10)
    ax[0].invert_yaxis() 

    # Now plot the warming scenario in the second subplot. Start with the temperature curves
    ax[1].plot(winter, x, label='winter')
    ax[1].plot(summer,x, label='summer')
    limits_w = ax[1].set_xlim([winter_w.min() - 2, summer_w.max() + 2]) # set limits based on temperature range

    # Plot horizontal lines at top and base of the permaforst
    ax[1].axhline(y = perm_top_w, color='black', linestyle='dashed')
    ax[1].axhline(y = perm_base_w, linestyle='dashed', color='black')

    # Fill the area between the top and bottom to shade the permafrost layer
    ax[1].fill_between(limits_w, perm_top_w, perm_base_w, color = 'deepskyblue', alpha = 0.5, # fill between the lines
                    label = f"Permafrost: top = {round(perm_top_w, 1)}m, " 
                    f"base = {round(perm_base_w, 1)}m, thickness = {round(perm_depth_w, 1)}m")
   
    # labels, title, legend, invert y-axis
    ax[1].set_xlabel('Temperature [\u00B0C]')
    ax[1].set_ylabel('Depth [m]')
    ax[1].set_title(f'Ground Temperature ({warming_shift}\u00B0C Surface Warming)')
    ax[1].legend(loc = 'lower left', frameon = True, fontsize = 10)
    ax[1].invert_yaxis() 

# Uncomment for figure 6-8

warming(50, warming_shift=3)
#warming(50, warming_shift=1)
#warming(50, warming_shift=0.5)