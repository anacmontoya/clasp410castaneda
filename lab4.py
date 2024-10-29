#1/usr/bin/env python3
'''
Tools and methods for solving our heat equation/diffusion. Solves permafrost
and wire problem from Lab 04. Uncomment lines to get figures from report.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker

plt.style.use('bmh')


def temp_kanger(t, warming=False, warming_shift=0.5): 
    '''
    For an array of times in days, return timeseries of temperature for Kangerlussuaq, Greenland.

    Parameters
    ----------
    t: time (array)
        Numpy array of times in days
    warming: (bool, defaults to False)
        Decides if a positive shift is added to the mean temperatures
    warming_shift: float (defaults to 0.5)
        value to be added to mean temperatures in degrees Celsius if Warming == True

    Returns
    -------

    '''
    # Kangerlussuaq average temperature:
    t_kanger = np.array([-19.7, -21.0, -17., -8.4, 2.3, 8.4,
                        10.7, 8.5, 3.1, -6.0, -12.0, -16.9])

    t_amp = (t_kanger - t_kanger.mean()).max()

    curve = t_amp*np.sin(np.pi/180 * t - np.pi/2) + t_kanger.mean()

    # Add a warming shift if applicable
    if warming == True:
        curve = curve + warming_shift

    return curve


def heatdiff(xmax, tmax, dx, dt, c2=1, conditions = 'permafrost', warming=False, 
             warming_shift=0.5, debug=True):
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
    warming: bool, defaults to False:
        Decides if a positive shift is added to the mean temperatures
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
        U[0, :] = temp_kanger(tgrid,warming=warming, warming_shift=warming_shift)
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
        index = np.where((change_U == max_change) | (change_U == -max_change))[0]
        print(f'the maximum change is {max_change} and it occurs at depth {(index+21)*dx}')

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
    plt.show()

# Uncomment for figure 1:
# plot_wire()

def plot_greenland(y,tick_interval, warming=False, warming_shift=0.5):
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
                              debug=False, warming=warming, warming_shift=warming_shift)

    # Create a figure/axes object
    fig, ax = plt.subplots(1, 1)
    # Create a color map and add a color bar
    map = ax.pcolor(t, x, heat, cmap='seismic', vmin=-25, vmax=25)
    plt.colorbar(map, ax=ax, label='Temperature ($C$)')

    # Sets locations and labels of ticks so that they do not appear every year \
    # and looks cluttered
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_interval))
    position_xticks = [tick_interval * i for i in range(y // (tick_interval // 365) + 1)]
    ax.set_xticks(position_xticks)
    ax.set_xticklabels([str(i * (tick_interval // 365)) for i in range(y // (tick_interval // 365) + 1)])
    ax.set_xlabel('Years')

    ax.set_ylabel('Depth [m]')
    ax.set_title("Ground Temperature: Kangerlussuaq")

    # Invert y_axis
    plt.gca().invert_yaxis()

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
    ax2.set_xlabel('Temperature [C]')
    ax2.set_ylabel('Depth [m]')
    ax2.set_title('Ground Temperature: Kangerlussuaq')
    fig2.legend()

    plt.gca().invert_yaxis()
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
    y = y
    #tick_interval = tick_interval # 1 year
    tmax = 365*y
    dx=0.2
    dt=0.5
    c2 = 0.25 * (1/1000)**2 * (86400) # 0.25 mm^2/s covert to m^2/days

    # Normal plot:
    x, t, heat = heatdiff(xmax=xmax, tmax=tmax, dx=dx, dt=dt, c2=c2, conditions='permafrost', 
                              debug=False, warming=False)
    # Warming plot:
    x_w, t_w, heat_w = heatdiff(xmax=xmax, tmax=tmax, dx=dx, dt=dt, c2=c2, conditions='permafrost', 
                              debug=False, warming=True, warming_shift=warming_shift)

    # Set indexing for the final year of results:
    loc = int(-365/dt) # Final 365 days of the result

    # Extract the min values over final year
    # Normal
    winter = heat[:,loc:].min(axis=1) # axis=1 is horizontal axis
    summer = heat[:,loc:].max(axis=1)
    change = summer - winter
    # warming:
    winter_w = heat_w[:,loc:].min(axis=1) # axis=1 is horizontal axis
    summer_w = heat_w[:,loc:].max(axis=1)
    change_w = summer_w - winter_w

    #find at what depth does ground starts to freeze in summer (active layer)
    # and where permafrost layer ends (temp goes above 0 again)
    #normal conditions:
    # find end of active layer:
    frozen = summer < 0
    frozen_depth = np.where(frozen==True)
    print(f'ground starts to freeze at: {frozen_depth[0][0]*dx} meters')
    # find end of permafrost layer:
    thaw = winter > 0
    thaw_depth = np.where(thaw == True)
    print(f'temp goes above 0 at {thaw_depth[0][0]*dx} meters')
    print(f'permafrost layer length: {thaw_depth[0][0]*dx - frozen_depth[0][0]*dx}')

    # warming conditions:
    # find end of active layer (start of permafrost):
    frozen_w = summer_w < 0
    frozen_depth_w = np.where(frozen_w==True)
    print(f'when warmed by {warming_shift} deg, ground starts to freeze at: {frozen_depth_w[0][0]*dx} meters')
    # find end of permafrost layer:
    thaw_w = winter_w > 0
    thaw_depth_w = np.where(thaw_w == True)
    print(f'temp goes above 0 at {thaw_depth_w[0][0]*dx} meters')
    print(f'permafrost layer length after warming:\
          {thaw_depth_w[0][0]*dx - frozen_depth_w[0][0]*dx}')
    
    # change of permafrost after warming: positive meand shrinking
    print(f'warming by {warming_shift} deg changed permafrost layer by:\
          {(thaw_depth[0][0]*dx - frozen_depth[0][0]*dx) - (thaw_depth_w[0][0]*dx - frozen_depth_w[0][0]*dx)}')

    #temp profiles for summer and winter
    # plot normal conditions
    fig, ax = plt.subplots(1,2)
    ax[0].plot(winter, x, label='winter')
    ax[0].plot(summer,x, label='summer')
    ax[0].axhline(y=x[frozen_depth[0][0]], color='black', linestyle='dashed')
    ax[0].axhline(y=x[thaw_depth[0][0]], linestyle='dashed', color='black')
    ax[0].text(1, x[frozen_depth[0][0]]+6, f'''Start Permafrost layer:
                {frozen_depth[0][0]*dx:.2f} m''')
    ax[0].text(1, x[thaw_depth[0][0]]+6, f'''End Permafrost layer:
               {thaw_depth[0][0]*dx:.2f} m''')
    ax[0].set_xlabel('Temperature [C]')
    ax[0].set_ylabel('Depth [m]')
    ax[0].set_title('Ground Temperature (Normal Conditions)')
    ax[0].invert_yaxis() 

    #plot warming conditions
    ax[1].plot(winter_w, x_w)
    ax[1].plot(summer_w ,x_w)
    ax[1].axhline(y=x[frozen_depth_w[0][0]], color='black', linestyle='dashed')
    ax[1].axhline(y=x[thaw_depth_w[0][0]], linestyle='dashed', color='black')
    ax[1].text(1, x[frozen_depth_w[0][0]]+6, f'''Start Permafrost layer:
                {frozen_depth_w[0][0]*dx:.2f} m''')
    ax[1].text(1, x[thaw_depth_w[0][0]]+6, f'''End Permafrost layer:
               {thaw_depth_w[0][0]*dx:.2f} m''')
    ax[1].set_xlabel('Temperature [C]')
    ax[1].set_ylabel('Depth [m]')
    ax[1].set_title(f'Ground Temperature (Warmer by {warming_shift} deg)')
    ax[1].invert_yaxis() 

    fig.legend()

    plt.show()


# Uncomment for figure 6-8

# warming(50, warming_shift=3)
# warming(50, warming_shift=1)
# warming(50, warming_shift=0.5)