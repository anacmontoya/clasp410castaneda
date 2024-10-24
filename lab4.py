#1/usr/bin/env python3
'''
Tools and methods for solving our heat equation/diffusion
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker

plt.style.use('bmh')


def temp_kanger(t, warming=False): 
    '''
    For an array of times in days, return timeseries of temperature for Kangerlussuaq, Greenland.
    '''
    # Kangerlussuaq average temperature:
    t_kanger = np.array([-19.7, -21.0, -17., -8.4, 2.3, 8.4,
                        10.7, 8.5, 3.1, -6.0, -12.0, -16.9])
    # if warming == True:
    #     add_values = np.array([0.5, 1, 3])
    #     t_kanger = t_kanger + add_values[np.arange(len(t_kanger)) % len(add_values)]

    t_amp = (t_kanger - t_kanger.mean()).max()

    curve = t_amp*np.sin(np.pi/180 * t - np.pi/2) + t_kanger.mean()

    if warming == True:
        add_values = np.array([0.5, 1, 3])
        curve = curve + add_values[np.arange(len(curve)) % len(add_values)]

    return curve


def heatdiff(xmax, tmax, dx, dt, c2=1, conditions = 'permafrost', warming=False, debug=True):
    '''
    Parameters:
    -----------


    Returns:
    --------

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
        U[0, :] = temp_kanger(tgrid,warming=warming)
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

    # Return grid and result:
    return xgrid, tgrid, U


def plot_wire():
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

#plot_wire()

def plot_greenland(y,tick_interval, warming=False):
    xmax = 100 # 100 meters
    y = y
    tick_interval = tick_interval # 1 year
    tmax = 365*y
    dx=1
    dt=0.5
    c2 = 0.25 * (1/1000)**2 * (86400) # 0.25 mm^2/s covert to m^2/days

    x, t, heat = heatdiff(xmax=xmax, tmax=tmax, dx=dx, dt=dt, c2=c2, conditions='permafrost', debug=False, warming=warming)
    # Create a figure/axes object
    fig, ax = plt.subplots(1, 1)
    # Create a color map and add a color bar
    map = ax.pcolor(t, x, heat, cmap='seismic', vmin=-25, vmax=25)
    plt.colorbar(map, ax=ax, label='Temperature ($C$)')

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

    fig2, ax2 = plt.subplots(1,1)
    ax2.plot(winter, x, label='winter')
    ax2.plot(summer,x, label='summer')
    ax2.set_xlabel('Temperature [C]')
    ax2.set_ylabel('Depth [m]')
    ax2.set_title('Ground Temperature: Kangerlussuaq')
    fig2.legend()

    plt.gca().invert_yaxis()
    plt.show()

# plot_greenland(y=5, tick_interval=365)

# plot_greenland(y=20, tick_interval=365*5)

# plot_greenland(y=50, tick_interval=365*5)

# plot_greenland(y=100, tick_interval=365*10)


plot_greenland(y=5, tick_interval=365, warming=True)
