#1/usr/bin/env python3
'''
Tools and methods for solving our heat equation/diffusion
'''

import numpy as np
import matplotlib.pyplot as plt


def heatdiff(xmax, tmax, dx, dt, c2=1, debug=True):
    '''
    Parameters:
    -----------


    Returns:
    --------

    '''
    if dt > dx**2 / (2*c2):
        raise ValueError('dt is too large. Must be less than dx**2 / (2*c2) for stability')

    # Start by calculating size of array: MxN
    M = int(np.round(xmax / dx + 1))
    N = int(np.round(tmax / dt + 1))

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

    # Return grid and result:
    return xgrid, tgrid, U

x, t, heat = heatdiff(xmax=1, tmax=0.2, dx=0.2, dt=0.02, debug=False)

# Create a figure/axes object
fig, ax = plt.subplots(1, 1)

# Create a color map and add a color bar.
map = ax.pcolor(t, x, heat, cmap='seismic', vmin=0, vmax=1)
plt.colorbar(map, ax=ax, label='Temperature ($C$)')
plt.show()