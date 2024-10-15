#!/usr/bin/env python3
'''
A set of tools and routines for solving the N-layer atmosphere energy
balance problem and perform some useful analysis. To reproduce figures
shown in report, uncomment calls to functions after they are defined.

'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

plt.style.use('bmh')

#  Define some useful constants here.
sigma = 5.67E-8  # Steffan-Boltzman constant.


def n_layer_atmos(N, epsilon, S0=1350, albedo=0.33, debug=False, nuclear=False):
    '''
    Solve the n-layer atmosphere problem and return temperature at each layer.

    Parameters
    ----------
    N : int
        Set the number of layers.
    epsilon : float, default=1.0
        Set the emissivity of the atmospheric layers.
    albedo : float, default=0.33
        Set the planetary albedo from 0 to 1.
    S0 : float, default=1350
        Set the incoming solar shortwave flux in Watts/m^2.
    debug : boolean, default=False
        Turn on debug output.

    Returns
    -------
    temps : Numpy array of size N+1
        Array of temperatures from the Earth's surface (element 0) through
        each of the N atmospheric layers, from lowest to highest.
    '''

    # Create matrices:
    A = np.zeros([N+1, N+1])
    b = np.zeros(N+1)
    if nuclear == False:
        b[0] = -S0/4 * (1-albedo)
    else:
        b[-1] = -S0/4 # Last layer absorbs ALL incoming radiation, we keep factor of 1/4 to account for Earth's shape


    if debug:
        print(f"Populating N+1 x N+1 matrix (N = {N})")

    # Populate our A matrix piece-by-piece.
    for i in range(N+1):
        for j in range(N+1):
            if debug:
                print(f"Calculating point i={i}, j={j}")
            # Diagonal elements are always -2 ('cept at Earth's surface.)
            if i == j:
                A[i, j] = -1*(i > 0) - 1
                # print(f"Result: A[i,j]={A[i,j]}")
            else:
                # This is the pattern we solved for in class!
                m = np.abs(j-i) - 1
                A[i, j] = epsilon * (1-epsilon)**m
    # At Earth's surface, epsilon =1, breaking our pattern.
    # Divide by epsilon along surface to get correct results.
    A[0, 1:] /= epsilon

    # Verify our A matrix.
    if debug:
        print(A)

    # Get the inverse of our A matrix.
    Ainv = np.linalg.inv(A)

    # Multiply Ainv by b to get Fluxes.
    fluxes = np.matmul(Ainv, b)

    # Convert fluxes to temperatures.
    # Fluxes for all atmospheric layers
    temps = (fluxes/epsilon/sigma)**0.25
    temps[0] = (fluxes[0]/sigma)**0.25  # Flux at ground: epsilon=1.

    return temps

# ====================================================================

# Part 1, experiment 1
def part1_exp1():
    '''
    Analayzes relationship between emissivity and surface temperature
    of Earth. Returns plot and prints R2 value of best-fit line of the
    second degree polynomial and estimated emissivity of atmosphere
    that yields a surface temperature of 288
    '''

    # array of emissivities
    emissivities = np.linspace(0.01,1,100)
    surf_temps = np.zeros(emissivities.size)
    # loop through emissivities and extract surface temp only
    for i in range(len(emissivities)):
        surf_temps[i] = n_layer_atmos(1,epsilon=emissivities[i])[0]

    print(emissivities)
    print(surf_temps)

    # Polynomial fit 2nd degree
    a, b, c = np.polyfit(emissivities, surf_temps, 2) # degree 2
    equation_text_1 = f'$y = {a:.2f}x^2 + {b:.2f}x + {c:.2f}$' 
    surf_temp_pred = a * emissivities**2 + b * emissivities + c 
    r2 = r2_score(surf_temps, surf_temp_pred)
    print('R2 value:'+str(r2))
    # Find emissivity for which the temp = 288
    # Subtract 288 so the point we are looking for goes to 0
    polynomial = np.poly1d((a,b,c-288))
    # Then find roots of polynomial, only positive value is valid
    print('Estimated Emissivity for Temp of 288K:'+str(polynomial.r))

    fig, ax = plt.subplots(1,1)
    ax.scatter(emissivities, surf_temps, label='Model Output')
    ax.plot(emissivities, surf_temp_pred, alpha=0.7, label=equation_text_1)
    ax.set_ylabel('Surface Temperature of Earth [K]')
    ax.set_xlabel('Emissivity of Atmosphere')
    ax.set_title('Surface Temperatures from 1-Layer Atmosphere Model')
    fig.legend()
    plt.show()


#part1_exp1()

# Part 1, experiment 2
def part1_exp2():
    '''
    Explores relationship between number of layers and surface temperature.
    Returns plot, prints R2 value of best-fit line of the second degree, and
    estimated number of layers for a surface temperature of 288K.

    Plots altitude profile of a 5 layer atmosphere.
    '''
    epsilon = 0.255
    layers = np.arange(1,10,1)
    surf_temps = np.zeros(layers.size)
    for i in range(len(layers)):
        surf_temps[i] = n_layer_atmos(N=layers[i], epsilon=epsilon)[0]

    a, b= np.polyfit(surf_temps, layers, 1) # degree 1
    equation_text_1 = f'$y = {a:.2f}x + {b:.2f}$' 
    layers_pred = a * surf_temps + b
    r2 = r2_score(layers, layers_pred)
    print('R2 value:'+str(r2))
    # Find number of layers for which the temp = 288
    polynomial = np.poly1d((a,b))
    print('Estimated # of layers for Temp of 288K: '+str(polynomial(288)))


    fig, ax = plt.subplots(1,1)
    ax.scatter(surf_temps, layers, label='Model Output')
    ax.plot(surf_temps, layers_pred, alpha=0.7, label=equation_text_1)
    ax.set_xlabel('Surface Temperature of Earth [K]')
    ax.set_ylabel('Number of Layers in Atmosphere')
    ax.set_title('N-Layer Atmosphere with emissivity of 0.255')
    fig.legend()
    plt.show()

    # To get a surface temp of 288 we need 5 layers:
    temps = n_layer_atmos(N=5,epsilon=epsilon)
    print('Surface Temperature: '+str(temps[0]))
    print('Temperature of last atmosphere layer: '+str(temps[-1]))
    layers2 = np.arange(0,6,1)

    fig, ax = plt.subplots(1,1)
    ax.plot(temps, layers2)
    ax.set_xlabel('Temperature [K]')
    ax.set_ylabel('Altitude [Atmospheric Layer Number]')
    ax.set_title('5-Layer Atmosphere Temperatures with emissivity of 0.255')
    plt.show()

# part1_exp2()


# Part 2 - Venus
def part2_venus(albedo=0.33):
    '''
    Explores number of layers necessary in planet Venus to achieve surface
    temperature of 700K. Returns plot of number of layers vs temperature.

    Parameters:
    ----------
    albedo: float, default = 0.33
        planetary surface albedo. if albedo == 0.8, the range of layers 
        extends to 120. Otherwise it plots up to 35.
    '''
    # Emissivity = 1, T_surf = 700 K, S0 = 2600, layers = ?
    if albedo == 0.8:
        layers = np.arange(1,120,1)
    else:
        layers = np.arange(1,35,1)
    t_surf_venus = np.zeros(layers.size)
    for i in range(len(layers)):
        t_surf_venus[i] = n_layer_atmos(layers[i], epsilon=1, S0=2600, albedo=albedo)[0]

    a, b, c = np.polyfit(t_surf_venus, layers, 2) # degree 2
    equation_text_1 = f'$y = {a:.4f}x^2 + {b:.2f}x + {c:.2f}$' 
    layers_pred = a * t_surf_venus**2 + b * t_surf_venus + c 
    r2 = r2_score(layers, layers_pred)
    print('R2 value:'+str(r2))
    # Find emissivity for which the temp = 288
    # Subtract 288 so the point we are looking for goes to 0
    polynomial = np.poly1d((a,b,c))
    # Then find roots of polynomial, only positive value is valid
    print('Estimated # of layers for Temp of 700K: '+str(polynomial(700)))

    fig, ax = plt. subplots(1,1)
    ax.scatter(t_surf_venus, layers, label='Model Output')
    ax.plot(t_surf_venus,layers_pred, label=equation_text_1, alpha=0.7)
    ax.set_xlabel('Surface Temperature of Venus [K]')
    ax.set_ylabel('Altitude of Atmosphere [# of layers]')
    ax.set_title(f'N-Layer Venus Atmosphere, $\\epsilon = 1$, $\\alpha = {albedo}$')
    fig.legend()
    plt.show()

# part2_venus()
# part2_venus(albedo=0.8)


def part3_nuclear():
    '''
    Explores altitude profiles of Earth in case of nuclear winter,
    and compares it to normal conditions. Plots both profiles
    side by side.
    '''
    layers = np.arange(0,6,1)
    temps = n_layer_atmos(N=5, epsilon=0.5, nuclear=True)
    print('Surface Temperature of Earth: '+str(temps[0]))
    print(temps)
    layers_normal = np.arange(0,6,1)
    temps_normal = n_layer_atmos(N=5, epsilon=0.5)

    fig, ax = plt.subplots(1,2)
    ax[0].scatter(temps, layers)
    ax[1].scatter(temps_normal,layers_normal)
    
    # Annotating the y values next to each marker
    for i, (temp, layer) in enumerate(zip(temps, layers)):
        ax[0].annotate(f'{temp:.2f}', (temp, layer), textcoords="offset points",
                       xytext=(1,5), ha='center')

    for i, (temps_normal, layers_normal) in enumerate(zip(temps_normal, layers_normal)):
        ax[1].annotate(f'{temps_normal:.2f}', (temps_normal, layers_normal), 
                       textcoords="offset points", xytext=(1,5), ha='center')

    ax[0].set_xlabel('Temperature [K]')
    ax[1].set_xlabel('Temperature [K]')
    ax[0].set_ylabel('Atmospheric Layer Number')    
    ax[1].set_ylabel('Atmospheric Layer Number')
    ax[0].set_title('Nuclear Winter')
    ax[1].set_title('Normal Conditions')
    fig.suptitle('Atmospheric Temperatures Under Different Conditions and Emissivity = 0.5')
    plt.show()

# part3_nuclear()