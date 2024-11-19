#!/usr/bin/env python3

'''
Draft code for Lab 5: SNOWBALL EARTH!!!
'''

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# Some constants:
radearth = 6357000.  # Earth radius in meters.
mxdlyr = 50.         # depth of mixed layer (m)
sigma = 5.67e-8      # Steffan-Boltzman constant
C = 4.2e6            # Heat capacity of water
rho = 1020           # Density of sea-water (kg/m^3)
# epsilon = 1          # Emissivity of blackbody
albedo_ice = 0.6
albedo_gnd = 0.3


def gen_grid(nbins=18):
    '''
    Generate a grid from 0 to 180 lat (where 0 is south pole, 180 is north)
    where each returned point represents the cell center.

    Parameters
    ----------
    nbins : int, defaults to 18
        Set the number of latitude bins.

    Returns
    -------
    dlat : float
        Grid spacing in degrees.
    lats : Numpy array
        Array of cell center latitudes.
    '''

    dlat = 180 / nbins  # Latitude spacing.
    lats = np.arange(0, 180, dlat) + dlat/2.

    # Alternative way to obtain grid:
    # lats = np.linspace(dlat/2., 180-dlat/2, nbins)

    return dlat, lats


def temp_warm(lats_in):
    '''
    Create a temperature profile for modern day "warm" earth.

    Parameters
    ----------
    lats_in : Numpy array
        Array of latitudes in degrees where temperature is required

    Returns
    -------
    temp : Numpy array
        Temperature in Celcius.
    '''

    # Get base grid:
    dlat, lats = gen_grid()

    # Set initial temperature curve:
    T_warm = np.array([-47, -19, -11, 1, 9, 14, 19, 23, 25, 25,
                       23, 19, 14, 9, 1, -11, -19, -47])
    coeffs = np.polyfit(lats, T_warm, 2)

    # Now, return fitting:
    temp = coeffs[2] + coeffs[1]*lats_in + coeffs[0] * lats_in**2

    return temp


def insolation(S0, lats):
    '''
    Given a solar constant (`S0`), calculate average annual, longitude-averaged
    insolation values as a function of latitude.
    Insolation is returned at position `lats` in units of W/m^2.

    Parameters
    ----------
    S0 : float
        Solar constant (1370 for typical Earth conditions.)
    lats : Numpy array
        Latitudes to output insolation. Following the grid standards set in
        the diffusion program, polar angle is defined from the south pole.
        In other words, 0 is the south pole, 180 the north.

    Returns
    -------
    insolation : numpy array
        Insolation returned over the input latitudes.
    '''

    # Constants:
    max_tilt = 23.5   # tilt of earth in degrees

    # Create an array to hold insolation:
    insolation = np.zeros(lats.size)

    #  Daily rotation of earth reduces solar constant by distributing the sun
    #  energy all along a zonal band
    dlong = 0.01  # Use 1/100 of a degree in summing over latitudes
    angle = np.cos(np.pi/180. * np.arange(0, 360, dlong))
    angle[angle < 0] = 0
    total_solar = S0 * angle.sum()
    S0_avg = total_solar / (360/dlong)

    # Accumulate normalized insolation through a year.
    # Start with the spin axis tilt for every day in 1 year:
    tilt = [max_tilt * np.cos(2.0*np.pi*day/365) for day in range(365)]

    # Apply to each latitude zone:
    for i, lat in enumerate(lats):
        # Get solar zenith; do not let it go past 180. Convert to latitude.
        zen = lat - 90. + tilt
        zen[zen > 90] = 90
        # Use zenith angle to calculate insolation as function of latitude.
        insolation[i] = S0_avg * np.sum(np.cos(np.pi/180. * zen)) / 365.

    # Average over entire year; multiply by S0 amplitude:
    insolation = S0_avg * insolation / 365

    return insolation

def snowball_earth(nbins=18, dt=1., tstop=10000, lam=100., epsilon = 1,
                   spherecorr=True, albedo=0.3, S0=1370,
                   dynamic_alb=False, debug=False):
    '''
    Perform snowball earth simulation.

    Parameters
    ----------
    nbins : int, defaults to 18
        Number of latitude bins.
    dt : float, defaults to 1
        Timestep in units of years
    tstop : float, defaults to 10,000
        Stop time in years
    lam : float, defaults to 100
        Diffusion coefficient of ocean in m^2/s
    spherecorr : bool, defaults to True
        Use the spherical coordinate correction term. This should always be
        true except for testing purposes.
    debug : bool, defaults to False
        Turn  on or off debug print statements.

    Returns
    -------
    lats : Numpy array
        Latitude grid in degrees where 0 is the south pole.
    Temp : Numpy array
        Final temperature as a function of latitude.
    '''

    # Get time step in seconds:
    dt_sec = 365 * 24 * 3600 * dt  # Years to seconds.

    # Generate grid:
    dlat, lats = gen_grid(nbins)

    # Get grid spacing in meters.
    dy = radearth * np.pi * dlat / 180.

    # Generate insolation:
    insol = insolation(S0, lats)

    # Create initial condition:
    Temp = temp_warm(lats)
    if debug:
        print('Initial temp = ', Temp)

    # Get number of timesteps:
    nstep = int(tstop / dt)

    # Debug for problem initialization
    if debug:
        print("DEBUG MODE!")
        print(f"Function called for nbins={nbins}, dt={dt}, tstop={tstop}")
        print(f"This results in nstep={nstep} time step")
        print(f"dlat={dlat} (deg); dy = {dy} (m)")
        print("Resulting Lat Grid:")
        print(lats)

    # Build A matrix:
    if debug:
        print('Building A matrix...')
    A = np.identity(nbins) * -2  # Set diagonal elements to -2
    A[np.arange(nbins-1), np.arange(nbins-1)+1] = 1  # Set off-diag elements
    A[np.arange(nbins-1)+1, np.arange(nbins-1)] = 1  # Set off-diag elements
    # Set boundary conditions:
    A[0, 1], A[-1, -2] = 2, 2

    # Build "B" matrix for applying spherical correction:
    B = np.zeros((nbins, nbins))
    B[np.arange(nbins-1), np.arange(nbins-1)+1] = 1  # Set off-diag elements
    B[np.arange(nbins-1)+1, np.arange(nbins-1)] = -1  # Set off-diag elements
    # Set boundary conditions:
    B[0, :], B[-1, :] = 0, 0

    # Set the surface area of the "side" of each latitude ring at bin center.
    Axz = np.pi * ((radearth+50.0)**2 - radearth**2) * np.sin(np.pi/180.*lats)
    dAxz = np.matmul(B, Axz) / (Axz * 4 * dy**2)

    if debug:
        print('A = ', A)
    # Set units of A derp
    A /= dy**2

    # Get our "L" matrix:
    L = np.identity(nbins) - dt_sec * lam * A
    L_inv = np.linalg.inv(L)

    if debug:
        print('Time integrating...')
    for i in range(nstep):
        # Add spherical correction term:
        if spherecorr:
            Temp += dt_sec * lam * dAxz * np.matmul(B, Temp)
            if debug:
                print(f'Temp after sphere corr: {Temp}')

        # Apply insolation and radiative losses:
        if dynamic_alb:
            loc_ice = Temp <= -10
            albedo = np.zeros_like(Temp)
            albedo[loc_ice] = albedo_ice
            albedo[~loc_ice] = albedo_gnd
            #set radiative variable here?

        radiative = (1-albedo) * insol - epsilon*sigma*(Temp+273.15)**4
        Temp += dt_sec * radiative / (rho*C*mxdlyr)
        
        # if insol:
        #     # Update albedo based on conditions if it is dynamic:
        #     if dynamic_alb:
        #         loc_ice = Temp <= -10
        #         albedo = np.zeros_like(Temp)
        #         albedo[loc_ice] = albedo_ice
        #         albedo[~loc_ice] = albedo_gnd
        #     ins = (dt_sec/(rho*C*mxdlyr)) * (insolation(lats)*(1-albedo)-epsilon*sigma*(Temp+273.15)**4)
        #     Temp += ins

        Temp = np.matmul(L_inv, Temp)

    return lats, Temp


def test_snowball(tstop=10000):
    '''
    Reproduce example plot in lecture/handout.

    Using our DEFAULT values (grid size, diffusion, etc.) and a warm-Earth
    initial condition, plot:
        - Initial condition
        - Plot simple diffusion only
        - Plot simple diffusion + spherical correction
        - Plot simple diff + sphere corr + insolation
    '''
    nbins = 18

    # Generate very simple grid
    # Generate grid:
    dlat, lats = gen_grid(nbins)

    # Create initial condition:
    initial = temp_warm(lats)

    # Get simple diffusion solution:
    lats, t_diff = snowball_earth(tstop=tstop, spherecorr=False, 
                                  S0=0, epsilon=0)

    # Get diffusion + spherical correction:
    lats, t_sphe = snowball_earth(tstop=tstop, S0=0, epsilon=0)

    # Get diff + spher corr + insolation
    lats, t_ins =  snowball_earth(tstop=tstop)

    # Create figure and plot!
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(lats, initial, label='Warm Earth Init. Cond.')
    ax.plot(lats, t_diff, label='Simple Diffusion')
    ax.plot(lats, t_sphe, label='Diffusion + Sphere. Corr.')
    ax.plot(lats, t_ins, label='Diff + Spherec Corr. + Insolation')

    ax.set_xlabel('Latitude (0=South Pole)')
    ax.set_ylabel('Temperature ($^{\circ} C$)')

    ax.legend(loc='best')

    fig.tight_layout()


#test_snowball()

def vary_epsilon():
    tstop = 10000
    emiss_array1 = np.linspace(0,1, 5) # plot some values to see general behavior as epsilon changes
    emiss_array2 = np.linspace(0.65,0.75,10) # values chosen after deciding a general range for epsilon

    # Generate grid:
    dlat, lats = gen_grid(18)
    # Create initial condition:
    initial = temp_warm(lats)


    fig, ax = plt. subplots(1,1)
    # plot initial conditions
    ax.plot(lats, initial, label='initial')

    # Plot temperatures for range of epsilon:
    for i in range(len(emiss_array1)):
        lats, temps = snowball_earth(tstop=tstop, epsilon=emiss_array1[i])
        ax.plot(lats, temps, label=f'$\\epsilon = {emiss_array1[i]}$')
        ax.set_xlabel('Latitude (0=South Pole)')
        ax.set_ylabel('Temperature ($^{\circ} C$)')

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Legend outside top-right


    # Empty Delta T array
    # Warm Earth temp - modeled temp:
    delta_T_eq = np.zeros_like(emiss_array2) # Equator
    delta_T_SP = np.zeros_like(emiss_array2) # South Pole
    delta_T_NP = np.zeros_like(emiss_array2) # North Pole

    for i in range(len(emiss_array2)):
        # compare values around equator (lat=85)
        lats, temps = snowball_earth(tstop=tstop, epsilon=emiss_array2[i])
        ind_eq = np.where(lats==85)[0][0]
        # Populate Delta T:
        delta_T_eq[i] = initial[ind_eq] - temps[ind_eq]
        delta_T_SP[i] = initial[0] - temps[0]
        delta_T_NP[i] = initial[-1] - temps[-1]

    fig2, ax2 = plt.subplots(1,1)
    ax2.plot(emiss_array2, delta_T_eq, label='Equator')
    ax2.plot(emiss_array2, delta_T_SP, label='South Pole')
    ax2.plot(emiss_array2, delta_T_NP, label='North Pole')
    ax2.set_xlabel('Emissivity ($\\epsilon$)')
    ax2.set_ylabel(r'$\Delta T$ (°C)')
    ax2.set_title('Warm Earth - Modeled Temperature', fontsize='medium')
    ax2.legend()

    plt.show()

#vary_epsilon()

def vary_lambda(plot1_start, plot1_end, plot2_start, plot2_end):
    tstop = 10000
    lam_array1 = np.linspace(plot1_start,plot1_end, 5) # plot some values to see general behavior as epsilon changes
    lam_array2 = np.linspace(plot2_start,plot2_end,10) # values chosen after deciding a general range for epsilon

    # Generate grid:
    dlat, lats = gen_grid(18)
    # Create initial condition:
    initial = temp_warm(lats)


    fig, ax = plt. subplots(1,1)
    # plot initial conditions
    ax.plot(lats, initial, label='initial', linestyle='dashed')

    # Plot temperatures for range of epsilon:
    for i in range(len(lam_array1)):
        lats, temps = snowball_earth(tstop=tstop, lam=lam_array1[i], epsilon=0.7)
        ax.plot(lats, temps, label=f'$\\lambda = {lam_array1[i]}$')
        ax.set_xlabel('Latitude (0=South Pole)')
        ax.set_ylabel('Temperature ($^{\circ} C$)')

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Legend outside top-right


    # Empty Delta T array
    # Warm Earth temp - modeled temp:
    delta_T_eq = np.zeros_like(lam_array2) # Equator
    delta_T_SP = np.zeros_like(lam_array2) # South Pole
    delta_T_NP = np.zeros_like(lam_array2) # North Pole

    for i in range(len(lam_array2)):
        # compare values around equator (lat=85)
        lats, temps = snowball_earth(tstop=tstop, lam=lam_array2[i], epsilon=0.7)
        ind_eq = np.where(lats==85)[0][0]
        # Populate Delta T:
        delta_T_eq[i] = initial[ind_eq] - temps[ind_eq]
        delta_T_SP[i] = initial[0] - temps[0]
        delta_T_NP[i] = initial[-1] - temps[-1]

    fig2, ax2 = plt.subplots(1,1)
    ax2.plot(lam_array2, delta_T_eq, label='Equator')
    ax2.plot(lam_array2, delta_T_SP, label='South Pole')
    ax2.plot(lam_array2, delta_T_NP, label='North Pole')
    ax2.set_xlabel('Diffusivity ($\\lambda$)')
    ax2.set_ylabel(r'$\Delta T$ (°C)')
    ax2.set_title('Warm Earth - Modeled Temperature', fontsize='medium')
    ax2.legend()

    plt.show()

vary_lambda(0,50, 0,50)

vary_lambda(50,100, 50,100)

vary_lambda(100,150, 100,150)