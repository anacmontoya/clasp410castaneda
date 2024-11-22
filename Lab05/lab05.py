#!/usr/bin/env python3

'''
code for Lab 5: SNOWBALL EARTH.

Uncomment lines at the end to reproduce report figures.
Note: question2() function will return 8 figures but only
5 were used in the report. Read docstring of function for
list of used figures.
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
                   spherecorr=True, albedo=0.3, S0=1370, initial_temp=None,
                   gamma=1, dynamic_alb=False, debug=False):
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
    epsilon: float, defaults to 1
        Set ground emissivity. Set to zero to turn off radiative cooling
    spherecorr : bool, defaults to True
        Use the spherical coordinate correction term. This should always be
        true except for testing purposes.
    albedo: float, defaults to 0.3
        Set the Earth's albedo
    S0: float, defaults to 1370
        Set incoming solar forcing constant. Change to zero to turn off 
        insolation
    initial_temp: array, defaults to None
        array used as the initial temperatures of each latitude point.
        Must match length of latitude array returned by gen_grid(nbins)
    gamma: float, defaults to 1
        Solar multiplier. Scales insolation.
    dynamic_alb: bool, defaults to False
        Determines whether albedo is recalculated each step based on 
        surface Temperature or is held constant.
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
    insol = gamma * insolation(S0, lats)

    # Create initial condition:
    if initial_temp is None:
        Temp = temp_warm(lats)
    else:
        if len(initial_temp) != nbins:
            raise ValueError(f"initial_temp must have the same length as the number of latitude bins (nbins={nbins}).")
        Temp = np.array(initial_temp)

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
            if debug:
                print(f'Dynamic albedo updated: {albedo}')
        else:
            # Use the user-specified albedo value (default or provided)
            if debug:
                print(f"Using fixed albedo: {albedo}")

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


def vary_epsilon(lam=100):
    '''
    Plots temperature profiles and errors for different emissivities.

    Parameters
    ----------
    lam: float, defaults to 100
        diffusivity (lambda) to be used in snowball_earth() function
    '''
    tstop = 10000
    emiss_array1 = np.linspace(0,1, 5) # plot some values to see general behavior as epsilon changes
    emiss_array2 = np.linspace(0.65,0.75,10) # values chosen after deciding a general range for epsilon

    nbins = 18

    # Generate very simple grid
    # Generate grid:
    dlat, lats = gen_grid(nbins)

    # Create initial condition:
    initial = temp_warm(lats)

    fig, ax = plt. subplots(1,1)
    # plot initial conditions
    ax.plot(lats, initial, label='Initial', linestyle='dashed')


    # Plot temperatures for range of epsilon:
    for i in range(len(emiss_array1)):
        lats, temps = snowball_earth(tstop=tstop, epsilon=emiss_array1[i], lam=lam)
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
        lats, temps = snowball_earth(tstop=tstop, epsilon=emiss_array2[i], lam=lam)
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


def vary_lambda(lam_array1, lam_array2, show_error=False):
    '''
    Plots temperature profiles and errors for different diffusivities.

    Parameters
    ----------
    lam_array1: array
        array of lambda values to be plotted in the temperature profile plot
    lam_array2: array
        array of lambda values to be plotted in the error plot
    show_error: bool, defaults to False
        determines if error for each lambda value in array 2 must be calculated
        and printed to screen
    '''
    tstop = 10000
    #lam_array1 = np.linspace(range_start,range_end, 5) # plot some values to see general behavior as epsilon changes
    #lam_array2 = np.linspace(range_start,range_end,10) # values chosen after deciding a general range for epsilon

    # Generate grid:
    dlat, lats = gen_grid(18)
    # Create initial condition:
    initial = temp_warm(lats)


    fig, ax = plt. subplots(1,1)
    # plot initial conditions
    # ax.plot(lats, initial, label='initial', linestyle='dashed')

    # Plot temperatures for range of epsilon:
    for i in range(len(lam_array1)):
        lats, temps = snowball_earth(tstop=tstop, lam=lam_array1[i], epsilon=0.7)
        ax.plot(lats, temps, label=f'$\\lambda = {lam_array1[i]}$')
        ax.set_xlabel('Latitude (0=South Pole)')
        ax.set_ylabel('Temperature ($^{\circ} C$)')
        error = initial - temps
        avg_error = np.mean(error)
        if show_error:
            print(f'average error for lambda={lam_array1[i]}: {avg_error}')

    ax.plot(lats, initial, label='initial', linestyle='dashed')
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

def hot_earth():
    '''
    Plots temperature profiles for a hot earth scenario. Simulations are run for 
    three different durations.
    '''
    hot_temp = np.full(18, 60, dtype=float)
    lats, temps_5k = snowball_earth(tstop=5000, lam=22., epsilon = 0.7, initial_temp=hot_temp,
                    dynamic_alb=True, debug=False)
    lats, temps_10k = snowball_earth(tstop=10000, lam=22., epsilon = 0.7, initial_temp=hot_temp,
                    dynamic_alb=True, debug=False)
    lats, temps_20k = snowball_earth(tstop=20000, lam=22., epsilon = 0.7, initial_temp=hot_temp,
                    dynamic_alb=True, debug=False)
    
    # Warm earth equilibrium for reference. Use same Eps and Lam values and dyn. albedo
    lats, t_warm =  snowball_earth(dynamic_alb=True, epsilon=0.7, lam=22.)


    fig, ax = plt.subplots(1,1)
    # ax.plot(lats, hot_temp, label='Initial')
    ax.plot(lats, temps_5k, label='5,000 yrs')
    ax.plot(lats, temps_10k, label='10,000 yrs')
    ax.plot(lats, temps_20k, label='20,000 yrs')
    ax.plot(lats, t_warm, label='Warm Earth Equilib.')
    ax.set_xlabel('Latitude (0=South Pole)')
    ax.set_ylabel('Temperature ($^{\circ} C$)')
    ax.set_title('''Temperature Equilibrium of "Hot Earth"
                 (all lats = 60 C)''', fontsize='medium')
    ax.legend()
    plt.show()


def cold_earth():
    '''
    Plots temperature profiles for a cold earth scenario. Simulations are run for 
    three different durations.
    '''
    cold_temp = np.full(18, -60, dtype=float)
    # lats, temps_5k = snowball_earth(tstop=5000, lam=22., epsilon = 0.7, initial_temp=cold_temp,
    #                 dynamic_alb=True, debug=False)
    lats, temps_10k = snowball_earth(tstop=10000, lam=22., epsilon = 0.7, initial_temp=cold_temp,
                    dynamic_alb=True, debug=False)
    lats, temps_20k = snowball_earth(tstop=20000, lam=22., epsilon = 0.7, initial_temp=cold_temp,
                    dynamic_alb=True, debug=False)
    lats, temps_40k = snowball_earth(tstop=40000, lam=22., epsilon = 0.7, initial_temp=cold_temp,
                    dynamic_alb=True, debug=False)
    
    # Warm earth equilibrium for reference. Use same Eps and Lam values and dyn. albedo
    lats, t_warm =  snowball_earth(dynamic_alb=True, epsilon=0.7, lam=22.)

    fig, ax = plt.subplots(1,1)
    # ax.plot(lats, cold_temp, label='Initial')
    # ax.plot(lats, temps_5k, label='5,000 yrs')
    ax.plot(lats, temps_10k, label='10,000 yrs')
    ax.plot(lats, temps_20k, label='20,000 yrs')
    ax.plot(lats, temps_40k, label='40,000 yrs')
    ax.plot(lats, t_warm, label='Warm Earth Equilib.')
    ax.set_xlabel('Latitude (0=South Pole)')
    ax.set_ylabel('Temperature ($^{\circ} C$)')
    ax.set_title('''Temperature Equilibrium of "Cold Earth"
                 (all lats = -60 C)''', fontsize='medium')
    ax.legend()
    plt.show()

def flash_freeze():
    '''
    Plots temperature profiles for a flash freeze earth scenario. Simulations are run for 
    three different durations.
    '''
    lats, temps = snowball_earth(tstop=10000, lam=22., epsilon = 0.7, albedo=0.6,
                    dynamic_alb=False, debug=False)
    # Warm earth equilibrium for reference. Use same Eps and Lam values and dyn. albedo
    lats, t_warm =  snowball_earth(dynamic_alb=True, epsilon=0.7, lam=22.)
    
    dlat, lats = gen_grid(18)
    # Create initial condition:
    initial = temp_warm(lats)

    fig, ax = plt.subplots(1,1)
    ax.plot(lats, initial, label='Initial')
    ax.plot(lats, t_warm, label='Warm Earth Equilib.')
    ax.plot(lats, temps, label='Flash Freeze Equilib.')
    ax.set_xlabel('Latitude (0=South Pole)')
    ax.set_ylabel('Temperature ($^{\circ} C$)')
    ax.set_title('Temperature Equilibrium of "Flash-Freeze Earth"',
                 fontsize='medium')
    ax.legend()
    plt.show()

def vary_gamma():
    '''
    Plot of global average temperature vs gamma
    '''
    gammas1 = np.arange(0.4,1.4,0.05)
    gammas2 = np.arange(1.4, 0.35, -0.05)
    # Append the arrays
    gamma = np.append(gammas1, gammas2)
    global_temp = np.zeros_like(gamma)
    # global_temp_warm
    dlat, lats = gen_grid(18)
    initial = np.full(18, -60, dtype=float) # cold Earth temps.
    #test_gamma = np.array((0.4,0.9,1.4,0.9,0.4))
    for i in range(len(gamma)):
        # print(f'calculating temps for gamma = {test_gamma[i]}:')
        # print(f'initial conditions are: {initial}')
        lats, temps = snowball_earth(tstop=20000, lam=22., epsilon=0.7,
                                    dynamic_alb=True, gamma=gamma[i], initial_temp=initial)
        # print(f'temperatures = {temps}')
        initial = temps.copy()
        global_temp[i] = np.mean(temps)
        # print(f'average global temp = {global_temp[i]}')
    #global temps of warm earth with varying gamma?
    fig, ax = plt.subplots(1,1)
    ax.plot(gamma[:21], global_temp[:21], label=r'Increasing $\gamma$')
    ax.plot(gamma[21:], global_temp[21:], label=r'Decreasing $\gamma$')
    ax.set_xlabel(r'$\gamma$')
    ax.set_ylabel('Average Global Temperature (C)')
    ax.legend()
    plt.show()


def question1():
    '''
    Reproduces figure 1 from report.
    '''
    test_snowball()


def question2():
    '''
    Reproduces figures 2, 3, 4, [unused figure], [unused figure], 
    5a, 5b, [unused figure] in that order.
    '''
    vary_epsilon()

    #General range of lambda:
    lam_array1 = np.linspace(0,150,5)
    lam_array2 = np.linspace(0,150,15)

    vary_lambda(lam_array1=lam_array1, lam_array2=lam_array2)
    
    # 15-30 range:
    lam_array1 = np.linspace(15,30,5)
    lam_array2 = np.linspace(15,30,15)
    
    vary_lambda(lam_array1=lam_array1, lam_array2=lam_array2) #NP=18, SP=26
    
    
    mean = np.mean((18,26))
    vary_lambda(np.array((18,mean,26)), np.array((18,mean,26)), show_error=True) #avg has smalles avg error

def question3():
    '''
    Reproduces figures 6,7 and 8 from the report
    '''
    hot_earth()
    cold_earth()
    flash_freeze()

def question4():
    '''Reproduces figure 9 from report'''
    vary_gamma()


# Uncomment for report figures:

# question1()
# question2()
# question3()
# question4()