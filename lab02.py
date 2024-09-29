#!/usr/bin/env python3

'''
This file contains tools and scripts for completing Lab 2 for CLaSP 410.
To reproduce the plots shown in the lab report, uncomment lines after functions.
All plots are in the same order as in the report, for the exception of the 
'recreate_plots()' which were not included in final report.
'''

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# defining L-V equations
def dNdt_comp(t, N, a=1, b=2, c=1, d=3): 
    '''
    This function calculates the Lotka-Volterra competition equations for
    two species. Given normalized populations, `N1` and `N2`, as well as the
    four coefficients representing population growth and decline,
    calculate the time derivatives dN_1/dt and dN_2/dt and return to the
    caller.
    This function accepts `t`, or time, as an input parameter to be
    compliant with Scipy's ODE solver. However, it is not used in this
    function.
    Parameters
    ----------
    t : float
        The current time (not used here).
    N : two-element list
        The current value of N1 and N2 as a list (e.g., [N1, N2]).
    a, b, c, d : float, defaults=1, 2, 1, 3
        The value of the Lotka-Volterra coefficients.
    Returns
    -------
    dN1dt, dN2dt : floats
        The time derivatives of `N1` and `N2`.
    '''
    # Here, N is a two-element list such that N1=N[0] and N2=N[1]
    dN1dt = a*N[0]*(1-N[0]) - b*N[0]*N[1]
    dN2dt = c*N[1]*(1-N[1]) - d*N[1]*N[0]

    return dN1dt, dN2dt

def dNdt_prey(t, N, a=1, b=2, c=1, d=3):
    '''
    This function calculates the Lotka-Volterra prey-predator equations for
    two species. Given normalized populations, `N1` and `N2`, as well as the
    four coefficients representing population growth and decline,
    calculate the time derivatives dN_1/dt and dN_2/dt and return to the
    caller.
    This function accepts `t`, or time, as an input parameter to be
    compliant with Scipy's ODE solver. However, it is not used in this
    function.
    Parameters
    ----------
    t : float
        The current time (not used here).
    N : two-element list
        The current value of N1 and N2 as a list (e.g., [N1, N2]).
    a, b, c, d : float, defaults=1, 2, 1, 3
        The value of the Lotka-Volterra coefficients.
    Returns
    -------
    dN1dt, dN2dt : floats
        The time derivatives of `N1` and `N2`.
    '''
    # Here, N is a two-element list such that N1=N[0] and N2=N[1]
    dN1dt = a*N[0] - b*N[0]*N[1]
    dN2dt = -c*N[1] + d*N[1]*N[0]

    return dN1dt, dN2dt

# Creating solvers
def euler_solve(func, N1_init=.3, N2_init=.6, dt=1, t_final=100.0, a=1, b=2, c=1, d=3):
    '''
    This function is an ODE solver that uses the first-order-accurate
    Euler's method. It takes a function equal to the time derivative of another function
    and returns arrays for time and equation solutions
    Parameters
    ----------
    func : function
        A python function that takes `time`, [`N1`, `N2`] as inputs and
        returns the time derivative of N1 and N2.
    N1_init : float
        Initial value for population 1
    N2_init : float
        Initial value for population 2
    dt : float, default = 1
        Size of the derivative step to use
    t_final : float, default = 100
        Integrate until this value is reached, in years.
    a, b, c, d : float
        Lotka-Volterra coefficient values that will be passed unto func
    '''
    # Create time array. We won't use that here, but will return it
    # to the caller for convenience.
    t = np.arange(0, t_final, dt)

    # Create container for the solution, set initial condition.
    N1 = np.zeros(t.size)
    N1[0] = N1_init
    N2 = np.zeros(t.size)
    N2[0] = N2_init

    for i in range(1, t.size):
        dN1, dN2 = func(i, [N1[i-1], N2[i-1]], a=a, b=b, c=c, d=d) # get time derivatives
        # get solutions with Euler formula
        N1[i] = N1[i-1] + dt * dN1
        N2[i] = N2[i-1] + dt * dN2

    return t, N1, N2

def solve_rk8(func, N1_init=.3, N2_init=.6, dT=10, t_final=100.0,
              a=1, b=2, c=1, d=3):
    '''
    Solve the Lotka-Volterra competition and predator/prey equations using
    Scipy's ODE class and the adaptive step 8th order solver.
    Parameters
    ----------
    func : function
        A python function that takes `time`, [`N1`, `N2`] as inputs and
        returns the time derivative of N1 and N2.
    N1_init, N2_init : float
        Initial conditions for `N1` and `N2`, ranging from (0,1]
    dT : float, default=10
        Largest timestep allowed in years.
    t_final : float, default=100
        Integrate until this value is reached, in years.
    a, b, c, d : float, default=1, 2, 1, 3
        Lotka-Volterra coefficient values
    Returns
    -------
    time : Numpy array
        Time elapsed in years.
    N1, N2 : Numpy arrays
        Normalized population density solutions.
    '''
    from scipy.integrate import solve_ivp
    # Configure the initial value problem solver
    result = solve_ivp(func, [0, t_final], [N1_init, N2_init],
                       args=[a, b, c, d], method='DOP853', max_step=dT)
    
    # Perform the integration
    time, N1, N2 = result.t, result.y[0, :], result.y[1, :] # Return values to caller.
    return time, N1, N2

# ======================================================================================
# Recreating plots from Lab02
def recreate_plots():
    '''
    Recreates the example plots from Lab 02 pdf
    '''
    t_comp_eu, N1_comp_eu, N2_comp_eu = euler_solve(dNdt_comp)
    t_comp_rk, N1_comp_rk, N2_comp_rk = solve_rk8(dNdt_comp, dT=1)

    t_prey_eu, N1_prey_eu, N2_prey_eu = euler_solve(dNdt_prey, dt=0.05)
    t_prey_rk, N1_prey_rk, N2_prey_rk = solve_rk8(dNdt_prey)

    # Plot
    n1_color = 'red'
    n2_color = 'blue'
    #Competition equation
    fig, ax = plt.subplots(1,1)
    ax.plot(t_comp_eu,N1_comp_eu, label='N1 - euler', color=n1_color)
    ax.plot(t_comp_eu,N2_comp_eu, label='N2 - euler', color=n2_color)
    ax.plot(t_comp_rk, N1_comp_rk, ':', label='N1 - rk8', color=n1_color)
    ax.plot(t_comp_rk, N2_comp_rk, ':',label='N2 - rk8', color=n2_color)
    ax.set_ylabel('Normalized population density')
    ax.set_xlabel('Time [years]')
    ax.set_title('Competition - dt=0.05')
    fig.legend()
    # Predator-Prey equation
    fig, ax = plt.subplots(1,1)
    ax.plot(t_prey_eu, N1_prey_eu, label='N1 - euler', color=n1_color)
    ax.plot(t_prey_eu, N2_prey_eu, label='N2 - euler', color=n2_color)
    ax.plot(t_prey_rk, N1_prey_rk, ':', label='N1 - rk8', color=n1_color)
    ax.plot(t_prey_rk, N2_prey_rk, ':',label='N2 - rk8', color=n2_color)
    ax.set_ylabel('Normalized population density')
    ax.set_xlabel('Time [years]')
    ax.set_title('Prey-Predator')
    fig.legend()
    plt.show()

# Uncomment following line to recreate example plots from lab02 pdf
#recreate_plots()

# ========================================================================

# Question 1: Compare Euler to RK8 for both sets of equations

def q1_competition():
    '''
    2x2 subplots showing solutions to the competition equations. Compares
    Euler and RK8 methods. Each subplot uses a different Euler step
    [0.05,0.1,0.5,1].
    '''
    n1_color = 'red'
    n2_color = 'blue'
    #  Varying time-step dt for Competition equation
    t_comp_rk, N1_comp_rk, N2_comp_rk = solve_rk8(dNdt_comp)

    t_comp_eu, N1_comp_eu, N2_comp_eu = euler_solve(dNdt_comp, dt=0.05)
    fig, ax = plt.subplots(2,2)
    ax[0,0].plot(t_comp_eu,N1_comp_eu, color=n1_color)
    ax[0,0].plot(t_comp_eu,N2_comp_eu, color=n2_color)
    ax[0,0].plot(t_comp_rk, N1_comp_rk, ':', color=n1_color)
    ax[0,0].plot(t_comp_rk, N2_comp_rk, ':', color=n2_color)
    ax[0,0].set_ylabel('Normalized population density', fontsize='small')
    ax[0,0].set_title('dt=0.05', fontsize='medium')

    t_comp_eu, N1_comp_eu, N2_comp_eu = euler_solve(dNdt_comp, dt=0.1)
    ax[0,1].plot(t_comp_eu,N1_comp_eu, label='N1 - euler', color=n1_color)
    ax[0,1].plot(t_comp_eu,N2_comp_eu, label='N2 - euler', color=n2_color)
    ax[0,1].plot(t_comp_rk, N1_comp_rk, ':', label='N1 - rk8', color=n1_color)
    ax[0,1].plot(t_comp_rk, N2_comp_rk, ':',label='N2 - rk8', color=n2_color)
    ax[0,1].set_title('dt=0.1', fontsize='medium')

    t_comp_eu, N1_comp_eu, N2_comp_eu = euler_solve(dNdt_comp, dt=0.5)
    ax[1,0].plot(t_comp_eu,N1_comp_eu, color=n1_color)
    ax[1,0].plot(t_comp_eu,N2_comp_eu, color=n2_color)
    ax[1,0].plot(t_comp_rk, N1_comp_rk, ':', color=n1_color)
    ax[1,0].plot(t_comp_rk, N2_comp_rk, ':', color=n2_color)
    ax[1,0].set_ylabel('Normalized population density', fontsize='small')
    ax[1,0].set_xlabel('Time [years]', fontsize='small')
    ax[1,0].set_title('dt=0.5', fontsize='medium')

    t_comp_eu, N1_comp_eu, N2_comp_eu = euler_solve(dNdt_comp, dt=1)
    ax[1,1].plot(t_comp_eu,N1_comp_eu, color=n1_color)
    ax[1,1].plot(t_comp_eu,N2_comp_eu, color=n2_color)
    ax[1,1].plot(t_comp_rk, N1_comp_rk, ':', color=n1_color)
    ax[1,1].plot(t_comp_rk, N2_comp_rk, ':', color=n2_color)
    ax[1,1].set_xlabel('Time [years]', fontsize='small')
    ax[1,1].set_title('dt=1', fontsize='medium')

    fig.legend(fontsize='small')
    fig.suptitle('Comparison of Euler and RK8 methods with different\n'\
                'time steps for solving the L-V competition equation', fontsize='medium')
    plt.show()

def q1_prey():
    '''
    2x2 subplots showing solutions to the prey-predator equations.
    Compares Euler and RK8 methods. Each subplot uses a different
    Euler step [0.05,0.1,0.005,0.1].
    '''
    n1_color = 'red'
    n2_color = 'blue'
    #  Varying time-step dt for Prey-Predator equation
    t_prey_rk, N1_prey_rk, N2_prey_rk = solve_rk8(dNdt_prey)

    fig, ax = plt.subplots(2,2)

    t_prey_eu, N1_prey_eu, N2_prey_eu = euler_solve(dNdt_prey, dt=0.05)
    ax[0,0].plot(t_prey_eu, N1_prey_eu, color=n1_color)
    ax[0,0].plot(t_prey_eu, N2_prey_eu, color=n2_color)
    ax[0,0].plot(t_prey_rk, N1_prey_rk, ':', color=n1_color)
    ax[0,0].plot(t_prey_rk, N2_prey_rk, ':', color=n2_color)
    ax[0,0].set_ylabel('Normalized population density', fontsize='small')
    ax[0,0].set_title('dt=0.05', fontsize='medium')

    t_prey_eu, N1_prey_eu, N2_prey_eu = euler_solve(dNdt_prey, dt=0.01)
    ax[0,1].plot(t_prey_eu, N1_prey_eu, label='N1 - euler', color=n1_color)
    ax[0,1].plot(t_prey_eu, N2_prey_eu, label='N2 - euler', color=n2_color)
    ax[0,1].plot(t_prey_rk, N1_prey_rk, ':', label='N1 - rk8', color=n1_color)
    ax[0,1].plot(t_prey_rk, N2_prey_rk, ':',label='N2 - rk8', color=n2_color)
    ax[0,1].set_title('dt=0.01', fontsize='medium')

    t_prey_eu, N1_prey_eu, N2_prey_eu = euler_solve(dNdt_prey, dt=0.005)
    ax[1,0].plot(t_prey_eu, N1_prey_eu, color=n1_color)
    ax[1,0].plot(t_prey_eu, N2_prey_eu, color=n2_color)
    ax[1,0].plot(t_prey_rk, N1_prey_rk, ':', color=n1_color)
    ax[1,0].plot(t_prey_rk, N2_prey_rk, ':', color=n2_color)
    ax[1,0].set_ylabel('Normalized population density', fontsize='small')
    ax[1,0].set_xlabel('Time [years]', fontsize='small')
    ax[1,0].set_title('dt=0.005', fontsize='medium')

    t_prey_eu, N1_prey_eu, N2_prey_eu = euler_solve(dNdt_prey, dt=0.1)
    ax[1,1].plot(t_prey_eu, N1_prey_eu, color=n1_color)
    ax[1,1].plot(t_prey_eu, N2_prey_eu, color=n2_color)
    ax[1,1].plot(t_prey_rk, N1_prey_rk, ':', color=n1_color)
    ax[1,1].plot(t_prey_rk, N2_prey_rk, ':', color=n2_color)
    #ax[1,1].set_ylabel('Normalized population density')
    ax[1,1].set_xlabel('Time [years]', fontsize='small')
    ax[1,1].set_title('dt=0.1', fontsize='medium')

    fig.legend(fontsize='small')
    fig.suptitle('Comparison of Euler and RK8 methods with different\n'\
                'time steps for solving the L-V prey-predator equation', fontsize='medium')
    plt.show()

# Uncomment following lines to recreate plot from question #1
#q1_competition()
# q1_prey()

# ======================================================================

# Question 2

def q2_comp(a,b,c,d, n1_list, n2_list):
    '''    
    2x2 subplots of population vs time for competition equations. 
    All subplots use same L-V coefficients but different initial conditions.
    Parameters
    ----------
    n1_list : list
        list of initial values for population of species 1 (floats)
    n2_list : list
        list of initial values for population of species 2 (floats)
    a, b, c, d, : float
    Lotka-Volterra coefficients
    '''

    n1_color = 'red'
    n2_color = 'blue'

    dt_eu = 0.05

    # Titles for each subplot
    subplot_titles = ['a', 'b', 'c', 'd']

    # Create the subplots
    fig, ax = plt.subplots(2, 2)

    # Store line handles for the legend
    lines = []

    # Loop over the indices and plot the graphs
    for i in range(4):
        n1 = n1_list[i]
        n2 = n2_list[i]
        delta_n = n2 - n1
        
        # Run the RK8 and Euler solvers
        t_comp_rk, N1_comp_rk, N2_comp_rk = solve_rk8(dNdt_comp, N1_init=n1, N2_init=n2, a=a, b=b, c=c, d=d)
        t_comp_eu, N1_comp_eu, N2_comp_eu = euler_solve(dNdt_comp, dt=dt_eu, N1_init=n1, N2_init=n2, a=a, b=b, c=c, d=d)
        
        # Plot the lines and collect the handles and labels
        l1, = ax.flat[i].plot(t_comp_eu, N1_comp_eu, color=n1_color, label='N1 - Euler')
        l2, = ax.flat[i].plot(t_comp_eu, N2_comp_eu, color=n2_color, label='N2 - Euler')
        l3, = ax.flat[i].plot(t_comp_rk, N1_comp_rk, ':', color=n1_color, label='N1 - RK8')
        l4, = ax.flat[i].plot(t_comp_rk, N2_comp_rk, ':', color=n2_color, label='N2 - RK8')
        
        # Collect the line handles for the first plot only (to avoid duplicating in the legend)
        if i == 0:
            lines = [l1, l2, l3, l4]
        
        # Set labels and titles dynamically
        if i in [0, 2]:  # First column
            ax.flat[i].set_ylabel('Normalized population density', fontsize='small')
        if i in [2, 3]:  # Last row
            ax.flat[i].set_xlabel('Time [years]', fontsize='small')
        
        # Set the title dynamically with the correct subplot label
        ax.flat[i].set_title(f'{subplot_titles[i]}) N1_init={n1}, N2_init={n2}, \u0394n={delta_n:.1f}', fontsize='small')

    # Add the main title
    fig.suptitle(f'L-V Competition Equation for Varying Initial Population Conditions\n'
                f'a = {a}, b = {b}, c = {c}, d = {d}', fontsize='medium')

    # Add one legend for the entire figure
    fig.legend(lines, ['N1 - Euler', 'N2 - Euler', 'N1 - RK8', 'N2 - RK8'], 
            loc='upper right', fontsize='small')

    # Show the plot
    plt.show()


# Uncomment for plots from q2 competition: varying initial conditions
# q2_comp(a=1, b=2, c=1, d=3, n1_list=[0.3,0.4,0.2,0.3], n2_list=[0.5,0.6,0.6,0.7]) # different impacts, diff reprod. rate
# q2_comp(a=1, b=2, c=1, d=2, n1_list=[0.3,0.4,0.2,0.3], n2_list=[0.5,0.6,0.6,0.7]) # same impacts and reprod rate
# q2_comp(a=1, b=2, c=1, d=2, n1_list=[0.4,0.3,0.2,0.1], n2_list=[0.4,0.3,0.2,0.1]) # same coeff. same init. cond.
# q2_comp(a=1, b=2, c=1, d=2, n1_list=[0.5,0.6,0.7,0.8], n2_list=[0.5,0.6,0.7,0.8]) # same coeff. same init. cond.

# ----------------------------------------------------------------------------------------------

def q3_prey(a,b,c,d, n1_list, n2_list):
    ''' 
    2x2 subplots of population vs time for prey-predator equations.
    All subplots use same L-V coefficients but different initial conditions.
    Parameters
    ----------
    n1_list : list
        list of initial values for population of species 1 (floats)
    n2_list : list
        list of initial values for population of species 2 (floats)
    a, b, c, d, : float
    Lotka-Volterra coefficients
    '''

    n1_color = 'red'
    n2_color = 'blue'

    dt_eu = 0.005

    # Titles for each subplot
    subplot_titles = ['a', 'b', 'c', 'd']

    # Create the subplots
    fig, ax = plt.subplots(2, 2)

    # Store line handles for the legend
    lines = []

    # Loop over the indices and plot the graphs
    for i in range(4):
        n1 = n1_list[i]
        n2 = n2_list[i]
        delta_n = n2 - n1
        
        # Run the RK8 and Euler solvers
        t_prey_rk, N1_prey_rk, N2_prey_rk = solve_rk8(dNdt_prey, N1_init=n1, N2_init=n2, a=a, b=b, c=c, d=d)
        t_prey_eu, N1_prey_eu, N2_prey_eu = euler_solve(dNdt_prey, dt=dt_eu, N1_init=n1, N2_init=n2, a=a, b=b, c=c, d=d)
        
        # Plot the lines and collect the handles and labels
        l1, = ax.flat[i].plot(t_prey_eu, N1_prey_eu, color=n1_color, label='N1 - Euler')
        l2, = ax.flat[i].plot(t_prey_eu, N2_prey_eu, color=n2_color, label='N2 - Euler')
        l3, = ax.flat[i].plot(t_prey_rk, N1_prey_rk, ':', color=n1_color, label='N1 - RK8')
        l4, = ax.flat[i].plot(t_prey_rk, N2_prey_rk, ':', color=n2_color, label='N2 - RK8')
        
        # Collect the line handles for the first plot only (to avoid duplicating in the legend)
        if i == 0:
            lines = [l1, l2, l3, l4]
        
        # Set labels and titles dynamically
        if i in [0, 2]:  # First column
            ax.flat[i].set_ylabel('Normalized population density', fontsize='small')
        if i in [2, 3]:  # Last row
            ax.flat[i].set_xlabel('Time [years]', fontsize='small')
        
        # Set the title dynamically with the correct subplot label
        ax.flat[i].set_title(f'{subplot_titles[i]}) N1_init={n1}, N2_init={n2}, \u0394n={delta_n:.1f}', fontsize='small')

    # Add the main title
    fig.suptitle(f'L-V Prey-Predator Equation for Varying Initial Population Conditions\n'
                f'a = {a}, b = {b}, c = {c}, d = {d}', fontsize='medium')

    # Add one legend for the entire figure
    fig.legend(lines, ['N1 - Euler', 'N2 - Euler', 'N1 - RK8', 'N2 - RK8'], 
            loc='upper right', fontsize='small')

    # Show the plot
    plt.show()

def phase_diagram(a, b, c, d, n1_list, n2_list):
    '''Plots a single phase diagram for the L-V Prey-Predator model using RK8 approximation.
    Parameters
    ----------
    n1_list : list
        list of initial values for population of species 1 (floats)
    n2_list : list
        list of initial values for population of species 2 (floats)
    a, b, c, d, : float
    Lotka-Volterra coefficients
    '''
    
    # Define the color scheme
    colors = ['red', 'blue', 'green', 'purple']
    
    # Create the figure and axis
    fig, ax = plt.subplots()
    
    # Loop over the initial conditions and plot the RK8 solutions
    for i in range(len(n1_list)):
        n1 = n1_list[i]
        n2 = n2_list[i]
        
        # Run the RK8 solver for each set of initial conditions
        t_rk, N1_rk, N2_rk = solve_rk8(dNdt_prey, N1_init=n1, N2_init=n2, a=a, b=b, c=c, d=d)
        
        # Plot the phase diagram (N1 vs N2) for each trajectory
        ax.plot(N1_rk, N2_rk, color=colors[i], label=f'N1={n1}, N2={n2}')
    
    # Set axis labels
    ax.set_xlabel('Prey population (N1)')
    ax.set_ylabel('Predator population (N2)')
    
    # Add the title with coefficients
    ax.set_title(f'Phase Diagram for L-V Prey-Predator Model\n'
                 f'with a={a}, b={b}, c={c}, d={d}', fontsize='medium')
    
    # Add a legend to differentiate the lines by initial conditions
    ax.legend(loc='best', fontsize='medium')
    
    # Show the plot
    plt.show()


# Part 3 graphs and phase diagrams in same order as report

# q3_prey(a=1, b=1, c=1, d=1, n1_list=[0.4, 0.5, 0.6, 0.7], n2_list=[0.4, 0.5, 0.6, 0.7])
# phase_diagram(a=1, b=1, c=1, d=1, n1_list=[0.4, 0.5, 0.6, 0.7], n2_list=[0.4, 0.5, 0.6, 0.7])

# q3_prey(a=1, b=1, c=1, d=1, n1_list=[0.4, 0.4, 0.4, 0.4], n2_list=[0.4, 0.5, 0.6, 0.7])
# phase_diagram(a=1, b=1, c=1, d=1, n1_list=[0.4, 0.4, 0.4, 0.4], n2_list=[0.4, 0.5, 0.6, 0.7])

# phase_diagram(a=1, b=1, c=1, d=1, n1_list=[0.4,0.5,0.6,0.7], n2_list=[0.4, 0.4, 0.4, 0.4])

# phase_diagram(a=3, b=1, c=1, d=1, n1_list=[0.4, 0.5, 0.6, 0.7], n2_list=[0.4, 0.5, 0.6, 0.7])

# phase_diagram(a=1, b=1, c=3, d=1, n1_list=[0.4, 0.5, 0.6, 0.7], n2_list=[0.4, 0.5, 0.6, 0.7])

# phase_diagram(a=1, b=3, c=1, d=1, n1_list=[0.4, 0.5, 0.6, 0.7], n2_list=[0.4, 0.5, 0.6, 0.7])

# phase_diagram(a=1, b=1, c=1, d=3, n1_list=[0.4, 0.5, 0.6, 0.7], n2_list=[0.4, 0.5, 0.6, 0.7])