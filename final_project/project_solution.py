import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

plt.style.use('bmh')

# to reproduce figures from report, uncomment lines that call functions

# load data or paste it from file
# co2_data = np.loadtxt('co2_concentrations.txt')
co2_data = np.array((315.98, 316.91, 317.64, 318.45, 318.99, 319.62, 320.04, 321.37, 
                     322.18, 323.05, 324.62, 325.68, 326.32, 327.46, 329.68, 330.19, 
                     331.13, 332.03, 333.84, 335.41, 336.84, 338.76, 340.12, 341.48, 
                     343.15, 344.87, 346.35, 347.61, 349.31, 351.69, 353.20, 354.45, 
                     355.70, 356.54, 357.21, 358.96, 360.97, 362.74, 363.88, 366.84, 
                     368.54, 369.71, 371.32, 373.45, 375.98, 377.70, 379.98, 382.09, 
                     384.02, 385.83, 387.64, 390.10, 391.85, 394.06, 396.74, 398.81, 
                     401.01, 404.41, 406.76, 408.72, 411.65, 414.21, 416.41, 418.53, 
                     421.08))

# create array of years from 1959 to 2023 (2024 not included)
yrs = np.arange(1959,2024,1)

# define functions that will be used for calculations:

def rad_forcing(co2):
    '''
    parameters
    ----------
    co2: array
        co2 concentrations, first value is used as c_0

    returns
    -------
    Q: array
        radiative forcing due to CO2, in units of W/m^2
    '''
    c_0 = co2[0]
    return 5.35 * np.log(co2/c_0)

def trapezoidal_rule(integrand, time):
    """
    Compute the integral using the trapezoidal rule.

    Parameters:
    integrand: array
        values of the function to integrate
    time: array
        corresponding time points in seconds

    Returns:
    Integral value
    """

    h = time[1] - time[0]  # Assumes uniform spacing
    return h * (0.5 * integrand[0] + np.sum(integrand[1:-1]) + 0.5 * integrand[-1])


def simpsons_rule(integrand, time):
    """
    Compute the integral using the Simpson's 1/3 rule.

    Parameters:
    integrand: array
        values of the function to integrate
    time: array
        corresponding time points in seconds

    Returns:
    Integral value
    """
    n = len(time)  # Number of data points (must be odd for Simpson's Rule)
    if n % 2 != 1:
        raise ValueError("Number of data points must be odd for Simpson's Rule.")
    
    h = time[1] - time[0]  # Step size (assuming uniform spacing)
    
    # Compute the sum of function values at endpoints and interior points
    result = integrand[0] + integrand[-1]  # First and last terms
    
    # Sum the interior points with appropriate coefficients (4 for odd, 2 for even indices)
    for i in range(1, n-1):
        result += (4 if i % 2 != 0 else 2) * integrand[i]
    
    # Multiply by the step size divided by 3 to get the final result
    result *= h / 3
    return result

def trans_temp_resp(dz, sensit, co2, time, method='trapz'):
    '''
    Calculates the transient temperature response due to forcing Q.

    Parameters:
    dz: float
        mixed layer depth of the ocean in meters
    sensit: float
        climate sensitivity parameter in K/(W/m^2)
    co2: array
        co2 concentrations
    time: array
        time in seconds that matches co2 concentrations
    method: str
        must be either 'trapz' or 'simpson'. decides which method
        of integration to use
    
    Returns
    responses: array
        temperature responses with respect to initial point
    '''

    responses = np.zeros(time.size)

    # set variables present in integrand
    Q = rad_forcing(co2)
    cw = 4218 # heat capacity of water [J K^-1 kg^-1]
    pw = 1000 # density of water
    ce = cw * pw * dz
    tao =  ce * sensit
    dt = time[1] - time [0]
    # calculate integrand
    integrand = Q * np.exp(time/tao)/ ce

    for i in range(len(time)):
        if method == 'trapz':
            integral = trapezoidal_rule(integrand=integrand[:i+2], time=time[:i+2])
        elif method == 'simpson':
            # Apply Simpson's rule only for an odd number of points
            if (i + 1) % 2 == 1 and i >= 2:  # Ensure at least 3 points for Simpson's rule
                integral = simpsons_rule(integrand=integrand[:i+1], time=time[:i+1])
            else:
                integral = 0

        exp = np.exp(-time[i]/tao)
        responses[i] = exp * integral

    return responses

def eq_temp(Q, sensit):
    '''
    Calculates the equilibrium temperature based on climate sensitivity
    and radiative forcing

    Parameters
    Q: array
        radiative forcing in W/m^2
    sensit: float
        climate sensitivity parameter in K/(W/m^2)

    returns
    Equilibrium Temperature change
    '''
    # equilibrium temp equals sensitivity times radiative forcing
    return Q * sensit

# =============================================================================

# First plot CO2 concentrations to visualize data

def plot_co2():
    '''Plots CO2 concentrations'''

    fig, ax = plt.subplots(1,1)
    ax.plot(yrs, co2_data, '.')
    ax.set_xlabel('years')
    ax.set_ylabel('CO2 concentrations [ppm]')
    plt.show()

# plot_co2()

# =============================================================================

# We want to predict CO2 concentrations for future years
# Let's find the line of best fit using degree 1 and 2 polynomials
# then compare R2 values to decided which one fits better

future_yrs = np.arange(1959, 2101) # new time array

# Linear regression (degree 1)
coeffs_1 = np.polyfit(yrs, co2_data, 1) # Fit a first degree polynomial
line_1 = np.polyval(coeffs_1, yrs) # Model data for known years
predictions_1 = np.polyval(coeffs_1, future_yrs) # Predict future values

# Quadratic regression (degree 2)
coeffs_2 = np.polyfit(yrs, co2_data, 2)
line_2 = np.polyval(coeffs_2, yrs)
predictions_2 = np.polyval(coeffs_2, future_yrs)

# Calculate R^2 values
r2_1 = r2_score(co2_data, line_1)
r2_2 = r2_score(co2_data, line_2)

# Plot data and predictions, and print R2 values:
def plot_predicted_co2():
    '''Plots Best-fit polynomials alongside observed data.
    prints equations and R2 values to screen
    '''
    # Plot data and predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(yrs, co2_data, color="blue", label="Observed Data", s=10)
    plt.plot(future_yrs, predictions_1, label="Linear Fit (Degree 1)", color="red")
    plt.plot(future_yrs, predictions_2, label="Quadratic Fit (Degree 2)", color="green")
    plt.xlabel("Year")
    plt.ylabel("CO₂ Concentration (ppm)")
    plt.title("CO₂ Concentrations and Predictions")
    plt.legend()
    plt.grid()
    plt.show()

    # Print equations and R2 values
    print(f"Linear Fit (Degree 1): y = {coeffs_1[0]:.2f}x + {coeffs_1[1]:.2f}")
    print(f"Quadratic Fit (Degree 2): y = {coeffs_2[0]:.6f}x² + {coeffs_2[1]:.2f}x + {coeffs_2[2]:.2f}")
    print(f"R² value for Linear Fit (Degree 1): {r2_1:.4f}")
    print(f"R² value for Quadratic Fit (Degree 2): {r2_2:.4f}")

# plot_predicted_co2()

# Quadratic fit has a higher R2 value, so we will use it from  now on

# Generate a CO2 value per year using the quadratic fit (1959-2100):
co2_modeled = np.polyval(coeffs_2, future_yrs)

# =============================================================================

# Using modeled concentrations, calculate radiative forcing:

Q = rad_forcing(co2_modeled)

def plot_rad_forcing():
    '''Plots Rad. forcing
    '''
    fig, ax = plt.subplots(1,1)
    ax.plot(future_yrs, Q)
    ax.set_xlabel('years')
    ax.set_ylabel(r'Q [$W m^{-2}$]')
    ax.set_title('Radiative forcing of CO2')
    plt.show()

# plot_rad_forcing()

# =============================================================================

# Calculate Transient Climate response using Q:

# First, compare integration methods:
t_sec = future_yrs * 3.154*10**7

def plot_compare_integration():
    '''Compares integration methods for dz=200 and sensit=0.5'''
    dz = 200
    sensit = 0.5
    temp_response_tp = trans_temp_resp(dz=dz, sensit=sensit, co2=co2_modeled, time=t_sec,
                                method='trapz')
    temp_response_sp = trans_temp_resp(dz=dz, sensit=sensit, co2=co2_modeled, time=t_sec,
                                method='simpson')
    # Simpson method onl works with odd number of steps so it skips every other year
    # Plot only the nonzero values in array:
    nonzero_ind = np.nonzero(temp_response_sp)[0]
    temp_response_sp = temp_response_sp[nonzero_ind]
    yrs_sp = future_yrs[nonzero_ind]

    fig, ax = plt.subplots(1,1)

    ax.plot(future_yrs, temp_response_tp, label='Trapz')
    ax.plot(yrs_sp, temp_response_sp, label='Simpson')
    ax.set_xlabel('Years')
    ax.set_ylabel('Temperature Response [K]')
    fig.suptitle(f'Transient Climate Response (dz={dz}m, λ={sensit}W/m2)')
    fig.legend(loc='upper left')
    plt.show()

# plot_compare_integration()

# =============================================================================

# how does dz and lambda affect T, respectively?

dz = np.array((50,200,350,500))
sensit = np.array((0.2,0.5,0.8,1))
method = np.array(('trapz', 'simpson'))

def vary_dz():
    '''Plots responses for varying dz's while sensit=1'''
    yrs = future_yrs
    fig, ax = plt. subplots(1,2)
    for j in range(2):
        for i in range(len(dz)):
            temp_resp = trans_temp_resp(dz=dz[i], sensit=1, co2=co2_modeled, time=t_sec,method=method[j])
            if method[j]=='simpson':
                nonzero_ind = np.nonzero(temp_resp)[0]
                temp_resp = temp_resp[nonzero_ind]
                yrs = future_yrs[nonzero_ind]
            ax[j].plot(yrs, temp_resp, label=f'dz={dz[i]}, λ=1')
            ax[j].set_xlabel('years')
            ax[j].set_ylabel('Temperature Response [K]')
            ax[j].set_title(f'Method: {method[j]}')
        if j == 0:
            fig.legend()
    fig.suptitle('Transient Climate Responses with varying dz')
    plt.show()

def vary_lambda():
    '''Plots responses for varying lambdas's while dz=200'''
    yrs = future_yrs
    fig, ax = plt. subplots(1,2)
    for j in range(2):
        for i in range(len(sensit)):
            temp_resp = trans_temp_resp(dz=200, sensit=sensit[i], co2=co2_modeled, time=t_sec,method=method[j])
            if method[j]=='simpson':
                nonzero_ind = np.nonzero(temp_resp)[0]
                temp_resp = temp_resp[nonzero_ind]
                yrs = future_yrs[nonzero_ind]
            ax[j].plot(yrs, temp_resp, label=f'λ={sensit[i]}, dz=200')
            ax[j].set_xlabel('years')
            ax[j].set_ylabel('Temperature Response [K]')
            ax[j].set_title(f'Method: {method[j]}')
        if j == 0:
            fig.legend()
    fig.suptitle('Transient Climate Responses with varying λ')
    plt.show()

# vary_dz()
# vary_lambda()

# =============================================================================

# If humans stopped emitting co2 to the atmosphere in 2023, how would temp
# response look?

#find index of year 2023:
ind = np.where(future_yrs == 2023)[0][0]

co2_const23 = co2_modeled.copy()
co2_const23[ind:] = co2_modeled[ind] # new co2 concentrations

dz = np.array((70,70,300,300))
sensit = np.array((0.4,1.0,0.4,1.0))

def plot_trapz_eq():
    '''Plots temp response for the constant co2 after 2023 using
    trapz integration method for each scenario defined in lab
    '''
    fig, ax = plt.subplots(1,1)
    for i in range(len(dz)):
        temp_resp = trans_temp_resp(dz=dz[i], sensit=sensit[i], co2=co2_const23, time=t_sec, method='trapz')
        equilibrium_temp = eq_temp(rad_forcing(co2_const23)[ind], sensit=sensit[i])
        print(f'''scenario {i+1}:
        Temp in 2100 = {temp_resp[-2]} K
            warming left after 2023 = {temp_resp[-2] - temp_resp[ind]}
            calculated equilibrium temp = {equilibrium_temp}
            Warming left until equilib = {equilibrium_temp - temp_resp[-2]}
        ''')
        ax.plot(future_yrs[:-1], temp_resp[:-1], label=f'{i+1}) dz={dz[i]}, λ={sensit[i]}')

    ax.axvline(2023, color='black', linestyle='dashed', alpha=0.5)
    ax.set_xlabel('years')
    ax.set_ylabel('Temperature Response [K]')
    ax.set_title('''Temperature Response after halted emissions in 2023
                 Integration method: Trapezoidal''')

    fig.legend()
    plt.show()

# plot_trapz_eq()

def plot_simp_eq():
    '''Plots temp response for the constant co2 after 2023 using
    simpson integration method for each scenario defined in lab
    '''

    fig, ax = plt.subplots(1,1)
    for i in range(len(dz)):
        temp_resp = trans_temp_resp(dz=dz[i], sensit=sensit[i], co2=co2_const23, time=t_sec, method='simpson')
        equilibrium_temp = eq_temp(rad_forcing(co2_const23)[ind], sensit=sensit[i])
        nonzero_ind = np.nonzero(temp_resp)[0]
        temp_resp = temp_resp[nonzero_ind]
        yrs = future_yrs[nonzero_ind]
        print(f'''scenario {i+1}:
        Temp in 2100 = {temp_resp[-1]} K
            warming left after 2023 = {temp_resp[-1] - temp_resp[ind]}
            calculated equilibrium temp = {equilibrium_temp}
            Warming left until equilib = {equilibrium_temp - temp_resp[-1]}
        ''')
        ax.plot(yrs, temp_resp, label=f'{i+1}) dz={dz[i]}, λ={sensit[i]}')

    ax.axvline(2023, color='black', linestyle='dashed', alpha=0.5)
    ax.set_xlabel('years')
    ax.set_ylabel('Temperature Response [K]')
    ax.set_title('''Temperature Response after halted emissions in 2023
                 Integration method: Simpson''')

    fig.legend()
    plt.show()

# plot_simp_eq()

# we see that simpson is overall a better method and gives correct solutions
# to three of the scenarios and we can see what the equilibrium temperatures 
# and how much warming is left after stopping emissions

# =============================================================================

# carbon sequestration from atmosphere. reduce concentration by 5 ppm each year
# starting in 2023

co2_sequestred = co2_modeled.copy()

for i in range(ind, len(co2_sequestred)):
    co2_sequestred[i] -= 5 * (i - ind + 1) # new co2 concentration

def plot_seq_co2_response():
    '''Plots new CO2 concentrations after sequestration, as well
    as the temperature response.
    prints to screen when temperature change becomes negative.    
    '''
    fig, ax = plt.subplots(1,1)
    ax.plot(future_yrs,co2_modeled, label='normal concentrations')
    ax.plot(future_yrs, co2_sequestred, label='sequestred concentrations')
    ax.set_xlabel('years')
    ax.set_ylabel('CO2 concentrations [ppm]')
    ax.set_title('CO2 concentrations after sequestration of 5ppm per yr')
    fig.legend()
    plt.show()

    temp_resp_nrml = trans_temp_resp(dz=70, sensit=1, co2=co2_modeled, time=t_sec, method='simpson')
    temp_resp_sqtd = trans_temp_resp(dz=70, sensit=1, co2=co2_sequestred, time=t_sec, method='simpson')
    nonzero_ind = np.nonzero(temp_resp_nrml)[0]
    temp_resp_nrml = temp_resp_nrml[nonzero_ind]
    temp_resp_sqtd =  temp_resp_sqtd[nonzero_ind]
    yrs = future_yrs[nonzero_ind]

    print(f'Temperature starts decreasing in {yrs[np.where(temp_resp_sqtd<0)[0][0]]}')

    fig, ax = plt.subplots(1,1)
    ax.plot(yrs, temp_resp_nrml, label='normal')
    ax.plot(yrs, temp_resp_sqtd, label='post-sequestration')
    ax.set_xlabel('years')
    ax.set_ylabel('Temperature Response [K]')
    ax.set_title('Temperature Response before and after start of Carbon Sequestration')
    fig.legend()
    plt.show()

# plot_seq_co2_response()