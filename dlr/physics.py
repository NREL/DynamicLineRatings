# -*- coding: utf-8 -*-
"""
References
* IEEE 738-2012 https://doi.org/10.1109/IEEESTD.2013.6692858
* Bartos et al 2016 https://dx.doi.org/10.1088/1748-9326/11/11/114008
"""

###### Imports
import numpy as np
import pandas as pd
import math

###### Constants
## Stefan-Boltzmann constant [W m^-2 K^-4]
STEFAN_BOLTZMANN = 5.67e-8
## Celsius to Kelvin
C2K = 273.15


###### Functions
### Solar heating ###
def solar_heating(solar_ghi, diameter_conductor, absorptivity_conductor):
    """Calculate solar heating of conductor [W m^-1]

    Args:
        solar_ghi (numeric): global horizonal solar irradiance [W m^-2]
        diameter_conductor (numeric): conductor diameter [m]
        absorptivity_conductor (numeric): absorptivity of conductor surface - Page 12 of Bartos [.]

    Returns:
        numeric: solar heating of conductor [W m^-1]
    """
    ## [W m^-2] * [m] * [.]
    out = solar_ghi * diameter_conductor * absorptivity_conductor
    ## -> [W m^-1]

    return out


### Radiative cooling ###
def radiative_cooling(diameter_conductor, emissivity_conductor, temp_surface, temp_ambient_air):
    """Calculate radiative cooling of conductor [W m^-1]

    Args:
        diameter_conductor (numeric): conductor diameter [m]
        emissivity_conductor (numeric): emissivity of conductor surface [.]
            - 0.7 page 8 of Bartos
        temp_surface (numeric): conductor surface temperature [K]
        temp_ambient_air (numeric): _description_

    Returns:
        numeric: radiative cooling of conductor [W m^-1]
    """
    out = (
        ## [.] * [m]
        math.pi
        * diameter_conductor
        ## * [.] * [W m^-2 K^-4]
        * emissivity_conductor
        * STEFAN_BOLTZMANN
        ## * [K^4]
        * (temp_surface ** 4 - temp_ambient_air ** 4)
    )
    ## -> [W m^-1]

    return out


### Convective cooling ###
def get_film_temperature(temp_conductor, temp_ambient_air):
    """Calculate average temperature of boundary layer

    Args:
        temp_conductor (numeric): Conductor surface temperature [K]
        temp_ambient_air (numeric): Ambient air temperature [K]

    Returns:
        T_film (numeric): Average temperature of boundary layer [K]
    """
    T_film = (temp_conductor + temp_ambient_air) / 2
    return T_film


def get_dynamic_viscosity(T_film):
    """Calculate dynamic viscosity of air using IEEE Std 738-2012 (13a)

    Args:
        T_film (numeric): Average temperature of boundary layer [K]
    Returns:
        dynamic_viscosity (numeric): Dynamic viscosity of air [kg/(m s)] or [(N s)/(m^2)]
    """
    dynamic_viscosity = 1.458e-6 * (T_film**1.5) / (T_film - C2K + 383.4)
    return dynamic_viscosity


def get_air_density(elevation, T_film):
    """Calculate air density using IEEE Std 738-2012 (14a)

    Args:
        elevation (numeric): Elevation of conductor above sea level [m]
        T_film (numeric): Average temperature of boundary layer [K]

    Returns:
        density_air (numeric): Air density [kg/(m^3)]
    """
    density_air = (1.293 - 1.525e-4 * elevation + 6.379e-9 * elevation**2) / (
        1 + 0.00367 * (T_film - C2K)
    )
    return density_air


def get_thermal_conductivity_air(T_film):
    """Calculate thermal conductivity of air using IEEE Std 738-2012 (15a)

    Args:
        T_film (numeric): Average temperature of boundary layer [K]

    Returns:
        thermal_conductivity_air (numeric): Thermal conductivity of air [W/(m °C)]
    """
    thermal_conductivity_air = (
        2.424e-2 + 7.477e-5 * (T_film - C2K) - 4.407e-9 * (T_film - C2K) ** 2
    )
    return thermal_conductivity_air


### Wind direction factor
def get_wind_direction_factor(phi):
    """Calculate wind direction factor from angle between wind and conductor axis
    using IEEE 738-2012 (4a)

    Args:
        phi (numeric): angle between the wind direction and the axis of the conductor in degrees

    Returns:
        numeric: Wind direction factor
    """
    ## Wrap around 90
    phi_in_180 = np.remainder(abs(phi), 180)
    phi_in_90 = 90 - abs(phi_in_180 - 90)
    ## Convert to radians for calculation
    _phi = np.deg2rad(phi_in_90)
    ## Calculate it
    K_angle = 1.194 - np.cos(_phi) + 0.194 * np.cos(2 * _phi) + 0.368 * np.sin(2 * _phi)
    ## Check it
    if isinstance(K_angle, float):
        assert K_angle <= 1
    else:
        assert all(K_angle <= 1)

    return K_angle


def get_wind_direction_factor_perpendicular(beta):
    """Calculate wind direction factor from angle between wind and perpendicular to
    conductor axis using IEEE 738-2012 (4b)

    Args:
        beta (numeric): angle between the wind direction and perpendicular
        to conductor axis in degrees

    Returns:
        numeric: Wind direction factor
    """
    ## Wrap around 90
    beta_in_180 = np.remainder(abs(beta), 180)
    beta_in_90 = 90 - abs(beta_in_180 - 90)
    ## Convert to radians for calculation
    _beta = np.deg2rad(beta_in_90)
    ## Calculate it
    K_angle = (
        1.194 - np.sin(_beta) - 0.194 * np.cos(2 * _beta) + 0.368 * np.sin(2 * _beta)
    )
    ## Check it
    if isinstance(K_angle, float):
        assert K_angle <= 1
    else:
        assert all(K_angle <= 1)

    return K_angle


### Air density
def p_wv_sat_tetens(T):
    """
    Determines the saturation vapor pressure of water from the temperature
    * input units: Kelvin
    * output units: Pa

    https://en.wikipedia.org/wiki/Vapour_pressure_of_water
    """
    t = T - C2K
    out = 610.78 * np.exp((17.27 * t) / (t + 237.3))
    return out


def p_wv(humidity, T):
    """
    Determines the partial pressure of water vapor from the
    relative humidity and temperature

    units
    -----
    humidity: percent [%]
    T:        Kelvin  [K]
    output:   Pascal  [Pa]
    """
    # return humidity * p_wv_sat_buck(T)
    out = 0.01 * humidity * p_wv_sat_tetens(T)
    return out


def get_air_density_ideal(
    pressure,
    temperature,
    humidity=0,
    R_dryair=287.058,
    R_wv=461.495,
):
    """Calculate air density using the ideal gas law"""
    p_wv_val = p_wv(humidity, temperature)
    density_total = (
        (pressure - p_wv_val) / R_dryair / temperature
    ) + (
        p_wv_val / R_wv / temperature
    )
    return density_total


def get_reynolds_number(diameter_conductor, density_air, windspeed, dynamic_viscosity):
    """Calculate Reynolds number using IEEE 738-2012 (2c)

    Args:
        diameter_conductor (numeric): _description_
        density_air (numeric): _description_
        windspeed (numeric): _description_
        dynamic_viscosity (numeric): _description_

    Returns:
        reynolds_number: _description_
    """
    reynolds_number = diameter_conductor * density_air * windspeed / dynamic_viscosity
    return reynolds_number


### Actual convective cooling equation
def convective_cooling_ieee(
    temp_conductor,
    diameter_conductor,
    windspeed,
    wind_direction_factor,
    temp_ambient_air,
    density_air,
):
    """Calculate convective cooling rate.
    Formulation is from:
        "IEEE Standard for Calculating the Current-Temperature Relationship of Bare
        Overhead Conductors," in IEEE Std 738-2012 (Revision of IEEE Std 738-2006 -
        Incorporates IEEE Std 738-2012 Cor 1-2013) , vol., no., pp.1-72, 23 Dec. 2013
        (https://ieeexplore.ieee.org/document/6692858)

    Args:
        density_air (numeric): density of air - function by temperature and pressure
        windspeed (numeric): speed of air stream at conductor - wind speed
        thermal_conductivity_air (numeric): thermal conductivity of air
            at temperature T_film (page 6 of Bartos) - page 7 of Bartos
        temp_conductor (numeric): conductor temperature - 75 Celcius
        temp_ambient_air (numeric): ambient air temperature (use the 100m one as approximation)
        diameter_conductor (numeric): conductor diameter [m]
        dynamic_viscosity (numeric): dynamic viscosity
        wind_direction_factor (numeric): _description_

    Returns:
        numeric: convective cooling of conductor [W m^-1]
    """
    # density_air: density of air [kg/m^3] (rho_f in IEEE)
    # wind_direction_factor: wind direction factor [.] (K_angle in IEEE)
    # diameter_conductor: Outside diameter of conductor [m] (D_0 in IEEE)
    # temp_conductor: Conductor surface temperature [°C] (T_s in IEEE)
    # temp_ambient_air: Ambient air temperature [°C]
    # reynolds_number: Dimensionless Reynolds number = D_0 * rho_f * V_w / mu_f [.] (N_Re in IEEE)
    # thermal_conductivity_air: Thermal conductivity of air at temperature T_film [W/(m °C)] (k_f in IEEE)
    # dynamic_viscosity: Absolute (dynamic) viscosity of air [kg/(m s)] (mu_f in IEEE)
    # windspeed: Speed of air stream at conductor [m/s] (V_w in IEEE)
    T_film = get_film_temperature(temp_conductor, temp_ambient_air)
    thermal_conductivity_air = get_thermal_conductivity_air(T_film)
    dynamic_viscosity = get_dynamic_viscosity(T_film)
    reynolds_number = get_reynolds_number(
        diameter_conductor, density_air, windspeed, dynamic_viscosity
    )

    # [W/m]
    convection_zero_windspeed = (
        3.645
        * density_air**0.5
        * diameter_conductor**0.75
        * (temp_conductor - temp_ambient_air) ** 1.25
    )
    # [W/m]
    convection_low_windspeed = (
        wind_direction_factor
        * (1.01 + 1.35 * reynolds_number**0.52)
        * thermal_conductivity_air
        * (temp_conductor - temp_ambient_air)
    )
    # [W/m]
    convection_high_windspeed = (
        wind_direction_factor
        * 0.754
        * reynolds_number**0.6
        * thermal_conductivity_air
        * (temp_conductor - temp_ambient_air)
    )
    ### Take maximum across all values
    if isinstance(convection_low_windspeed, float):
        out = max(convection_zero_windspeed,
                  convection_low_windspeed,
                  convection_high_windspeed)
    else:
        out = pd.DataFrame(
            {
                "zero": convection_zero_windspeed,
                "low": convection_low_windspeed,
                "high": convection_high_windspeed,
            }
        ).max(axis=1)

    return out


### Ampacity considering all heating and cooling types
def ampacity(
    windspeed=0.61,
    pressure=101325,
    temp_ambient_air=40+C2K,
    wind_conductor_angle=90,
    solar_ghi=1000,
    temp_conductor=75+C2K,
    diameter_conductor=0.02814,
    resistance_conductor=8.688e-5,
    emissivity_conductor=0.8,
    absorptivity_conductor=0.8,
    forecast_margin={},
    check_units=True,
):
    """Calculate ampacity as a function of weather and conductor parameters.

    Args:
        windspeed (numeric): Windspeed [m/s]
        pressure (numeric): Air pressure [Pa]
        temp_ambient_air (numeric): Ambient air temperature [K]
        wind_conductor_angle (numeric): Angle between wind direction and line segment [°]
        solar_ghi (numeric): Solar global horizontal irradiance [W m^-2]
        temp_conductor (float): Maximum allowable temperature of conductor [K].
            Default of 75°C + C2K = 348.15 K is a rule of thumb for ACSR conductors.
        diameter_conductor (float): Diameter of conductor [m]
        resistance_conductor (float): Resistance of conductor [Ω/m]
        absorptivity_conductor (float): Absorptivity of conductor. Defaults to 0.8.
        emissivity_conductor (float): Emissivity of conductor. Defaults to 0.8.

    Returns:
        current (numeric): Rated ampacity [A]
    """
    ### Check units (in case °C or kPa are provided instead of K and Pa)
    if check_units:
        check_params = dict(zip(
            ['pressure', 'temp_ambient_air', 'temp_conductor'],
            [pressure, temp_ambient_air, temp_conductor],
        ))
        lowest_realistic_value = {
            'pressure': 40e3,
            'temp_ambient_air': C2K - 90,
            'temp_conductor': C2K,
        }
        for name, param in check_params.items():
            err = f"{name} should be above {lowest_realistic_value[name]}; check units"
            if isinstance(param, pd.DataFrame):
                if (param < lowest_realistic_value[name]).any().any():
                    raise Exception(err)
            elif isinstance(param, (pd.Series, np.ndarray)):
                if (param < lowest_realistic_value[name]).any():
                    raise Exception(err)
            else:
                if param < lowest_realistic_value[name]:
                    raise Exception(err)

    ### Modify inputs by forecast margins, if provided
    if len(forecast_margin):
        forecast_params = dict(zip(
            ['windspeed','pressure','temp_ambient_air','wind_conductor_angle','solar_ghi'],
            [windspeed, pressure, temp_ambient_air, wind_conductor_angle, solar_ghi],
        ))
        for key in forecast_margin.keys():
            assert key in forecast_params, (
                f"forecast_margin keys must be in {','.join(forecast_params.keys())} "
                f"but {key} was provided"
            )
            forecast_params[key] += forecast_margin[key]
            if key in ['windspeed','pressure','temp_ambient_air','solar_ghi']:
                if isinstance(forecast_params[key], (pd.Series, pd.DataFrame)):
                    forecast_params[key] = forecast_params[key].clip(lower=0)
                elif isinstance(forecast_params[key], (np.ndarray)):
                    forecast_params[key] = forecast_params[key].clip(min=0)
                else:
                    forecast_params[key] = max(forecast_params[key], 0)

    ### Process other inputs
    phi = abs(wind_conductor_angle)

    T_film = get_film_temperature(temp_conductor, temp_ambient_air)
    density_air = get_air_density_ideal(
        pressure=pressure,
        temperature=T_film,
        humidity=0.5,
    )
    wind_direction_factor = get_wind_direction_factor(phi)

    ### Convective Cooling
    q_c = convective_cooling_ieee(
        temp_conductor=temp_conductor,
        diameter_conductor=diameter_conductor,
        windspeed=windspeed,
        wind_direction_factor=wind_direction_factor,
        temp_ambient_air=temp_ambient_air,
        density_air=density_air,
    )

    ### Radiative Cooling
    q_r = radiative_cooling(
        diameter_conductor=diameter_conductor,
        emissivity_conductor=emissivity_conductor,
        temp_surface=temp_conductor,
        temp_ambient_air=temp_ambient_air,
    )

    ### Solar Heating
    q_s = solar_heating(
        solar_ghi=solar_ghi,
        diameter_conductor=diameter_conductor,
        absorptivity_conductor=absorptivity_conductor,
    )

    ### Ampacity for equivalent rate of Joule heating
    current = np.sqrt((q_c + q_r - q_s) / resistance_conductor)

    return current
