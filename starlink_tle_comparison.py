# Load standard modules
import numpy as np
import requests
import matplotlib
matplotlib.use('Agg') 
from matplotlib import pyplot as plt

from datetime import datetime, timedelta, timezone
from typing import List

# Load tudatpy modules
from tudatpy.interface import spice
from tudatpy import dynamics
from tudatpy.dynamics import environment
from tudatpy.dynamics import environment_setup, propagation_setup, simulator
from tudatpy.astro import element_conversion, frame_conversion, time_conversion
from tudatpy import constants
from tudatpy.util import result2array
from tudatpy.astro.time_representation import DateTime

from sgp4.api import Satrec, jday
from datetime import datetime, timedelta

spice.load_standard_kernels()


def tle_epoch_to_ymdhms(epoch: float):
    # Split year and day-of-year
    year = int(epoch // 1000)      # first two digits (YY)
    doy_fractional = epoch % 1000  # DDD.DDDDDD
    
    # Expand year to 4 digits (TLE convention: year < 57 → 2000s, else 1900s)
    year += 2000 if year < 57 else 1900

    # Separate day and fractional part
    day_of_year = int(doy_fractional)
    fractional_day = doy_fractional - day_of_year
    
    # Start of year
    start_of_year = datetime(year, 1, 1)
    
    # Compute datetime
    dt = start_of_year + timedelta(days=day_of_year - 1,
                                   seconds=fractional_day * 86400.0)
    
    return dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second


def generate_query_times(anchor_time: datetime, lookback: timedelta, step: timedelta) -> List[datetime]:
    """
    Generate a list of UTC datetimes spanning backwards from `anchor_time`
    at fixed intervals, covering the specified lookback window.
    """
    if step <= timedelta(0):
        raise ValueError("step must be positive")

    if lookback <= timedelta(0):
        raise ValueError("lookback must be positive")

    # Normalize anchor time to UTC
    if anchor_time.tzinfo is None:
        anchor_time = anchor_time.replace(tzinfo=timezone.utc)
    else:
        anchor_time = anchor_time.astimezone(timezone.utc)

    earliest_time = anchor_time - lookback

    times = []
    t = anchor_time

    while t >= earliest_time:
        times.append(t)
        t -= step

    return list(reversed(times))


if __name__ == "__main__":
    # Starlink-32174 TLE (NORAD ID: 61509)
    STARLINK_ID = "1262" #31728
    # TODO integrate api and time function to extract TLEs automatically
    now = datetime.now(timezone.utc)
    # now = datetime(2025, 11, 10, 12, 0, 0, tzinfo=timezone.utc)

    # Generate query times
    times = generate_query_times(
        anchor_time=now,
        lookback=timedelta(days=10),
        step=timedelta(days=1),
    )

    # Convert to Satcat-compatible date strings
    query_dates = [t.strftime("%Y-%m-%d") for t in times]

    BASE_URL = "https://www.satcat.com/api/sats/tles"

    tles = []
    for date in query_dates:
        params = {
            # "q": 61509,     # NORAD ID
            "q": 45366,
            "date": date    # ISO date string
        }

        response = requests.get(BASE_URL, params=params)

        # Raise an error if the request failed
        response.raise_for_status()

        data = response.json()

        raw_tle = data[0]["tle"]
        line1, line2 = raw_tle.splitlines()[1:]
        tles.append((line1, line2))


    tudat_pos_error = []
    sgp4_pos_error = []
    time_diffs = []

    target_tle = tles[-1]
    for tle in tles[:-1]:
        t0_line_1, t0_line_2 = tle
        t1_line_1, t1_line_2 = target_tle

            
        # === SGP4 Propagation ===
        # Create a satellite record
        sat = Satrec.twoline2rv(t0_line_1, t0_line_2)

        # Extract epochs from TLEs
        # Start time
        epoch0 = float(t0_line_1[18:32])
        year0, month0, day0, hour0, minute0, second0 = tle_epoch_to_ymdhms(epoch0)
        jd0, fr0 = jday(year0, month0, day0, hour0, minute0, second0)

        # End time
        epoch1 = float(t1_line_1[18:32])
        year1, month1, day1, hour1, minute1, second1 = tle_epoch_to_ymdhms(epoch1)
        jd1, fr1 = jday(year1, month1, day1, hour1, minute1, second1)

        error_sgp4, prop_position1_sgp4, prop_velocity1_sgp4 = sat.sgp4(jd1, fr1)

        print(f"Epoch0: {year0}-{month0}-{day0} {hour0}:{minute0}:{second0}")
        print(f"Epoch1: {year1}-{month1}-{day1} {hour1}:{minute1}:{second1}")

        # === Tudat Propagation ===
        # Convert to position and velocity state in TEME frame 
        error, position0_teme, velocity0_teme = sat.sgp4(jd0, fr0)

        # if error == 0:
        #     print("Position TEME [km]:", position0_teme)
        #     print("Velocity TEME [km/s]:", velocity0_teme,"\n")
        # else:
        #     print("Error code:", error)

        # Convert TEME state to J2000
        epoch = time_conversion.julian_day_to_seconds_since_epoch(jd0 + fr0)
        R_teme_to_j2000 = element_conversion.teme_to_j2000(epoch)

        pos0_j2000 = R_teme_to_j2000 @ position0_teme
        vel0_j2000 = R_teme_to_j2000 @ velocity0_teme

        # Create default body settings for "Earth"
        bodies_to_create = ["Earth", "Sun", "Moon"]

        state0 = np.concatenate((pos0_j2000, vel0_j2000)) * 1e3  # Convert to meters and m/s
        simulation_start_epoch = DateTime(year0, month0, day0, hour0, minute0, second0).to_epoch()
        simulation_end_epoch = DateTime(year1, month1, day1, hour1, minute1, second1).to_epoch()

        time_difference = float((simulation_end_epoch - simulation_start_epoch) / (60*60*24))  # in days
        time_diffs.append(time_difference)
        print(f"Epoch0: {simulation_start_epoch} to Epoch1: {simulation_end_epoch} (difference: {time_difference:.4f} days)\n")


        # IMPORTANT: Use TEME frame to match SGP4 output
        global_frame_origin = "Earth"
        global_frame_orientation = "J2000"  # Changed from J2000

        body_settings = environment_setup.get_default_body_settings(
            bodies_to_create, global_frame_origin, global_frame_orientation)

        # Add empty settings to body settings
        body_settings.add_empty_settings("Starlink")
        body_settings.get("Starlink").constant_mass = 250 #715

        # Aerodynamic coefficients
        reference_area_drag = 5 #15
        drag_coefficient = 2.2
        aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
            reference_area_drag, [drag_coefficient, 0.0, 0.0])
        body_settings.get("Starlink").aerodynamic_coefficient_settings = aero_coefficient_settings

        # Radiation pressure coefficients
        reference_area_radiation = 7 #15
        radiation_pressure_coefficient = 1.2
        occulting_bodies = dict()
        occulting_bodies["Sun"] = ["Earth"]
        radiation_pressure_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
            reference_area_radiation, radiation_pressure_coefficient, occulting_bodies)
        body_settings.get("Starlink").radiation_pressure_target_settings = radiation_pressure_settings

        # Create system of bodies
        bodies = environment_setup.create_system_of_bodies(body_settings)

        # Define bodies that are propagated
        bodies_to_propagate = ["Starlink"]

        # Define central bodies of propagation
        central_bodies = ["Earth"]

        # Enhanced acceleration model to better match SGP4
        acceleration_settings_starlink = dict(
            Sun=[
                propagation_setup.acceleration.radiation_pressure(),
                propagation_setup.acceleration.point_mass_gravity()
            ],
            Moon=[
                # propagation_setup.acceleration.spherical_harmonic_gravity(8, 8),
                propagation_setup.acceleration.point_mass_gravity()
            ],
            Earth=[
                # propagation_setup.acceleration.point_mass_gravity(),
                propagation_setup.acceleration.spherical_harmonic_gravity(5, 5), # Changing to spherical harmonics made a big difference
                propagation_setup.acceleration.aerodynamic(),
            ]
        )

        acceleration_settings = {"Starlink": acceleration_settings_starlink}

        # Create acceleration models
        acceleration_models = propagation_setup.create_acceleration_models(
            bodies, acceleration_settings, bodies_to_propagate, central_bodies
        )

        # Create termination settings
        termination_settings = propagation_setup.propagator.time_termination(simulation_end_epoch)

        integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
            time_step=1.0,
            coefficient_set=propagation_setup.integrator.CoefficientSets.rkf_78
        )
        propagator_settings = propagation_setup.propagator.translational(
            central_bodies,
            acceleration_models,
            bodies_to_propagate,
            state0,
            simulation_start_epoch,
            integrator_settings,
            termination_settings,
        )

        dynamics_simulator = simulator.create_dynamics_simulator(bodies, propagator_settings)
        print("Tudat simulation completed successfully")
        

        # Extract the resulting state history
        states = dynamics_simulator.propagation_results.state_history
        # print(f"Number of output states: {len(states)}")
        states_array = result2array(states)

        tudat_state = states_array[-1, 1:] / 1E3  # Convert to km and km/s
        # tudat_state = states_array[-1] / 1E3
        print(tudat_state)
        # tudat_state = states[simulation_end_epoch]
        pos1_tudat = tudat_state[:3]
        vel1_tudat = tudat_state[3:]

        # === Comparison === 
        sat1 = Satrec.twoline2rv(t1_line_1, t1_line_2)
        error, gt_pos1_sgp4_teme, gt_vel1_sgp4_teme = sat1.sgp4(jd1, fr1)
        epoch1 = time_conversion.julian_day_to_seconds_since_epoch(jd1 + fr1)
        R_teme_to_j2000 = element_conversion.teme_to_j2000(epoch1)

        gt_pos1_j2000 = R_teme_to_j2000 @ gt_pos1_sgp4_teme
        gt_vel1_j2000 = R_teme_to_j2000 @ gt_vel1_sgp4_teme

        prop_position1_sgp4_j2000 = R_teme_to_j2000 @ prop_position1_sgp4
        prop_velocity1_sgp4_j2000 = R_teme_to_j2000 @ prop_velocity1_sgp4

        # print(f"\nGT final position [km]: {gt_pos1_j2000}")
        # print(f"GT final velocity [km/s]: {gt_vel1_j2000}")

        # print(f"\nSGP4 final position [km]: {prop_position1_sgp4_j2000}")
        # print(f"SGP4 final velocity [km/s]: {prop_velocity1_sgp4_j2000}")

        # print(f"\nTudat final position [km]: {pos1_tudat}")
        # print(f"Tudat final velocity [km/s]: {vel1_tudat}")

        pos_diff_sgp4 = np.linalg.norm(gt_pos1_j2000 - prop_position1_sgp4_j2000)
        vel_diff_sgp4 = np.linalg.norm(gt_vel1_j2000 - prop_velocity1_sgp4_j2000)
        sgp4_pos_error.append(pos_diff_sgp4)
        # print(f"\nSGP4 Position difference [km]: {pos_diff_sgp4}")
        # print(f"SGP4 Velocity difference [km/s]: {vel_diff_sgp4}")

        pos_diff_tudat = np.linalg.norm(gt_pos1_j2000 - pos1_tudat)
        vel_diff_tudat = np.linalg.norm(gt_vel1_j2000 - vel1_tudat)
        tudat_pos_error.append(pos_diff_tudat)
        # print(f"\nTudat Position difference [km]: {pos_diff_tudat}")
        # print(f"Tudat Velocity difference [km/s]: {vel_diff_tudat}")


    # Plot position errors vs time differences
    plt.figure(figsize=(10, 6))
    plt.semilogy(time_diffs, sgp4_pos_error, 'o-', label='SGP4 Position Error (km)')
    plt.semilogy(time_diffs, tudat_pos_error, 's-', label='Tudat Position Error (km)')
    plt.xlabel('Propagation Time (days)')
    plt.ylabel('Position Error (km)')
    # plt.title(f'Starlink-{STARLINK_ID} Position Error Comparison: Tudat vs SGP4')
    plt.title(f'{STARLINK_ID} Position Error Comparison: Tudat vs SGP4')
    plt.legend()
    plt.grid()
    plt.savefig(f'starlink_{STARLINK_ID}_position_error_comparison.png', dpi=300)
    # plt.show()