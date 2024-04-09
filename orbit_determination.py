from google.colab import drive
drive.mount('/content/drive',force_remount=True)

# Directory containing FITS files
fits_directory = '/content/drive/MyDrive/2024_001_trial/fits'

fits_file = '/content/drive/MyDrive/Colab-Debris/please4.fits'

fits_filenames = ['Copy of NEOS_SCI_2024001001404.fits','Copy of NEOS_SCI_2024001001121.fits','Copy of NEOS_SCI_2024001000838.fits','Copy of NEOS_SCI_2024001000555.fits','Copy of NEOS_SCI_2024001000312.fits']

import os
import pandas as pd
from astropy.io import fits


# Create a list to store DataFrames for each FITS file
dfs = []

for root, dirs, files in os.walk(fits_directory):
    for fits_filename in files:
        if fits_filename.endswith('.fits'):
            # Full path to the FITS file
            full_path_fits = os.path.join(root, fits_filename)
            # Open the FITS file
            with fits.open(full_path_fits) as hdul:
                # Access the header of the primary HDU
                header = hdul[0].header

                # Extract relevant information
                try:
                    # Extracting Date of observation from header
                    dateobs = header['DATE-OBS']
                except KeyError:
                    dateobs = "Not available"

                try:
                    # Extracting Number of axis from the header
                    EPOS_X = header['EPOS1_1']
                except KeyError:
                    EPOS_X = "Not available"

                try:
                    # Extracting Wavelength from the header
                    EPOS_Y = header['EPOS1_2']
                except KeyError:
                    EPOS_Y = "Not available"

                try:
                    # Extracting Pixel Scale from the header
                    EPOS_Z = header['EPOS1_3']
                except KeyError:
                    EPOS_Z = "Not available"

                try:
                    # Extracting Average Velocity from the header
                    AVG_VEL = header['AVG_VEL']
                except KeyError:
                    AVG_VEL = "Not available"

                try:
                    # Extracting RA Velocity from the header
                    RA_VEL = header['RA_VEL']
                except KeyError:
                    RA_VEL = "Not available"

                try:
                    # Extracting DEC Velocity from the header
                    DEC_VEL = header['DEC_VEL']
                except KeyError:
                    DEC_VEL = "Not available"

                try:
                    # Extracting ROL Velocity from the header
                    ROL_VEL = header['ROL_VEL']
                except KeyError:
                    ROL_VEL = "Not available"

                # Create a DataFrame for the current FITS file
                df = pd.DataFrame({
                    'FITS File': [fits_filename],
                    'Date time': [dateobs],
                    'EPOS_X': [EPOS_X],
                    'EPOS_Y': [EPOS_Y],
                    'EPOS_Z': [EPOS_Z],
                    'AVG_VEL': [AVG_VEL],
                    'RA_VEL': [RA_VEL],
                    'DEC_VEL': [DEC_VEL],
                    'ROL_VEL': [ROL_VEL],
                })

                # Append the DataFrame to the list
                dfs.append(df)

# Concatenate the list of DataFrames into a single DataFrame
if dfs:
    df_one = pd.concat(dfs, ignore_index=True)
    df_one = df_one.sort_values(by='FITS File')
    # Display the resulting DataFrame
    print(df_one)
else:
    print("No FITS files found in the directory.")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
M_Earth = 5.972e24  # Mass of Earth (kg)
R_Earth = 6371e3  # Radius of Earth (m)

# Number of frames in the animation
num_frames = 5

def calculate_satellite_position(frame, datetime_obs, EPOS_X, EPOS_Y, EPOS_Z, AVG_VEL, RA_VEL, DEC_VEL, ROL_VEL):
    # Convert date_obs and time_obs to datetime object for calculations
    datetime_obs = np.datetime64(datetime_obs)

    # Compute satellite position in space using ECEF coordinates
    # Example: For simplicity, we assume a circular orbit in the x-y plane
    # You need to replace this with actual calculations based on the provided data
    # Here, we just use the y-component from EPOS_Y as the radius of the orbit
    radius = EPOS_Y * 1000  # Convert km to m
    angular_speed = np.sqrt(G * M_Earth / radius**3)  # Angular speed (radians per second)

    # Adjust angular speed based on average angular slew velocity
    angular_speed += AVG_VEL * np.pi / (180 * 3600)  # Convert arcsec/s to rad/s

    # Compute the change in angle considering the signed angular slew velocities
    delta_theta = (RA_VEL / 3600) * frame  # Convert hours/s to degrees
    delta_phi = (DEC_VEL / 3600) * frame  # Convert degrees/s to degrees
    delta_psi = (ROL_VEL / 3600) * frame  # Convert degrees/s to degrees

    # Adjust angular position based on the change in angle
    theta = angular_speed * frame + delta_theta
    phi = delta_phi
    psi = delta_psi

    x_satellite = radius * np.cos(theta) * np.cos(phi)
    y_satellite = radius * np.sin(theta) * np.cos(phi)
    z_satellite = radius * np.sin(phi)

    return x_satellite, y_satellite, z_satellite


# Function to update the plot for each frame
def update(frame):
    # Calculate satellite position for the current frame
    x_satellite, y_satellite, z_satellite = calculate_satellite_position(frame, df_one['Date time'][frame], df_one['EPOS_X'][frame], df_one['EPOS_Y'][frame], df_one['EPOS_Z'][frame],df_one['AVG_VEL'][frame] , df_one['RA_VEL'][frame] , df_one['DEC_VEL'][frame] , df_one['ROL_VEL'][frame])

    # Clear the previous plot
    ax.clear()

    # Plot the Earth
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_earth = R_Earth * np.outer(np.cos(u), np.sin(v))
    y_earth = R_Earth * np.outer(np.sin(u), np.sin(v))
    z_earth = R_Earth * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_earth, y_earth, z_earth, color='blue', alpha=0.2)

    # Plot the satellite
    ax.plot([x_satellite], [y_satellite], [z_satellite], marker='o', color='red', markersize=10, label='Satellite')

    # Add title and legend
    ax.set_title('Satellite Orbit around Earth')
    ax.legend()

    return ax

# Create a figure and axis for the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50)

# Save the animation as a video
ani.save('satellite_orbit_3dd.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()



import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from scipy.optimize import minimize
import os


# Lists to accumulate times and positions from all FITS files
all_times = []
all_positions = []

# Iterate over FITS files and extract header information
for fits_filename in os.listdir(fits_directory):
    if fits_filename.endswith('.fits'):
        # Full path to the FITS file
        full_path_fits = os.path.join(fits_directory, fits_filename)

        # Open the FITS file
        with fits.open(full_path_fits) as hdul:
            # Access the header of the primary HDU
            header = hdul[0].header

            # Extract necessary information
            time = Time(header['DATE-OBS'], format='fits', scale='utc')  # Assuming time is in FITS format
            all_times.append(time)

            # Convert RA from hours:minutes:seconds to degrees
            ra_str = header['RA']
            ra_hour, ra_min, ra_sec = map(float, ra_str.split(':'))
            ra_deg = (ra_hour + ra_min / 60 + ra_sec / 3600) * u.deg

            # Convert DEC from degrees:arcminutes:arcseconds to degrees
            dec_str = header['DEC']
            dec_deg, dec_min, dec_sec = map(float, dec_str.split(':'))
            dec_deg = (dec_deg + dec_min / 60 + dec_sec / 3600) * u.deg

            position = SkyCoord(ra=ra_deg, dec=dec_deg)
            all_positions.append(position)

# Concatenate all times and positions
times = Time(all_times)
positions = SkyCoord(all_positions)

# Define the objective function for optimization
def objective_function(params):
    a, ecc, inc, raan, argp, nu = params
    residuals = []
    for i, time in enumerate(times):
        # Check if the orbit is hyperbolic (eccentricity > 1)
        if ecc > 1:
            # For hyperbolic orbits, semimajor axis should be negative
            orbit = Orbit.from_classical(Earth, -a*u.km, ecc*u.one, inc*u.deg, raan*u.deg, argp*u.deg, nu*u.deg, time)
        else:
            orbit = Orbit.from_classical(Earth, a*u.km, ecc*u.one, inc*u.deg, raan*u.deg, argp*u.deg, nu*u.deg, time)
        residuals.append(positions[i].represent_as('cartesian').xyz.value - orbit.propagate(time).r.value)
    return np.linalg.norm(np.concatenate(residuals))

# Initial guess for orbital parameters
initial_guess = [7000, 0.01, 30, 45, 60, 90]

# Perform optimization to find best-fit parameters
result = minimize(objective_function, initial_guess)

# Extract the optimized parameters
a_opt, ecc_opt, inc_opt, raan_opt, argp_opt, nu_opt = result.x

# Create the final orbit with the optimized parameters
final_orbit = Orbit.from_classical(Earth, a_opt*u.km, ecc_opt*u.one, inc_opt*u.deg, raan_opt*u.deg, argp_opt*u.deg, nu_opt*u.deg, times[0])

# Print the orbital parameters
print("Semi-major axis (a):", a_opt, "km")
print("Eccentricity (e):", ecc_opt)
print("Inclination (i):", inc_opt, "deg")
print("Right ascension of the ascending node (RAAN):", raan_opt, "deg")
print("Argument of periapsis (ω):", argp_opt, "deg")
print("True anomaly (ν):", nu_opt, "deg")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
M = 5.972e24  # Mass of Earth (kg)
R = 6371e3  # Radius of Earth (m)

# Function to calculate satellite position
def calculate_satellite_position(theta, inclination, eccentricity, semi_major_axis):
    r = (semi_major_axis * (1 - eccentricity**2)) / (1 + eccentricity * np.cos(theta))
    x = r * (np.cos(theta))
    y = r * (np.sin(theta)) * np.sin(inclination)  # Incorporate inclination
    z = r * (np.sin(theta)) * np.cos(inclination)
    return x, y, z

# Orbital parameters
semi_major_axis = a_opt  # Semi-major axis in meters
eccentricity = ecc_opt
inclination = inc_opt  # Inclination angle in radians

# Time array
theta_values = np.linspace(0, 2*np.pi, 100)

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1,1,1])  # Equal aspect ratio for all axes

# Function to update plot for each frame
def update(frame):
    ax.clear()
    ax.set_box_aspect([1,1,1])

    # Calculate satellite position
    x, y, z = calculate_satellite_position(theta_values[frame], inclination, eccentricity, semi_major_axis)

    # Plot Earth
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    earth_x = R * np.outer(np.cos(u), np.sin(v))
    earth_y = R * np.outer(np.sin(u), np.sin(v))
    earth_z = R * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(earth_x, earth_y, earth_z, color='b', alpha=0.5)

    # Plot satellite position
    ax.scatter(x, y, z, color='r', label='Satellite Position')

# Create animation
ani = FuncAnimation(fig, update, frames=len(theta_values), interval=100)

# Save animation as video
ani.save('satellite_orbit.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define orbital parameters (semi-major axis, eccentricity, inclination, etc.)
# Replace these values with actual data
a = 7000  # Semi-major axis (km)
e = 0.1   # Eccentricity
i = 60    # Inclination (degrees)
# Add other orbital parameters as needed

# Generate time array (e.g., 1 orbit)
t = np.linspace(0, 2*np.pi, 1000)

# Calculate satellite's position in 3D space
x = a * (np.cos(t) - e)
y = np.zeros_like(t)  # Assuming circular orbit in this example
z = np.zeros_like(t)  # Assuming equatorial orbit in this example

# Plot the orbit
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z)

# Plot the Earth (optional)
earth_radius = 6371  # Earth's radius in km
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x_earth = earth_radius * np.cos(u) * np.sin(v)
y_earth = earth_radius * np.sin(u) * np.sin(v)
z_earth = earth_radius * np.cos(v)
ax.plot_surface(x_earth, y_earth, z_earth, color='b', alpha=0.2)

# Customize plot appearance (axis labels, title, etc.)
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('Satellite Orbit')

# Show plot
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation

# Number of frames in the animation
num_frames = 100

# Function to update the plot for each frame
def update(frame):
    # Calculate satellite position for the current frame
    orbit_angle = frame * 0.1  # Example angular speed
    x_satellite = np.cos(orbit_angle)  # X-coordinate of satellite
    y_satellite = np.sin(orbit_angle)  # Y-coordinate of satellite

    # Clear the previous plot
    ax.clear()

    # Plot the Earth circle
    earth_circle = plt.Circle((0, 0), 1, color='blue', alpha=0.2)  # Assuming radius of Earth circle as 1
    ax.add_patch(earth_circle)

    # Plot the satellite
    ax.plot(x_satellite, y_satellite, marker='o', color='red', markersize=10, label='Satellite')

    # Set plot limits and aspect ratio
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')

    # Add title and legend
    ax.set_title('Satellite Orbit around Imaginary Earth Circle')
    ax.legend()

    return ax

# Create a figure and axis for the plot
fig, ax = plt.subplots()

# Create the animation
ani = FuncAnimation(fig, update, frames=num_frames, interval=50)

# Set up video writer
Writer = matplotlib.animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

# Save the animation as a video
ani.save('satellite_orbit_circular.mp4', writer=writer)

plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Number of frames in the animation
num_frames = 100

# Function to update the plot for each frame
def update(frame):
    # Calculate satellite position for the current frame
    orbit_angle = frame * 0.1  # Example angular speed
    x_satellite = np.cos(orbit_angle)  # X-coordinate of satellite
    y_satellite = np.sin(orbit_angle)  # Y-coordinate of satellite
    z_satellite = 0  # Z-coordinate of satellite (assuming it's in the xy-plane)

    # Clear the previous plot
    ax.clear()

    # Plot the Earth sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_earth = np.outer(np.cos(u), np.sin(v))
    y_earth = np.outer(np.sin(u), np.sin(v))
    z_earth = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_earth, y_earth, z_earth, color='blue', alpha=0.2)

    # Plot the satellite
    ax.plot([x_satellite], [y_satellite], [z_satellite], marker='o', color='red', markersize=10, label='Satellite')

    # Set plot limits and aspect ratio
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.set_box_aspect([1,1,1])  # Equal aspect ratio

    # Add title and legend
    ax.set_title('Satellite Orbit around Imaginary Earth Sphere')
    ax.legend()

    return ax

# Create a figure and axis for the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50)

# Set up video writer
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

# Save the animation as a video
ani.save('satellite_orbit_3d.mp4', writer=writer)

plt.show()

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import Angle
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.constants import GM_earth

def compute_orbital_elements(fits_file):
    # Open FITS file and extract header
    with fits.open(fits_file) as hdul:
        header = hdul[0].header

        # Extract relevant parameters
        date_obs = header['DATE-OBS']
        obj_ra = header['OBJCTRA']
        obj_dec = header['OBJCTDEC']
        obj_rol = header['OBJCTROL']
        exposure = header['EXPOSURE']

        # Compute position vector
        ra = obj_ra
        dec = obj_dec
        rol = obj_rol
        r = Earth.R

        pos = [r * np.cos(ra) * np.cos(dec),
               r * np.sin(ra) * np.cos(dec),
               r * np.sin(dec)] * u.km

        # Compute velocity vector
        v = np.sqrt(GM_earth / r)
        v_x = -v * np.sin(ra) * np.cos(dec) * u.km / u.s
        v_y = v * np.cos(ra) * np.cos(dec) * u.km / u.s
        v_z = 0 * u.km / u.s  # Since there's no z-component

        vel = [v_x, v_y, v_z]

        # Compute orbital elements
        orb = Orbit.from_vectors(Earth, pos, vel, epoch=date_obs)
        a = orb.semi_major_axis.to(u.km).value
        ecc = orb.eccentricity.value
        inc = orb.inclination.to(u.deg).value
        arg_pe = orb.argument_of_periapsis.to(u.deg).value
        raan = orb.raan.to(u.deg).value
        mean_anomaly = orb.nu.to(u.deg).value

    return a, ecc, inc, arg_pe, raan, mean_anomaly

# Example usage
fits_file = '/content/drive/MyDrive/Colab-Debris/please4.fits'
a, ecc, inc, arg_pe, raan, mean_anomaly = compute_orbital_elements(fits_file)
print("Semi-major axis (a):", a, "km")
print("Eccentricity (ecc):", ecc)
print("Inclination (inc):", inc, "degrees")
print("Argument of periapsis (arg_pe):", arg_pe, "degrees")
print("Longitude of ascending node (raan):", raan, "degrees")
print("Mean anomaly (mean_anomaly):", mean_anomaly, "degrees")
