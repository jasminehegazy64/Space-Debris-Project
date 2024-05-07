import os
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

class SatelliteAnalyzer:
    def __init__(self, fits_directory):
        self.fits_directory = fits_directory
        self.R_Earth = 6371e3  # Radius of Earth (m)

    def extract_header_info(self, full_path_fits):
        with fits.open(full_path_fits) as hdul:
            header = hdul[0].header
            dateobs = header.get('DATE-OBS', "Not available")
            epos_x = header.get('EPOS1_1', "Not available")
            epos_y = header.get('EPOS1_2', "Not available")
            epos_z = header.get('EPOS1_3', "Not available")
            avg_vel = header.get('AVG_VEL', "Not available")
            ra_vel = header.get('RA_VEL', "Not available")
            dec_vel = header.get('DEC_VEL', "Not available")
            rol_vel = header.get('ROL_VEL', "Not available")
        return dateobs, epos_x, epos_y, epos_z, avg_vel, ra_vel, dec_vel, rol_vel

    def extract_orbital_parameters(self, full_path_fits):
        with fits.open(full_path_fits) as hdul:
            header = hdul[0].header
            date_obs = header.get('DATE-OBS', None)
            obj_ra = header.get('OBJCTRA', None)
            obj_dec = header.get('OBJCTDEC', None)
            obj_rol = header.get('OBJCTROL', None)
        return date_obs, obj_ra, obj_dec, obj_rol

    def calculate_satellite_position(self, theta, inclination, eccentricity, semi_major_axis):
        r = (semi_major_axis * (1 - eccentricity**2)) / (1 + eccentricity * np.cos(theta))
        x = r * np.cos(theta)
        y = r * np.sin(theta) * np.sin(np.deg2rad(inclination))
        z = r * np.sin(theta) * np.cos(np.deg2rad(inclination))
        return x, y, z

    def update_plot(self, frame, theta_values, inclination, eccentricity, semi_major_axis, ax):
        ax.clear()
        ax.set_box_aspect([1,1,1])

        # Calculate satellite position
        x, y, z = self.calculate_satellite_position(theta_values[frame], inclination, eccentricity, semi_major_axis)

        # Plot Earth
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        earth_x = self.R_Earth * np.outer(np.cos(u), np.sin(v))
        earth_y = self.R_Earth * np.outer(np.sin(u), np.sin(v))
        earth_z = self.R_Earth * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(earth_x, earth_y, earth_z, color='blue', alpha=0.2)

        # Plot satellite position
        ax.scatter(x, y, z, color='red', label='Satellite Position')
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

     

    def analyze(self):
        fits_files = [f for f in os.listdir(self.fits_directory) if f.endswith('.fits')]
        
        dfs = []
        for fits_filename in fits_files:
            full_path_fits = os.path.join(self.fits_directory, fits_filename)
            dateobs, epos_x, epos_y, epos_z, avg_vel, ra_vel, dec_vel, rol_vel = self.extract_header_info(full_path_fits)
            df = pd.DataFrame({
                'FITS File': [fits_filename],
                'Date time': [dateobs],
                'EPOS_X': [epos_x],
                'EPOS_Y': [epos_y],
                'EPOS_Z': [epos_z],
                'AVG_VEL': [avg_vel],
                'RA_VEL': [ra_vel],
                'DEC_VEL': [dec_vel],
                'ROL_VEL': [rol_vel],
            })
            dfs.append(df)

        if dfs:
            df_one = pd.concat(dfs, ignore_index=True)
            df_one = df_one.sort_values(by='FITS File')
            print(df_one)
        else:
            print("No FITS files found in the directory.")

        # Example animation
        num_frames = 100
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        theta_values = np.linspace(0, 2*np.pi, 100)
        inclination = 30  # Example inclination (degrees)
        eccentricity = 0.1  # Example eccentricity
        semi_major_axis = 7000  # Example semi-major axis (km)
        ani = animation.FuncAnimation(fig, self.update_plot, frames=num_frames, fargs=(theta_values, inclination, eccentricity, semi_major_axis, ax), interval=50)
        plt.show()


