import os
import numpy as np
import plotly.graph_objs as go
from astropy.io import fits
from astropy.wcs import WCS
from photutils import DAOStarFinder
from astropy.stats import mad_std
from astropy.time import Time
from pykalman import KalmanFilter

def load_fits_files(folder_path):
    fits_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.fits')]
    return fits_files

def process_fits_file(fits_file):
    hdulist = fits.open(fits_file)
    data = hdulist[0].data

    # Get WCS information
    w = WCS(hdulist[0].header)

    # Detect stars in the image
    mean, median, std = np.mean(data), np.median(data), mad_std(data)
    daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)
    sources = daofind(data - median)

    # Get observation time from the FITS header
    date_obs = hdulist[0].header['DATE-OBS']
    jd_obs = Time(date_obs, format='isot', scale='utc').jd

    # Convert pixel coordinates to celestial coordinates
    observations = []
    for source in sources:
        x = source['xcentroid']
        y = source['ycentroid']
        ra, dec = w.wcs_pix2world(x, y, 1)
        observations.append((jd_obs, ra, dec))
    
    return observations

def spherical_to_cartesian(ra, dec, r=6371 + 400):  # Assuming objects are 400 km above Earth's surface
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    x = r * np.cos(dec_rad) * np.cos(ra_rad)
    y = r * np.cos(dec_rad) * np.sin(ra_rad)
    z = r * np.sin(dec_rad)
    return x, y, z

def apply_kalman_filter(observations, num_future_predictions=10):
    initial_state = np.array([observations[0][1], 0, observations[0][2], 0])  # initial state (ra, ra_velocity, dec, dec_velocity)
    initial_state_covariance = np.eye(4) * 1e-4  # initial state covariance

    transition_matrix = np.array([
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1]
    ])
    observation_matrix = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0]
    ])

    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        initial_state_mean=initial_state,
        initial_state_covariance=initial_state_covariance
    )

    # Extract RA and DEC observations
    observations_array = np.array([(obs[1], obs[2]) for obs in observations])

    # Apply the filter
    states, state_covariances = kf.filter(observations_array)

    # Predict future positions
    future_states = []
    state = states[-1]
    covariance = state_covariances[-1]
    for _ in range(num_future_predictions):
        state, covariance = kf.filter_update(state, covariance)
        future_states.append(state)

    return states, state_covariances, future_states

def plot_earth():
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 6371 * np.outer(np.cos(u), np.sin(v))
    y = 6371 * np.outer(np.sin(u), np.sin(v))
    z = 6371 * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z

def calculate_distances(positions1, positions2):
    positions1 = np.array(positions1)
    positions2 = np.array(positions2)
    distances = np.linalg.norm(positions1 - positions2, axis=1)
    return distances

def main(folder_path, collision_threshold=1.0):
    fits_files = load_fits_files(folder_path)

    all_observations = []
    for fits_file in fits_files:
        observations = process_fits_file(fits_file)
        all_observations.extend(observations)
    
    all_observations.sort()  # Sort by observation time

    states, state_covariances, future_states = apply_kalman_filter(all_observations)

    # Get Earth data
    x_earth, y_earth, z_earth = plot_earth()

    # Create scatter plot for object positions
    x_positions = []
    y_positions = []
    z_positions = []
    x_predicted = []
    y_predicted = []
    z_predicted = []
    x_future_predicted = []
    y_future_predicted = []
    z_future_predicted = []

    for obs in all_observations:
        jd, ra, dec = obs
        x, y, z = spherical_to_cartesian(ra, dec)
        x_positions.append(x)
        y_positions.append(y)
        z_positions.append(z)

    for state in states:
        ra_pred, dec_pred = state[0], state[2]
        x_pred, y_pred, z_pred = spherical_to_cartesian(ra_pred, dec_pred)
        x_predicted.append(x_pred)
        y_predicted.append(y_pred)
        z_predicted.append(z_pred)

    for state in future_states:
        ra_pred, dec_pred = state[0], state[2]
        x_pred, y_pred, z_pred = spherical_to_cartesian(ra_pred, dec_pred)
        x_future_predicted.append(x_pred)
        y_future_predicted.append(y_pred)
        z_future_predicted.append(z_pred)

    # Calculate distances between all pairs of future predicted positions
    collisions = []
    for i in range(len(x_future_predicted)):
        for j in range(i+1, len(x_future_predicted)):
            dist = calculate_distances(
                [(x_future_predicted[i], y_future_predicted[i], z_future_predicted[i])],
                [(x_future_predicted[j], y_future_predicted[j], z_future_predicted[j])]
            )
            if dist < collision_threshold:
                collisions.append((i, j, dist))

    # Create 3D plot
    fig = go.Figure()

    # Add Earth
    fig.add_trace(go.Surface(x=x_earth, y=y_earth, z=z_earth, colorscale='Blues', showscale=False, opacity=0.6))

    # Add object positions
    fig.add_trace(go.Scatter3d(
        x=x_positions, y=y_positions, z=z_positions,
        mode='markers',
        marker=dict(size=6, color='red', symbol='circle'),
        name='Observed Positions'
    ))

    # Add predicted positions
    fig.add_trace(go.Scatter3d(
        x=x_predicted, y=y_predicted, z=z_predicted,
        mode='lines',
        line=dict(color='green', width=4),
        name='Predicted Positions'
    ))

    # Add future predicted positions
    fig.add_trace(go.Scatter3d(
        x=x_future_predicted, y=y_future_predicted, z=z_future_predicted,
        mode='lines',
        line=dict(color='blue', width=4),
        name='Future Predicted Positions'
    ))

    # Add collision points
    if collisions:
        colliding_points_x = [x_future_predicted[i] for (i, j, dist) in collisions]
        colliding_points_y = [y_future_predicted[i] for (i, j, dist) in collisions]
        colliding_points_z = [z_future_predicted[i] for (i, j, dist) in collisions]
        fig.add_trace(go.Scatter3d(
            x=colliding_points_x, y=colliding_points_y, z=colliding_points_z,
            mode='markers',
            marker=dict(size=10, color='yellow', symbol='diamond'),
            name='Collisions'
        ))

    # Set plot layout
    fig.update_layout(
        title='Object Motion Around the Earth with Kalman Filter Prediction and Collision Detection',
        scene=dict(
            xaxis_title='X (km)',
            yaxis_title='Y (km)',
            zaxis_title='Z (km)',
            aspectmode='data'
        )
    )

    fig.show()

# Example usage
if __name__ == "__main__":
    folder_path = r'C:\Users\USER\Desktop\finalGPbegad\fits2\OG'
    main(folder_path)
