import os
import numpy as np
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
    w = WCS(hdulist[0].header)

    mean, median, std = np.mean(data), np.median(data), mad_std(data)
    daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)
    sources = daofind(data - median)

    date_obs = hdulist[0].header['DATE-OBS']
    jd_obs = Time(date_obs, format='isot', scale='utc').jd

    observations = []
    for source in sources:
        x = source['xcentroid']
        y = source['ycentroid']
        ra, dec = w.wcs_pix2world(x, y, 1)
        observations.append((jd_obs, ra, dec))
    
    return observations

def apply_kalman_filter(observations, num_predictions=10):
    initial_state = np.array([observations[0][1], observations[0][2], 0, 0])  # initial state (ra, dec, ra_velocity, dec_velocity)
    initial_state_covariance = np.eye(4) * 1e-4  # initial state covariance

    transition_matrix = np.eye(4)
    observation_matrix = np.eye(4)[:2, :4]

    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        initial_state_mean=initial_state,
        initial_state_covariance=initial_state_covariance,
    )

    observations_array = np.array([(obs[1], obs[2]) for obs in observations])

    kf = kf.em(observations_array, n_iter=5)

    states, state_covariances = kf.filter(observations_array)

    future_predictions = []
    for _ in range(num_predictions):
        next_state, next_state_covariance = kf.filter_update(states[-1], state_covariances[-1])
        states = np.vstack((states, next_state))
        state_covariances = np.concatenate((state_covariances, [next_state_covariance]))
        future_predictions.append((next_state[0], next_state[1]))  # Predicted RA, DEC
    
    return states, future_predictions

def main(folder_path):
    fits_files = load_fits_files(folder_path)

    all_observations = []
    for fits_file in fits_files:
        observations = process_fits_file(fits_file)
        all_observations.append(observations)
    
    all_trajectories = []
    for observations in all_observations:
        states, future_predictions = apply_kalman_filter(observations)
        trajectory = {
            'observations': observations,
            'filtered_states': states,
            'future_predictions': future_predictions
        }
        all_trajectories.append(trajectory)

    return all_trajectories

if __name__ == "__main__":
    folder_path = r'C:\Users\USER\Desktop\finalGPbegad\fits'
    trajectories = main(folder_path)
    
    for i, trajectory in enumerate(trajectories):
        print(f"Object {i+1}:")
        print("Observations:")
        for obs in trajectory['observations']:
            print(f"  Time: {obs[0]}, RA: {obs[1]}, DEC: {obs[2]}")
        print("Filtered States:")
        for state in trajectory['filtered_states']:
            print(f"  RA: {state[0]}, DEC: {state[1]}, RA_velocity: {state[2]}, DEC_velocity: {state[3]}")
        print("Future Predictions:")
        for pred in trajectory['future_predictions']:
            print(f"  RA: {pred[0]}, DEC: {pred[1]}")
        print()
