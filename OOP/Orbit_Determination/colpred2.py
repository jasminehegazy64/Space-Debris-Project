# # import numpy as np
# # import plotly.graph_objs as go
# # from pykalman import KalmanFilter

# # # Constants
# # NUM_OBJECTS = 5
# # TIME_STEPS = 10
# # COLLISION_THRESHOLD = 50  # Distance threshold in km

# # # Generate fake debris orbits (circular orbits around the Earth)
# # def generate_fake_orbits(num_objects, time_steps):
# #     orbits = []
# #     for _ in range(num_objects):
# #         radius = 6671  # Roughly 300 km above Earth's surface (Earth radius + 300 km)
# #         theta = np.linspace(0, 2 * np.pi, time_steps)
# #         x = radius * np.cos(theta) + np.random.uniform(-10, 10)
# #         y = radius * np.sin(theta) + np.random.uniform(-10, 10)
# #         z = np.random.uniform(-5, 5, time_steps)
# #         orbits.append(np.vstack((x, y, z)).T)
# #     return orbits

# # # Apply Kalman filter to predict future positions
# # def apply_kalman_filter(orbits, num_future_predictions=10):
# #     future_orbits = []
# #     for orbit in orbits:
# #         initial_state = np.array([orbit[0][0], 0, orbit[0][1], 0, orbit[0][2], 0])
# #         initial_state_covariance = np.eye(6) * 1e-4

# #         transition_matrix = np.array([
# #             [1, 1, 0, 0, 0, 0],
# #             [0, 1, 0, 0, 0, 0],
# #             [0, 0, 1, 1, 0, 0],
# #             [0, 0, 0, 1, 0, 0],
# #             [0, 0, 0, 0, 1, 1],
# #             [0, 0, 0, 0, 0, 1]
# #         ])
# #         observation_matrix = np.array([
# #             [1, 0, 0, 0, 0, 0],
# #             [0, 0, 1, 0, 0, 0],
# #             [0, 0, 0, 0, 1, 0]
# #         ])

# #         kf = KalmanFilter(
# #             transition_matrices=transition_matrix,
# #             observation_matrices=observation_matrix,
# #             initial_state_mean=initial_state,
# #             initial_state_covariance=initial_state_covariance
# #         )

# #         states, _ = kf.filter(orbit)
# #         future_states = []
# #         state = states[-1]
# #         covariance = np.eye(6) * 1e-4
# #         for _ in range(num_future_predictions):
# #             state, covariance = kf.filter_update(state, covariance)
# #             future_states.append(state[:3])
# #         future_orbits.append(np.array(future_states))
# #     return future_orbits

# # # Calculate distances between objects
# # def calculate_distances(positions1, positions2):
# #     distances = np.linalg.norm(positions1 - positions2, axis=1)
# #     return distances

# # # Detect potential collisions
# # def detect_collisions(future_orbits, collision_threshold):
# #     num_objects = len(future_orbits)
# #     collisions = []
# #     for i in range(num_objects):
# #         for j in range(i + 1, num_objects):
# #             distances = calculate_distances(future_orbits[i], future_orbits[j])
# #             if np.any(distances < collision_threshold):
# #                 collisions.append((i, j, np.min(distances)))
# #     return collisions

# # # Main function
# # def main():
# #     # Generate fake debris orbits
# #     orbits = generate_fake_orbits(NUM_OBJECTS, TIME_STEPS)

# #     # Predict future positions
# #     future_orbits = apply_kalman_filter(orbits)

# #     # Detect potential collisions
# #     collisions = detect_collisions(future_orbits, COLLISION_THRESHOLD)

# #     # Generate warnings
# #     if collisions:
# #         for i, j, distance in collisions:
# #             print(f"Warning: Potential collision detected between object {i} and object {j} with minimum distance {distance:.2f} km")
# #     else:
# #         print("No potential collisions detected")

# # # Run the main function
# # if __name__ == "__main__":
# #     main()







# import numpy as np
# import matplotlib.pyplot as plt
# from pykalman import KalmanFilter

# # Constants
# EARTH_RADIUS = 6371  # km
# ORBIT_ALTITUDE = 400  # km
# THRESHOLD_DISTANCE = 50.0  # km
# NUM_OBJECTS = 10
# NUM_OBSERVATIONS = 5
# NUM_FUTURE_PREDICTIONS = 10

# # Generate fake data with orbital parameters
# def generate_fake_data_with_collisions(num_objects=NUM_OBJECTS, num_observations=NUM_OBSERVATIONS):
#     np.random.seed(42)
#     data = []
#     collision_point = (180, 0)  # RA=180, DEC=0
#     for obj_id in range(num_objects):
#         if obj_id < num_objects // 2:
#             ra = np.linspace(collision_point[0] - 1, collision_point[0], num_observations)
#             dec = np.linspace(collision_point[1] - 1, collision_point[1], num_observations)
#         else:
#             ra = np.linspace(collision_point[0] + 1, collision_point[0], num_observations)
#             dec = np.linspace(collision_point[1] + 1, collision_point[1], num_observations)
#         for obs_id in range(num_observations):
#             data.append((obj_id, ra[obs_id], dec[obs_id]))
#     return data

# # Convert RA and DEC to Cartesian coordinates
# def spherical_to_cartesian(ra, dec, r=EARTH_RADIUS + ORBIT_ALTITUDE):
#     ra_rad = np.radians(ra)
#     dec_rad = np.radians(dec)
#     x = r * np.cos(dec_rad) * np.cos(ra_rad)
#     y = r * np.cos(dec_rad) * np.sin(ra_rad)
#     z = r * np.sin(dec_rad)
#     return x, y, z

# # Apply Kalman filter to predict future positions
# def apply_kalman_filter(observations, num_future_predictions=NUM_FUTURE_PREDICTIONS):
#     initial_state = np.array([observations[0][1], 0, observations[0][2], 0])  # initial state (ra, ra_velocity, dec, dec_velocity)
#     initial_state_covariance = np.eye(4) * 1e-4  # initial state covariance

#     transition_matrix = np.array([
#         [1, 1, 0, 0],
#         [0, 1, 0, 0],
#         [0, 0, 1, 1],
#         [0, 0, 0, 1]
#     ])
#     observation_matrix = np.array([
#         [1, 0, 0, 0],
#         [0, 0, 1, 0]
#     ])

#     kf = KalmanFilter(
#         transition_matrices=transition_matrix,
#         observation_matrices=observation_matrix,
#         initial_state_mean=initial_state,
#         initial_state_covariance=initial_state_covariance
#     )

#     # Extract RA and DEC observations
#     observations_array = np.array([(obs[1], obs[2]) for obs in observations])

#     # Apply the filter
#     states, state_covariances = kf.filter(observations_array)

#     # Predict future positions
#     future_states = []
#     state = states[-1]
#     covariance = state_covariances[-1]
#     for _ in range(num_future_predictions):
#         state, covariance = kf.filter_update(state, covariance)
#         future_states.append(state)

#     return states, future_states

# # Calculate distances between future positions
# def calculate_distances(positions1, positions2):
#     positions1 = np.array(positions1)
#     positions2 = np.array(positions2)
#     distances = np.linalg.norm(positions1 - positions2, axis=1)
#     return distances

# def main():
#     # Generate fake data with collisions
#     data = generate_fake_data_with_collisions()

#     # Group observations by object
#     grouped_observations = {}
#     for obj_id, ra, dec in data:
#         if obj_id not in grouped_observations:
#             grouped_observations[obj_id] = []
#         grouped_observations[obj_id].append((obj_id, ra, dec))
    
#     # Apply Kalman filter and predict future positions
#     future_positions = {}
#     for obj_id, observations in grouped_observations.items():
#         _, future_states = apply_kalman_filter(observations)
#         future_positions[obj_id] = [spherical_to_cartesian(state[0], state[2]) for state in future_states]

#     # Check for potential collisions
#     collisions = []
#     for i in range(len(future_positions)):
#         for j in range(i + 1, len(future_positions)):
#             dist = calculate_distances(future_positions[i], future_positions[j])
#             if np.any(dist < THRESHOLD_DISTANCE):
#                 collisions.append((i, j))

#     # Plot collisions
#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.set_title('Collision Risks Between Debris Objects', fontsize=16)
#     ax.set_xlabel('Debris Object Index', fontsize=14)
#     ax.set_ylabel('Debris Object Index', fontsize=14)

#     # Add collision markers
#     for (i, j) in collisions:
#         ax.plot(i, j, 'ro', markersize=10)
#         ax.plot(j, i, 'ro', markersize=10)

#     # Add grid lines
#     ax.grid(True, which='both', linestyle='--', linewidth=0.5)

#     # Highlight collision points dramatically
#     ax.scatter(*zip(*collisions), s=200, c='red', marker='X')
    
#     plt.show()

# if __name__ == "__main__":
#     main()



import os
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from astropy.io import fits
from astropy.coordinates import Angle
import astropy.units as u

# Constants
EARTH_RADIUS = 6371  # km
ORBIT_ALTITUDE = 400  # km
THRESHOLD_DISTANCE = 50.0  # km
NUM_FUTURE_PREDICTIONS = 10

# Convert RA and DEC from strings to decimal degrees
def convert_ra_dec_to_degrees(ra_str, dec_str):
    ra_angle = Angle(ra_str, unit=u.hourangle)
    dec_angle = Angle(dec_str, unit=u.deg)
    return ra_angle.deg, dec_angle.deg

# Function to read FITS files from a folder and extract RA and DEC
def read_fits_folder(folder_path):
    data = []
    obj_id = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.fits'):
            file_path = os.path.join(folder_path, file_name)
            try:
                with fits.open(file_path) as hdul:
                    ra_str = hdul[0].header['RA']
                    dec_str = hdul[0].header['DEC']
                    ra, dec = convert_ra_dec_to_degrees(ra_str, dec_str)
                    data.append((obj_id, ra, dec))
                    obj_id += 1
            except Exception as e:
                print(f"Error reading file '{file_name}': {e}")
    return data

# Convert RA and DEC to Cartesian coordinates
def spherical_to_cartesian(ra, dec, r=EARTH_RADIUS + ORBIT_ALTITUDE):
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    x = r * np.cos(dec_rad) * np.cos(ra_rad)
    y = r * np.cos(dec_rad) * np.sin(ra_rad)
    z = r * np.sin(dec_rad)
    return x, y, z

# Apply Kalman filter to predict future positions
def apply_kalman_filter(observations, num_future_predictions=NUM_FUTURE_PREDICTIONS):
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

    return states, future_states

# Calculate distances between future positions
def calculate_distances(positions1, positions2):
    positions1 = np.array(positions1)
    positions2 = np.array(positions2)
    distances = np.linalg.norm(positions1 - positions2, axis=1)
    return distances

def main():
    folder_path = r'C:\Users\USER\Desktop\Space-Debris-Project\OOP\2024-001'  # Replace with your folder path
    data = read_fits_folder(folder_path)

    # Group observations by object
    grouped_observations = {}
    for obj_id, ra, dec in data:
        if obj_id not in grouped_observations:
            grouped_observations[obj_id] = []
        grouped_observations[obj_id].append((obj_id, ra, dec))
    
    # Apply Kalman filter and predict future positions
    future_positions = {}
    for obj_id, observations in grouped_observations.items():
        _, future_states = apply_kalman_filter(observations)
        future_positions[obj_id] = [spherical_to_cartesian(state[0], state[2]) for state in future_states]

    # Check for potential collisions
    collisions = []
    for i in range(len(future_positions)):
        for j in range(i + 1, len(future_positions)):
            dist = calculate_distances(future_positions[i], future_positions[j])
            if np.any(dist < THRESHOLD_DISTANCE):
                collisions.append((i, j))

    # Plot collisions
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title('Collision Risks Between Debris Objects', fontsize=16)
    ax.set_xlabel('Debris Object Index', fontsize=14)
    ax.set_ylabel('Debris Object Index', fontsize=14)

    # Add collision markers
    for (i, j) in collisions:
        ax.plot(i, j, 'ro', markersize=10)
        ax.plot(j, i, 'ro', markersize=10)

    # Add grid lines
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Highlight collision points dramatically
    if collisions:
        ax.scatter(*zip(*collisions), s=200, c='red', marker='X')
    
    plt.show()

if __name__ == "__main__":
    main()


