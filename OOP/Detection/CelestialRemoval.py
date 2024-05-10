from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR  # Import Support Vector Regression instead of SVC
import astropy
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cv2
from PIL import Image
from google.colab.patches import cv2_imshow
import os
from scipy.optimize import curve_fit
import csv
from skimage import feature
import pandas as pd
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
import joblib
import lightgbm as lgb
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def convert_fits_to_image(fits_filename, output_image_filename):
    # Open the FITS file
    with fits.open(fits_filename) as hdul:
        # Get the data from the primary HDU (Header Data Unit)
        data = hdul[0].data

        # Noise reduction using Gaussian filter
        data = cv2.GaussianBlur(data, (5, 5), 0)

        # Sharpening using Laplacian filter
        laplacian = cv2.Laplacian(data, cv2.CV_64F)
        sharpened = data - 0.8 * laplacian

        # Plot the data as an image without grid
        plt.imshow(sharpened, cmap='gray')
        plt.axis('off')  # Turn off the axis (including grid)

        # Save the preprocessed image as a PNG file
        plt.savefig(output_image_filename, bbox_inches='tight', pad_inches=0)
        plt.close()


def iterative_thresholding(image, initial_threshold=128, max_iterations=50, tolerance=1e-3):
    threshold = initial_threshold

    for iteration in range(max_iterations):
        # Segment the image into foreground and background based on the threshold
        foreground = image >= threshold
        background = image < threshold

        # Compute the mean intensity of each group
        foreground_mean = np.mean(image[foreground])
        background_mean = np.mean(image[background])

        # Compute the new threshold as the average of the means
        new_threshold = (foreground_mean + background_mean) / 2.0

        # Check for convergence
        if abs(new_threshold - threshold) < tolerance:
            break

        threshold = new_threshold

    return threshold


def otsu_thresholding(image):
    # Ensure the image is of type np.uint8
    image = np.uint8(image)

    # Apply Otsu's thresholding
    # minimize intraclass variance
    _, thresholded_img = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Get the optimal threshold value determined by Otsu's method
    optimal_threshold = _

    return thresholded_img, optimal_threshold


def momentOfInertia(xWidth, yHeight, xCG, yCG):
    Ixx = sum((y - yCG)**2 for y in yHeight)
    Iyy = sum((x - xCG)**2 for x in xWidth)
    Ixy = sum((x - xCG)*(y - yCG) for x, y in zip(xWidth, yHeight))

    return Ixx, Iyy, Ixy


def mainInteria(Ixx, Iyy, Ixy, yHeight, xWidth):
    Imain1 = 0.5 * (Ixx + Iyy + np.sqrt((Ixx - Iyy)**2 + 4*(Ixy)**2))
    Imain2 = 0.5 * (Ixx + Iyy - np.sqrt((Ixx - Iyy)**2 + 4*(Ixy)**2))

    epsilonn = 10

    finalInteria = Imain1 / Imain2
    if finalInteria > epsilonn:
        print(f"This object  is predicted to be debris")
    else:
        print(f"This object  is predicted to be a Celestial object")

    return finalInteria


# Directory containing FITS files
fits_directory = '/content/drive/MyDrive/2024TOTAL/001'

# Output directory for PNG images
output_directory = '/content/drive/MyDrive/2024_output/001'

csv_file_path = '/content/drive/MyDrive/2024_output/001/InetriaOutPut.csv'

# Open the CSV file in write mode
with open(csv_file_path, 'w', newline='') as csvfile:
    # Create a CSV writer
    csvwriter = csv.writer(csvfile)

    # Write the header row
    csvwriter.writerow(['Image', 'Object ID', 'Area', 'Edges',
                       'Center_x', 'Center_y', 'Width', 'Height', 'Prediction'])

    for fits_filename in os.listdir(fits_directory):
        if fits_filename.endswith('.fits'):
            # Full path to the FITS file
            full_path_fits = os.path.join(fits_directory, fits_filename)

            # Output PNG filename (assuming the same name with a different extension)
            output_image_filename = os.path.join(
                output_directory, os.path.splitext(fits_filename)[0] + '_preprocessed.png')
            convert_fits_to_image(full_path_fits, output_image_filename)

            image = cv2.imread(output_image_filename)
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply the iterative thresholding algorithm to the image
            optimal_threshold = iterative_thresholding(img)

            # Threshold the image using the optimal threshold
            thresholded_img = (img >= optimal_threshold).astype(np.uint8) * 255

            num_labels_iterative, labels_iterative, stats_iterative, centroids_iterative = cv2.connectedComponentsWithStats(
                thresholded_img, connectivity=8)

            # Create a mask to exclude celestial objects from the image
            mask = np.ones_like(img) * 255

            # Reset object_id for each new image
            object_id = 1

            # List to store deleted Object IDs
            deleted_object_ids = []

            for label in range(1, num_labels_iterative):
                area_iterative = stats_iterative[label, cv2.CC_STAT_AREA]
                component_mask = (labels_iterative == label).astype(np.uint8)
                center_x, center_y = centroids_iterative[label]

                # Multiply the component mask with the edges to get edges within the component
                edges = cv2.Canny(thresholded_img, 30, 100)
                edges_in_component = cv2.bitwise_and(
                    edges, edges, mask=component_mask)

                # Get the coordinates of the bounding box for the current object
                x, y, w, h, area = stats_iterative[label]

                # Count the number of edges in the component
                edge_count = np.count_nonzero(edges_in_component)

                # Extract the region of interest (ROI)
                roi = img[y:min(y + h, img.shape[0]),
                          x:min(x + w, img.shape[1])]

                # Ensure xWidth and yHeight are iterable (lists)
                xWidth = list(range(w))
                yHeight = list(range(h))

                # Print the coordinates of the bounding box
                print(f"Object {object_id} in {fits_filename}:")

                cv2.putText(image, str(object_id), (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Increment Id
                object_id += 1

                Ixx, Iyy, Ixy = momentOfInertia(
                    xWidth, yHeight, center_x, center_y)
                finalint = mainInteria(Ixx, Iyy, Ixy, yHeight, xWidth)

                # Exclude celestial objects by creating a mask
                if finalint <= 10:
                    mask[y:y+h, x:x+w] = 0

                    # Add the ID of the deleted object to the list
                    deleted_object_ids.append(object_id - 1)

                prediction = 'Celestial Object' if finalint <= 10 else 'Debris'

                # Write the row to the CSV file
                csvwriter.writerow([fits_filename, object_id - 1, area_iterative,
                                   edge_count, center_x, center_y, w, h, prediction])

            # Apply the mask to the original image
            image_masked = cv2.bitwise_and(img, img, mask=mask)

            # Print the IDs of deleted celestial objects
            print(
                f"Deleted Celestial Object IDs in {fits_filename}: {deleted_object_ids}")

            # Display the original and masked images
            print("Original Image")
            cv2_imshow(img)
            print("Masked Image")
            cv2_imshow(image_masked)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# APPLY MODELS AND ACCURACY VALUES FOR "CONTINIOUS"
csv_file_path = '/content/drive/MyDrive/2024_output/001/InetriaOutPut.csv'
df = pd.read_csv(csv_file_path)
df.info()


def outlier_percent(data):
    numeric_columns = data.select_dtypes(include=[np.number])
    Q1 = numeric_columns.quantile(0.25)
    Q3 = numeric_columns.quantile(0.75)
    IQR = Q3 - Q1
    minimum = Q1 - (1.5 * IQR)
    maximum = Q3 + (1.5 * IQR)
    num_outliers = ((numeric_columns < minimum) | (
        numeric_columns > maximum)).sum().sum()
    num_total = numeric_columns.count().sum()
    return (num_outliers / num_total) * 100


outlier_percent(df)

# encoding Prediction column
df["Prediction"].replace(to_replace="Celestial Object", value=0, inplace=True)
df["Prediction"].replace(to_replace="Debris", value=1, inplace=True)

scaler = StandardScaler()

# Perform scaling
df[['Area', 'Edges', 'Center_x', 'Center_y', 'Width', 'Height', 'Prediction']] = scaler.fit_transform(
    df[['Area', 'Edges', 'Center_x', 'Center_y', 'Width', 'Height', 'Prediction']])

# Calculate correlation matrix
correlation_matrix = df.drop(["Object ID", "Image"], axis=1).corr()

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Create a heatmap of the correlation matrix
plt.figure(figsize=(14, 14))
sns.heatmap(data=correlation_matrix, annot=True,
            cmap='coolwarm', center=0, mask=mask)
plt.title('Correlation Matrix')
plt.show()

df = df.sample(frac=1, random_state=42)

X = df.drop(['Object ID', 'Image', 'Prediction'], axis=1)
y = df['Prediction']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


model = []
acc = []

# Shuffle the DataFrame
df = df.sample(frac=1, random_state=42)

# Define features (X) and target variable (y)
X = df.drop(['Object ID', 'Image', 'Prediction'], axis=1)
y = df['Prediction']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initialize SVR model
svm_model = SVR()  # Use SVR for regression instead of SVC for classification

# Train SVR model
svm_model.fit(X_train, y_train)
# Make predictions on the testing set
svm_predictions = svm_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, svm_predictions)
print("Mean Squared Error:", mse)

# Predictions on training set
svm_predictions_train = svm_model.predict(X_train)

# Calculate R-squared (coefficient of determination) as a measure of train accuracy
train_r2 = r2_score(y_train, svm_predictions_train)
print("Train R-squared:", train_r2)

model.append("SVR")
acc.append(train_r2)

# k-Nearest Neighbors (k-NN)

# Initialize and train KNN model
knn_model = KNeighborsRegressor()
knn_model.fit(X_train, y_train)

# Make predictions on the testing set
knn_predictions = knn_model.predict(X_test)

# Evaluate the model
knn_accuracy = knn_model.score(X_test, y_test)
print("KNN Test Accuracy:", knn_accuracy)
model.append("KNN")
acc.append(knn_accuracy)

# Linear Regression

# Initialize and train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions on the testing set
lr_predictions = lr_model.predict(X_test)

# Evaluate the model
lr_accuracy = lr_model.score(X_test, y_test)
print("Linear Regression Test Accuracy (R-squared):", lr_accuracy)
model.append("LR")
acc.append(lr_accuracy)

# Random Forest Classifier

# Initialize and train Random Forest model
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

# Make predictions on the testing set
rf_predictions = rf_model.predict(X_test)

# Evaluate the model
rf_accuracy = rf_model.score(X_test, y_test)
print("Random Forest Test Accuracy (R-squared):", rf_accuracy)
model.append("RF")
acc.append(rf_accuracy)

# Decision Tree Classifier

# Initialize and train Decision Tree model
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)

# Make predictions on the testing set
dt_predictions = dt_model.predict(X_test)

# Evaluate the model
dt_accuracy = dt_model.score(X_test, y_test)
print("Decision Tree Test Accuracy (R-squared):", dt_accuracy)
model.append("DT")
acc.append(dt_accuracy)

# Create XGBoost classifier

# Initialize and train XGBoost model
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_train, y_train)

# Make predictions on the testing set
xgb_predictions = xgb_model.predict(X_test)

# Evaluate the model
xgb_accuracy = xgb_model.score(X_test, y_test)
print("XGBoost Test Accuracy (R-squared):", xgb_accuracy)
model.append("xgb")
acc.append(xgb_accuracy)

# Create LightGBM classifier

# Set verbosity to -1 to suppress LightGBM informational messages
lgb_model = lgb.LGBMRegressor(verbosity=-1)
lgb_model.fit(X_train, y_train)

# Make predictions on the testing set
lgb_predictions = lgb_model.predict(X_test)

# Evaluate the model
lgb_accuracy = lgb_model.score(X_test, y_test)
print("LightGBM Test Accuracy (R-squared):", lgb_accuracy)
model.append("LGBM")
acc.append(lgb_accuracy)

plt.figure(figsize=(10, 8))
plt.bar(model, acc)
plt.title('conclusion')
plt.xlabel('model')
plt.ylabel('accuracy')

plt.figure(figsize=(10, 10))
plt.plot(model, acc, 'r*-')  # 'r' is the color red
plt.xlabel('model')
plt.ylabel('accuracy')
plt.title('Conclusion')


# Assuming 'best_model' is your trained model
joblib.dump(rf_model, 'best_model.pkl')

loaded_model = joblib.load('best_model.pkl')
# do preprocessing needed

# Assuming 'new_data' is a DataFrame with the same features as the training data
predictions = loaded_model.predict(
    df.drop(['Object ID', 'Image', 'Prediction'], axis=1))
predictions
