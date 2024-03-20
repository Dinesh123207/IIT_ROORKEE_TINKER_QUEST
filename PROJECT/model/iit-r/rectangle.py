import cv2
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Function to detect rectangles or tables in an image
def detect_rectangles(img_path):
    image = cv2.imread(img_path)
    gray = None  # Default value

    if image is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if gray is not None:
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Sort contours by area and keep the 10 largest
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        rectangles = [cv2.boundingRect(cnt) for cnt in sorted_contours]
        return rectangles
    else:
        return []

# Function to extract SIFT features from rectangles in an image
def extract_sift_features_from_rectangles(image, rectangles):
    descriptors_list = []
    for rect in rectangles:
        x, y, w, h = rect
        roi = image[y:y+h, x:x+w]
        _, descriptors = sift.detectAndCompute(roi, None)
        if descriptors is not None:
            descriptors_list.append(descriptors)
    return np.vstack(descriptors_list) if descriptors_list else None

# Function to build a feature database from rectangles in a directory of images
def build_feature_database(directory_path):
    image_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    database_descriptors = []
    image_names = []
    for path in image_paths:
        image = cv2.imread(path)
        rectangles = detect_rectangles(path)  # Pass the image path to detect_rectangles
        descriptors = extract_sift_features_from_rectangles(image, rectangles)
        if descriptors is not None:
            database_descriptors.append(descriptors)
            image_names.append(path)
    return database_descriptors, image_names

def find_top_matches(input_descriptors, database_descriptors, image_names, top_n=30):
    # Flatten the database descriptors for NearestNeighbors
    all_database_descriptors = np.vstack(database_descriptors)
    
    # Fit NearestNeighbors
    neighbors = NearestNeighbors(n_neighbors=top_n, algorithm='auto', metric='euclidean').fit(all_database_descriptors)
    
    # Find the k nearest neighbors for each descriptor in the input image
    distances, indices = neighbors.kneighbors(input_descriptors)

    # Calculate the cumulative sum of descriptor counts to map flat indices to image indices
    descriptor_counts = np.array([len(desc) for desc in database_descriptors])
    cumulative_descriptor_counts = np.cumsum(descriptor_counts)

    # Count the occurrences of database image indices in the nearest neighbors
    image_index_counts = np.zeros(len(image_names), dtype=np.int32)
    for flat_idx in indices.flatten():
        # Find the image index that corresponds to the current flat index
        image_index = np.searchsorted(cumulative_descriptor_counts, flat_idx, side='right')
        image_index_counts[image_index] += 1

    # Get the top N indices with the highest counts
    top_image_indices = np.argsort(-image_index_counts)[:top_n]
    top_image_paths = [image_names[i] for i in top_image_indices if image_index_counts[i] > 0]
    top_image_scores = [image_index_counts[i] for i in top_image_indices if image_index_counts[i] > 0]

    return top_image_paths, top_image_scores



def runner1(img_path,database_directory):
    # Load input image
    input_image = cv2.imread(img_path)

    # Directory where the database images are stored

    # Detect rectangles in the input image
    input_rectangles = detect_rectangles(img_path)

    # Extract SIFT features from the detected rectangles
    input_descriptors = extract_sift_features_from_rectangles(input_image, input_rectangles)

    # Build feature database
    db_descriptors, db_image_names = build_feature_database(database_directory)

    # Find the top matching images
    top_matches, match_scores = find_top_matches(input_descriptors, db_descriptors, db_image_names)

    res = {}
    # Print the top matching images and their scores
    for match, score in zip(top_matches, match_scores):
        # print(f'Match: {match}, Score: {score}')
        match = match[14:]
        res[match] = score
    
    return res
