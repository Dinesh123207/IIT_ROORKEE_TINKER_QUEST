import cv2
import os

class ImageMatcherFLANN:
    def __init__(self, sample_images_folder):
        self.sample_images_folder = sample_images_folder
        self.orb = cv2.ORB_create()
        self.incoming_keypoints = None
        self.incoming_descriptors = None
        self.sample_keypoints_descriptors = {}
        self.load_incoming_image()
        self.extract_sample_images_features()

    def load_incoming_image(self, incoming_image_path='D:\\Shashank_SIH\\aditya\\static\\civil_hospital.jpg'):
        self.incoming_image = cv2.imread(incoming_image_path, 0)
        self.incoming_keypoints, self.incoming_descriptors = self.orb.detectAndCompute(self.incoming_image, None)

    def extract_sample_images_features(self):
        for filename in os.listdir(self.sample_images_folder):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                sample_image_path = os.path.join(self.sample_images_folder, filename)
                sample_image = cv2.imread(sample_image_path, 0)
                kp, des = self.orb.detectAndCompute(sample_image, None)
                self.sample_keypoints_descriptors[sample_image_path] = {'keypoints': kp, 'descriptors': des}

    def find_similar_images_flann(self):
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches_list = []

        for image_path, image_data in self.sample_keypoints_descriptors.items():
            kp_sample, des_sample = image_data['keypoints'], image_data['descriptors']

            if self.incoming_descriptors is None or des_sample is None:
                continue

            matches = flann.knnMatch(self.incoming_descriptors, des_sample, k=2)
            ss=len(matches)

            good_matches = []
            if len(matches) > 0:  # Ensure matches exist
                for match in matches:
                    if len(match) >= 2:
                        m, n = match[:2]
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)

            matches_list.append((image_path, len(good_matches)))

        matches_list = sorted(matches_list, key=lambda x: x[1], reverse=True)
        sorted_image_paths_with_scores = [(image_path, score) for image_path, score in matches_list]
        return sorted_image_paths_with_scores,ss

# Example usage:
    


def runner6(sample_images_folder):
    sample_images_folder = sample_images_folder
    image_matcher_flann = ImageMatcherFLANN(sample_images_folder)

    similar_images_with_scores_flann,ss = image_matcher_flann.find_similar_images_flann()
    res = {}
    for index, (image_path, score) in enumerate(similar_images_with_scores_flann, start=1):
        print(f"Top {index} Similar Image Path (FLANN): {image_path}, Score: {score/ss}")
        res[image_path] = (score)