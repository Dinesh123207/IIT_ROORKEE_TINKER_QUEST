import cv2
import numpy as np
from skimage import io, color, img_as_ubyte, img_as_float
from skimage.transform import resize
from skimage.feature import hog
from skimage.metrics import structural_similarity as ssim
from Levenshtein import distance as levenshtein_distance
from matplotlib import pyplot as plt
from skimage.metrics import normalized_root_mse
import imutils
from PIL import Image
import pytesseract
from easyocr import Reader

class MedicalClaimTemplateMatcher:
    def __init__(self, template_path, test_path):
        self.template_path = template_path
        self.test_path = test_path
        self.template_image = io.imread(template_path)
        self.test_image = io.imread(test_path)
        self.reader = Reader(['en'])

    def extract_text_from_image(self, image_path):
        reader = Reader(['en'])
        result = reader.readtext(image_path)
        extracted_text = ' '.join([entry[1] for entry in result])
        return extracted_text


    def analyze_text_properties(self, text):
        lines = text.split('\n')
        font_sizes, horizontal_spacings, vertical_positions, spacings = [], [], [], []

        for line in lines:
            words = line.split()
            if len(words) >= 2:
                try:
                    start, end = words[:2]
                    font_size = float(words[-1])
                    horizontal_spacing = abs(float(end) - float(start))
                    vertical_position = float(words[-2])

                    if len(spacings) > 0:
                        prev_end = float(lines[-1].split()[1])
                        spacing_value = self.calculate_euclidean_distance(
                            (prev_end, vertical_position), (float(start), vertical_position))
                        spacings.append(spacing_value)

                    font_sizes.append(font_size)
                    horizontal_spacings.append(horizontal_spacing)
                    vertical_positions.append(vertical_position)
                    spacings.append(spacing_value)

                except ValueError:
                    pass

        horizontal_spacing_variance = np.var(horizontal_spacings)
        vertical_position_variance = np.var(vertical_positions)

        return font_sizes, horizontal_spacings, vertical_positions, spacings, horizontal_spacing_variance, vertical_position_variance

    def textual_comparison(self, img_path):
        extracted_text = self.extract_text_from_image(img_path)
        font_sizes, horizontal_spacings, vertical_positions, spacings, horizontal_spacing_variance, vertical_position_variance = self.analyze_text_properties(
            extracted_text)

        font_size_threshold = 2.0
        spacing_threshold = 5.0
        alignment_threshold = 10.0
        vertical_position_threshold = 2.0

        if all(size >= font_size_threshold for size in font_sizes):
            print("Font sizes are consistent.")
        else:
            print("Inconsistent font sizes detected.")

        if all(spacing <= spacing_threshold for spacing in horizontal_spacings):
            print("Horizontal spacings are consistent.")
        else:
            print("Inconsistent horizontal spacings detected.")

        if np.var(vertical_positions) <= vertical_position_threshold:
            print("Vertical positions are consistent.")
        else:
            print("Inconsistent vertical positions detected.")

        if horizontal_spacing_variance <= alignment_threshold and vertical_position_variance <= alignment_threshold:
            print("Text is aligned.")
        else:
            print("Text is not aligned.")

    def convert_to_template(self, image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        plt.subplot(121), plt.imshow(cv2.cvtColor(
            img, cv2.COLOR_BGR2RGB)), plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(edges, cmap='gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        plt.show()
        cv2.imwrite("tem.png", edges)

    def extract_hog_features(self, image):
      if len(image.shape) == 3 and image.shape[2] == 4:  # RGBA format
          gray = color.rgba2rgb(image)
          gray = color.rgb2gray(gray)
      elif len(image.shape) == 3:  # RGB format
          gray = color.rgb2gray(image)
      else:
          gray = image

      resized_image = resize(gray, (128, 64))

      hog_features, _ = hog(resized_image, orientations=9, pixels_per_cell=(
          8, 8), cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')

      return hog_features


    def extract_sift_features(self, image):
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        return descriptors

    def calculate_textual_similarity(self, text1, text2):
        lev_distance = levenshtein_distance(text1, text2)
        if(text1>text2):
            lev_dis= (lev_distance/len(text1))
        else:
            lev_dis= (lev_distance/len(text2))
        return lev_dis

    # def match_header_and_logo(self):
    #     # Read images
    #     template_img = self.template_image
    #     test_img = self.test_image
    #     # Read images
    #     #template_img = cv2.imread(template_path)
    #     #test_img = cv2.imread(test_path)

    #     # Extract top 15% of the template image.
    #     header = template_img[:int(template_img.shape[0] * 0.15), :]

    #     # Perform dilation and erosion to blur the text.
    #     d_kernel = np.ones((5, 5), np.uint8)
    #     header = cv2.dilate(header, d_kernel)
    #     e_kernel = np.ones((15, 15), np.uint8)
    #     header = cv2.erode(header, e_kernel)

    #     # Use Canny to find edges.
    #     canny = cv2.Canny(header, 100, 100 * 2)

    #     # Find the contours using edges.
    #     contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #     # If no contours were found, return None.
    #     if not contours:
    #         return None, None

    #     contours = np.vstack(c for c in contours)
    #     x, y, w, h = cv2.boundingRect(contours)
    #     logo = template_img[y:y+h, x:x+w]

    #     # Extract top 15% of the test image.
    #     header2 = test_img[:int(test_img.shape[0] * 0.15), :]

    #     # Perform template matching of the template logo with header 2.
    #     match = cv2.matchTemplate(header2, logo, cv2.TM_CCOEFF)
    #     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)
    #     #_, _, _, max_loc = cv2.minMaxLoc(match)

    #     # Extract the matched region.
    #     matched = header2[max_loc[1]:max_loc[1]+h, max_loc[0]:max_loc[0]+w]

    #     # Draw rectangle on the second image.
    #     bottom_right = (max_loc[0] + w, max_loc[1] + h)
    #     cv2.rectangle(test_img, max_loc, bottom_right, (0, 0, 255), 10)

    #     # Calculate normalized root mean squared error.
    #     nmse = normalized_root_mse(logo, matched)

        # Calculate Structural Similarity Index (SSI) for header and header2.
        #score, diff = ssim(header, header2, full=True)
        #win_size = min(header.shape[0], header.shape[1])
        #if win_size % 2 == 0:
        #    win_size -= 1  # Ensure win_size is odd
        #ssi_index, _ = structural_similarity(header, header2, full=True, win_size=win_size)
        #ssi_index, _ = structural_similarity(header, header2, full=True, win_size=min(header.shape[0], header.shape[1]))

      #  return nmse, min_val, matched, test_img

    def calculate_ssim_and_display(self, template_path, test_path, desired_width=810, desired_height=610):
      original = cv2.imread(template_path)
      tampered = cv2.imread(test_path)

      # Resize the images to the desired dimensions
      original_height, original_width = original.shape[:2]
      tampered_height, tampered_width = tampered.shape[:2]

      if original_height != tampered_height or original_width != tampered_width:
          if original_height < tampered_height or original_width < tampered_width:
              original = cv2.resize(original, (tampered_width, tampered_height))
          else:
              tampered = cv2.resize(tampered, (original_width, original_height))

      originalr = cv2.resize(original, (desired_width, desired_height))
      tamperedr = cv2.resize(tampered, (desired_width, desired_height))

      original_gray = cv2.cvtColor(originalr, cv2.COLOR_BGR2GRAY)
      tampered_gray = cv2.cvtColor(tamperedr, cv2.COLOR_BGR2GRAY)

      # Convert images to uint8 to prevent matchTemplate error
      original_gray = cv2.convertScaleAbs(original_gray)
      tampered_gray = cv2.convertScaleAbs(tampered_gray)

      score, diff = ssim(original_gray, tampered_gray, full=True)
      diff = (diff * 255).astype("uint8")
      thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

      cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      cnts = imutils.grab_contours(cnts)

      for c in cnts:
          (x, y, w, h) = cv2.boundingRect(c)
          cv2.rectangle(originalr, (x, y), (x + w, y + h), (0, 0, 255), 2)
          cv2.rectangle(tamperedr, (x, y), (x + w, y + h), (0, 0, 255), 2)

      #print("Original format image")
      #plt.imshow(cv2.cvtColor(originalr, cv2.COLOR_BGR2RGB))
      #plt.axis('off')
      #plt.show()

    #   print("Tampered image")
    #   plt.imshow(cv2.cvtColor(tamperedr, cv2.COLOR_BGR2RGB))
    #   plt.axis('off')
    #   plt.show()

      # print('Different Image')
      # plt.imshow(diff, cmap='gray')
      # plt.axis('off')
      # plt.show()
      cv2.imwrite("static/tempered.jpg",originalr)
      return score

    # def match_header_and_logo(self):
    #     # Read images
    #     template_img = self.template_image
    #     test_img = self.test_image

    #     # Extract top 15% of the template image.
    #     header = template_img[:int(template_img.shape[0] * 0.15), :]

    #     # Perform dilation and erosion to blur the text.
    #     d_kernel = np.ones((5, 5), np.uint8)
    #     header = cv2.dilate(header, d_kernel)
    #     e_kernel = np.ones((15, 15), np.uint8)
    #     header = cv2.erode(header, e_kernel)

    #     # Use Canny to find edges.
    #     canny = cv2.Canny(header, 100, 100 * 2)

    #     # Find the contours using edges.
    #     contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #     # If no contours were found, return None.
    #     if not contours:
    #         return None, None, None, None

    #     contours = np.vstack(c for c in contours)
    #     x, y, w, h = cv2.boundingRect(contours)
    #     logo = template_img[y:y+h, x:x+w]

    #     # Ensure the images have the same depth and type
    #     logo = cv2.convertScaleAbs(logo)
    #     header2 = cv2.convertScaleAbs(test_img[:int(test_img.shape[0] * 0.15), :])

    #     # Ensure both images are in grayscale
    #     if len(logo.shape) == 3:
    #         logo = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    #     if len(header2.shape) == 3:
    #         header2 = cv2.cvtColor(header2, cv2.COLOR_BGR2GRAY)

    #     # Perform template matching of the template logo with header 2.
    #     match = cv2.matchTemplate(header2, logo, cv2.TM_CCOEFF)
    #     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)

    #     # Extract the matched region.
    #     matched = header2[max_loc[1]:max_loc[1]+h, max_loc[0]:max_loc[0]+w]

    #     # Draw rectangle on the second image.
    #     bottom_right = (max_loc[0] + w, max_loc[1] + h)
    #     cv2.rectangle(test_img, max_loc, bottom_right, (0, 0, 255), 10)

    #     # Calculate normalized root mean squared error.
    #     nmse = normalized_root_mse(logo, matched)

    #     return nmse, max_val

    def match_header_and_logo(template_path, test_path):
        # Read images
        template_img = cv2.imread(template_path)
        test_img = cv2.imread(test_path)

        # Resize images to the same dimensions
        #template_img = cv2.resize(template_img, (test_img.shape[1], test_img.shape[0]))

        # Extract top 15% of the template image.
        header = template_img[:int(template_img.shape[0] * 0.15), :]

        # Perform dilation and erosion to blur the text.
        d_kernel = np.ones((5, 5), np.uint8)
        header = cv2.dilate(header, d_kernel)
        e_kernel = np.ones((15, 15), np.uint8)
        header = cv2.erode(header, e_kernel)

        # Use Canny to find edges.
        canny = cv2.Canny(header, 100, 100 * 2)

        # Find the contours using edges.
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # If no contours were found, return None.
        if not contours:
            return None, None

        contours = np.vstack(c for c in contours)
        x, y, w, h = cv2.boundingRect(contours)
        logo = template_img[y:y+h, x:x+w]

        # Ensure the images have the same depth and type
        logo = cv2.convertScaleAbs(logo)
        header2 = cv2.convertScaleAbs(test_img[:int(test_img.shape[0] * 0.15), :])

        # Ensure both images are in grayscale
        if len(logo.shape) == 3:
            logo = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
        if len(header2.shape) == 3:
            header2 = cv2.cvtColor(header2, cv2.COLOR_BGR2GRAY)

        # Perform template matching of the template logo with header 2.
        match = cv2.matchTemplate(header2, logo, cv2.TM_CCOEFF)
        _, max_val, _, _ = cv2.minMaxLoc(match)

        # Extract the matched region.
        matched = header2[int(y):int(y+h), int(x):int(x+w)]

        # Draw rectangle on the second image.
        bottom_right = (int(x + w), int(y + h))
        cv2.rectangle(test_img, (int(x), int(y)), bottom_right, (0, 0, 255), 10)

        # Calculate normalized root mean squared error.
        nmse = normalized_root_mse(logo, matched)
        print("nmse score:", nmse)
        print('max matched by nmse:', max_val)
        return nmse, max_val



    def calculate_distances_and_similarity(self):
        hog_features_temp = self.extract_hog_features(self.template_image)
        hog_features_test = self.extract_hog_features(self.test_image)

        hog_distance = np.linalg.norm(hog_features_temp - hog_features_test)
        print("HOG Distance:", hog_distance/10)

        text_reference = self.extract_text_from_image(self.template_path)
        text_target = self.extract_text_from_image(self.test_path)

        text_similarity = self.calculate_textual_similarity(text_reference, text_target)
        print("Textual Similarity (Levenshtein distance):", text_similarity)
        threshold_text = 0.5
        # Text Discrepancy Analysis
        if text_similarity < threshold_text:
            res = "Text discrepancy detected!"
        else:
            res = "Original format image!"
            # print("Text discrepancy detected!")
        SSIm=template_matcher.calculate_ssim_and_display(template_path, test_path)
        print(" Structural Similarity Index:", SSIm)

        
        nmse,score=template_matcher.match_header_and_logo(template_path, test_path)
        print("nmse score ",nmse/10)
        print('max matched my nmse ', max_val ) 
        return hog_distance/10, text_similarity, SSIm,res,nmse/10, score


# def runner5(test_path,template_path):
#     template_image = io.imread(template_path)
#     test_image = io.imread(test_path)
#     template_matcher = MedicalClaimTemplateMatcher(template_path, test_path)
#     hog_distance, text_similarity, SSIm,res = template_matcher.calculate_distances_and_similarity()
#     print(hog_distance, text_similarity, SSIm,res)

template_path = "static/combined Dataset/civil hospital.jpg"
test_path = "static/combined Dataset/modified_template_87.png"
template_image = io.imread(template_path)
test_image = io.imread(test_path)
template_matcher = MedicalClaimTemplateMatcher(template_path, test_path)
hog_distance, text_similarity, SSIm,res,nmse, score = template_matcher.calculate_distances_and_similarity()
print(hog_distance, text_similarity, SSIm,res,nmse, score)