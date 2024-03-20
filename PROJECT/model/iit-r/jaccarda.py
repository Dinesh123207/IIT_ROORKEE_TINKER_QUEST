import os
import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
class ImageTextComparator:
    def __init__(self, reference_folder):
        self.reference_folder = reference_folder
        self.reference_texts = self._extract_reference_texts()

    def _extract_text(self, image_path):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        custom_config = r'--oem 3 --psm 6'
        text_data = pytesseract.image_to_data(gray, config=custom_config, output_type=pytesseract.Output.DICT)

        extracted_text = []
        for i, word_text in enumerate(text_data['text']):
            if text_data['conf'][i] > 0:
                extracted_text.append(word_text)

        return set(extracted_text)

    def _extract_reference_texts(self):
        reference_texts = {}
        for filename in os.listdir(self.reference_folder):
            if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
                file_path = os.path.join(self.reference_folder, filename)
                reference_texts[filename] = self._extract_text(file_path)
        return reference_texts

    def jaccard_similarity(self, text1, text2):
        intersection = len(text1.intersection(text2))
        union = len(text1.union(text2))
        return intersection / union if union != 0 else 0

    def compare_user_text(self, image_path_user):
        text_user = self._extract_text(image_path_user)
        similarity_scores = {}
        for ref_filename, ref_text in self.reference_texts.items():
            similarity_scores[ref_filename] = self.jaccard_similarity(text_user, ref_text)

        top_matches = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:10]

        return top_matches




def runner2(image_path_user,reference_folder):

    comparator = ImageTextComparator(reference_folder)
    top_matches = comparator.compare_user_text(image_path_user)

    res = {}
    for match, score in top_matches:
        # print(f"Image Path: {os.path.join(reference_folder, match)} | Similarity Score: {score}")
        path = os.path.join(reference_folder, match)[14:]
        res[path] = float(str(score)[0:6])
    
    return res
