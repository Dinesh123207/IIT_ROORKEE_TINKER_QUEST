from rectangle import runner1
from jaccarda import runner2
from sift_knn import runner3
from esdd import runner4
from flann import runner6
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import shutil

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'upload_folder'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def serialize_result(result):
    try:
        # Try to convert the result to a JSON-serializable format
        return json.dumps(result)
    except TypeError:
        # If not serializable, handle accordingly (e.g., convert to string)
        return str(result)
    
@app.route('/upload1', methods=['POST'])
def upload_files1():
    
    files = request.files.getlist('files')

    for file in files:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
    
    source_folder = "upload_folder"
    destination_folder = "static/static"
    for filename in os.listdir(source_folder):
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, filename)
        shutil.copy2(source_path, destination_path)
    data = {
        'message': 'Files uploaded successfully',
        'upload_file': UPLOAD_FOLDER
    }
    return jsonify(data)

@app.route('/upload2', methods=['POST'])
def upload_files2():
    
    file = request.files['file']
    file_path = os.path.join('static', file.filename)
    file.save(file_path)    
    # Path to the query image
    query_image_path = file_path  # Change to your query image path

    # Image directory
    image_dir = UPLOAD_FOLDER  # Change to your image directory

    # res_1 = serialize_result(runner1(query_image_path,image_dir))
    
    # res_2 = serialize_result(runner2(query_image_path,image_dir))

    res_4 = runner4(query_image_path,image_dir)

    length_of_esd = len(res_4)
    
    if length_of_esd >= 6:
        res_3 = serialize_result(runner6(image_dir))
        data = {
            "status": 200,
            "res_4": res_4,
            "res_3": res_3
        }
    else:
         res_3 = serialize_result(runner3(query_image_path,image_dir))
         data = {
            "status": 200,
            "res_4": res_3
        }

    return jsonify(data)   

if __name__ == '__main__':
    app.run(debug=True,port=5002)