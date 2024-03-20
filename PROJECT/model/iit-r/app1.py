from flask import Flask, jsonify,request
from flask_cors import CORS
import os
import shutil
from catnet import main

app = Flask(__name__)
CORS(app)

@app.route('/upload2', methods=['POST'])
def index():
    file = request.files['file']
    image_file_name = file.filename
    file_path = os.path.join('../CAT-Net-main/input', image_file_name)
    file.save(file_path)

    main()

    source_file = "../CAT-Net-main/output_pred"
    destination_directory = "static"
    shutil.copy(source_file, destination_directory)

    data = {
        "status": 200,
        "op-file-name": image_file_name
    }

    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True,port=5001)