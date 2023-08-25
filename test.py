from flask import Flask, request, render_template
import requests
from PIL import Image
from io import BytesIO
import os
import cv2
import json
from plotting import add_bboxs_on_img, get_image_from_bytes
from better_detect import better_detect, better_detect_on_preprocessed_image
import pandas as pd
app = Flask(__name__)

@app.route("/",methods=["POST","GET"])
def index():
    return render_template("index.html")

PROCSESS_FOLDER = r'./static/display'

@app.route("/result", methods=["POST"])
def uploadjson():
    try:
        # Get the image from the request
        image = request.files["image"]
        # save our image in upload folder
        original_image_path = os.path.join(PROCSESS_FOLDER,"demo.jpg")
        image.save(original_image_path) # save image into upload folder
        # Upload the image to the server to get JSON
        url = "http://10.182.220.134:8000/detection/img_object_detection_to_json"     
        files = {'file': open(original_image_path, 'rb')}
        response = requests.post(url, files=files)
        data = response.json()    
        new_data = pd.DataFrame.from_dict(data['detect_objects'])
        with open(original_image_path, "rb") as im:
            f = im.read()
            bytes_im = bytearray(f)
            input_image = get_image_from_bytes(bytes_im)
            final_image = add_bboxs_on_img(input_image, new_data)
            result_image_path = os.path.join(PROCSESS_FOLDER,"all_demo.jpg")
            final_image.save(result_image_path)
        return render_template("result.html", jsondata=data, result_image_path=result_image_path, original_image_path=original_image_path)
    except Exception as e:
        return f"An error occurred: {str(e)}", 500
    
@app.route('/enhanced-output-1', methods=['GET', 'POST'])
def enhanced_output_1():   
        original_image_path = os.path.join(PROCSESS_FOLDER,"demo.jpg")
        response = better_detect(original_image_path)
        display_image_path = os.path.join(PROCSESS_FOLDER,"display.jpg")
        jsondata = response["detect_objects_names"]
        data = pd.DataFrame.from_dict(response["detect_objects"])
        with open(display_image_path, "rb") as im:
            f = im.read()
            bytes_im = bytearray(f)
            input_image = get_image_from_bytes(bytes_im)
            final_image = add_bboxs_on_img(input_image, data)
            result_image_path = os.path.join(PROCSESS_FOLDER,"test.jpg")
            final_image.save(result_image_path)        
        return render_template('enhanced-output-1.html',jsondata=jsondata)  
    

@app.route('/enhanced-output-2', methods=['GET', 'POST'])
def enhanced_output_2():  
    # pd.set_option("display.max_rows", None, "display.max_columns", None)
    original_image_path = os.path.join(PROCSESS_FOLDER,"demo.jpg")
    preprocessed_image_path = os.path.join(PROCSESS_FOLDER,"preprocessed.jpg")
    display_image_path = os.path.join(PROCSESS_FOLDER,"display.jpg")
    objectlist = request.form.getlist('object')
    response = better_detect_on_preprocessed_image(display_image_path, preprocessed_image_path)
    # response = better_detect(original_image_path)
    jsondata = response["detect_objects_names"]
    data = pd.DataFrame.from_dict(response["detect_objects"])
    new_data = data[data["name"].isin(objectlist)]
    with open(display_image_path, "rb") as im:
        f = im.read()
        bytes_im = bytearray(f)
        input_image = get_image_from_bytes(bytes_im)
        final_image = add_bboxs_on_img(input_image, new_data)
        result_image_path = os.path.join(PROCSESS_FOLDER,"test.jpg")
        final_image.save(result_image_path)        
    return render_template('enhanced-output-2.html',jsondata=jsondata)
    
@app.route('/output', methods=['GET', 'POST'])
def output():
        original_image_path = os.path.join(PROCSESS_FOLDER,"demo.jpg")
        objectlist = request.form.getlist('object')
        url = "http://10.182.220.134:8000/detection/img_object_detection_to_json"
        files = {'file': open(original_image_path, 'rb')}
        response = requests.post(url, files=files)
        data = response.json()
        df_data = pd.DataFrame.from_dict(data["detect_objects"])
        new_data = df_data[df_data["name"].isin(objectlist)]
        with open(original_image_path, "rb") as im:
            f = im.read()
            bytes_im = bytearray(f)
            input_image = get_image_from_bytes(bytes_im)
            final_image = add_bboxs_on_img(input_image, new_data)
            result_image_path = os.path.join(PROCSESS_FOLDER,"test.jpg")
            final_image.save(result_image_path)
        return render_template('output.html',jsondata=data, result_image_path=result_image_path, original_image_path=original_image_path)


if __name__ == "__main__":
    app.run(host='0.0.0.0',port='80',debug=True)
