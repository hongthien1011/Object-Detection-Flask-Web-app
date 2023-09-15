from flask import Flask, request, render_template
import requests
import os
from plotting import add_bboxs_on_img, get_image_from_bytes
from better_detect import better_detect_on_preprocessed_image, onetotwoimage, better_detect,combine_json
import pandas as pd

app = Flask(__name__)

result_count_dict = {
    'door' : 0,
    '2door' : 0,
    'window1' : 0,
    'window2' : 0,
    'baywindow' : 0,
    'window4' : 0,
    'window5' : 0,
    'window6' : 0,
    'black_wall' : 0,
    'white_wall' : 0,
    'grey_wall' : 0,
    'cross_wall' : 0
}
@app.route("/",methods=["POST","GET"])
def index():
    return render_template("index.html")

PROCSESS_FOLDER = r'./static/display'

@app.route("/result", methods=["POST"])
def uploadjson():
    url_org = "http://10.182.220.137:8000/detection/img_object_detection_to_json"
    url_sliced = "http://10.182.220.137:8000/detection/sliced_img_object_detection_to_json"
    original_image_path = os.path.join(PROCSESS_FOLDER,"original.jpg")
    try:
        image = request.files["image"]                 
        image.save(original_image_path)  

        files_1 = {'file': open(original_image_path, 'rb')}
        response_org = requests.post(url_org, files=files_1)       
        jsondata_org = response_org.json()

        files_2 = {'file': open(original_image_path, 'rb')}
        response_sliced = requests.post(url_sliced, files=files_2)    
        jsondata_sliced = response_sliced.json()

        jsondata_combine = combine_json(jsondata_org,jsondata_sliced)
    
        new_data = pd.DataFrame.from_dict(jsondata_combine['detect_objects'])

        global global_json 
        global_json = jsondata_combine
        count_dict = dict(new_data.value_counts("name"))
        for x in count_dict.keys():
            result_count_dict[x] = count_dict[x]

        with open(original_image_path, "rb") as im:
            f = im.read()
            bytes_im = bytearray(f)
            input_image = get_image_from_bytes(bytes_im)
            final_image = add_bboxs_on_img(input_image, new_data)
            result_image_path = os.path.join(PROCSESS_FOLDER,"original_pred.jpg")
            final_image.save(result_image_path)        
    except:
        objectlist = request.form.getlist('object')
        # files_1 = {'file': open(original_image_path, 'rb')}
        # response_org = requests.post(url_org, files=files_1)       
        # jsondata_org = response_org.json()

        # files_2 = {'file': open(original_image_path, 'rb')}
        # response_sliced = requests.post(url_sliced, files=files_2)    
        # jsondata_sliced = response_sliced.json()
        jsondata_combine = global_json
        new_data = pd.DataFrame.from_dict(jsondata_combine["detect_objects"])
        count_dict = dict(new_data.value_counts("name"))
        for x in count_dict.keys():
            result_count_dict[x] = count_dict[x]

        df_data = new_data[new_data["name"].isin(objectlist)]
        with open(original_image_path, "rb") as im:
            f = im.read()
            bytes_im = bytearray(f)
            input_image = get_image_from_bytes(bytes_im)
            final_image = add_bboxs_on_img(input_image, df_data)
            result_image_path = os.path.join(PROCSESS_FOLDER,"original_pred.jpg")
            final_image.save(result_image_path)
    
    return render_template("result.html", jsondata=jsondata_combine
                           ,door_count = result_count_dict['door']
                           ,doubledoor_count = result_count_dict['2door']
                           ,window1_count = result_count_dict['window1']
                           ,window2_count = result_count_dict['window2']
                           ,baywindow_count = result_count_dict['baywindow']
                           ,window4_count = result_count_dict['window4']
                           ,window5_count = result_count_dict['window5']
                           ,window6_count = result_count_dict['window6']
                           ,black_wall_count = result_count_dict['black_wall']
                           ,white_wall_count = result_count_dict['white_wall']
                           ,grey_wall_count = result_count_dict['grey_wall']
                           ,cross_wall_count = result_count_dict['cross_wall'])

@app.route('/preview_preprocessing', methods=['GET', 'POST'])
def preview_preprocessing():  
        try: 
            image = request.files["image"]
            original_image_path = os.path.join(PROCSESS_FOLDER,"original.jpg")
            image.save(original_image_path)
            listPreprocessing = request.form.getlist('preprocessing')
            onetotwoimage(original_image_path, listPreprocessing)
        except:
            original_image_path = os.path.join(PROCSESS_FOLDER,"original.jpg")
            listPreprocessing = request.form.getlist('preprocessing')         
            onetotwoimage(original_image_path, listPreprocessing)       
        return render_template('preview_preprocessing.html', listPreprocessing=listPreprocessing)

@app.route('/enhanced-output-1', methods=['GET', 'POST'])
def enhanced_output_1():   
        preprocessed_image_path = os.path.join(PROCSESS_FOLDER,"preprocessed.jpg")
        display_image_path = os.path.join(PROCSESS_FOLDER,"display.jpg")
        
        response_preprocessing = better_detect_on_preprocessed_image(display_image_path, preprocessed_image_path)
        json_preprocessing = response_preprocessing

        files_1 = {'file': open(preprocessed_image_path, 'rb')}
        url_sliced = "http://10.182.220.137:8000/detection/sliced_img_object_detection_to_json"
        response_sliced = requests.post(url_sliced, files=files_1)    
        jsondata_sliced = response_sliced.json()

        jsondata_combine = combine_json(json_preprocessing,jsondata_sliced)

        global global_json_preprocessing 
        global_json_preprocessing = jsondata_combine
        data = pd.DataFrame.from_dict(jsondata_combine["detect_objects"])
        count_dict = dict(data.value_counts("name"))
        for x in count_dict.keys():
            result_count_dict[x] = count_dict[x]
        with open(display_image_path, "rb") as im:
            f = im.read()
            bytes_im = bytearray(f)
            input_image = get_image_from_bytes(bytes_im)
            final_image = add_bboxs_on_img(input_image, data)
            result_image_path = os.path.join(PROCSESS_FOLDER,"preprocessed_pred.jpg")
            final_image.save(result_image_path)        
        return render_template('enhanced-output-1.html',jsondata=jsondata_combine
                               ,door_count = result_count_dict['door']
                           ,doubledoor_count = result_count_dict['2door']
                           ,window1_count = result_count_dict['window1']
                           ,window2_count = result_count_dict['window2']
                           ,baywindow_count = result_count_dict['baywindow']
                           ,window4_count = result_count_dict['window4']
                           ,window5_count = result_count_dict['window5']
                           ,window6_count = result_count_dict['window6']
                           ,black_wall_count = result_count_dict['black_wall']
                           ,white_wall_count = result_count_dict['white_wall']
                           ,grey_wall_count = result_count_dict['grey_wall']
                           ,cross_wall_count = result_count_dict['cross_wall'])
    

@app.route('/enhanced-output-2', methods=['GET', 'POST'])
def enhanced_output_2():  
    # preprocessed_image_path = os.path.join(PROCSESS_FOLDER,"preprocessed.jpg")
    display_image_path = os.path.join(PROCSESS_FOLDER,"display.jpg")
    objectlist = request.form.getlist('object')

    # response_preprocessing = better_detect_on_preprocessed_image(display_image_path, preprocessed_image_path)
    # json_preprocessing = response_preprocessing
    
    # files_1 = {'file': open(preprocessed_image_path, 'rb')}
    # url_sliced = "http://10.182.220.137:8000/detection/sliced_img_object_detection_to_json"
    # response_sliced = requests.post(url_sliced, files=files_1)    
    # jsondata_sliced = response_sliced.json()

    jsondata_combine = global_json_preprocessing
    data = pd.DataFrame.from_dict(jsondata_combine["detect_objects"])
    new_data = data[data["name"].isin(objectlist)]
    
    with open(display_image_path, "rb") as im:
        f = im.read()
        bytes_im = bytearray(f)
        input_image = get_image_from_bytes(bytes_im)
        final_image = add_bboxs_on_img(input_image, new_data)
        result_image_path = os.path.join(PROCSESS_FOLDER,"preprocessed_pred.jpg")
        final_image.save(result_image_path)        
    return render_template('enhanced-output-2.html',jsondata=jsondata_combine
                           ,door_count = result_count_dict['door']
                           ,doubledoor_count = result_count_dict['2door']
                           ,window1_count = result_count_dict['window1']
                           ,window2_count = result_count_dict['window2']
                           ,baywindow_count = result_count_dict['baywindow']
                           ,window4_count = result_count_dict['window4']
                           ,window5_count = result_count_dict['window5']
                           ,window6_count = result_count_dict['window6']
                           ,black_wall_count = result_count_dict['black_wall']
                           ,white_wall_count = result_count_dict['white_wall']
                           ,grey_wall_count = result_count_dict['grey_wall']
                           ,cross_wall_count = result_count_dict['cross_wall'])

if __name__ == "__main__":
    app.run(host='0.0.0.0',port='80',debug=True)
