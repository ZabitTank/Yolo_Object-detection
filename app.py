# for UI
from PIL import Image
import streamlit as st

# for loading  and process detect
import cv2
import numpy as np
import os
from pathlib import Path
#for download model
import urllib3

MODEL_PATH = os.path.join(Path(__file__).parent, 'model')

URL_PRETRAIN_MODEL = {
"yolov4-tiny.weights" : "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
"yolov4-tiny.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
"yolov4-tiny.names" : "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
}

PRETRAIN_MODEL_NAME = "yolov4-tiny"

URL_CUSTOM_MODEL = {
"yolov4-custom.weights" : "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-p5.weights",
"yolov4-custom.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-p5.cfg",
"yolov4-custom.names" : "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
}

MY_MODEL_NAME = "yolo-custom"

http = urllib3.PoolManager()

def download_file(file_name,url):
    weights_warning, progress_bar = None, None
    MEGABYTES = 2.0 ** 20.0
    # check if existed
    path = os.path.join(MODEL_PATH,file_name)
    
    response = http.request('GET',url,preload_content=False)
    length = int(response.headers.get("Content-Length"))
    counter = 0.0
            
    weights_warning = st.warning("Downloading %s..." % path)
    progress_bar = st.progress(0)
            
    with open(path,"wb") as output_file:
        while True:
            data =  response.read(32768)
            if not data:
                break
                
            output_file.write(data)
            counter += len(data)
                    
            weights_warning.warning("Downloading %s... (%.2f/%.2f MB)" %
            (path, counter / MEGABYTES, length / MEGABYTES))
            progress_bar.progress(min(counter / length, 1.0))
           
    weights_warning.empty()
    progress_bar.empty()
    response.release_conn()        


def check_missing_model(url_model,download_file):
    flag_download = False
    
    for file_name in url_model:
        path = path = os.path.join(MODEL_PATH,file_name)
        if not os.path.exists(path):
            flag_download = True        
            st.warning("Missing %s" % file_name)
            download_file.update({file_name:url_model[file_name]})
            
    return flag_download

def check_update(url_model,update_file):
    flag_update = False

    for file_name in url_model:
        path = os.path.join(MODEL_PATH,file_name)
       
        os.path.getsize(path)
            
        response = http.request('GET',url_model[file_name],preload_content=False)
        length = int(response.headers.get("Content-Length"))
        response.release_conn()
        
        if(length != os.path.getsize(path)):
            flag_update = True
            st.warning("Updata avaiable %s" % file_name)
            update_file.update({file_name:url_model[file_name]})
            

def show_download_model_button(model_name,url_model,files,is_update):
    button = st.button("download %s" % model_name)
    
    if(button):
        if is_update:
            check_update(url_model,files)
            
        for file_name in files.keys():
            download_file(file_name,files[file_name])
        
        if(is_update):
            st.legacy_caching.clear_cache()
        st.experimental_rerun()
        

    
def main():
    download_file = {}
    update_file = {}
    
    st.title("Yolo detection object")
    
    # SLIDE BAR : using to setting model
    st.sidebar.image("./yolo_icon.png",width=100)
    
    st.sidebar.write("19127385(Phạm Lê Hạ)")
    st.sidebar.write("19127114(Phạm Thành Đăng)")
    
    # setting model
    st.sidebar.header("Setting Model")
    
    modelName = st.sidebar.selectbox("What model you want to use?",
    ("pre-train model", "custom model")) 
    
        
    if (modelName == "pre-train model" ):
        handleNavigate(modelName,URL_PRETRAIN_MODEL,PRETRAIN_MODEL_NAME,download_file,update_file)
    elif (modelName == "custom model"):
        handleNavigate(modelName,URL_CUSTOM_MODEL,MY_MODEL_NAME,download_file,update_file)

#####################################################################################
    
# INPUT IMAGE PAGE: to insert image and process dectection

def handleNavigate(modelName,url_model,model_original_name,download_file,update_file):
    if(check_missing_model(url_model,download_file)):
        show_download_model_button(modelName,url_model,download_file,False)
    else:
        check_update_button = st.button("check update")
        if check_update_button:
            check_update(url_model,update_file)
        show_download_model_button(modelName,url_model,download_file,True)
            
        model,labels = load_model(model_original_name)
        if (labels is None or model is None):
            st.warning("Loading failed: Path could be incorrect or model %s not existed" % modelName)
        else:
            main_page_render(modelName,model,labels)

def main_page_render(modelName,model,labels):
    #col1,col2 = st.columns(2)
    
    input_form = st.form(key="input") 
    
    input_form.header("Current model: " + modelName)
    
    input_form.selectbox("Object can be detected on this model",labels)
    
    confThreshold = input_form.slider("confThreshold: ",0.01,0.99,0.1,0.1)
    nmsThreshold = input_form.slider("nmsThreshold: ", 0.001,0.99,0.4,0.1)
    
    imgSource = selectImage(input_form)
        
    detect_button = input_form.form_submit_button("Detect")
    if (detect_button):
        if imgSource is not None:
            img = load_image(imgSource)
            st.image(detect_image(model,labels,img,confThreshold,nmsThreshold))
        
#Note: cache model
#@st.cache(allow_output_mutation=True)        
def load_model(model_name):
    with st.spinner('Loading model...'):
        try:
            cfg = model_name+".cfg"
            weight = model_name+".weights"
            label = model_name+".names"
            
            cfg_path = os.path.join(MODEL_PATH,cfg)
            weights_path = os.path.join(MODEL_PATH,weight)
            labels_path = os.path.join(MODEL_PATH,label)
            
            net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
            
            model = cv2.dnn_DetectionModel(net)
            
            model.setInputSize(704, 704)
            model.setInputScale(1.0 / 255)
            model.setInputSwapRB(True)
            
            with open(labels_path, 'rt') as f:
                labels = f.read().strip().split("\n")
               
            return model, labels
        except Exception as ex:
            st.write(cfg_path)
            st.write(weights_path)
            st.write(labels_path)
            st.write(MODEL_PATH)
            st.warning(ex)
            return None, None
            
    
#Note:cache image 
@st.cache(suppress_st_warning=True)
def load_image(image_source):
    with st.spinner('Loading image...'):
        try:
            image = Image.open(image_source)
            return np.array(image)
        except Exception:
            input_form.error("Error: Invalid Image")
            return None

def selectImage(form):
    with st.spinner('Waiting loading model'):
        image_destination = form.file_uploader("Select image from file explorer",type = ["png","jpg","jpeg"])
        return image_destination


def detect_image(model,labels,img,confThreshold,nmsThreshold):
    img = cv2.resize(img, dsize=(704, 704), interpolation=cv2.INTER_AREA)
    classes, confidences, boxes = model.detect(img, confThreshold, nmsThreshold)
   
    for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
        label = '%.2f' % confidence
        label = '%s: %s' % (labels[classId], label)
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        left, top, width, height = box
        top = max(top, labelSize[1])
        
        cv2.rectangle(img, box, color=(0, 255, 0), thickness=3)
        cv2.rectangle(img, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine),
                      (255, 255, 255), cv2.FILLED)
        cv2.putText(img, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    return img

  
    
if __name__ == '__main__':
    main()
