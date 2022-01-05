# for UI
from PIL import Image
import streamlit as st

# for building button inside button
import SessionState

# for loading  and process detect
import cv2
import numpy as np
import os

#for download model
import urllib3

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model')

URL_PRETRAIN_MODEL = {
"yolov4-p5.weights" : "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-p5.weights",
"yolov4-p5.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-p5.cfg",
"yolov4-p5.names" : "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
}

PRETRAIN_MODEL_NAME = "yolov4-p5"

URL_CUSTOM_MODEL_1 = {
"yolov4-tiny-custom.weights" : "https://github.com/ZabitTank/Store-training-model/raw/main/yolov4-tiny-custom_best.weights",
"yolov4-tiny-custom.cfg": "https://raw.githubusercontent.com/ZabitTank/Store-training-model/main/yolo-obj.cfg",
"yolov4-tiny-custom.names" : "https://raw.githubusercontent.com/ZabitTank/Store-training-model/main/obj.names"
}

MY_MODEL_NAME_1 = "yolov4-tiny-custom"

URL_CUSTOM_MODEL_2 = {
"yolov4-custom.weights" : "https://drive.google.com/u/0/uc?export=download&confirm=Ep9U&id=1fxOZYOdbCx5zP_kJZmWbtHfFvpe-5_7n",
"yolov4-custom.cfg": "https://raw.githubusercontent.com/ZabitTank/Store-training-model/main/yolov4-custom.cfg",
"yolov4-custom.names" : "https://raw.githubusercontent.com/ZabitTank/Store-training-model/main/obj.names"
}

MY_MODEL_NAME_2 = "yolov4-custom"


URL_FASTFOOD_MODEL = {
"yolov4-tiny-fastfood.weights" : "https://github.com/ZabitTank/Store-training-model/raw/main/yolov4-tiny-fast-food.weights",
"yolov4-tiny-fastfood.cfg": "https://raw.githubusercontent.com/ZabitTank/Store-training-model/main/yolov4-tiny-fast-food.cfg",
"yolov4-tiny-fastfood.names" : "https://raw.githubusercontent.com/ZabitTank/Store-training-model/main/fast-food.names"
}

FASTFOOD_MODEL_NAME = "yolov4-tiny-fastfood"



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
    if(is_update):
        button = st.button("Update %s" % model_name)
    else:
        button = st.button("Download %s" % model_name)
    
    if(button):
        if is_update:
            check_update(url_model,files)
            
        for file_name in files.keys():
            download_file(file_name,files[file_name])
        
        if(is_update):
            st.legacy_caching.clear_cache()
        st.experimental_rerun()


def load_evaluation_img(path):
    st.header("Fast food evaluation chart")
    st.image(path+"food-early-chart.png")
    st.image(path+"food-mid-chart.png")
    st.image(path+"food-final-chart.png")
    st.header("Fast food evaluation value")
    st.image(path+"food-final-score.png")
    st.header("Test model evaluation chart")            
    st.image(path+"test-evaluation.png")
    st.header("Fast food evaluation value")
    st.image(path+"test-final-score.png")
    
    
    
        
def main():
    download_file = {}
    update_file = {}
    
    # SLIDE BAR : using to setting model
    with st.sidebar:
        st.sidebar.image("./yolo_icon.png",width=100)
        st.sidebar.write("19127385(Phạm Lê Hạ)")
        st.sidebar.write("19127114(Phạm Thành Đăng)")
        
        # setting model
        with st.form(key="size"):
            st.header("Setting Model")
            image_size = st.slider("Width and Height: ",32,896,416,32)
            
            modelName = st.selectbox("What model you want to use?",
            ("Pre-train model", "Train model (yolov4-tiny)","Fast food detection","Evaluation"))
            
            loading_model_button = st.form_submit_button("Load")
        
        clear_cache_button = st.sidebar.button("clear cache")
        if(clear_cache_button): st.legacy_caching.clear_cache()
    
    ss = SessionState.get(loading_model_button = False)
    
    if(loading_model_button):
        ss.loading_model_button = True
    
    if(ss.loading_model_button):
        
        if (modelName == "Pre-train model" ):
            st.title("Yolo pre-train model: yolov4-p5")
            handleNavigate(modelName,image_size,URL_PRETRAIN_MODEL,PRETRAIN_MODEL_NAME,download_file,update_file)
            
        elif (modelName == "Train model (yolov4-tiny)"):
            st.title(modelName)
            handleNavigate(modelName,image_size,URL_CUSTOM_MODEL_1,MY_MODEL_NAME_1,download_file,update_file)
            
        elif (modelName == "Fast food detection"):
            st.title(modelName)
            handleNavigate(modelName,image_size,URL_FASTFOOD_MODEL,FASTFOOD_MODEL_NAME,download_file,update_file)

        elif (modelName == "Evaluation"):
            st.write("File size is too large. I can't upload to github. So this model is only demo in local machine")
            st.write("One more thing is this model has no evaluation information")
            st.write("I am using this page to show evaluation chart of another training model")
            
            load_evaluation_img("./image/")
            

#####################################################################################
    
# INPUT IMAGE PAGE: to insert image and process dectection

def handleNavigate(modelName,image_size,url_model,model_original_name,download_file,update_file):
    if(check_missing_model(url_model,download_file)):
        show_download_model_button(modelName,url_model,download_file,False)
    else:
        check_update_button = st.button("Check update")
        if check_update_button:
            check_update(url_model,update_file)
        show_download_model_button(modelName,url_model,download_file,True)
            
        model,labels = load_model(model_original_name,image_size)
        if (labels is None or model is None):
            st.warning("Loading failed: Path could be incorrect or model %s not existed" % modelName)
        else:
            main_page_render(modelName,image_size,model,labels)

def main_page_render(modelName,image_size,model,labels):
    #col1,col2 = st.columns(2)
    st.write("Current Input size: " , image_size)
    input_form = st.form(key="input") 
    
    input_form.header("Current model: " + modelName)
    
    input_form.selectbox("Object can be detected on this model",labels)
    
    confThreshold = input_form.slider("confThreshold: ",0.0,1.0,0.1,0.01)
    nmsThreshold = input_form.slider("nmsThreshold: ", 0.0,1.0,0.4,0.01)
    
    imgSource = selectImage(input_form)
        
    detect_button = input_form.form_submit_button("Detect")
    if (detect_button):
        if imgSource is not None:
            img = load_image(imgSource)
            predict_img = detect_image(model,labels,img,confThreshold,nmsThreshold,image_size)
            if(predict_img is not None):
                st.image(predict_img)
        
#Note: cache model
@st.cache(allow_output_mutation=True)        
def load_model(model_name,image_size):
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
            
            model.setInputSize(image_size, image_size)
            model.setInputScale(1.0 / 255)
            model.setInputSwapRB(True)
            
            with open(labels_path, 'rt') as f:
                labels = f.read().strip().split("\n")
               
            return model, labels
        except Exception as ex:
            print(ex)
            return None, None
            
    
#Note:cache image 
@st.cache(show_spinner=False)
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


def detect_image(model,labels,img,confThreshold,nmsThreshold,image_size):
    try:
    
        img = cv2.resize(img, dsize=(image_size, image_size), interpolation=cv2.INTER_AREA)
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
    except Exception as e:
        st.warning("can't detect this image, try change Threshold value or image size" )
        return None

  
    
if __name__ == '__main__':
    main()
