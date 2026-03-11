import sys
sys.path.append(r"C:\Users\babuk\Documents\HOPE AI\Deep Learning\Week11-Deep Learning Module\Pytorch_Retinaface")

import cv2
import torch
import numpy as np
#from tensorflow.keras.models import load_model
import tensorflow as tf
from models.retinaface import RetinaFace
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms
from data import cfg_re50

cfg = cfg_re50

mask_model = tf.keras.models.load_model("mask_classifier_mobilenetv2.h5", compile=False)

class_names= ['mask','no_mask']

device = torch.device("cpu")   # using CPU version
print("Using device:", device)

net = RetinaFace(cfg=cfg, phase='test')
net.load_state_dict(torch.load("weights/Resnet50_Final_fixed.pth", map_location=device))
net.to(device)
net.eval()


def detect_faces(frame):
    
    img= frame.copy()
    h,w,_=img.shape
    
    image = np.float32(img)
    image -=(104,117,123)
    image = image.transpose(2,0,1)
    image = torch.from_numpy(image).unsqueeze(0).to(device)
    
    
    loc,conf,landms=net(image)
    
    priorbox = PriorBox(cfg,image_size=(h,w))
    priors = priorbox.forward().to(device)
    
    boxes = decode(loc.data.squeeze(),priors.data,cfg['variance'])
    boxes = boxes*torch.Tensor([w,h,w,h]).to(device)
    
    scores = conf.squeeze()[:,-1].detach().cpu().numpy()
    
    inds = np.where(scores >0.7)[0]
    if (len(inds))==0:
        return []
    boxes= boxes[inds].cpu().numpy()
    scores = scores[inds]
    
    dets = np.hstack((boxes,scores[:,None]))
    keep = py_cpu_nms(dets, 0.4)
    return dets[keep]

def classify_face(face_img):
    face = cv2.resize(face_img, (224, 224))
    face = face[..., ::-1] / 255.0   # BGR → RGB and scale
    pred = mask_model.predict(face[np.newaxis, ...])[0]
    class_id = np.argmax(pred)
    confidence = pred[class_id]
    return class_names[class_id], confidence

cap = cv2.VideoCapture(0)
print("Webcam started...")

while True:
    ret,frame = cap.read()
    if not ret:
        break;
    
    detections = detect_faces(frame)
    
    for det in detections:
        x1,y1,x2,y2,score = det.astype(int)
        
        # Clip coordinates to avoid errors
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        face_crop = frame[y1:y2,x1:x2]
        
        if face_crop.size ==0:
            continue
        
        label,conf = classify_face(face_crop)
        if label is None:
            continue
        
        color = (0,255,0) if label =="mask" else (0,0,225)
        
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        
        text = f"{label}: {conf:.2f}"
        
        cv2.putText(frame, text, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Display the webcam window
    cv2.imshow("Mask Detection", frame)

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
        
        