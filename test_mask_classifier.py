import cv2
import numpy as np
from tensorflow.keras.models import load_model

# load model
model = load_model("mask_classifier_mobilenetv2.h5", compile=False)

labels = ["mask", "no_mask"]

def predict(img_path):

    img = cv2.imread(img_path)

    img_resized = cv2.resize(img, (224,224))

    # BGR → RGB and normalize
    img_resized = img_resized[..., ::-1] / 255.0

    img_resized = np.expand_dims(img_resized, axis=0)

    pred = model.predict(img_resized)[0]

    print(f"Prediction: {labels[np.argmax(pred)]}, Confidence: {np.max(pred):.4f}")


predict(r"C:\Users\ASUS\Documents\Hope AI\Deep Learning\Week11-Deep Learning Module\Pytorch_Retinaface\dataset_cropped\with_mask\0_0_0 copy 5.jpg")