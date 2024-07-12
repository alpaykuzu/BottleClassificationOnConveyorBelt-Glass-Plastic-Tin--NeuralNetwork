import cv2
import numpy as np
import pandas as pd
from joblib import load
import torch
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import serial
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

model_classifier = load('trained_model.joblib')

ser = serial.Serial('COM3', 9600)  

def detect_objects(image, model):
    results = model(image)
    labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    target_labels = ['bottle', 'wine glass', 'cup', 'can', 'vase']
    objects = []
    coordinates = []
    for i in range(len(labels)):
        if model.names[int(labels[i])] in target_labels:
            x1, y1, x2, y2, conf = cords[i]
            x1, y1, x2, y2 = int(x1 * image.shape[1]), int(y1 * image.shape[0]), int(x2 * image.shape[1]), int(y2 * image.shape[0])
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = height / width if width > 0 else 0
            objects.append(model.names[int(labels[i])])
            coordinates.append((x1, y1, x2, y2, aspect_ratio))
    return objects, coordinates

def extract_color_distribution(image):
    r, g, b = cv2.split(image)
    hist_r, _ = np.histogram(r, bins=256, range=(0, 256))
    hist_g, _ = np.histogram(g, bins=256, range=(0, 256))
    hist_b, _ = np.histogram(b, bins=256, range=(0, 256))
    return np.concatenate([hist_r, hist_g, hist_b])

def extract_texture_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    return hist_lbp

def extract_features(frame):
    image = frame
    if image is None:
        return None

    image = cv2.resize(image, (128, 128))

    features = []

    for color_space in ['RGB', 'HSV']:
        if color_space == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
        features.extend(hist)

    color_distribution_features = extract_color_distribution(image)
    texture_features = extract_texture_features(image)

    mean_color_val = np.mean(image, axis=(0, 1))
    edges = cv2.Canny(image, 100, 200)
    edge_hist = cv2.calcHist([edges], [0], None, [256], [0, 256]).flatten()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
    glcm_features = []
    glcm_props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    for prop in glcm_props:
        glcm_props_val = graycoprops(glcm, prop)[0, 0]
        glcm_features.append(glcm_props_val)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments).flatten()
    else:
        area = 0
        perimeter = 0
        hu_moments = np.zeros(7)
    shape_features = [area, perimeter] + list(hu_moments)
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    stat_features = [mean_val, std_val]

    features += list(mean_color_val) + list(edge_hist) + glcm_features + shape_features + stat_features + list(color_distribution_features) + list(texture_features)
    
    return features

cap = cv2.VideoCapture(1)
isPredict = False
last_predict_time = time.time()

while True:
    ret, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    objects, coordinates = detect_objects(frame, model)

    current_time = time.time()

    for obj, coord in zip(objects, coordinates):
        if obj in ['bottle', 'wine glass', 'cup', 'can', 'vase'] and not isPredict:
            x1, y1, x2, y2, aspect_ratio = coord
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            region = frame[y1:y2, x1:x2]
            features = extract_features(region)
            features_df = pd.DataFrame(features).T
            features_scaled = features_df
            prediction = model_classifier.predict(features_scaled)
            if prediction == 0:
                ser.write(b'2')
            elif prediction == 1:
                ser.write(b'3')
            elif prediction == 2:
                ser.write(b'4')
            prediction_text = "Predicted Class: " + str(prediction)
            print("Predicted Class: " + str(prediction))
            cv2.putText(frame, prediction_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            isPredict = True
            last_predict_time = current_time

            ser.write(b'1')  

    if isPredict and (current_time - last_predict_time >= 12):
        ser.write(b'0')  
        isPredict = False

    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
ser.close()
