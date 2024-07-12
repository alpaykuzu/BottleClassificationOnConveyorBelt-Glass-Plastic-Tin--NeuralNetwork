import os
import torch
import cv2
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_and_save_object(image_path, save_dir):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return
    
    print(f"Original image shape: {image.shape}")

    results = model(image)

    labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    target_labels = ['bottle', 'can']

    objects = []
    for i in range(len(labels)):
        if model.names[int(labels[i])] in target_labels:
            x1, y1, x2, y2, conf = cords[i]
            x1, y1, x2, y2 = int(x1 * image.shape[1]), int(y1 * image.shape[0]), int(x2 * image.shape[1]), int(y2 * image.shape[0])
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = height / width if width > 0 else 0
            objects.append((x1, y1, x2, y2, aspect_ratio))

    if objects:
        best_object = max(objects, key=lambda x: x[4])
        x1, y1, x2, y2, aspect_ratio = best_object
        detected_object = image[y1:y2, x1:x2]
        if detected_object.size != 0:
            save_path = os.path.join(save_dir, os.path.basename(image_path))
            cv2.imwrite(save_path, detected_object)
            print(f"Saved detected object to {save_path}")

base_dir = 'Data'
categories = ['Cam', 'Plastik', 'Teneke']
save_base_dir = 'DedectionData'

os.makedirs(save_base_dir, exist_ok=True)

for category in categories:
    category_path = os.path.join(base_dir, category)
    save_category_dir = os.path.join(save_base_dir, category)
    os.makedirs(save_category_dir, exist_ok=True)
    
    for filename in os.listdir(category_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(category_path, filename)
            detect_and_save_object(file_path, save_category_dir)
