```markdown
# 🧴 Bottle Classification on Conveyor Belt

This project is a smart vision system that classifies bottles (e.g., plastic, glass, metal) on a conveyor belt using both **YOLOv5 object detection** and a **neural network classifier** (MLPClassifier). The system supports real-time video stream processing and is extensible with Arduino for actuator control.

---

## ✨ Features

- 🧠 Neural network-based classification using tabular image features
- 🧪 YOLOv5s model for object detection and localization
- 📊 CSV dataset-based training and evaluation
- 🎥 Real-time detection with OpenCV camera feed
- ✅ Model persistence with joblib
- 🤖 Arduino support for mechanical control (e.g., sorting arm)

---

## 🧱 Tech Stack

| Component       | Technology                         |
|------------------|-------------------------------------|
| ML Model         | MLPClassifier (scikit-learn)        |
| Object Detection | YOLOv5s (PyTorch-based .pt file)    |
| Real-Time Input  | OpenCV + Camera Stream              |
| Data Format      | CSV (tabular features)              |
| Hardware Logic   | Arduino Uno (optional)              |
| Model I/O        | joblib + pandas                     |

---

## 🗂️ Project Structure

```

BottleClassificationOnConveyorBelt/
├── Code/
│   ├── camera.py                    # Real-time detection script
│   ├── objectdedection.py           # YOLOv5 integration
│   ├── yapaysiniraglari.py         # Neural network training script
│   ├── tabulardata.py              # Dataset preparation
│   ├── DedectionData\_image\_features.csv  # Feature data
│   ├── trained\_model.joblib        # Saved MLP model
│   └── yolov5s.pt                  # Pre-trained YOLOv5s model
│
└── ArduinoCode/                    # Arduino logic for actuators

````

---

## 🚀 Getting Started

### ✅ Requirements

- Python 3.10+
- Install Python dependencies:

```bash
pip install pandas scikit-learn opencv-python joblib torch torchvision
````

---

## 🧠 Model Training

To train the neural network classifier:

```bash
python Code/yapaysiniraglari.py
```

This will:

* Train an MLPClassifier on CSV feature data
* Output accuracy and evaluation reports
* Save the model as `trained_model.joblib`

---

## 🎥 Real-Time Detection

To start classifying bottles in real-time using camera feed:

```bash
python Code/camera.py
```

Or to use YOLOv5 for object detection:

```bash
python Code/objectdedection.py
```

Ensure `yolov5s.pt` and `trained_model.joblib` are in the `Code/` directory.

---

## 📊 Example Output

```
Confusion Matrix:
[[30  2]
 [ 1 27]]

Classification Report:
              precision    recall  f1-score   support
       Glass       0.96      0.94      0.95        32
     Plastic       0.93      0.96      0.94        28

Accuracy Score: 0.95
```

---

## 🔌 Arduino Integration

* The `ArduinoCode/` folder contains code to trigger motors/servos.
* Connect Arduino to PC via USB and interface using `pyserial` (if applicable).
* Possible use: push bottle left/right based on classification.

---

## 🧺 Future Improvements

* 🔍 Replace MLP with CNN-based classifier
* 📈 Integrate YOLO output with classifier fusion
* 🌐 Add web dashboard (e.g., Flask or FastAPI)
* 🤖 Integrate full IoT pipeline (Edge device + Cloud)
* 🧠 Real-time retraining on edge

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgments

* [Scikit-learn](https://scikit-learn.org/)
* [YOLOv5 by Ultralytics](https://github.com/ultralytics/yolov5)
* [OpenCV](https://opencv.org/)

```
```
