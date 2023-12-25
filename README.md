Vehicle-Detection-main 

Vehicle-Detection-main\models_zoo\yolov5\s\vehicle-license-plate-detection-s-320x320-v3.7z =>you need unzip

python license_plate_recognition.py -vmp models_zoo/yolov5/s/vehicle-license-plate-detection-s-320x320-v3.onnx -lmp models_zoo/yolov5/n/license-plate-text-detection-n-224-v2.onnx -wip ./images/warm_up_image.png -vcp ./classes/classes.txt -ltcp ./classes/classes-license-plate-text.txt -vconf 0.5 -viou 0.4 -lconf 0.5 -liou 0.4 -ltconf 0.5 -ltiou 0.4 --source ./images/charger_in.mp4 -op ./outputs/ -show -fps -debug

tkinter
python3 uimain.py
