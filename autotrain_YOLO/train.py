from ultralytics import YOLO

model = YOLO(r'E:\aHieu\autotrain_YOLO\models\yolov8n.pt')


results = model.train(data=r'E:\aHieu\autotrain_YOLO\config.yaml', epochs=20, imgsz=640)