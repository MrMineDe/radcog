from ultralytics import YOLO

model = YOLO('radcog-0.1.3.pt')

model.train(
        data='data.yaml',
        epochs=20,
        imgsz=512,
        batch=4,
        half=True,
        workers=2,
        patience=5,
        name='radcog-0.1'
        )
