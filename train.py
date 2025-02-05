from ultralytics import YOLO

model = YOLO('yolo11s.pt')

model.train(
        data='data.yaml',
        epochs=200,
        imgsz=1028,
        batch=-1, # auto batch so 70% of GPU Memory is used
        half=True,
        workers=6, # should be half of your cpu cores (find out with "nproc")
        patience=5, # stop when the model doesnt get better 5 epochs in a row
        cache='disk', # cache on disk instead of GPU to save data
        scale=0.5, # enable autoscaling images from 0.5-1.5 scale
        name='radcog-0.1'
)
