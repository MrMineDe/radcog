from ultralytics import YOLO

model = YOLO('yolo11s.pt')

model.train(
        data='data.yaml',
        epochs=200,
        imgsz=512,
        batch=1, # auto batch so 70% of GPU Memory is used
        half=True,
        workers=6, # should be half of your cpu cores (find out with "nproc")
        patience=5, # stop when the model doesnt get better 5 epochs in a row
        cache='disk', # cache on disk instead of GPU to save data
        scale=0.5, # enable autoscaling images from 0.5-1.5 scale
        amp=True, # enable mixed Precision Training
        momentum=0.9, # set the momentum optimizer to 0.9 as it provides "good balance"
        verbose=True,
        name='radcog-0.1'
)
