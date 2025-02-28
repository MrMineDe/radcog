from ultralytics import YOLO

model = YOLO('yolo11s')

model.train(
        data='data.yaml',
        epochs=200,
        imgsz=1028,
        batch=-1, # auto batch so 70% of GPU Memory is used
        # half=True,
        workers=6, # should be half of your cpu cores (find out with "nproc")
        patience=10, # stop when the model doesnt get better 5 epochs in a row
        cache='disk', # cache on disk instead of GPU to save data

        # optimizer='AdamW', # Handles rapid movements better than SGD
        # lr0=0.001, # Lower learning rate for fine detection
        # cos_lr=True, # Cosine learning rate scheduler
        # augment=True, # enable strong data augmentation
        # mosaic=1.0, # Increases object variation
        # hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, # Color jittering
        # perspective=0.001, flipud=0.0, fliplr=0.5, # Flipping and warping
        # degrees=15, shear=2.0, scale=0.5, translate=0.2, #Helps with blurred and occluded views
        name='radcog-0.6'
)
