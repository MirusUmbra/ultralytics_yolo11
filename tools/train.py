import sys
sys.path.insert(0, './')

from ultralytics import YOLO
from multiprocessing import freeze_support
import torch

if __name__ == '__main__':
    freeze_support()

    # train
    # model = YOLO("ultralytics/cfg/models/v8/yolov8_tune.yaml").load("stage/pre_train/yolov8n_tune.pt")

    # finetune
    # model = YOLO("./stage/prune/yolov8n_prune_start.pt")
    model = YOLO("./runs/detect/train45/weights/last.pt")

    # # Train the model
    results = model.train(data="data/human_all/data.yaml", epochs=50, save=True)
    # results = model.val(data="data/human_all/data.yaml")
    # model.predict(source="https://ultralytics.com/images/bus.jpg", save=True)  # predict on an image
    model.export(format="onnx")  # export the model to ONNX format


    model = YOLO("ultralytics/cfg/models/v8/yolov8_nano.yaml").load("yolov8n.pt")

    results = model.train(data="data/human_all/data.yaml", epochs=50, save=True)
    model.export(format="onnx")  # export the model to ONNX format
    # results = model.val(data="data/human_all/data.yaml")
    # model.predict(source="https://ultralytics.com/images/bus.jpg", save=True)  # predict on an image
    model.export(format="onnx")  # export the model to ONNX format
