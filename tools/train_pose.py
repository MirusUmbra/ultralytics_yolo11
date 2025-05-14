import sys
sys.path.insert(0, './')

from ultralytics import YOLO
from multiprocessing import freeze_support
import torch

if __name__ == '__main__':
    freeze_support()

    # train teacher
    model = YOLO("runs/pose/train4/weights/last.pt")
    # model = YOLO("ultralytics/cfg/models/v8/yolov8_tune_teacher.yaml").load("runs/detect/train4/weights/last.pt")
    
    # model = YOLO("ultralytics/cfg/models/11/yolo11s_tune.yaml").load("yolo11s.pt")

    # results = model.train(data="ultralytics/cfg/datasets/coco-pose.yaml", epochs=100, save=True)
    # model.predict(source="https://ultralytics.com/images/bus.jpg", save=True)  # predict on an image
    model.export(format="onnx")  # export the model to ONNX format


    # train student
    # model_s = YOLO("ultralytics/cfg/models/v8/yolov8n_pose_test.yaml").load("stage/pre_train/yolov8n.pt")
    # model_t = YOLO("ultralytics/cfg/models/11/yolo11s_tune.yaml").load("runs/detect/train/weights/last.pt")

    # results = model_s.train(data="data/human_all/data.yaml", teacher=model_t,
    #                        distillation_loss="CWDLoss", epochs=50, save=True)
    # model_s.predict(source="https://ultralytics.com/images/bus.jpg", save=True)  # predict on an image
    # model_s.export(format="onnx")  # export the model to ONNX format


