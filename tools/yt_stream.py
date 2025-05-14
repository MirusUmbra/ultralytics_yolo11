import sys
sys.path.insert(0, './')

import os
import cv2
import time
import subprocess
import numpy as np

from ultralytics import YOLO
import torch
# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:', device)
model = YOLO('runs/pose/train4/weights/last.pt')
# model = YOLO('yolov8n-pose.pt')
# 切换计算设备
model.to(device)

# 框（rectangle）可视化配置
bbox_color = (150, 0, 0)             # 框的 BGR 颜色
bbox_thickness = 1                   # 框的线宽

# 框类别文字
bbox_labelstr = {
    'font_size':1,         # 字体大小
    'font_thickness':1,    # 字体粗细
    'offset_x':0,          # X 方向，文字偏移距离，向右为正
    'offset_y':-10,        # Y 方向，文字偏移距离，向下为正
}
# 关键点 BGR 配色
# COCO 17 关键点 BGR 配色
kpt_color_map = {
    0: {'name':'nose', 'color':[255, 0, 0], 'radius':4},
    1: {'name':'left_eye', 'color':[0, 255, 0], 'radius':4},
    2: {'name':'right_eye', 'color':[0, 0, 255], 'radius':4},
    3: {'name':'left_ear', 'color':[255, 255, 0], 'radius':4},
    4: {'name':'right_ear', 'color':[0, 255, 255], 'radius':4},
    5: {'name':'left_shoulder', 'color':[255, 0, 255], 'radius':4},
    6: {'name':'right_shoulder', 'color':[128, 128, 0], 'radius':4},
    7: {'name':'left_elbow', 'color':[128, 0, 128], 'radius':4},
    8: {'name':'right_elbow', 'color':[0, 128, 128], 'radius':4},
    9: {'name':'left_wrist', 'color':[128, 128, 255], 'radius':4},
    10: {'name':'right_wrist', 'color':[255, 128, 128], 'radius':4},
    11: {'name':'left_hip', 'color':[128, 255, 128], 'radius':4},
    12: {'name':'right_hip', 'color':[255, 128, 0], 'radius':4},
    13: {'name':'left_knee', 'color':[0, 128, 255], 'radius':4},
    14: {'name':'right_knee', 'color':[255, 0, 128], 'radius':4},
    15: {'name':'left_ankle', 'color':[128, 0, 255], 'radius':4},
    16: {'name':'right_ankle', 'color':[0, 255, 128], 'radius':4}
}

# 点类别文字
kpt_labelstr = {
    'font_size':1,             # 字体大小
    'font_thickness':1,       # 字体粗细
    'offset_x':10,             # X 方向，文字偏移距离，向右为正
    'offset_y':0,            # Y 方向，文字偏移距离，向下为正
}

# COCO 骨架连接配置
skeleton_map = [
    (5, 7), (7, 9), (6, 8), (8, 10), # 肩膀-肘-腕
    (5, 6), (11, 12), # 左右肩膀-髋部
    (11, 13), (13, 15), # 左髋-膝-踝
    (12, 14), (14, 16), # 右髋-膝-踝
    (0, 1), (0, 2), # 鼻子-左右眼
    (1, 3), (2, 4) # 左右眼-耳
]

# 配置 YouTube 直播流 URL
# YOUTUBE_URL = "https://www.youtube.com/watch?v=VR-x3HdhKLQ" # town
# YOUTUBE_URL = "https://www.youtube.com/watch?v=KY4Yd5QR570" # bar
YOUTUBE_URL = "https://www.youtube.com/watch?v=i3w7qZVSAsY" # market

# 配置保存路径和文件名
OUTPUT_DIR = "./processed_recordings"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 文件名使用当前时间，避免覆盖
def get_output_filename():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(OUTPUT_DIR, f"processed_capture_{timestamp}.mp4")

# yt-dlp 获取流 URL
def get_stream_url():
    command = ["yt-dlp", "-g", YOUTUBE_URL]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout.strip()

# COCO 人体关键点检测推理函数
def keypoint_predict(img_bgr):
    start_time = time.time()
    # results = model(img_bgr, verbose=False) # verbose设置为False，不单独打印每一帧预测结果
    conf_threshold = 0.5
    results = model(img_bgr, verbose=False, conf=conf_threshold)
    
    # 预测框的个数
    num_bbox = len(results[0].boxes.cls)
    
    # 预测框的 xyxy 坐标

    bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('uint32') 
    
    # 关键点的 xy 坐标
    bboxes_keypoints = results[0].keypoints.xy.cpu().numpy().astype('uint32')
    # bboxes_keypoints = results[0].keypoints.cpu().numpy().astype('uint32')
    
    for idx in range(num_bbox): # 遍历每个框

        # 获取该框坐标
        bbox_xyxy = bboxes_xyxy[idx] 

        # 获取框的预测类别（对于关键点检测，只有一个类别）
        bbox_label = results[0].names[0]

        # 画框
        img_bgr = cv2.rectangle(img_bgr, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), bbox_color, bbox_thickness)

        # 写框类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
        img_bgr = cv2.putText(img_bgr, bbox_label, (bbox_xyxy[0]+bbox_labelstr['offset_x'], bbox_xyxy[1]+bbox_labelstr['offset_y']), cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color, bbox_labelstr['font_thickness'])

        bbox_keypoints = bboxes_keypoints[idx] # 该框所有关键点坐标和置信度

        # 画该框的骨架连接
        for skeleton in skeleton_map:

            bbox_keypoints = bboxes_keypoints[idx]
        for kpt_id, kpt_info in kpt_color_map.items():
            kpt_x, kpt_y = bbox_keypoints[kpt_id][0], bbox_keypoints[kpt_id][1]
            cv2.circle(img_bgr, (kpt_x, kpt_y), kpt_info['radius'], kpt_info['color'], -1)
            cv2.putText(img_bgr, kpt_info['name'], (kpt_x + 5, kpt_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, kpt_info['color'], 1)

        # 绘制骨架
        for srt_id, dst_id in skeleton_map:
            srt_x, srt_y = bbox_keypoints[srt_id][0], bbox_keypoints[srt_id][1]
            dst_x, dst_y = bbox_keypoints[dst_id][0], bbox_keypoints[dst_id][1]

            # 检查坐标是否有效且在图像范围内
            if (np.isnan(srt_x) or np.isnan(srt_y) or np.isnan(dst_x) or np.isnan(dst_y)):
                continue

            if (0 < srt_x < img_bgr.shape[1] and 0 < srt_y < img_bgr.shape[0] and 
                0 < dst_x < img_bgr.shape[1] and 0 < dst_y < img_bgr.shape[0]):
                cv2.line(img_bgr, (int(srt_x), int(srt_y)), (int(dst_x), int(dst_y)), (0, 255, 0), 2)
            
        # 画该框的关键点
        for kpt_id in kpt_color_map:

            # 获取该关键点的颜色、半径、XY坐标
            kpt_color = kpt_color_map[kpt_id]['color']
            kpt_radius = kpt_color_map[kpt_id]['radius']
            kpt_x = bbox_keypoints[kpt_id][0]
            kpt_y = bbox_keypoints[kpt_id][1]

            # 画圆：图片、XY坐标、半径、颜色、线宽（-1为填充）
            img_bgr = cv2.circle(img_bgr, (kpt_x, kpt_y), kpt_radius, kpt_color, -1)

            # 写关键点类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
            kpt_label = str(kpt_id) # 写关键点类别 ID（二选一）
            # kpt_label = str(kpt_color_map[kpt_id]['name']) # 写关键点类别名称（二选一）
            img_bgr = cv2.putText(img_bgr, kpt_label, (kpt_x+kpt_labelstr['offset_x'], kpt_y+kpt_labelstr['offset_y']), cv2.FONT_HERSHEY_SIMPLEX, kpt_labelstr['font_size'], kpt_color, kpt_labelstr['font_thickness'])
            
    # 记录该帧处理完毕的时间
    end_time = time.time()
    # 计算每秒处理图像帧数FPS
    time_err = end_time - start_time
    if time_err <= 0 : time_err = 1e9
    FPS = 1/time_err

    # 在画面上写字：图片，字符串，左上角坐标，字体，字体大小，颜色，字体粗细
    FPS_string = 'FPS  '+str(int(FPS)) # 写在画面上的字符串
    img_bgr = cv2.putText(img_bgr, FPS_string, (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 255), 2)
    
    return img_bgr


# 逐帧处理视频流
def process_stream():
    stream_url = get_stream_url()
    cap = cv2.VideoCapture(stream_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)

    FPS = 1/60
    FPS_MS = int(FPS * 1000)

    if not cap.isOpened():
        print("无法打开视频流")
        return

    output_file = get_output_filename()
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 60.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # print(f"开始逐帧处理直播流，保存到：{output_file}")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 降低分辨率（如 720p）
        frame = cv2.resize(frame, (1280, 720))

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        lab_planes = list(cv2.split(lab))

        lab_planes[0] = clahe.apply(lab_planes[0])

        lab = cv2.merge(lab_planes)

        processed_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        processed_frame = keypoint_predict(processed_frame)
        # out.write(processed_frame)
        cv2.imshow("Processed Stream", processed_frame)
        if cv2.waitKey(FPS_MS) & 0xFF == ord('q'):
            break

    cap.release()
    # out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("YouTube 直播流逐帧处理器 - 按 Q 停止")
    process_stream()
