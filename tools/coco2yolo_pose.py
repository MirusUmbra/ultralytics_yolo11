import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument(
    "--coco_json_path",
    default="/home/hodor/dev/data/coco_test/person_keypoints_train2017.json",
    type=str,
    help="input: coco format(json)",
)

parser.add_argument(
    "--yolo_save_root_dir",
    default="/home/hodor/dev/data/coco_test/yolo/",
    type=str,
    help="specify where to save the output dir of labels",
)


def convert_bbox_to_yolo(size, box):
    dw = 1.0 / (size[0])
    dh = 1.0 / (size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]
    # The round function determines the number of decimal places in (xmin, ymin, xmax, ymax)
    x = round(x * dw, 6)
    w = round(w * dw, 6)
    y = round(y * dh, 6)
    h = round(h * dh, 6)
    return (x, y, w, h)


def convert_keypoints2_list(keypoints, img_width, img_height):
    xiaoshu = 10 ** 6  
    arry_x = np.zeros([17, 1])
    num_1 = 0
    for x in keypoints[0:51:3]:
        arry_x[num_1, 0] = int((x / img_width) * xiaoshu) / xiaoshu
        num_1 += 1

    arry_y = np.zeros([17, 1])
    num_2 = 0
    for y in keypoints[1:51:3]:
        arry_y[num_2, 0] = int((y / img_height) * xiaoshu) / xiaoshu
        num_2 += 1

    arry_v = np.zeros([17, 1])
    num_3 = 0
    for v in keypoints[2:51:3]:
        arry_v[num_3, 0] = v
        num_3 += 1

    list_1 = []
    num_4 = 0
    for i in range(17):
        list_1.append(float(arry_x[num_4]))
        list_1.append(float(arry_y[num_4]))
        list_1.append(float(arry_v[num_4]))
        num_4 += 1
    return list_1



def main(root_dir, ana_txt_save_path_txt, json_file):
    xiaoshu = 10 ** 6  
    data = json.load(open(json_file, "r"))
    # if not os.path.exists(ana_txt_save_path_txt):
    #     os.makedirs(ana_txt_save_path_txt)

    id_map = (
        {}
    )  # The ids of the coco dataset are not continuous! Remap and output again!
    with open(os.path.join(root_dir, "classes.txt"), "w") as f:
        for i, category in enumerate(data["categories"]):
            f.write(f"{category['name']}\n")
            id_map[category["id"]] = i

    fn = Path(ana_txt_save_path_txt)

    images = {"%g" % x["id"]: x for x in data["images"]}
    # Create image-annotations dict
    imgToAnns = defaultdict(list)
    for ann in data["annotations"]:
        imgToAnns[ann["image_id"]].append(ann)

    list_file = open(os.path.join(root_dir, "train.txt"), "w")

    # Write labels file
    for img_id, anns in tqdm(imgToAnns.items(), desc=f"Annotations {json_file}"):
        img = images["%g" % img_id]
        h, w, f = img["height"], img["width"], img["file_name"]

        bboxes = []
        segments = []
        for ann in anns:
            # if ann['iscrowd']:
            #     continue
            # The COCO box format is [top left x, top left y, width, height]
            box = np.array(ann["bbox"], dtype=np.float64)
            box[:2] += box[2:] / 2  # xy top-left corner to center
            box[[0, 2]] /= w  # normalize x
            box[[1, 3]] /= h  # normalize y
            if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                continue

            # keypoints
            keypoints = ann["keypoints"]
            keypoints_list = convert_keypoints2_list(keypoints, w, h)
            # conver_keypoins2_list(keypoints,w,h)
            # print(keypoints)

            cls = id_map[ann["category_id"]]
            # cls = coco80[ann['category_id'] - 1] if cls91to80 else ann['category_id'] - 1  # class
            box = [cls] + box.tolist() + keypoints_list
            if box not in bboxes:
                bboxes.append(box)
            # Segments
            # if use_segments:
            #     if len(ann['segmentation']) > 1:
            #         s = merge_multi_segment(ann['segmentation'])
            #         s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
            #     else:
            #         s = [j for i in ann['segmentation'] for j in i]  # all segments concatenated
            #         s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
            #     s = [cls] + s
            #     if s not in segments:
            #         segments.append(s)

            # Write
            # head, tail = os.path.splitext(filename)
            # ana_txt_name = head + ".txt"
            # replace_txt_name = os.path.join(ana_txt_save_path_txt, ana_txt_name)
        with open((fn / f).with_suffix(".txt"), "w") as file:
            for i in range(len(bboxes)):
                line = (*(bboxes[i]),)  # cls, box,keypoins
                file.write(("%g " * len(line)).rstrip() % line + "\n")
        list_file.write("./images/train/%s\n" % (f))
    list_file.close()


if __name__ == "__main__":
    args = parser.parse_args()
    print("Parsing and creating directories...")
    # ROOT_DIR = args.yolo_save_root_dir
    # COCO_JSON_FILE = args.coco_json_path
    ROOT_DIR = "F:/coco/2017"
    COCO_JSON_FILE = "F:/coco/2017/annotations/person_keypoints_train2017.json"
    YOLO_ANNO_TXT_SAVE_PATH = ROOT_DIR+"/labels/train"
    print(ROOT_DIR,COCO_JSON_FILE, YOLO_ANNO_TXT_SAVE_PATH)
    # try:
    #     os.makedirs(YOLO_ANNO_TXT_SAVE_PATH, exist_ok=True)
    # except:
    #     print("Permission error!")
    main(ROOT_DIR, YOLO_ANNO_TXT_SAVE_PATH, COCO_JSON_FILE)
    

  

## USAGE 
# python coco_keypointjson2yolo.py  --coco_json_path /home/hodor/dev/data/coco_test/person_keypoints_train2017.json  --yolo_save_root_dir /home/hodor/dev/data/coco_test/yolo/