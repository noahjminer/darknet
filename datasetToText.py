import os
import json

#training set
with open('train.txt', 'w') as f:
    img_id = 0
    for city in os.listdir("train/train"):
        for image in os.listdir(f"train/train/{city}"):
            img_id += 1
            img_dir = f"train/train/{city}/{image}"
            line = f"{img_id} {img_dir}"
            json_file = image.split(".")[0] + ".json"
            json_path = f"ECP_day_labels_train/ECP/day/labels/train/{city}/{json_file}"
            with open(json_path, 'r') as jf:
                data = json.load(jf)
            width = data["imagewidth"]
            height = data["imageheight"]
            line += f" {width} {height}"
            count = 1
            for object in data["children"]:
                if (object["identity"] == "pedestrian"):
                    label = str(count)
                    xMin = object["x0"]
                    yMin = object["y0"]
                    xMax = object["x1"]
                    yMax = object["y1"]
                    line += f" {label} {xMin} {yMin} {xMax} {yMax}"
                    count += 1
            f.write(line + '\n')
f.close()

#validation set
with open('val.txt', 'w') as f:
    img_id = 0
    for city in os.listdir("val/val"):
        for image in os.listdir(f"val/val/{city}"):
            img_id += 1
            img_dir = f"val/val/{city}/{image}"
            line = f"{img_id} {img_dir}"
            json_file = image.split(".")[0] + ".json"
            json_path = f"ECP_day_labels_val/ECP/day/labels/val/{city}/{json_file}"
            with open(json_path, 'r') as jf:
                data = json.load(jf)
            width = data["imagewidth"]
            height = data["imageheight"]
            line += f" {width} {height}"
            count = 1
            for object in data["children"]:
                if (object["identity"] == "pedestrian"):
                    label = str(count)
                    xMin = object["x0"]
                    yMin = object["y0"]
                    xMax = object["x1"]
                    yMax = object["y1"]
                    line += f" {label} {xMin} {yMin} {xMax} {yMax}"
                    count += 1
            f.write(line + '\n')
f.close()