import json
import re
from PIL import Image

def eval(preds, gts):
    num = len(preds)
    right = 0
    for i in range(len(preds)):
        point = preds[i]
        bbox = gts[i]
        #print(point, bbox)
        if cal_res(bbox, point):
            right += 1
    print(f"### {right} / {num} = {right/num}")

def cal_res(gt_bbox, pred_point):
    # print(gt_bbox, pred_point)
    x1, y1, x2, y2 = gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]
    c_x, c_y = pred_point[0], pred_point[1]

    if c_x >= x1 and c_x <= x2 and c_y >= y1 and c_y <= y2:
        return 1
    else:
        return 0

def extract_point(text):
    numbers = [float(num) for num in re.findall(r"\d*\.\d+|\d+", text)]
    numbers = numbers[1:]
    if len(numbers) > 2:
        return numbers[:2]
    else:
        return numbers

def convert_to_prediction(file_path):
    ret_list = []

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    for index, item in enumerate(data):
        ret_list.append(extract_point(item['pred']))

    return ret_list


def convert_to_reference(file_path):
    ret_list = []

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for index, item in enumerate(data):
        if isinstance(item['gt'][0], str):
            item['gt'] = [int(x) for x in item['gt'].split('[')[1].split(']')[0].split(',')]
        #w, h = Image.open(item['image_path']).size
        #item['gt'][0] = item['gt'][0] * w
        #item['gt'][2] = item['gt'][2] * w
        #item['gt'][1] = item['gt'][1] * h
        #item['gt'][3] = item['gt'][3] * h

        ret_list.append(item['gt'])
        
    return ret_list

##Data Format in Your Json File:
"""
[
    {
        "image_path": xxx.png, (Optional)
        "question": UI element description text, (Optional)
        "pred": Model prediction value, e.g., "{'point_2d': [236, 146], 'label': 'click on the search icon'}",
        "gt": Ground Truth bbox, e.g., [161, 94, 245,157]
    },
    {...},
    ...

]
"""

path = './your_grounding_data.json'

predictions = convert_to_prediction(path)
references = convert_to_reference(path)

eval(predictions, references)