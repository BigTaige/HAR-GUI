import json
from metric import evaluate
from metric import compute_iou
import re
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 13
import pandas as pd

def get_resolution(image_path):
    with Image.open(image_path) as img:
        return img.size


def is_valid_integer(s):
    try:
        value = float(s)
        return 0 <= value <= 1000
    except ValueError:
        return False

## You can scale based on the coordinate range.
def extract_bbox(text):
    if '[' not in text or ']' not in text:
        match = text.split('<box>')[1].split('</box>')[0]
        match = match.replace(')','').replace('(','')
        match = match.replace(' ', ',')
        x1, y1, x2, y2 = match.split(",")
        coordinates_list = [int(x1), int(y1), int(x2), int(y2)]
    else:
        if '[[' in text:
            match = text.split('[[')[1].split(']]')[0]
        else:
            match = text.split('[')[1].split(']')[0]
        coordinates_list = [int(eval(x)) for x in match.split(',')]
    return coordinates_list

def extract_bbox_gpt4o(text):
   
    if '[' not in text or ']' not in text:
        coordinates_list = [0,0,0,0]
    else:
        if '[' in text and ']' in text:
            match = text.split('[')[1].split(']')[0]
            coordinates_list = [eval(x) for x in match.split(',')]
        else:
            if ']' in text:
                match = text.split(']')[0]
                coordinates_list = [eval(x) for x in match.split(',')]
            else:
                if '[' in text:
                    match = text.split('[')[1]
                    coordinates_list = [eval(x) for x in match.split(',')]
                else:
                     coordinates_list = [0,0,0,0]
    return coordinates_list

def extract_bbox_llama(text):
   
    if '[' in text and ']' in text:
        match = text.split('[')[1].split(']')[0]
        try:
            coordinates_list = [eval(x) for x in match.split(',')]
        except:
            coordinates_list = [0,0,0,0]
    else:
        coordinates_list = [0,0,0,0]
    if len(coordinates_list) != 4:
        coordinates_list = [0,0,0,0]
    return coordinates_list

def convert_to_prediction(file_path):
    ret_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    for index, item in enumerate(data):        
        ret_list.append(extract_bbox(item['pred']))

    return ret_list

def convert_to_reference(file_path):
    ret_list = []

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for index, item in enumerate(data):
        if isinstance(item['gt'][0], str):
            item['gt'] = [int(x) for x in item['gt'].split('[')[1].split(']')[0].split(',')]
        resolution = get_resolution(item['image_path'])

        item['gt'][0] = item['gt'][0] / resolution[0]  
        item['gt'][2] = item['gt'][2] / resolution[0]
        item['gt'][1] = item['gt'][1] / resolution[1] 
        item['gt'][3] = item['gt'][3] / resolution[1]
  
        
        # # 999 for CogAgent
        # item['gt'][0] = item['gt'][0] * 999 / resolution[0]
        # item['gt'][2] = item['gt'][2] * 999 / resolution[0]
        # item['gt'][1] = item['gt'][1] * 999 / resolution[1]
        # item['gt'][3] = item['gt'][3] * 999 / resolution[1]

        if isinstance(item['gt'], str):
            ret_list.append([int(x) for x in item['gt'].replace('[','').replace(']','').split(',')])
        else:
            ret_list.append(item['gt'])
        
    return ret_list


def test(path):
    predictions = convert_to_prediction(path)
    references = convert_to_reference(path)
    # print(predictions)
    evaluate(predictions, references, metric='iou')

path = './your_grounding_data.json'
test(path)

##Data Format in Your Json File:
"""
[
    {
        "image_path": xxx.png, (Optional)
        "question": UI element description text, (Optional)
        "pred": Model prediction value, e.g., "[159, 100, 240, 140]",
        "gt": Ground Truth bbox, e.g., [161, 94, 245,157]
    },
    {...},
    ...
]
"""

