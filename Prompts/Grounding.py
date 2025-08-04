import requests
import json
import torch
import random
import re
import numpy as np
from PIL import Image
from tqdm import tqdm

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)

url = "http://localhost:8000/v1/chat/completions"
headers = {
     "Content-Type": "application/json"
}

GROUNDING_TEMPLATE =  "Locate the element on the screen with the function or description: \"(Description)\". Keep the following output format: {\"point_2d\": [x,y], \"label\": description of the target element.}."""

def chat(img_url, description):
    query = GROUNDING_TEMPLATE.replace("(Description)", description)
    content = []
    content.append({"type": "image_url", "image_url": {"url": img_url}})
    content.append({"type": "text", "text": query})
    data = {
        "model": "Qwen2.5-VL-3B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}
        ],
        "temperature":0}
        
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response = response.json()
    response = response['choices'][0]['message']['content']

    return response

def extract_point(response):
    numbers = [float(num) for num in re.findall(r"\d*\.\d+|\d+", response)]
    numbers = numbers[1:]
    if len(numbers) > 2:
        return numbers[:2]
    elif len(numbers) == 2:
        return numbers
    else:
        return None

def cal_res(gt_bbox, pred_point):
    # print(gt_bbox, pred_point)
    x1, y1, x2, y2 = gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]
    c_x, c_y = pred_point[0], pred_point[1]

    if c_x >= x1 and c_x <= x2 and c_y >= y1 and c_y <= y2:
        return 1.
    else:
        return 0.

def evaluate(gts, preds):
    score = 0.
    for i in range(len(gts)):
        score += cal_res(gts[i], preds[i])

    return score / len(gts)

if __name__ == "__main__":
  your_path = ""
  data = json.load(open(your_path, 'r'))

  preds, gts = [], []
  for row in data:
    label, description, img = row
    res = extract_point(chat(img, description))
    gts.append(label)
    preds.append(res)

    score = evaluate(gts, preds)
    

## You can use vllm to deploy locally and mount the local images via "nohup python3 -m http.server 6666 &"".
# Or accelerate inference via swift.