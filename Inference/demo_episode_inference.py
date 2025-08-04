import requests
import json
from tqdm import tqdm

#############################################################################################
ACTION_SPACE = """
CLICK:(x,y): Click on the element at the coordinate point (x,y) on the screen, e.g., CLICK:(1980,224).
TYPE:typed_text: An action of typing a piece of text, e.g., TYPE:"Macbook-Pro 16G Black".
COMPLETE: The goal has been completed in the current screen state.
SCROLL:UP/DOWN/LEFT/RIGHT: Scroll in a specific direction, e.g., SCROLL:UP.
LONG_PRESS:(x,y): Long press at a specific point (x,y) on the screen, e.g., LONG_PRESS:(345,2218).
BACK: Go back to the previous screen, e.g, BACK.
HOME: Go to the home screen, e.g., HOME.
=========================================
OTHER_CUSTOM_ACTIONS: ...
"""

INFERENCE_INSTRUCTION = f"""
You are a skilled assistant, interacting with the screen to accomplish the user's goals.
Here is the action space:
{ACTION_SPACE}
Your overall goal is: <goal>(goal)</goal>
Actions completed at previous steps: <history>(history)</history>

The output format should be as follows:
<think>Analyze step by step based on guidance and screen state to choose the action.</think>
<answer>The action you finally choose from "action space".</answer>"""

ACT2SUM_INSTRUCTION = f"""
Step-by-step GUI navigation task. Briefly summarize the current action.
Action space:
{ACTION_SPACE}
Goal: <goal>(goal)</goal>
Current action: <action>(action)</action>

Output Format: <summary>One-sentence summary of the action based on the screen image.</summary>"""
#############################################################################################
def execute(meta_data):
    goal, hist, img_url = meta_data
    inference_temp = INFERENCE_INSTRUCTION.replace("(goal)", goal).replace("(history)", hist)
    pred = chat_HAR_GUI_3B(img_url, inference_temp)
    return pred

def act2sum_fn(meta_data):
    goal, cur_action, img_url = meta_data
    act2sum_temp = ACT2SUM_INSTRUCTION.replace("(goal)", goal).replace("(action)", cur_action)
    pred = chat_HAR_GUI_3B(img_url, act2sum_temp)
    # pred = chat_72B(img_url, act2sum_temp)
    return pred
#############################################################################################


url = "http://localhost:8000/v1/chat/completions"
headers = {
     "Content-Type": "application/json"
}

def chat_HAR_GUI_3B(img_url, query):
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

def chat_72B(img_url, query):
    content = []
    content.append({"type": "image_url", "image_url": {"url": img_url}})
    content.append({"type": "text", "text": query})
    data = {
        "model": "Qwen2.5-VL-72B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}
        ],
        "temperature":0}
        
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response = response.json()
    response = response['choices'][0]['message']['content']

    return response

## You can also use its model loading method, such as the following (or use the Swift inference framework for faster speed),
##################################################################################
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# MAX_IMAGE_PIXELS = 2048*28*28
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "./models/HAR-GUI-3B", 
#     torch_dtype=torch.bfloat16, 
#     attn_implementation="flash_attention_2", 
#     device_map="auto"
# )
# processor = AutoProcessor.from_pretrained("./HAR-GUI-3B", max_pixels=MAX_IMAGE_PIXELS, padding_side="left")
##################################################################################

if __name__ == "__main__":

    folder = "./your_data_folder/"
    episodes = json.load(open(folder + "your_data_path.json", "r"))
    k = 4
    inference_data = []
    for i, episode in tqdm(enumerate(episodes)):
        hist_horizon = []
        for t, step in tqdm(enumerate(episode)):
            cur_hist = ""
            # You can also build ADB pipeline for online execution  #https://developer.android.com/tools/adb
            goal, gt_action, img, ep_id = step["goal"], step["ground_truth"], step["image_path"], step["episode_id"]
            img_url = 'http://localhost:6666/' + img
            
            if len(hist_horizon) == 0:
                 cur_hist = "This is the task's initial state."
            else:
                for i, act2sum_ in enumerate(hist_horizon[-k:]):
                    cur_hist += 'Step' + str(i+1) + ': ' + act2sum_ + ".\n"
            
            pred = execute((goal, cur_hist, img_url))
            think, pred_action = pred.split("<think>")[-1].split("</think>")[0].strip(), pred.split("<answer>")[-1].split("</answer>")[0].strip()
            
            #############
            # act2sum = act2sum_fn((goal, gt_action, img_url))  # Can be used for static inference
            act2sum = act2sum_fn((goal, pred_action, img_url)) # Can be used for online inference
            hist_horizon.append(act2sum.split("<summary>")[-1].split("</summary>")[0])
            
            inference_data.append({
                "episode_id": ep_id,
                "image_path": img,
                "goal": goal,
                "pred": pred,
                "history": cur_hist,
                "ground_truth": gt_action
            })
    # evaluate(inference_data)
    with open("your_saving_path.json", "w") as f:
        f.write(json.dumps(inference_data, indent=4))
