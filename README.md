---
# Error-Centric Learning for Reasoning Improvement in GUI Agent (AAAI 2026)
<div align="center">
  <img width="85%" src="LFF-Framework.png">
</div>

## Introduction

**LFF-GUI-3B** is a GUI-tailored basic model (native end-to-end GUI agent) built upon Qwen2.5-VL-3B-Instruct. It was developed through our LFF Framework, incorporating a series of tailored training strategies. LFF-GUI-3B integrates a stable short-term memory for episodic reasoning, which can perceive the sequential clues of the episode flexibly and make reasonable use of it. This enhancement of reasoning can assist the GUI agent in executing long-horizon interaction and achieving consistent and persistent growth across GUI-oriented tasks. Further details can be found in our article.

## Quick Start

### 1. Download Model Weights

We provide the foundational model and several fine-tuned versions for specific tasks. **LFF-GUI-3B** is the base model, while the others are fine-tuned on top of it.

**To comply with AAAI Org's anonymity guidelines, the model weight download address will be provided upon paper acceptance. (HuggingFace/ModelScope/Github)**

| Model | Description | Download Link |
| :--- | :--- | :--- |
| **LFF-GUI-3B** | The GUI-tailored basic model via our LFF Framework. | [Link] |
| **LFF-GUI-3B-AITW** | Fine-tuned on the AITW dataset. | [Link] |
| **LFF-GUI-3B-Mind2Web**| Fine-tuned on the Mind2Web dataset. | [Link] |
| **LFF-GUI-3B-GUI-Odyssey**| Fine-tuned on the GUI-Odyssey dataset. | [Link] |
| **LFF-GUI-3B-GUI-Understanding**| Fine-tuned for comprehensive GUI understanding tasks. | [Link] |

### 2. Environment Setup

We recommend using a virtual environment to manage dependencies.

```bash
# Clone the repository
git clone [https://github.com/xxxxxxxxxxxx/LFF-GUI.git]

# Create and activate a virtual environment (optional but recommended)
conda create -n LFF-GUI python=3.10
conda activate LFF-GUI

# Install the required packages
pip install -r requirements.txt
```

## Custom Fine-tuning

We provide a script to fine-tune the model on your own custom datasets or tasks. Before running, you may need to modify the script to point to your data and set your desired training parameters.

```bash
# Run the training script (w/ LoRA fine-tuning)
sh training_scripts/scripts/finetune_lora.sh
# Run the training script (full-parameter fine-tuning)
sh training_scripts/scripts/finetune.sh
```

## Instruction Templates

All instruction templates used in our paper and experiments can be found in the `/Prompts` directory. These are crucial for replicating our results and for prompting the model effectively.

## Data Sample

We list all types of training data we use in the LFF framework in `./Data` directory.

## Deployment and Inference

To accelerate inference speed, we support two primary methods:
1.  **Local vLLM Deployment:**
   ```bash
    sh Inference/swift_inference.sh
   ```
2.  **Swift Framework:**
  ```bash
    sh Inference/swift_inference.sh
  ```

## Usage Example: GUI Episode Reasoning (GUI Automation)

The following Python script demonstrates how to use the model for a GUI automation task. This example assumes you have a local vLLM server running the model. You can adapt the code to fit your specific needs.
```bash
# Start vllm service
nohup python -m vllm.entrypoints.openai.api_server --served-model-name Qwen2.5-VL-3B-Instruct --model ./LFF-GUI-3B -tp 4 > log.txt &
#nohup python -m vllm.entrypoints.openai.api_server --served-model-name Qwen2.5-VL-72B-Instruct --model ./Qwen2.5-VL-72B-Instruct -tp 8 > log.txt &

# Mount your local directory
cd ./your_directory/
python3 -m http.server 6666
```
python -m vllm.entrypoints.openai.api_server --served-model-name Qwen2-VL-72B-Instruct --model Qwen/Qwen2-VL-72B-Instruct -tp 8
```python
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
    pred = chat_LFF_GUI_3B(img_url, inference_temp)
    return pred

def act2sum_fn(meta_data):
    goal, cur_action, img_url = meta_data
    act2sum_temp = ACT2SUM_INSTRUCTION.replace("(goal)", goal).replace("(action)", cur_action)
    pred = chat_LFF_GUI_3B(img_url, act2sum_temp)
    # pred = chat_72B(img_url, act2sum_temp)
    return pred
#############################################################################################

url = "http://localhost:8000/v1/chat/completions"
headers = {
     "Content-Type": "application/json"
}

def chat_LFF_GUI_3B(img_url, query):
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
#     "./LFF-GUI-3B", 
#     torch_dtype=torch.bfloat16, 
#     attn_implementation="flash_attention_2", 
#     device_map="auto"
# )
# processor = AutoProcessor.from_pretrained("./LFF-GUI-3B", max_pixels=MAX_IMAGE_PIXELS, padding_side="left")
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
```

## Citation

If you use LFF-GUI-3B or its derivatives in your research, please cite our paper (coming soon) 🙏


