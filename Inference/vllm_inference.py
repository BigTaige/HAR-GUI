import requests
import json
import sys
sys.path.append("../Prompts")
from Act2Sum import ODYSSEY_ACT2SUM_PROMPT, MIND2WEB_ACT2SUM_PROMPT, AITW_ACT2SUMSUM_PROMPT
from GuidelinesSynthesis import GUIDELINE_SYNTHESIS_TEMPLATE
from Inference import MIND2WEB_EXECUTION_PROMPT, ODYSSEY_EXECUTION_PROMPT, AITW_EXECUTION_PROMPT

url = "http://localhost:8000/v1/chat/completions"
headers = {
     "Content-Type": "application/json"
}
def chat(img_url, query):
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


#### Take Act2Sum as an example

if __name__ == "__main__":
    folder = "./image_folder"
    episode = json.load(open("your_path.json", "r"))
    training_data = []
    for i, step in enumerate(episode):
        goal, action, img = step["goal"], step["ground_truth"], step["image_path"]
        query = ODYSSEY_ACT2SUM_PROMPT.replace("(goal)", goal).replace("(action)", action)
        img_url = 'http://localhost:6666/' + img
        act2sum = chat(img_url, query)
        item = {
            "image": folder + img,
            "conversations":[
                {
                    "from": "human",
                    "value": query
                },
                {
                    "from": "gpt",
                    "value": act2sum
                }
            ]
        }
        training_data.append(item)
    
    with open("your_saving_path.json", "w") as f:
        f.write(json.dumps(training_data, indent=4))
