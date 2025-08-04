###### Mind2Web
MIND2WEB_ACTION_SPACE_PROMPT ="""
CLICK:(x,y): Click on the element at the coordinate point (x,y) on the screen, e.g., CLICK:(1980,224).
TYPE:(x,y,typed_text): An action of typing a piece of text at the coordinate point (x,y), e.g., TYPE:(208,1082,Macbook-Pro 16G Black).  
SELECT:(x,y,option): Opens the Select Menu or Drop-Down List at coordinate (x, y) and selects the option you specify, e.g., SELECT:(59,892,Chicago)."""

###### GUI-ODYSSEY
ODYSSEY_ACTION_SPACE_PROMPT ="""
CLICK:(x,y): Click on the element at the coordinate point (x,y) on the screen, e.g., CLICK:(1980,224).
TYPE:typed_text: An action of typing a piece of text, e.g., TYPE:"Macbook-Pro 16G Black".
COMPLETE: The goal has been completed in the current screen state.
SCROLL:UP/DOWN/LEFT/RIGHT: Scroll in a specific direction, e.g., SCROLL:UP.
LONG_PRESS:(x,y): Long press at a specific point (x,y) on the screen, e.g., LONG_PRESS:(345,2218).
BACK: Go back to the previous screen, e.g, BACK.
HOME: Go to the home screen, e.g., HOME.
PRESS_RECENT: The action to go to the previous app.
IMPOSSIBLE: No action can be taken to achieve the goal based on current screen."""

###### AITW
AITW_ACTION_SPACE_PROMPT ="""
CLICK:(x,y): Click on the element at the coordinate point (x,y) on the screen, e.g., CLICK:(1980,224).
TYPE:typed_text: An action of typing a piece of text, e.g., TYPE:"Macbook-Pro 16G Black".
COMPLETE: The goal has been completed in the current screen state.
SCROLL:UP/DOWN/LEFT/RIGHT: Scroll in a specific direction, e.g., SCROLL:UP.
BACK: Go back to the previous screen, e.g, BACK.
HOME: Go to the home screen, e.g., HOME.
ENTER: Press the ENTER key to submit input content."""

######
AITW_ACT2SUMSUM_PROMPT = f"""Step-by-step GUI navigation task. Briefly summarize the current action.
Action space:
{AITW_ACTION_SPACE_PROMPT}

Goal: <goal>(goal)</goal>
Current action: <action>(action)</action>

Output Format: <summary>One-sentence summary of the action based on the screen image.</summary>"""

######
ODYSSEY_ACT2SUM_PROMPT = f"""Step-by-step GUI navigation task. Briefly summarize the current action.
Action space:
{ODYSSEY_ACTION_SPACE_PROMPT}

Goal: <goal>(goal)</goal>
Current action: <action>(action)</action>

Output Format: <summary>One-sentence summary of the action based on the screen image.</summary>"""

######
MIND2WEB_ACT2SUM_PROMPT = f"""Step-by-step GUI navigation task. Briefly summarize the current action.
Action space:
{MIND2WEB_ACTION_SPACE_PROMPT}

Goal: <goal>(goal)</goal>
Current action: <action>(action)</action>

Output Format: <summary>One-sentence summary of the action based on the screen image.</summary>"""



## You can use vllm to deploy locally and mount the local images via "nohup python3 -m http.server 6666 &"".
# Or accelerate inference via swift.

# import requests
# import json

# url = "http://localhost:8000/v1/chat/completions"
# headers = {
#      "Content-Type": "application/json"
# }
# def chat(img_url, goal, action):
#     query = MIND2WEB_SUM_PROMPT.replace("(goal)", goal).replace("(action)", action)
#     content = []
#     content.append({"type": "image_url", "image_url": {"url": img_url}})
#     content.append({"type": "text", "text": query})
#     data = {
#         "model": "Qwen2.5-VL-72B-Instruct",
#         "messages": [
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": content}
#         ],
#         "temperature":0}
        
#     response = requests.post(url, headers=headers, data=json.dumps(data))
#     response = response.json()
#     response = response['choices'][0]['message']['content']

#     return response

# Act2Sum = []
# for step in episode:
#   img, goal, action = step
#   act2sum = chat(img, goal, action)
#   ...
#   Act2Sum.append(act2sum)
# ...
# Save and format data