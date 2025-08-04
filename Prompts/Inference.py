########### Action Space ########### 
###### Mind2Web
MIND2WEB_ACTION_SPACE_PROMPT = """
CLICK:(x,y): Click on the element at the coordinate point (x,y) on the screen, e.g., CLICK:(1980,224).
TYPE:(x,y,typed_text): An action of typing a piece of text at the coordinate point (x,y), e.g., TYPE:(208,1082,Macbook-Pro 16G Black).  
SELECT:(x,y,option): Opens the Select Menu or Drop-Down List at coordinate (x, y) and selects the option you specify, e.g., SELECT:(59,892,Chicago)."""

###### GUI-ODYSSEY
ODYSSEY_ACTION_SPACE_PROMPT = """
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
AITW_ACTION_SPACE_PROMPT = """
CLICK:(x,y): Click on the element at the coordinate point (x,y) on the screen, e.g., CLICK:(1980,224).
TYPE:typed_text: An action of typing a piece of text, e.g., TYPE:"Macbook-Pro 16G Black".
COMPLETE: The goal has been completed in the current screen state.
SCROLL:UP/DOWN/LEFT/RIGHT: Scroll in a specific direction, e.g., SCROLL:UP.
BACK: Go back to the previous screen, e.g, BACK.
HOME: Go to the home screen, e.g., HOME.
ENTER: Press the ENTER key to submit input content."""

###### In-House Bench
IN_HOUSE_ACTION_SPACE_PROMPT = """
CLICK:(x,y): Click on the element at the coordinate point (x,y) on the screen, e.g., CLICK:(1980,224).
"""

##### CUSTOMERIZED Action Space
CUSTOMERIZED_ACTION_SPACE_PROMPT = """
Here you can define the actions you need to perform with the device.
"""


########### Inference ########### 
###### In-house Bench
IN_HOUSE_EXECUTION_PROMPT = f"""You are a skilled assistant, interacting with the screen to accomplish the user's goals.
Here is the action space:
{IN_HOUSE_ACTION_SPACE_PROMPT}
Your overall goal is: <goal>(goal)</goal>
Actions completed at previous steps: <history>(history)</history>
The output format should be as follows:
<think>Analyze step by step based on guidance and screen state to choose the action.</think>
<answer>The action you finally choose from "action space".</answer>"""

###### GUI-ODYSSEY
ODYSSEY_EXECUTION_PROMPT = f"""You are a skilled assistant, interacting with the screen to accomplish the user's goals.
Here is the action space:
{ODYSSEY_ACTION_SPACE_PROMPT}
Your overall goal is: <goal>(goal)</goal>
Actions completed at previous steps: <history>(history)</history>
The output format should be as follows:
<think>Analyze step by step based on guidance and screen state to choose the action.</think>
<answer>The action you finally choose from "action space".</answer>"""

###### AITW
AITW_EXECUTION_PROMPT = f"""You are a skilled assistant, interacting with the screen to accomplish the user's goals.
Here is the action space:
{AITW_ACTION_SPACE_PROMPT}
Your overall goal is: <goal>(goal)</goal>
Actions completed at previous steps: <history>(history)</history>
The output format should be as follows:
<think>Analyze step by step based on guidance and screen state to choose the action.</think>
<answer>The action you finally choose from "action space".</answer>"""

###### MIND2WEB
MIND2WEB_EXECUTION_PROMPT = f"""You are a skilled assistant, interacting with the screen to accomplish the user's goals.
Here is the action space:
{MIND2WEB_ACTION_SPACE_PROMPT}
Your overall goal is: <goal>(goal)</goal>
Actions completed at previous steps: <history>(history)</history>
The output format should be as follows:
<think>Analyze step by step based on guidance and screen state to choose the action.</think>
<answer>The action you finally choose from "action space".</answer>"""


###### Customized
CUSTOMERIZED_EXECUTION_PROMPT = f"""You are a skilled assistant, interacting with the screen to accomplish the user's goals.
Here is the action space:
{CUSTOMERIZED_ACTION_SPACE_PROMPT}
Your overall goal is: <goal>(goal)</goal>
Actions completed at previous steps: <history>(history)</history>
The output format should be as follows:
<think>Analyze step by step based on guidance and screen state to choose the action.</think>
<answer>The action you finally choose from "action space".</answer>"""



###### System-1 Inference Format
CUSTOMERIZED_EXECUTION_SYS1_PROMPT = f"""You are a skilled assistant, interacting with the screen to accomplish the user's goals.
Here is the action space:
{CUSTOMERIZED_ACTION_SPACE_PROMPT}
Your overall goal is: <goal>(goal)</goal>
Actions completed at previous steps: <history>(history)</history>
The output format should be as follows:
<answer>The action you finally choose from "action space".</answer>"""