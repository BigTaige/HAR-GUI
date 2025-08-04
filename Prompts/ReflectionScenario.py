REFLECTION_PROMPT = f"""[INPUT]

However, you made an incorrect action prediction:
[ERROR PREDICTION]

Here are some guidelines to help you select the right action:
<guidelines> [GUIDELINES] </guidelines>

The output format should be as follows:
<statement>Explain here why you made incorrect decision in the past and summarize your mistakes here</statement>
<think>Analyze step by step based on guidance and screen state to choose the action</think>
<answer>The action you finally choose from "action space".</answer>"""




## You can use vllm to deploy locally and mount the local images via "nohup python3 -m http.server 6666 &"".
# Or accelerate inference via swift.

# guidelines = json.load(open('your_path.json', 'r'))
# for row in hard_samples:
#   img, instruction, pred, label = row
#   reflection_instruction = REFLECTION_PROMPT.replace("[INPUT]", instruction).replace("[ERROR PREDICTION]", pred).replace("[EGUIDELINES]", guidelines[img])
#   ...   
# Save and format data (reflection_instruction, lable, img) for Round-1 RL training.