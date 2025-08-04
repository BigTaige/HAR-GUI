GUIDELINE_SYNTHESIS_TEMPLATE = f"""Below is a model's incorrect predictions when processing a GUI navigation task. Your goal is to identify the error's cause and provide helpful clues for correct future predictions.
Model Input: [INPUT]
Model Output: [ERROR_PREDICTION]
Ground Truth: [GROUND_TRUTH]

Please keep the following output format:
<think>Please refer to the "Model Input" and compare the "Model Output" with the "Ground Truth" to analyze and output the reasons for the model prediction errors.</think>
<guidelines>Provide up to three helpful guidelines to guide the model to make correct prediction (keep it short and concise), formatted as a numbered list (e.g., 1. Guideline 1; 2. Guideline 2). Do not include content from the "Ground Truth".</guidelines>"""




## You can use vllm to deploy locally and mount the local images via "nohup python3 -m http.server 6666 &"".
# Or accelerate inference via swift.

# G = {}
# for row in hard_samples:
#   img, instruction, pred, label = row
#   res = GUIDELINE_SYNTHESIS_TEMPLATE.replace("[INPUT]", instruction).replace("[ERROR_PREDICTION]", pred).replace("[GROUND_TRUTH]", label)
#   guidelines = res.split("<guidelines>")[-1].split("</guidelines>")[0]
#   G[img] = guidelines

# Save guidelines G