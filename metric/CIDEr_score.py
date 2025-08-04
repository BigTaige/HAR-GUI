import json
from metric import evaluate
from metric import compute_cider

def get_imgid(image_path):
    return image_path.split('/')[-1].split('.')[0]

def convert_to_prediction(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    ret_dict = {}
    for index, item in enumerate(data):
        imgid = get_imgid(item['image_path'])
        if "Qwen2" in file_path:
            ret_dict[imgid] = [item['pred']]
        elif "MiniCPM" in file_path:
            ret_dict[imgid] = [item['pred'].replace('<|endoftext|>','')]
        else:
            ret_dict[imgid] = [item['pred']]
    
    return ret_dict


def convert_to_reference(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    ret_dict = {}
    for index, item in enumerate(data):
        imgid = get_imgid(item['image_path'])
        ret_dict[imgid] = item['gt']
    
    return ret_dict


def test(path):
    predictions = convert_to_prediction(path)
    references = convert_to_reference(path)
    print(len(predictions), len(references))
    evaluate(predictions, references, metric='cider')


path = './your_grounding_data.json'
test(path)


##Data Format in Your Json File:
"""
[
    {
        "image_path": "./Rico/24378.jpg",
        "question": "Use a phrase to describe the function of the page with not more than 10 words",
        "pred": "help page",
        "gt": [
            "display of help options for an app",
            "list of options available in the app",
            "page displaying with different options to know about the application",
            "page displaying with list of help options",
            "screen is showing help page"
        ]
    },
    { ... },
    ...
]
"""




# predictions= {'0':["the"], '1':["menu"]}
# references = {'0':["menu", "open main menu", "open the menu"], 
#                 '1':["menu", "open main menu", "open menu"]
#                }
# predictions = { '1': ["A man riding a horse."],
#                 '2': ["A woman holding a cat."]}
# references ={ '1':    ["A man is riding a horse.",
#                          "There is a man on a horse."],
#                 '2':    ["A lady is holding a kitty.",
#                          "A woman has a cat in her hands."]}
# evaluate(predictions, references, metric='cider')