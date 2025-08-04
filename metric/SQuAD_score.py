import json
from metric import evaluate
from metric import compute_squad_v2

def convert_to_prediction(file_path):
    ret_list = []

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    for index, item in enumerate(data):
        prediction = {}
        prediction['prediction_text'] = item['pred']
        prediction['id'] = str(index)
        prediction['no_answer_probability'] = 0.
        ret_list.append(prediction)

    return ret_list


def convert_to_reference(file_path):
    ret_list = []

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for index, item in enumerate(data):
        reference = {}
        reference['answers'] = {}
        reference['answers']['answer_start'] = []
        reference['answers']['text'] =  item['gt']
        reference['id'] = str(index)
        ret_list.append(reference)

    return ret_list


def test(path):
    predictions = convert_to_prediction(path)
    references = convert_to_reference(path)
    evaluate(predictions, references, metric='rouge')
    evaluate(predictions, references, metric='squad_v2')

path = './your_grounding_data.json'
test(path)

##Data Format in Your Json File:
"""
[
    {
        "image_path": "./Rico/10.jpg",
        "question": "What are the different categories under the \"Plans\" section? answer with numbers or phrases rather than sentence.",
        "pred": "Healthy, Sporty, Weightloss",
        "gt": [
            "Healthy, Sporty and Weightloss",
            "Healthy, Sporty, Weightloss"
        ]
    },
    {...},
    ...
]
"""