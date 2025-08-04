from cider.cider import Cider
from collections import defaultdict
import torch
from torchvision.ops import box_iou
from sklearn.metrics import f1_score
from evaluate import load
from rouge import Rouge


def evaluate(input1, input2, metric="cider"):
    assert metric in ["cider", "f1", "iou", "squad_v2", "center", 'rouge']
    assert input1 is not None and input2 is not None

    score = eval(f"compute_{metric}(input1, input2)")
    return score

def compute_rouge(preds, labels):
    rouge = Rouge()
    total_num = len(preds)
    print('total_num: ',total_num)
    total_score_1 = 0.0
    total_score_2 = 0.0
    total_score_L = 0.0
    for i in range(total_num):
        pred = preds[i]['prediction_text']
        one_score_1 = 0
        one_score_2 = 0
        one_score_L = 0
        for label in labels[i]['answers']['text']:
            rouge_res = rouge.get_scores(pred.lower(), label.lower(), avg=True)
            # print(pred)
            # print(label)
            # print(rouge_res)
            one_score_1 = max(one_score_1, rouge_res['rouge-1']['f'])
            one_score_2 = max(one_score_2, rouge_res['rouge-2']['f'])
            one_score_L = max(one_score_L, rouge_res['rouge-l']['f'])
        total_score_1 += one_score_1
        total_score_2 += one_score_2
        total_score_L += one_score_L

    print('score-1: ',total_score_1 / total_num)
    print('score-2: ',total_score_2 / total_num)
    print('score-L: ',total_score_L / total_num)


def compute_f1(preds, labels):
    '''
    Binary classification F1 score
    Return:
        score(float): computed F1 score
    '''
    score = f1_score(preds, labels) * 100.0
    print(f'binary f1: {score}')
    return score


def compute_squad_v2(predictions, references):
    '''
    Use squad_v2 version to calculate SQuAD F1
    Args:
        predictions
            [{  'prediction_text': '1976',  prediction
                'id': '',                   need to be same with the corresponding reference_id
                'no_answer_probability': 0.},
             {'prediction_text': 'Beyonce', 'id': '', 'no_answer_probability': 0.}]
        references
            [{  'answers': 
                    {'answer_start': [],    could be empty
                     'text': ['1976']       ground truth
                    },
                'id': ''                    need to be same with the corresponding prediction_id
             },
             {'answers': {'answer_start': [], 'text': ['Beyonce and Bruno Mars']}, 'id': ''}]
    '''
    squad_metric = load("squad_v2/squad_v2.py")
    result = squad_metric.compute(predictions=predictions, references=references)
    
    print(f"squad_v2 f1: {result['best_f1']}")
    return result['best_f1'], result['f1_raw']


def compute_iou(predict_bbox, target_bbox, threshold=0.1):
    '''
    Compute Precision@IoU=threshold
    Args:
        predict_bbox ([N, 4])
        target_bbox ([N, 4])
            [[0, 0, 0.9, 0.9],
             [0.1, 0.1, 1, 1]]
    Return:
        precision(float)
    '''
    if not isinstance(predict_bbox, torch.Tensor):
        predict_bbox = torch.tensor(predict_bbox)
    if not isinstance(target_bbox, torch.Tensor):
        target_bbox = torch.tensor(target_bbox)

    ious = box_iou(predict_bbox, target_bbox)
    correct = (ious.diag() > threshold).sum().item()
    precision = correct / predict_bbox.size(0) * 100.0

    print(f'precision@IoU={threshold}: {precision}')
    return precision, ious.diag() > threshold


def compute_center(predict_bbox, target_bbox):
     
    correct_num = 0
    total_num = len(predict_bbox)
    for index in range(total_num):
        #print(index)
        center_x = (predict_bbox[index][0] + predict_bbox[index][2]) // 2
        center_y = (predict_bbox[index][1] + predict_bbox[index][3]) // 2
        if center_x >= target_bbox[index][0] and center_x <= target_bbox[index][2] and center_x >= target_bbox[index][0] and center_x <= target_bbox[index][2]:
            correct_num += 1
    
    print(f'correct_num = {correct_num}')
    print(f'precision = {correct_num/total_num}')
    return correct_num/total_num


def compute_cider(generated_captions, ground_truth):
    '''
    Compute CIDEr score
    Args:
        generated_captions(dict)
            generated_captions = {
                '1': ["A man riding a horse."],
                '2': ["A woman holding a cat."]
            }
        ground_truth(dict)
            ground_truth = {
                '1':    ["A man is riding a horse.",
                         "There is a man on a horse."],
                '2':    ["A lady is holding a kitty.",
                         "A woman has a cat in her hands."]
            }
    Return:
        score(float): CIDEr score
    '''
    assert len(generated_captions) == len(ground_truth)

    cider_scorer = Cider()
    score, scores = cider_scorer.compute_score(ground_truth, generated_captions)
    score *= 100.0
    
    print(f'CIDEr score: {score}')
    return score, scores


if __name__ == "__main__":
    predict = [[0.1, 0.1, 1, 1],[0, 0, 0.9, 0.9]]
    target = [[0, 0, 0.9, 0.9],[0.1, 0.1, 1, 1]]

    generated_captions = {
        '1': ["a mobile app's forum page displaying the latest real estate topics."],
        '2': ["page displaying post comments on a mobile app."]
    }
    ground_truth = {
        '1':    ["page displaying information about real estate application."],
        '2':    ["page displaying a post and option to comment on it."]
    }


    predictions = [{'prediction_text': '12', 'id': '0', 'no_answer_probability': 0.0}, {'prediction_text': '2', 'id': '1', 'no_answer_probability': 0.0}, {'prediction_text': 'wall sit', 'id': '2', 'no_answer_probability': 0.0}, {'prediction_text': 'REST', 'id': '3', 'no_answer_probability': 0.0}, {'prediction_text': 'Healthy, Sporty, Weightloss', 'id': '4', 'no_answer_probability': 0.0}]
    references = [{'answers': {'answer_start': [], 'text': ['a total of 12 exercises', '12', '12 exercises', '12 exercises in total']}, 'id': '0'}, {'answers': {'answer_start': [], 'text': ['2 exercises', '2', 'two']}, 'id': '1'}, {'answers': {'answer_start': [], 'text': ['Wall Sit', 'wall sit', 'WALL SIT']}, 'id': '2'}, {'answers': {'answer_start': [], 'text': ['rest', 'REST', '"REST"', '<no answer>']}, 'id': '3'}, {'answers': {'answer_start': [], 'text': ['Healthy, Sporty and Weightloss', 'Healthy, Sporty, Weightloss']}, 'id': '4'}]
    # predictions = [{'prediction_text': '1976', 'id': '', 'no_answer_probability': 0.}, {'prediction_text': 'Beyonce', 'id': '', 'no_answer_probability': 0.}]
    # references = [{'answers': {'answer_start': [], 'text': ['1976']}, 'id': ''}, {'answers': {'answer_start': [], 'text': ['Beyonce']}, 'id': ''}]

    # evaluate(predictions, references, metric='squad_v2')
    evaluate(predict, target, metric="iou")
    # evaluate(generated_captions, ground_truth, metric="cider")