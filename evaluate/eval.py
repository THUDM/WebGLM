import json
from tqdm import tqdm

def eval(model, args):
    ds = [json.loads(data_str) for data_str in open(args.evaluate_task_data_path).readlines()]
    
    correct, total = 0, 0
    
    for ix, sample in enumerate(tqdm(ds)):
        predict = model.query(sample['question'])['answer']
        for label in sample['answer']:
            if label in predict:
                correct += 1
                break
        total += 1
    
    return correct / total