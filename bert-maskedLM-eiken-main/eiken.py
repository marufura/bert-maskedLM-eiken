import re
import json
import torch
from argparse import ArgumentParser
from unidecode import unidecode
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM


def to_clean(text):
    return unidecode(text.strip())


def show(question, candidates, predict_idx, answer=None):
    print("")
    print('[Question] : %s' % question)
    print('\t1) %s 2) %s 3) %s 4) %s\n' %
          (candidates[0], candidates[1], candidates[2], candidates[3]))
    print("-> BERT's Answer : %s" % candidates[predict_idx])
    if answer != None:
        print('->   Real Answer : %s\n' % answer)


def get_score(model, tokenizer, question_tensors, segment_tensors, masked_index, candidate):
    candidate_tokens = tokenizer.tokenize(candidate)  # warranty -> ['warrant', '##y']
    candidate_ids = tokenizer.convert_tokens_to_ids(candidate_tokens)

    predictions = model(question_tensors, segment_tensors)
    predictions_candidates = predictions[0, masked_index, candidate_ids].mean()

    return predictions_candidates.item()


def solve(row, bertmodel='bert-base-uncased'):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    tokenizer = BertTokenizer.from_pretrained(bertmodel)
    model = BertForMaskedLM.from_pretrained(bertmodel).to(device)
    model.eval()

    question = re.sub('\_+', ' [MASK] ', to_clean(row['question']))
    question_tokens = tokenizer.tokenize(question)
    masked_index = question_tokens.index('[MASK]')

    # make segment which is divided with sentence A or B, but we set all '0' as sentence A
    segment_ids = [0] * len(question_tokens)
    segment_tensors = torch.tensor([segment_ids]).to(device)

    # question tokens convert to ids and tensors
    question_ids = tokenizer.convert_tokens_to_ids(question_tokens)
    question_tensors = torch.tensor([question_ids]).to(device)

    candidates = [to_clean(row['1']), to_clean(row['2']), to_clean(row['3']), to_clean(row['4'])]
    predict_tensor = torch.tensor([get_score(model, tokenizer, question_tensors, segment_tensors,　masked_index, candidate) for candidate in candidates])
    predict_idx = torch.argmax(predict_tensor).item()

    if 'answer' in row:
        show(row['question'], candidates, predict_idx, row['answer'])
    else:
        show(row['question'], candidates, predict_idx, None)


parser = ArgumentParser()
parser.add_argument("-f", '--file', type=str, required=True)
args = parser.parse_args()

with open("./data/" + args.file + ".json") as data_file:
    file = json.load(data_file)

for (key, row) in file.items():
    if 'question' in row and '1' in row and '2' in row and '3' in row and '4' in row:
        solve(row, "bert-base-uncased")
    else:
        print('key of %s : No required options.' % key)
        continue
