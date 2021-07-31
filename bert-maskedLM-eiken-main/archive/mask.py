import torch
from transformers import BertTokenizer, BertForPreTraining

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForPreTraining.from_pretrained('bert-base-uncased')


def part5_slover(text, candidate):
    tokens = tokenizer.tokenize(text)
    masked_index = tokens.index("*")
    tokens[masked_index] = "[MASK]"
    tokens = ["[CLS]"] + tokens + ["[SEP]"]

    ids = tokenizer.convert_tokens_to_ids(tokens)
    ids = torch.tensor(ids).reshape(1, -1)
    with torch.no_grad():
        outputs, _ = model(ids)
    predictions = outputs[0]

    print(outputs)
    print(predictions)

    _, predicted_indexes = torch.topk(predictions[masked_index + 1], k=1000)
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_indexes.tolist())

    for i, v in enumerate(predicted_tokens):
        if v in candidate:
            return (i, v)
    return "don't know"


text = "I have a * for you."
candidate = ["question", "sorry", "thirsty", "like"]

part5_slover(text, candidate)
