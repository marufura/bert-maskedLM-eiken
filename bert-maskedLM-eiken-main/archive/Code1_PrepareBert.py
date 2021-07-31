import torch
from transformers import BertJapaneseTokenizer, BertForMaskedLM

model_name_or_path = "cl-tohoku/bert-base-japanese-v2"
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)

model = BertForMaskedLM.from_pretrained(model_name_or_path)

input_ids = tokenizer.encode(f"BERTは{tokenizer.mask_token}のひとつの技術です。", return_tensors="pt")

# print(input_ids)

# print(tokenizer.convert_ids_to_tokens(input_ids[0].tolist()))

masked_index = torch.where(input_ids == tokenizer.mask_token_id)[1][0].tolist()
# print(masked_index)

result = model(input_ids)
pred_ids = result[0][:, masked_index].topk(5).indices.tolist()[0]
for pred_id in pred_ids:
    output_ids = input_ids.tolist()[0]
    output_ids[masked_index] = pred_id
    print(tokenizer.decode(output_ids))