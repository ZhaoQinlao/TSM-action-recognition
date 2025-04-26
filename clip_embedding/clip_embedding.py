from transformers import AutoTokenizer, CLIPTextModelWithProjection
import torch

model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

from tqdm import tqdm
with open('/home/fitz_joye/TSM-action-recognition/data/assembly101/assembly101-annotations/fine-grained-annotations/actions.csv') as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines[1:]] # 跳过标题行
lines = [line.split(',')[4] for line in lines] # 只保留动作名称
assert len(lines) == 1380, "动作名称数量不正确"

embeddings = torch.zeros((1380, 512), dtype=torch.float32)
template = "this is a photo of {}"
for i, line in enumerate(tqdm(lines)):
    texts = template.format(line)   
    inputs = tokenizer([texts], padding=True, return_tensors="pt")

    outputs = model(**inputs)
    text_embeds = outputs.text_embeds
    # print(text_embeds.shape)  # (batch_size, sequence_length, hidden_size)
    embeddings[i] = text_embeds.detach().cpu()

torch.save(embeddings, 'clip_embeddings.pt')