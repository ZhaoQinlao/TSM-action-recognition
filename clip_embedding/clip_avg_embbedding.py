import torch
from transformers import CLIPTokenizer, CLIPTextModel

class ClipTextEmbedder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # 设置为评估模式

    @torch.no_grad()
    def encode(self, texts):
        """
        提取一批文本的CLIP嵌入
        Args:
            texts (str 或 List[str]): 输入的文本或文本列表
        Returns:
            Tensor: shape = (batch_size, embedding_dim)，默认是 (batch_size, 512)
        """
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
        # pooled = last_hidden_state[:, 0, :]  # 取 [CLS] token（第0位）作为全局文本特征
        pooled = last_hidden_state.mean(dim=1)
        pooled = pooled / pooled.norm(dim=-1, keepdim=True)  # L2归一化（非常重要）

        return pooled  # (batch_size, 512)

# -------------------------------
# 测试
if __name__ == "__main__":
    from tqdm import tqdm
    with open('/home/fitz_joye/TSM-action-recognition/data/assembly101/assembly101-annotations/fine-grained-annotations/actions.csv') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines[1:]] # 跳过标题行
    lines = [line.split(',')[4] for line in lines] # 只保留动作名称
    assert len(lines) == 1380, "动作名称数量不正确"

    embeddings = torch.zeros((1380, 512), dtype=torch.float32)
    embedder = ClipTextEmbedder()

    template = "this is a photo of {}"

    # embeddings = embedder.encode(texts)
    # print(f"Embedding shape: {embeddings.shape}")  # 应该是 (3, 512)

    for i, line in enumerate(tqdm(lines)):
        texts = template.format(line)
        # print(embedder.encode(texts).detach().cpu().shape)  # 应该是 (1, 512)
        embeddings[i] = embedder.encode(texts).detach().cpu()
    
    print(embeddings.shape)  # 应该是 (1380, 512)
    torch.save(embeddings, 'clip_embeddings.pt')
