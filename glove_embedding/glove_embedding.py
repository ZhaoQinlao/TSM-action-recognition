import numpy as np

# 加载 GloVe 词向量
def load_glove_embeddings(glove_path):
    embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    print(f"Loaded {len(embeddings)} word vectors from GloVe.")
    return embeddings

# 获取动作标签的语义嵌入向量（平均池化）
def get_text_embedding(text, embeddings, dim=300):
    words = text.lower().split()
    vecs = []
    for word in words:
        if word in embeddings:
            vecs.append(embeddings[word])
        else:
            vecs.append(np.zeros(dim))  # OOV处理
            print(f"Warning: Word '{word}' not found in GloVe embeddings. Using zero vector.")
    if len(vecs) == 0:
        print(f"Warning: No valid words found in text '{text}'. Returning zero vector.")
        return np.zeros(dim)
    return np.mean(vecs, axis=0)

# 示例使用
if __name__ == "__main__":
    glove_path = 'data/assembly101/glove/glove.6B.300d.txt'
    embeddings = load_glove_embeddings(glove_path)

    action_label = "shaking head"
    embedding_vector = get_text_embedding(action_label, embeddings)
    print(f"Embedding shape: {embedding_vector.shape}")
    print(embedding_vector)  # 输出 X_q ∈ ℝ³⁰⁰
