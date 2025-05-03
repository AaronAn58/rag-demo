from datasets import load_dataset
from datasets import config
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
print(config.HF_DATASETS_CACHE)

# # 清除缓存（注意：这会删除所有本地缓存的 dataset）
# rm -rf ~/.cache/huggingface/datasets/wiki_qa

# 加载 wiki_qa 数据集
dataset1 = load_dataset("wiki_qa")
# dataset2 = load_dataset("squad")
# print(dataset1)
# 合并并构建统一知识库hi
# combined = dataset1["train"].concatenate(dataset2["train"])

# 构建知识库
knowledge_base = dataset1["train"].map(lambda x: {"text": x["answer"]})
knowledge_texts = [item["text"] for item in knowledge_base if item["text"]]

# 加载嵌入模型（如 multi-qa-MiniLM-L6-cos-v1）
model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

# 向量化
embeddings = model.encode(knowledge_texts)

# 构建 FAISS 索引
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 保存索引
faiss.write_index(index, "wiki_qa_index.faiss")

# 构建检索器
def retrieve(query, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [knowledge_texts[i] for i in indices[0]]

# 构建生成器
# 加载 Qwen 模型
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)
model_gen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True).to("cuda")

# 生成答案
def generate_answer(query, retrieved_docs):
    context = "\n\n".join(retrieved_docs)
    prompt = f"问题：{query}\n\n上下文：{context}\n\n答案："
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model_gen.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def rag_pipeline(query, top_k=5):
    # 1. 检索相关文档
    retrieved_docs = retrieve(query, top_k=top_k)

    # 2. 生成答案
    answer = generate_answer(query, retrieved_docs)
    return answer

print(rag_pipeline("康熙和雍正"))