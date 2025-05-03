from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import DashScopeEmbeddings  # 使用 Qwen 的 Embedding

# 1. 模拟文档数据（可替换为实际读取的 PDF/文本文件）
raw_documents = [
    "量子计算是一种利用量子力学原理进行计算的新型计算范式。",
    "量子比特是量子计算的基本单位，可以同时处于 0 和 1 的叠加态。",
    "量子纠缠是量子计算中的关键现象，允许远距离粒子状态的关联。",
]

# 2. 文本切分
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
docs = text_splitter.create_documents(raw_documents)

# 3. 初始化 Qwen Embedding 模型（需替换为你的 API Key）
embeddings = DashScopeEmbeddings(model="text-embedding-v1", dashscope_api_key="YOUR_API_KEY")

# 4. 构建 FAISS 向量数据库
vectorstore = FAISS.from_documents(docs, embeddings)

# 5. 保存本地索引（可选）
vectorstore.save_local("faiss_index")