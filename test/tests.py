from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings  # 使用 Qwen 的 Embedding
from langchain.chains import RetrievalQA
from langchain_community.llms import Tongyi

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
embeddings = DashScopeEmbeddings(model="text-embedding-v1", dashscope_api_key="sk-7f104e21fda743829fa3f55f01e99377")

# 4. 构建 FAISS 向量数据库
vectorstore = FAISS.from_documents(docs, embeddings)

# 5. 保存本地索引（可选）
vectorstore.save_local("faiss_index")

# 1. 加载向量数据库
vectorstore = FAISS.load_local("../faiss_index", embeddings, allow_dangerous_deserialization=True)

# 2. 初始化 Qwen LLM（需替换为你的 API Key）
llm = Tongyi(model="qwen-max", dashscope_api_key="sk-7f104e21fda743829fa3f55f01e99377")

# 3. 构建检索器
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# 4. 构建 RAG 链（集成检索+生成）
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True  # 返回检索到的上下文
)

# 5. 执行查询
query = "量子计算的基本原理是什么？"
result = qa_chain({"query": query})

# 6. 输出结果
print("问题:", query)
print("答案:", result["result"])
print("检索到的上下文:", [doc.page_content for doc in result["source_documents"]])