from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.settings import Settings
from dotenv import load_dotenv 
load_dotenv()
import os 
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from local_llamaindex_llm import QwenLLM
from remote_embedding import CachedRemoteEmbedding
import csv

api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE")
model = os.getenv("OPENAI_API_MODEL")
embed_model = os.getenv("OPENAI_API_EMBEDDING_MODEL")

Settings.llm = QwenLLM(api_key=api_key, base_url=api_base, model=model, temperature=0.6)

Settings.embed_model = CachedRemoteEmbedding(
    api_key=api_key,
    api_base=api_base,
    model_name=embed_model,
    batch_size=64,
    max_retries=3,
    retry_delay=1.0
)

client = chromadb.Client()

from llama_index.core import SimpleDirectoryReader


documents = SimpleDirectoryReader(
    input_dir="./datas/文本数据",
    recursive=True
).load_data()

chroma_collection = client.get_or_create_collection("emergency_docs")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

query_engine = index.as_query_engine()

questions = []
with open("./question.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if len(row) >= 2:
            questions.append((row[0], row[1]))

answers = []
for qid, question in questions:
    print(f"正在回答问题 {qid}: {question}")
    try:
        response = query_engine.query(question)
        answer = str(response)
        answers.append((qid, answer))
        print(f"问题 {qid} 回答完成")
    except Exception as e:
        print(f"问题 {qid} 回答失败: {e}")
        answers.append((qid, f"回答失败: {str(e)}"))

with open("answer.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ID", "answer"])
    for qid, answer in answers:
        writer.writerow([qid, answer])

print(f"已处理 {len(answers)} 个问题，结果已保存到 answer.csv")