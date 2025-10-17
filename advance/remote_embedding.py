import os
import hashlib
import json
import time
from typing import List, Any, Optional
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
import numpy as np
from dotenv import load_dotenv


class CachedRemoteEmbedding(BaseEmbedding):
    
    api_key: str
    api_base: str
    model_name: str = "BAAI/bge-m3"
    cache_dir: str = "./embedding_cache_remote"
    batch_size: int = 64
    max_retries: int = 3
    retry_delay: float = 1.0
    
    _client: Any = PrivateAttr()
    _cache: dict = PrivateAttr()
    
    def __init__(
        self,
        api_key: str,
        api_base: str,
        model_name: str = "BAAI/bge-m3",
        cache_dir: str = "./embedding_cache_remote",
        batch_size: int = 64,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key,
            api_base=api_base,
            model_name=model_name,
            cache_dir=cache_dir,
            batch_size=batch_size,
            max_retries=max_retries,
            retry_delay=retry_delay,
            **kwargs
        )
        
        self._cache = {}
        self._init_client()
        self._load_cache()
    
    def _init_client(self) -> None:
        try:
            from openai import OpenAI
            
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base
            )
            print(f"远程 Embedding API 初始化完成")
            print(f"模型: {self.model_name}")
            print(f"API Base: {self.api_base}")
            
        except ImportError as e:
            raise ImportError(
                "请安装必要的依赖: pip install openai"
            ) from e
    
    def _load_cache(self) -> None:
        import pickle
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = os.path.join(self.cache_dir, "embedding_cache.pkl")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    self._cache = pickle.load(f)
                print(f"已加载 {len(self._cache)} 条缓存记录")
            except Exception as e:
                print(f"加载缓存失败: {e}")
                print("将创建新的缓存")
                self._cache = {}
    
    def _save_cache(self) -> None:
        import pickle
        import builtins
        cache_file = os.path.join(self.cache_dir, "embedding_cache.pkl")
        try:
            with builtins.open(cache_file, 'wb') as f:
                pickle.dump(self._cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"保存缓存失败: {e}")
    
    def _get_cache_key(self, text: str) -> str:
        cache_str = f"{self.model_name}:{text}"
        return hashlib.md5(cache_str.encode('utf-8')).hexdigest()
    
    def _get_text_embedding(self, text: str) -> List[float]:
        embeddings = self._get_text_embeddings([text])
        return embeddings[0]
    
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                embeddings.append(None)
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                embeddings.append(None)
        
        if uncached_texts:
            print(f"需要请求 {len(uncached_texts)}/{len(texts)} 个新 embedding")
            
            for batch_start in range(0, len(uncached_texts), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(uncached_texts))
                batch_texts = uncached_texts[batch_start:batch_end]
                batch_indices = uncached_indices[batch_start:batch_end]
                
                print(f"请求批次: {batch_start//self.batch_size + 1}/{(len(uncached_texts)-1)//self.batch_size + 1}")
                
                success = False
                for attempt in range(self.max_retries):
                    try:
                        response = self._client.embeddings.create(
                            model=self.model_name,
                            input=batch_texts
                        )
                        
                        for idx, text, emb_data in zip(batch_indices, batch_texts, response.data):
                            embedding = np.array(emb_data.embedding)
                            cache_key = self._get_cache_key(text)
                            self._cache[cache_key] = embedding
                            embeddings[idx] = embedding
                        
                        success = True
                        break
                        
                    except Exception as e:
                        error_msg = str(e)
                        print(f"API 请求失败 (尝试 {attempt + 1}/{self.max_retries}): {error_msg}")
                        
                        if "rate_limit" in error_msg.lower() or "429" in error_msg:
                            print(f"检测到速率限制，等待 {self.retry_delay} 秒后重试...")
                            time.sleep(self.retry_delay)
                            continue
                        else:
                            break
                
                if not success:
                    print(f"批次 {batch_start//self.batch_size + 1} 请求最终失败，使用零向量")
                    for idx in batch_indices:
                        embeddings[idx] = np.zeros(1024)
        
        for i, text in enumerate(texts):
            if embeddings[i] is None:
                cache_key = self._get_cache_key(text)
                embeddings[i] = self._cache[cache_key]
        
        if uncached_texts:
            self._save_cache()
        
        return [emb.tolist() if isinstance(emb, np.ndarray) else emb for emb in embeddings]
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)



if __name__ == "__main__":
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")
    embed_model = os.getenv("OPENAI_API_EMBEDDING_MODEL", "BAAI/bge-m3")
    
    if not api_key or not api_base:
        print("错误: 请在 .env 文件中设置 OPENAI_API_KEY 和 OPENAI_API_BASE")
        exit(1)
    
    print("初始化远程 Embedding 模型...")
    embedding_model = CachedRemoteEmbedding(
        api_key=api_key,
        api_base=api_base,
        model_name=embed_model,
        batch_size=64
    )
    
    test_text = "这是一个测试文本"
    print(f"\n测试文本: {test_text}")
    embedding = embedding_model._get_query_embedding(test_text)
    print(f"Embedding 维度: {len(embedding)}")
    print(f"Embedding 前5个值: {embedding[:5]}")
    
    test_texts = [
        "危险化学品安全管理",
        "应急预案编制指南",
        "安全生产法律法规",
        "应急响应流程规范",
        "事故调查处理办法"
    ]
    print(f"\n测试批量文本 ({len(test_texts)} 条)...")
    embeddings = embedding_model._get_text_embeddings(test_texts)
    print(f"生成了 {len(embeddings)} 个 embedding")
    
    print("\n测试缓存功能...")
    embedding2 = embedding_model._get_query_embedding(test_text)
    print(f"缓存命中: {embedding == embedding2}")
    
    print(f"\n再次测试批量文本（测试缓存）...")
    embeddings2 = embedding_model._get_text_embeddings(test_texts)
    print(f"获取了 {len(embeddings2)} 个 embedding（应该全部来自缓存）")
    
    print("\n保存缓存...")
    embedding_model._save_cache()
    print(f"已保存 {len(embedding_model._cache)} 条缓存记录")
    
    print("\n测试完成！")
