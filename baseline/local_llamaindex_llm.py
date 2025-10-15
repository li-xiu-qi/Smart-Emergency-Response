from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.bridge.pydantic import PrivateAttr
from openai import OpenAI as OpenAIClient


class QwenLLM(CustomLLM):
    context_window: int = 32768
    num_output: int = 2048
    max_tokens: int = 2048

    _client: OpenAIClient = PrivateAttr()
    _model: str = PrivateAttr()

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        max_tokens: int = 2048,
        **kwargs
    ):
        super().__init__(max_tokens=max_tokens, **kwargs)
        self._client = OpenAIClient(api_key=api_key, base_url=base_url)
        self._model = model

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self._model,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.1),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return CompletionResponse(text=response.choices[0].message.content)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.1),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True,
        )

        def gen():
            text = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    text += chunk.choices[0].delta.content
                    yield CompletionResponse(
                        text=text, delta=chunk.choices[0].delta.content
                    )

        return gen()
