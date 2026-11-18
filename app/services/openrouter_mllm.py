
import os
import json
from openai import OpenAI, AsyncOpenAI
from deepeval.models import DeepEvalBaseMLLM
from deepeval.test_case import MLLMImage


class OpenRouterMLLM(DeepEvalBaseMLLM):
    def __init__(self):
        self.model_name = os.getenv(
            "OPENROUTER_VISION_MODEL",
            "google/gemma-3-27b-it"
        )
        api_key = os.environ["OPENROUTER_API_KEY"]
        base_url = "https://openrouter.ai/api/v1"

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        super().__init__(model_name=self.model_name)

    def get_model_name(self):
        return self.model_name

    def prepare_messages(self, prompt, schema=None):
        content = []

        for item in prompt if isinstance(prompt, list) else [prompt]:
            if isinstance(item, MLLMImage):
                url = (
                    f"data:{item.mimeType};base64,{item.dataBase64}"
                    if item.local else item.url
                )
                content.append({
                    "type": "image_url",
                    "image_url": {"url": url}
                })
            else:
                content.append({"type": "text", "text": str(item)})

        if schema is not None:
            content.append({
                "type": "text",
                "text": "Return only JSON matching this schema: "
                        + json.dumps(schema.model_json_schema())
            })

        return [{"role": "user", "content": content}]

    def parse_response(self, response, schema):
        text = response.choices[0].message.content or ""

        if schema is not None:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start < 0 or end <= start:
                raise ValueError("No JSON in model response")
            return schema.model_validate_json(text[start:end])

        return text

    def generate(self, prompt, schema=None, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.prepare_messages(prompt, schema),
            temperature=0,
            max_tokens=800
        )
        return self.parse_response(response, schema)

    async def a_generate(self, prompt, schema=None, **kwargs):
        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=self.prepare_messages(prompt, schema),
            temperature=0,
            max_tokens=800
        )
        return self.parse_response(response, schema)
