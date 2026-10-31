
import os
import base64
import json
from openai import AsyncOpenAI


async def evaluate_image(image_bytes: bytes, content_type: str, description: str):
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not configured")

    model = os.getenv(
        "OPENROUTER_VISION_MODEL",
        "google/gemma-3-27b-it:free"
    )

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    image_data_url = f"data:{content_type};base64,{image_base64}"

    prompt = f"""
    Evaluate whether the provided image visually matches the following description.

    Description:
    {description}

    Consider whether the visible content, layout, logos, text and other
    visual elements are consistent with the description.

    Important:
    - Evaluate only what can be seen in the image.
    - Do not assume that the image actually originates from the named website.
    - Do not penalize minor visual differences.
    - Give a score between 0 and 1.
    - 1 means a very strong visual match.
    - 0 means no meaningful visual match.

    Return only valid JSON in this format:
    {{
        "score": 0.0,
        "reason": "Short explanation of the evaluation"
    }}
    """

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url}
                    }
                ]
            }
        ],
        temperature=0,
        max_tokens=300
    )

    result_text = response.choices[0].message.content

    if not result_text:
        raise ValueError("The image model returned an empty response")

    result = json.loads(result_text)

    score = float(result["score"])
    reason = str(result["reason"])

    if not 0 <= score <= 1:
        raise ValueError("The returned score must be between 0 and 1")

    return {
        "score": score,
        "reason": reason
    }
