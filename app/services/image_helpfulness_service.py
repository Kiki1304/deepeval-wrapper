
import tempfile
from pathlib import Path

from deepeval.metrics import ImageHelpfulnessMetric
from deepeval.test_case import MLLMTestCase, MLLMImage

from .openrouter_mllm import OpenRouterMLLM


async def evaluate_image_helpfulness(image_bytes, content_type, description):
    extensions = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp"
    }

    with tempfile.TemporaryDirectory() as folder:
        image_path = Path(folder) / ("image" + extensions[content_type])
        image_path.write_bytes(image_bytes)

        test_case = MLLMTestCase(
            input=["Evaluate image helpfulness."],
            actual_output=[
                description,
                MLLMImage(url=str(image_path), local=True)
            ]
        )

        metric = ImageHelpfulnessMetric(
            model=OpenRouterMLLM(),
            async_mode=True,
            strict_mode=False
        )

        await metric.a_measure(test_case)

        return {
            "score": metric.score,
            "reason": metric.reason
        }
