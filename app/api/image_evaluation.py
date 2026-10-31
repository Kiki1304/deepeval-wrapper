from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status

from ..services.image_judge_service import evaluate_image


router = APIRouter(prefix="/evaluate-image", tags=["Image Evaluation"])


@router.post("/")
async def evaluate_uploaded_image(
    image: UploadFile = File(...),
    description: str = Form(...)
):
    from ..auth import get_current_user
    current_user = await get_current_user()

    allowed_types = ["image/jpeg", "image/png", "image/webp"]

    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG, PNG and WEBP images are supported."
        )

    image_bytes = await image.read(10 * 1024 * 1024 + 1)

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty."
        )

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="The maximum image size is 10 MB."
        )

    if not description.strip():
        raise HTTPException(
            status_code=400,
            detail="Description must not be empty."
        )

    try:
        result = await evaluate_image(
            image_bytes=image_bytes,
            content_type=image.content_type,
            description=description
        )

        return result

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image evaluation failed: {str(e)}"
        )
