from .evaluation import router as evaluation_router
from .auth import router as auth_router
from .metrics import router as metrics_router
from .jobs import router as jobs_router
from .health import router as health_router
from .image_evaluation import router as image_evaluation_router

__all__ = [
    "evaluation_router",
    "auth_router",
    "metrics_router",
    "jobs_router",
    "health_router",
    "image_evaluation_router",
]
