"""Version 1 URL analysis route; orchestration stays in AnalysisService."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_analysis_service
from src.api.schemas.analyze import AnalyzeRequest, AnalyzeResponse

router = APIRouter(tags=["analysis"])


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze a URL",
    description=(
        "Analyze URL-only features with the existing model and SHAP implementation, "
        "then query configured threat-intelligence providers. The submitted destination "
        "is not fetched by this API."
    ),
    responses={500: {"description": "The analysis could not be completed."}},
)
def post_analyze(
    payload: AnalyzeRequest,
    service: Any = Depends(get_analysis_service),
) -> AnalyzeResponse:
    """Delegate analysis to the application service and hide internal failures."""
    try:
        return service.analyze(payload.url)
    except Exception:
        raise HTTPException(status_code=500, detail="Analysis could not be completed.") from None
