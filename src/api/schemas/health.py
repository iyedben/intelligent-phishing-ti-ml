"""Health endpoint response schema."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    """Minimal process-liveness response."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
