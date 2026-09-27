"""
UPI-SHIELD API request and response schemas.
"""

from typing import Any

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """Transaction data submitted to the API."""

    TransactionDT: float = Field(
        default=86400.0
    )

    TransactionAmt: float = Field(
        default=25.0,
        ge=0,
    )

    ProductCD: str = "W"

    card1: float | None = None
    card2: float | None = None
    card3: float | None = None
    card4: str | None = None
    card5: float | None = None
    card6: str | None = None

    addr1: float | None = None
    addr2: float | None = None

    P_emaildomain: str | None = None
    R_emaildomain: str | None = None

    DeviceType: str | None = None
    DeviceInfo: str | None = None


class PredictionResponse(BaseModel):
    """UPI-SHIELD prediction response."""

    fraud_probability: float
    risk_score: int
    risk_band: str
    decision_threshold: float
    decision: str
    probability_source: str
    explanations: list[dict[str, Any]]