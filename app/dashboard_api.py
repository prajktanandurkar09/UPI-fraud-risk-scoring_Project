"""
UPI-SHIELD dashboard API client.

This module is the ONLY place that knows the FastAPI base URL and the
ONLY place that makes HTTP calls to it. Every other dashboard module
goes through the functions defined here.

The dashboard never loads or retrains the model itself: every fraud
prediction shown to the user comes from the existing FastAPI
``/predict`` endpoint.
"""

from __future__ import annotations

import os
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_API_URL = "http://127.0.0.1:8000"

# Read once at import time. Override with:
#   set UPI_SHIELD_API_URL=http://127.0.0.1:8000   (Windows)
#   export UPI_SHIELD_API_URL=http://127.0.0.1:8000 (macOS/Linux)
API_BASE_URL = os.environ.get("UPI_SHIELD_API_URL", DEFAULT_API_URL).rstrip("/")

HEALTH_ENDPOINT = f"{API_BASE_URL}/health"
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"

HEALTH_TIMEOUT_SECONDS = 3
PREDICT_TIMEOUT_SECONDS = 20


class ApiResult:
    """
    Uniform result wrapper for API calls.

    Exactly one of ``data`` / ``error`` is populated. Keeping this as an
    explicit object (rather than a bare tuple) makes call sites read
    clearly: ``if result.ok: ... else: st.error(result.error)``.
    """

    def __init__(self, ok: bool, data: Any = None, error: str | None = None):
        self.ok = ok
        self.data = data
        self.error = error


def check_health() -> ApiResult:
    """
    Call GET /health.

    Returns an ApiResult wrapping the health payload on success, or a
    short, user-facing error message on failure. Never raises.
    """

    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=HEALTH_TIMEOUT_SECONDS)
        response.raise_for_status()
        return ApiResult(ok=True, data=response.json())

    except requests.exceptions.ConnectionError:
        return ApiResult(
            ok=False,
            error="Could not connect to the UPI-SHIELD API. Is it running?",
        )
    except requests.exceptions.Timeout:
        return ApiResult(ok=False, error="Health check timed out.")
    except requests.exceptions.RequestException as exc:
        return ApiResult(ok=False, error=f"Health check failed: {exc}")


def predict_transaction(payload: dict) -> ApiResult:
    """
    Call POST /predict with a transaction payload.

    ``payload`` must match the FastAPI TransactionRequest schema
    (app/schemas.py). Field values that are unknown/missing should be
    sent as ``None`` rather than omitted, since the schema declares
    them as Optional.
    """

    try:
        response = requests.post(
            PREDICT_ENDPOINT,
            json=payload,
            timeout=PREDICT_TIMEOUT_SECONDS,
        )

        if response.status_code == 422:
            # Pydantic validation error — surface the real detail
            # instead of a generic message so the user can fix input.
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            return ApiResult(
                ok=False,
                error=f"The API rejected the transaction (validation error): {detail}",
            )

        response.raise_for_status()
        return ApiResult(ok=True, data=response.json())

    except requests.exceptions.ConnectionError:
        return ApiResult(
            ok=False,
            error=(
                "Could not connect to the UPI-SHIELD API at "
                f"{API_BASE_URL}. Make sure it is running "
                "(uvicorn app.main:app --reload)."
            ),
        )
    except requests.exceptions.Timeout:
        return ApiResult(
            ok=False,
            error="The prediction request timed out. The API may be slow to respond.",
        )
    except requests.exceptions.HTTPError as exc:
        return ApiResult(ok=False, error=f"API returned an error: {exc}")
    except requests.exceptions.RequestException as exc:
        return ApiResult(ok=False, error=f"API request failed: {exc}")
    except ValueError:
        # response.json() failed — API returned a non-JSON body.
        return ApiResult(ok=False, error="The API returned an invalid (non-JSON) response.")