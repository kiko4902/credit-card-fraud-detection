"""Fraud scoring API.

    uvicorn app.api:app            (or ``make api``)

All model logic lives in :mod:`fraud.serving`; this file only defines the HTTP interface.
The model is loaded once at startup. Every response carries the time the request took in
the ``X-Process-Time-Ms`` header, and the same number is written to the log.
"""

import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, create_model

from fraud.config import AMOUNT_COLUMN, MAX_BATCH_SIZE, PCA_COLUMNS
from fraud.features import SECONDS_PER_DAY
from fraud.serving import SECONDS_OF_DAY, FraudModel, Prediction, get_model

logger = logging.getLogger("fraud.api")

# Input limits. The PCA components of the dataset lie within about +-120 and the largest
# amount is 25,691; the limits are generous but reject obviously broken input.
MAX_ABS_COMPONENT: float = 1_000.0
MAX_AMOUNT: float = 1_000_000.0
LATENCY_HEADER: str = "X-Process-Time-Ms"


class _TransactionBase(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


_component = (
    float,
    Field(ge=-MAX_ABS_COMPONENT, le=MAX_ABS_COMPONENT, description="PCA component"),
)
Transaction = create_model(  # type: ignore[call-overload]
    "Transaction",
    __base__=_TransactionBase,
    __doc__="One card transaction: the 28 PCA components, the amount and the time of day.",
    **{column: _component for column in PCA_COLUMNS},
    **{
        AMOUNT_COLUMN: (float, Field(ge=0, le=MAX_AMOUNT, description="Amount in euros")),
        SECONDS_OF_DAY: (
            float,
            Field(ge=0, lt=SECONDS_PER_DAY, description="Seconds since midnight (0 to 86,399)"),
        ),
    },
)


class Batch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transactions: list[Transaction] = Field(min_length=1, max_length=MAX_BATCH_SIZE)  # type: ignore[valid-type]


class FeatureContribution(BaseModel):
    feature: str
    label: str
    value: float = Field(description="Value of the feature for this transaction")
    contribution: float = Field(description="SHAP value in log-odds of the fraud score")
    direction: Literal["increases", "decreases"] = Field(
        description="Whether the feature pushes the fraud score up or down"
    )


class PredictionResponse(BaseModel):
    fraud_score: float = Field(
        description="Score between 0 and 1 that ranks transactions. Not a probability."
    )
    flagged: bool = Field(description="True when fraud_score >= threshold")
    threshold: float
    top_features: list[FeatureContribution]
    model_version: str


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int
    flagged_count: int
    model_version: str


class Health(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    model_version: str


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Load the model once, before the first request is accepted."""
    if not logger.handlers:  # uvicorn configures only its own loggers
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s - %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    start = time.perf_counter()
    application.state.model = get_model()
    logger.info(
        "model %s loaded in %.0f ms",
        application.state.model.version,
        1000 * (time.perf_counter() - start),
    )
    yield


app = FastAPI(
    title="Credit card fraud scoring",
    description=(
        "Scores card transactions with the model selected in this project and flags those "
        "at or above a cost-based threshold. The output is a fraud score, not a calibrated "
        "probability."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def measure_latency(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = 1000 * (time.perf_counter() - start)
    response.headers[LATENCY_HEADER] = f"{elapsed_ms:.2f}"
    logger.info(
        "%s %s -> %d in %.2f ms", request.method, request.url.path, response.status_code, elapsed_ms
    )
    return response


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, error: RequestValidationError) -> JSONResponse:
    """422 with the location and reason of every problem.

    The rejected input is not echoed back: it can hold NaN or infinity, which cannot be
    written as JSON and would turn the 422 into a 500.
    """
    detail = [
        {"type": problem["type"], "loc": list(problem["loc"]), "msg": problem["msg"]}
        for problem in error.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": detail})


def _model(request: Request) -> FraudModel:
    return request.app.state.model


def _response(prediction: Prediction) -> PredictionResponse:
    return PredictionResponse(**prediction.as_dict())


# Plain ``def`` endpoints: FastAPI runs them in a thread pool, so scoring does not block
# the event loop.
@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction, request: Request) -> PredictionResponse:  # type: ignore[valid-type]
    """Score one transaction."""
    [prediction] = _model(request).predict([transaction.model_dump()])  # type: ignore[attr-defined]
    return _response(prediction)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: Batch, request: Request) -> BatchResponse:
    """Score up to ``MAX_BATCH_SIZE`` transactions in one request."""
    model = _model(request)
    predictions = model.predict([transaction.model_dump() for transaction in batch.transactions])
    return BatchResponse(
        predictions=[_response(prediction) for prediction in predictions],
        count=len(predictions),
        flagged_count=sum(prediction.flagged for prediction in predictions),
        model_version=model.version,
    )


@app.get("/health", response_model=Health)
def health(request: Request) -> Health:
    return Health(status="ok", model_loaded=True, model_version=_model(request).version)


@app.get("/model-info")
def model_info(request: Request) -> dict[str, Any]:
    """Version, threshold, cost model and the headline metrics of the served model."""
    return {**_model(request).info(), "max_batch_size": MAX_BATCH_SIZE}
