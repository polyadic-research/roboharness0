"""FastAPI route modules for the Evaluation control plane."""

from fastapi import FastAPI

from rpent.ui.routes import cases, evaluations, internal, simulators


def include_evaluation_routes(app: FastAPI) -> None:
    app.include_router(evaluations.router)
    app.include_router(cases.router)
    app.include_router(simulators.router)
    app.include_router(internal.router)


__all__ = ["include_evaluation_routes"]
