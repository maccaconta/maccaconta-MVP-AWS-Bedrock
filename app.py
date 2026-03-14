# =============================================================================
# Arquivo: app.py
# Projeto: Backend de IA (MVP) - EMS GenAI
# Finalidade:
#   Programa principal FastAPI.
#   Registra endpoints do Backend de IA:
#     - POST /v1/ai/turn
#     - POST /v1/ai/evaluate
#     - POST /v1/ai/summarize-session
#     - POST /v1/ai/turn-graph
# Observações:
#   Em produção, este app pode ser executado com uvicorn/gunicorn.
# =============================================================================

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent  # pasta onde app.py está
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from config import Config

from endpoints.turn_endpoint import router as turn_router
from endpoints.evaluate_endpoint import router as evaluate_router
from endpoints.summarize_session_endpoint import router as summarize_router
from endpoints.turn_graph_endpoint import router as turn_graph_router


def create_app() -> FastAPI:
    """
    Cria e configura a aplicação FastAPI.
    """
    app = FastAPI(
        title="Backend de IA - EMS GenAI",
        version="1.0.0",
    )

    # Equivalente funcional ao app.config.from_object(Config) do Flask
    # Acessar depois via request.app.state.config
    app.state.config = Config

    # Registro dos routers sob /v1/ai
    app.include_router(turn_router, prefix="/v1/ai")
    app.include_router(evaluate_router, prefix="/v1/ai")
    app.include_router(summarize_router, prefix="/v1/ai")
    app.include_router(turn_graph_router, prefix="/v1/ai")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)