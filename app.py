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
#   Em produção, este app pode ser executado com:
#     gunicorn app:app -k uvicorn.workers.UvicornWorker -w 2 -b 0.0.0.0:8080
# =============================================================================

import logging
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI

# garantir import local do pacote (se estiver executando do diretório do projeto)
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# import local
from config import Config  # Config deve ser inicializável (veja nota abaixo)

from endpoints.turn_endpoint import router as turn_router
from endpoints.evaluate_endpoint import router as evaluate_router
from endpoints.summarize_session_endpoint import router as summarize_router
from endpoints.turn_graph_endpoint import router as turn_graph_router

logger = logging.getLogger("ems_mvp")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def create_app(config: Any | None = None) -> FastAPI:
    """
    Cria e configura a aplicação FastAPI.

    Args:
        config: opcional. Se fornecido, será armazenado em app.state.config.
                Caso None, cria uma instância de Config() aqui.
    """
    app = FastAPI(
        title="Backend de IA - EMS GenAI",
        version="1.0.0",
    )

    # Inicializa e disponibiliza a configuração central da aplicação
    # Preferível passar uma instância (Config()) em vez da classe.
    app.state.config = config if config is not None else Config()

    # registro dos routers sob /v1/ai
    app.include_router(turn_router, prefix="/v1/ai")
    app.include_router(evaluate_router, prefix="/v1/ai")
    app.include_router(summarize_router, prefix="/v1/ai")
    app.include_router(turn_graph_router, prefix="/v1/ai")

    # health check simples
    @app.get("/health", tags=["health"])
    def health() -> dict:
        return {"status": "ok"}

    logger.info("Aplicação FastAPI criada e routers registrados.")
    return app


# instância global do app (necessária para gunicorn/uvicorn)
app = create_app()

# execução direta (apenas para dev). Em produção use gunicorn + UvicornWorker.
if __name__ == "__main__":
    import uvicorn

    # reload=True apenas para desenvolvimento local (não em container de produção)
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)