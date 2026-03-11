# =============================================================================
# Arquivo: app.py
# Projeto: Backend de IA (MVP) - EMS GenAI
# Finalidade:
#   Programa principal Flask.
#   Registra endpoints do Backend de IA:
#     - POST /v1/ai/turn
#     - POST /v1/ai/evaluate
#     - POST /v1/ai/summarize-session
# Observações:
#   Em produção, este app deve ser executado por gunicorn dentro de container ECS.
# =============================================================================

# app.py (no topo, antes de from endpoints... )
# app.py (topo)
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent  # pasta onde app.py está
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

 
from flask import Flask
from config import Config

from endpoints.turn_endpoint import turn_bp
from endpoints.evaluate_endpoint import evaluate_bp
from endpoints.summarize_session_endpoint import summarize_bp

def create_app() -> Flask:
    """
    Cria e configura a aplicação Flask.
    """
    app = Flask(__name__)
    app.config.from_object(Config)

    # Registro dos blueprints sob /v1/ai
    app.register_blueprint(turn_bp, url_prefix="/v1/ai")
    app.register_blueprint(evaluate_bp, url_prefix="/v1/ai")
    app.register_blueprint(summarize_bp, url_prefix="/v1/ai")

    return app


if __name__ == "__main__":
    # Execução local (DEV). Em produção use gunicorn.
    app = create_app()
    app.run(host="0.0.0.0", port=8080, debug=True)
