# =============================================================================
# config.py
# Configuração central da aplicação
# =============================================================================

import os
from pathlib import Path


class Config:
    # Diretório base do projeto
    BASE_DIR = Path(__file__).resolve().parent

    # =============================
    # Diretórios
    # =============================
    TEMPLATES_ROOT = os.environ.get(
        "TEMPLATES_ROOT",
        str(BASE_DIR / "templates")
    )

    # =============================
    # AWS - Região
    # =============================
    AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

    # =============================
    # Bedrock - Knowledge Base
    # =============================
    BEDROCK_KB_ID = os.environ.get("BEDROCK_KB_ID", None)

    # =============================
    # Bedrock - Modelos
    # =============================
    # Modelo usado para TURN (geração principal)
    BEDROCK_TURN_MODEL_ID = os.environ.get(
        "BEDROCK_TURN_MODEL_ID",
        "amazon.nova-pro-v1:0"
    )

    # Modelo usado para EVALUATE
    BEDROCK_EVALUATE_MODEL_ID = os.environ.get(
        "BEDROCK_EVALUATE_MODEL_ID",
        "amazon.nova-pro-v1:0"
    )

    # Modelo usado para SUMMARIZE
    BEDROCK_SUMMARIZE_MODEL_ID = os.environ.get(
        "BEDROCK_SUMMARIZE_MODEL_ID",
        "amazon.nova-pro-v1:0"
    )

    # =============================
    # Timeout Bedrock
    # =============================
    BEDROCK_TIMEOUT_SECONDS = int(
        os.environ.get("BEDROCK_TIMEOUT_SECONDS", "20")
    )

    # =============================
    # Defaults RAG / Retrieval
    # =============================
    DEFAULT_TOP_K = int(os.environ.get("DEFAULT_TOP_K", "3"))
    DEFAULT_SCORE_THRESHOLD = float(os.environ.get("DEFAULT_SCORE_THRESHOLD", "0.1"))

    COST_PER_1000_TOKENS_USD = '1'

    DEFAULT_TOP_K = int(os.environ.get("DEFAULT_TOP_K", "3"))
    DEFAULT_SCORE_THRESHOLD = float(os.environ.get("DEFAULT_SCORE_THRESHOLD", "0.1"))

    DEFAULT_MAX_OUTPUT_TOKENS = int(os.environ.get("DEFAULT_MAX_OUTPUT_TOKENS", "512"))
    DEFAULT_TEMPERATURE = float(os.environ.get("DEFAULT_TEMPERATURE", "0.2"))
    DEFAULT_TOP_P = float(os.environ.get("DEFAULT_TOP_P", "0.9"))