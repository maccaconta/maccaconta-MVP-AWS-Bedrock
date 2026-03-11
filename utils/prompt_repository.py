# =============================================================================
# Arquivo: utils/prompt_repository.py
# Projeto: Backend de IA (MVP) - EMS GenAI
# Finalidade:
#   Repositório de templates (arquivos JSON) armazenados no bundle do container.
#   - Carrega blueprint de Turn
#   - Resolve componentes (persona, especialidade, cenário, políticas, saída)
# =============================================================================

import json
import os
from typing import Dict, Any


class PromptRepository:
    """
    Carrega templates/blueprints a partir de arquivos JSON.

    Estratégia MVP:
    - Templates residem no repositório Git e são empacotados no container.
    - Evolução ocorre via alteração de arquivo + deploy (controle de mudanças).
    """

    def __init__(self, templates_root: str):
        self.templates_root = templates_root
        self._cache: Dict[str, Dict[str, Any]] = {}

    def load_blueprint(self, blueprint_id: str, blueprint_version: str) -> Dict[str, Any]:
        """
        Carrega blueprint do Turn (receita de composição).
        Convenção: templates/blueprints/{blueprint_id}_{blueprint_version}.json
        """
        filename = f"{blueprint_id}_{blueprint_version}.json"
        path = os.path.join(self.templates_root, "blueprints", filename)
        return self._load_json(path)

    def load_component(self, component_type: str, file_name: str) -> Dict[str, Any]:
        """
        Carrega um componente por tipo.
        component_type: personas | especialidades | cenarios | politicas | saida
        file_name: nome do arquivo JSON (ex.: persona_risco_v1.json)
        """
        path = os.path.join(self.templates_root, "componentes", component_type, file_name)
        return self._load_json(path)

    def load_evaluate_template(self, file_name: str) -> Dict[str, Any]:
        """
        Carrega template de avaliação por rubricas.
        """
        path = os.path.join(self.templates_root, "evaluate", file_name)
        return self._load_json(path)

    def load_summarize_template(self, file_name: str) -> Dict[str, Any]:
        """
        Carrega template de sumarização.
        """
        path = os.path.join(self.templates_root, "summarize", file_name)
        return self._load_json(path)

    def _load_json(self, path: str) -> Dict[str, Any]:
        """
        Leitura com cache em memória.
        """
        if path in self._cache:
            return self._cache[path]

        if not os.path.exists(path):
            raise FileNotFoundError(f"Template/arquivo não encontrado: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._cache[path] = data
        return data
