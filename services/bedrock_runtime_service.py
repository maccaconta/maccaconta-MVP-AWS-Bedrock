# services/bedrock_runtime_service.py
import json
import os
from typing import Any, Dict, List, Union, Optional

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError


class BedrockRuntimeService:
    """
    Serviço para invocar modelos de texto no Amazon Bedrock Runtime.

    Uso:
        svc = BedrockRuntimeService(region="us-east-1", timeout_seconds=20)
        resp = svc.invoke_text_model(model_id="amazon.nova-pro-v1:0", prompt={"system": "...", "user":"..."}, generation={...})
        resp -> {"replyText": "...", "modelId": model_id, "raw": <raw json>}
    """

    def __init__(self, region: str, timeout_seconds: int = 20):
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=BotoConfig(
                read_timeout=timeout_seconds,
                connect_timeout=timeout_seconds,
                retries={"max_attempts": 2, "mode": "standard"},
            ),
        )
        # debug flag controlado por env var (opcional)
        self.debug = bool(os.environ.get("BEDROCK_RUNTIME_DEBUG", "") in ("1", "true", "True"))

    # ---- helpers ----------------------------------------------------------------
    @staticmethod
    def _merge_system_into_user_text(system_text: str, user_text: str) -> str:
        """
        Junta system_text e user_text em um único texto adequado para ser enviado como
        uma única mensagem 'user' (alguns modelos Bedrock não aceitam role='system').
        """
        system_text = (system_text or "").strip()
        user_text = (user_text or "").strip()
        if not system_text:
            return user_text
        # separadores   para o modelo entender o que são instruções
        return f"INSTRUÇÕES DO SISTEMA:\n{system_text}\n\nCONTEÚDO A SER PROCESSADO:\n{user_text}"

    @staticmethod
    def _messages_to_valid_user_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Converte uma lista de messages (potencialmente com vários roles) em
        uma lista contendo UMA message role='user' (concatenação).
        - concatena todos os textos de role 'system' como prefixo
        - concatena todos os textos de outros roles como corpo
        """
        system_parts: List[str] = []
        other_parts: List[str] = []

        for m in messages:
            try:
                role = (m.get("role") or "").lower()
            except Exception:
                role = ""
            # extrai texto do content: aceita tanto list->[{"text": "..."}] quanto string
            content_text = ""
            try:
                content = m.get("content", "")
                if isinstance(content, list) and content:
                    # pega o primeiro item com campo text
                    first = content[0]
                    if isinstance(first, dict):
                        content_text = first.get("text", "") or ""
                    else:
                        content_text = str(first)
                elif isinstance(content, str):
                    content_text = content
                else:
                    # fallback: 
                    content_text = json.dumps(content, ensure_ascii=False)
            except Exception:
                content_text = str(m.get("content", "") or "")

            if role == "system":
                system_parts.append(content_text)
            else:
                other_parts.append(content_text)

        merged_user = ""
        if system_parts:
            merged_user += "\n".join(system_parts) + "\n\n"
        if other_parts:
            merged_user += "\n\n".join(p for p in other_parts if p)

        merged_user = merged_user.strip() or ""  # garante string
        return [{"role": "user", "content": [{"text": merged_user}]}]

    @staticmethod
    def _extract_text_from_raw(data: Dict[str, Any]) -> str:
        """
        Extrai de forma defensiva o texto do payload bruto retornado pelo Bedrock Runtime.
        Testa alguns caminhos comuns e usa fallbacks.
        """
        if not data or not isinstance(data, dict):
            return ""

 
        try:
            text = data.get("output", {}).get("message", {}).get("content", [])[0].get("text", "")
            if isinstance(text, str) and text.strip():
                return text
        except Exception:
            pass

        # alguns modelos retornam results[0].outputText
        try:
            results = data.get("results")
            if isinstance(results, list) and len(results) > 0:
                ot = results[0].get("outputText") or results[0].get("generatedText") or ""
                if ot:
                    return ot
        except Exception:
            pass

        # outros fallbacks
        for key in ("completion", "generated_text", "outputText", "body", "text"):
            v = data.get(key)
            if isinstance(v, str) and v.strip():
                return v

        # último recurso: stringify data (curto)
        try:
            s = json.dumps(data, ensure_ascii=False)
            return s[:2000]
        except Exception:
            return ""
 
    def invoke_text_model(self, model_id: str, prompt: Union[str, Dict[str, str], List[Dict[str, Any]]], generation: Dict[str, Any]):
        """
        Invoca o modelo de texto no Bedrock Runtime.

        - model_id: id do modelo (ex.: 'amazon.nova-pro-v1:0')
        - prompt:
            * str -> enviado como uma única mensagem user
            * dict {"system": "...", "user": "..."} -> system será mesclado em user
            * list[message] -> convertida para UMA mensagem user concatenada (remove roles inválidos)
        - generation: dict com 'maxOutputTokens', 'temperature', 'topP' (nomes usados internamente)
        """
        # 1) Normaliza prompt -> mensagens compatíveis (sem role 'system')
        if isinstance(prompt, list):
            messages = self._messages_to_valid_user_messages(prompt)
        elif isinstance(prompt, dict):
            system_text = prompt.get("system", "") or ""
            user_text = prompt.get("user", "") or ""
            merged = self._merge_system_into_user_text(system_text, user_text)
            messages = [{"role": "user", "content": [{"text": merged}]}]
        else:
            messages = [{"role": "user", "content": [{"text": str(prompt)}]}]

        # 2) Monta body para invoke_model
        body = {
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": int(generation.get("maxOutputTokens", generation.get("max_tokens", 300))),
                "temperature": float(generation.get("temperature", 0.0)),
                "topP": float(generation.get("topP", 0.1)),
            },
        }

        # Debug opcional: print curto do body (primeira mensagem + infer config)
        if self.debug:
            try:
                preview = {
                    "messages_preview": [messages[0] if messages else {}],
                    "inference": body["inferenceConfig"],
                }
                print("BEDROCK_RUNTIME DEBUG >>>", json.dumps(preview, ensure_ascii=False))
            except Exception:
                pass

        # 3) Chamada ao Bedrock
        try:
            resp = self.client.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "Unknown")
            msg = e.response.get("Error", {}).get("Message", str(e))
            # lança erro mais legível para o endpoint capturar
            raise RuntimeError(f"Bedrock InvokeModel falhou: {code} - {msg}") from e

        # 4) Ler e parsear resposta
        try:
            raw = json.loads(resp["body"].read())
        except Exception:
            # some boto variants may already present dict, fallback:
            raw = resp.get("body") or {}

        # 5) extrai texto de forma defensiva
        text = self._extract_text_from_raw(raw)

        return {"replyText": text, "modelId": model_id, "raw": raw}