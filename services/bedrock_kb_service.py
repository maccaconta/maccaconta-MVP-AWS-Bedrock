# =============================================================================
# Arquivo: services/bedrock_kb_service.py
# Projeto: Backend de IA (MVP) - EMS GenAI
# a. Finalidade:
#   Recuperação de conhecimento (RAG) via Amazon Bedrock Knowledge Bases.
#   A Knowledge Base é configurada no Console apontando para S3 Vectors.
#   Este serviço executa "retrieve" e normaliza para evidências consumíveis no prompt.
# b. Objetivo principal: 
#   fornecer evidências (trechos) e citações de forma padronizada para a etapa seguinte 
#   (composição do prompt + geração).
# ------------------------------------------------------------------------------
# Este script implementa a camada de recuperação de conhecimento (RAG) usando Amazon Bedrock Knowledge Bases.
# Ele encapsula o cliente boto3 do serviço bedrock-agent-runtime e expõe duas responsabilidades bem delimitadas:
#
# 1. retrieve(...): monta e executa uma chamada de busca vetorial na Knowledge Base, enviando knowledgeBaseId, 
#  a query em retrievalQuery.text e a configuração de busca vectorSearchConfiguration.numberOfResults (top-k). 
# bedrock_kb_service
# 2. normalize(resp, score_threshold): transforma o retorno “bruto” do Bedrock KB em um formato consumível 
#  pelo prompt e pela API. Ele extrai score, gera um snippet (até 500 caracteres), identifica a origem do 
#  documento (ex.: URI no S3) e um identificador do chunk, produzindo:
# - evidences: lista com {docId, chunkId, score, snippet}
# - citations: lista de strings no formato docId:chunkId
# - flags e no_evidence: sinalizam ausência/baixa qualidade de evidências (ex.: best_score < score_threshold) 
# =============================================================================

import boto3
from botocore.config import Config as BotoConfig
from typing import Dict, Any, List, Optional


class BedrockKnowledgeBaseService:
    def __init__(self, region: str, timeout_seconds: int):
        self.client = boto3.client(
            "bedrock-agent-runtime",
            region_name=region,
            config=BotoConfig(
                read_timeout=timeout_seconds,
                connect_timeout=timeout_seconds,
                retries={"max_attempts": 2, "mode": "standard"},
            ),
        )

    def retrieve(
        self,
        knowledge_base_id: str,
        query_text: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Recupera evidências via Knowledge Base.
        Para MVP, filtros são opcionais e dependem do schema de metadados definido na KB.
        """
        req: Dict[str, Any] = {
            "knowledgeBaseId": knowledge_base_id,
            "retrievalQuery": {"text": query_text},
            "retrievalConfiguration": {
                "vectorSearchConfiguration": {
                    "numberOfResults": top_k
                }
            },
        }

        if filters:
            req["retrievalConfiguration"]["vectorSearchConfiguration"]["filter"] = filters

        return self.client.retrieve(**req)



    # @staticmethod
    # def normalize(resp: Dict[str, Any], score_threshold: float) -> Dict[str, Any]:
    #     """
    #     Converte o retorno do retrieve para um formato simples de evidências.
    #     """
    #     results = resp.get("retrievalResults", [])
    #     evidences: List[Dict[str, Any]] = []
    #     citations: List[str] = []

    #     best_score = 0.0
    #     for r in results:
    #         score = float(r.get("score", 0.0))
    #         best_score = max(best_score, score)

    #         content = r.get("content", {})
    #         snippet = content.get("text", "")[:500]

    #         location = r.get("location", {})
    #         doc_id = str(location.get("s3Location", {}).get("uri", "")) or str(location)
    #         #chunk_id = r.get("metadata", {}).get("chunkId", "desconhecido")
            
    #         meta = r.get("metadata", {}) or {}
    #         chunk_id = meta.get("chunkId") or meta.get("chunk_id") or meta.get("id") or r.get("id") or "desconhecido"

    #         evidences.append({
    #             "docId": doc_id,
    #             "chunkId": chunk_id,
    #             "score": score,
    #             "snippet": snippet,
    #         })
    #         citations.append(f"{doc_id}:{chunk_id}")

    #     citations = list(dict.fromkeys(citations))

    #     flags: List[str] = []
    #     no_evidence = (len(evidences) == 0) or (best_score < score_threshold)
    #     if no_evidence:
    #         flags.append("no_evidence")

    #     return {
    #         "evidences": evidences,
    #         "citations": citations,
    #         "flags": flags,
    #         "no_evidence": no_evidence,
    #     }
    

    @staticmethod
    def normalize(resp: Dict[str, Any], score_threshold: float) -> Dict[str, Any]:
        """
        Converte o retorno do retrieve para um formato simples de evidências.
        Melhorias:
        - fallback robusto para chunkId
        - inclui 'full_content' para debug
        - deduplication de citations preservando ordem
        - snippet maior (até 2000 chars)
        """
        results = resp.get("retrievalResults", []) or []
        evidences: List[Dict[str, Any]] = []
        citations: List[str] = []

        best_score = 0.0
        for r in results:
            score = float(r.get("score", 0.0))
            best_score = max(best_score, score)

            content = r.get("content", {}) or {}
            # Many KBs store the text in content.text or content['body'] etc.
            snippet = ""
            if isinstance(content, dict):
                snippet = content.get("text") or content.get("body") or ""
            else:
                snippet = str(content)

            # limit snippet length for safety
            snippet = snippet[:2000]

            location = r.get("location", {}) or {}
            doc_id = ""
            # Try s3Location.uri first, then fallbacks
            try:
                doc_id = str(location.get("s3Location", {}).get("uri", "")) or str(location)
            except Exception:
                doc_id = str(location)

            # robust chunk id selection: check various possible names
            meta = r.get("metadata", {}) or {}
            chunk_id = (
                meta.get("chunkId")
                or meta.get("chunk_id")
                or meta.get("id")
                or meta.get("chunk")
                or r.get("id")
                or doc_id  # last resort: use doc id so citation not empty
            )

            evidences.append({
                "docId": doc_id,
                "chunkId": chunk_id,
                "score": score,
                "snippet": snippet,
                "full_content": content,
                "location": location,
            })

            citations.append(f"{doc_id}:{chunk_id}")

        # dedupe citations preserving order
        seen = set()
        deduped_citations = []
        for c in citations:
            if c not in seen:
                deduped_citations.append(c)
                seen.add(c)

        flags: List[str] = []
        no_evidence = (len(evidences) == 0) or (best_score < score_threshold)
        if no_evidence:
            flags.append("no_evidence")

        return {
            "evidences": evidences,
            "citations": deduped_citations,
            "flags": flags,
            "no_evidence": no_evidence,
        }
