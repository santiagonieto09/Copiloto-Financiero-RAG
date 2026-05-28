"""
Guardrails de dominio para el Copiloto Financiero.

Estas validaciones no reemplazan al prompt: imponen limites desde la aplicacion
antes de llamar al LLM o de indexar contenido no financiero.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


FINANCIAL_TERMS = {
    # Finanzas generales
    "finanza", "financiero", "financial", "banco", "bancario", "bank", "banking",
    "credito", "credit", "prestamo", "loan", "hipoteca", "mortgage", "deuda",
    "debt", "interes", "interest", "tasa", "rate", "cuota", "fee", "comision",
    "commission", "capital", "liquidez", "liquidity", "solvencia", "solvency",
    "balance", "activo", "asset", "pasivo", "liability", "patrimonio", "equity",
    "ingreso", "income", "egreso", "gasto", "expense", "costo", "cost",
    "presupuesto", "budget", "flujo", "cashflow", "cash", "caja",
    # Mercados e inversion
    "inversion", "investment", "invertir", "portfolio", "portafolio", "accion",
    "acciones", "stock", "stocks", "bono", "bond", "bonds", "fondo", "fund",
    "etf", "dividendo", "dividend", "rendimiento", "return", "rentabilidad",
    "volatilidad", "volatility", "riesgo", "risk", "mercado", "market",
    "trading", "broker", "derivado", "derivative", "opcion", "option",
    "futuro", "future", "forex", "cripto", "crypto", "bitcoin",
    # Contabilidad, auditoria, compliance
    "contabilidad", "accounting", "auditoria", "audit", "compliance",
    "cumplimiento", "regulacion", "regulation", "regulatorio", "regulatory",
    "impuesto", "tax", "tributario", "fiscal", "aml", "kyc", "antilavado",
    "lavado", "laundering", "fraude", "fraud", "basilea", "basel", "ifrs",
    "niif", "sox", "riesgo crediticio", "riesgo operacional",
    # Productos y operaciones
    "transferencia", "transfer", "pago", "payment", "tarjeta", "card",
    "debito", "credito", "cuenta", "account", "nomina", "payroll",
    "seguro", "insurance", "poliza", "claim", "siniestro", "factura",
    "invoice", "cobranza", "collection", "tesoreria", "treasury",
}

PROMPT_INJECTION_PATTERNS = (
    r"ignora\s+(las\s+)?instrucciones",
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"olvida\s+(las\s+)?reglas",
    r"forget\s+(the\s+)?rules",
    r"system\s+prompt",
    r"developer\s+message",
    r"reveal\s+(your\s+)?instructions",
    r"muestra\s+(tus\s+)?instrucciones",
    r"actua\s+como",
    r"act\s+as",
)

DOCUMENT_REFERENCE_TERMS = {
    "documento", "documentos", "archivo", "archivos", "texto", "fuente",
    "fuentes", "contexto", "contenido", "resumen", "resumir", "resume",
    "summarize", "summary", "analiza", "analizar", "analysis", "explica",
    "explicar", "explain", "lo anterior", "la informacion", "la información",
}

REFUSAL_MESSAGE = (
    "Solo puedo ayudar con temas financieros. Puedo analizarlo desde una "
    "perspectiva de costos, ingresos, presupuesto, inversion, riesgo, "
    "cumplimiento, banca, seguros, impuestos o impacto financiero."
)


@dataclass(frozen=True)
class DomainDecision:
    """Resultado de una clasificacion de dominio."""

    allowed: bool
    reason: str
    score: int
    matched_terms: list[str]
    prompt_injection_detected: bool = False


def _normalize(text: str) -> str:
    return text.casefold()


def _count_financial_terms(text: str) -> tuple[int, list[str]]:
    normalized = _normalize(text)
    matches = sorted(
        term for term in FINANCIAL_TERMS
        if re.search(rf"\b{re.escape(term)}\b", normalized)
    )
    return len(matches), matches[:12]


def detect_prompt_injection(text: str) -> bool:
    """Detecta instrucciones sospechosas dentro de texto no confiable."""
    normalized = _normalize(text)
    return any(re.search(pattern, normalized) for pattern in PROMPT_INJECTION_PATTERNS)


def is_document_reference_query(text: str) -> bool:
    """
    Detecta consultas referidas al corpus RAG.

    Permite preguntas como "haz un resumen del documento" cuando ya existen
    documentos financieros indexados, sin abrir el chat directo a temas libres.
    """
    normalized = _normalize(text)
    return any(term in normalized for term in DOCUMENT_REFERENCE_TERMS)


def classify_financial_domain(text: str, *, min_score: int = 1) -> DomainDecision:
    """
    Clasifica si un texto pertenece al dominio financiero.

    Es una barrera deterministica y conservadora. El prompt del LLM sigue
    aplicando reglas mas detalladas, pero esta funcion evita llamadas obvias
    fuera de alcance y reduce la indexacion de documentos irrelevantes.
    """
    score, matched_terms = _count_financial_terms(text)
    prompt_injection = detect_prompt_injection(text)
    allowed = score >= min_score

    if allowed:
        reason = "Contenido financiero detectado."
    else:
        reason = "No se detectaron terminos financieros suficientes."

    return DomainDecision(
        allowed=allowed,
        reason=reason,
        score=score,
        matched_terms=matched_terms,
        prompt_injection_detected=prompt_injection,
    )


def refusal_response() -> str:
    """Mensaje unico de rechazo fuera de dominio."""
    return REFUSAL_MESSAGE
