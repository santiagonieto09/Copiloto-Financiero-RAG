"""
Esquemas Pydantic para request/response de la API.
Definen la estructura de datos del Copiloto Financiero.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


# ============================================
# Enums
# ============================================

class DocumentType(str, Enum):
    """Tipos de documentos financieros soportados."""
    PDF = "pdf"
    TXT = "txt"
    DOCX = "docx"


# ============================================
# Chat Schemas
# ============================================

class ChatRequest(BaseModel):
    """Solicitud de chat al copiloto."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Pregunta del usuario sobre servicios financieros",
        examples=["¿Cuáles son los requisitos para una auditoría de riesgo bancario?"],
    )
    session_id: Optional[str] = Field(
        default=None,
        description="ID de sesión para mantener contexto de conversación",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "¿Qué regulaciones aplican para transferencias internacionales?",
                    "session_id": "session-abc-123",
                }
            ]
        }
    }


class SourceDocument(BaseModel):
    """Documento fuente usado para generar la respuesta."""
    content: str = Field(description="Extracto del documento")
    source: str = Field(description="Nombre o ruta del archivo fuente")
    page: Optional[int] = Field(default=None, description="Número de página (si aplica)")
    relevance_score: Optional[float] = Field(
        default=None,
        description="Score de relevancia (0-1)",
    )


class ChatResponse(BaseModel):
    """Respuesta del copiloto financiero."""
    answer: str = Field(description="Respuesta generada por el copiloto")
    sources: list[SourceDocument] = Field(
        default_factory=list,
        description="Documentos fuente utilizados",
    )
    session_id: str = Field(description="ID de sesión de la conversación")
    model: str = Field(description="Modelo LLM utilizado")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Marca de tiempo de la respuesta",
    )
    guardrail_triggered: bool = Field(
        default=False,
        description="Indica si se aplicó una regla de seguridad o dominio",
    )
    guardrail_reason: Optional[str] = Field(
        default=None,
        description="Motivo de la regla aplicada, si existe",
    )


# ============================================
# Document Schemas
# ============================================

class DocumentUploadResponse(BaseModel):
    """Respuesta tras subir un documento."""
    filename: str = Field(description="Nombre del archivo procesado")
    chunks_created: int = Field(description="Número de chunks generados")
    chunks_rejected: int = Field(
        default=0,
        description="Chunks descartados por estar fuera del dominio financiero",
    )
    suspicious_chunks: int = Field(
        default=0,
        description="Chunks con posibles instrucciones maliciosas detectadas",
    )
    doc_type: DocumentType = Field(description="Tipo de documento")
    message: str = Field(description="Mensaje de estado")


class CollectionStats(BaseModel):
    """Estadísticas de la colección de documentos."""
    total_documents: int = Field(description="Total de chunks en la colección")
    collection_name: str = Field(description="Nombre de la colección")


# ============================================
# Health Check
# ============================================

class HealthResponse(BaseModel):
    """Respuesta del health check."""
    status: str = Field(default="healthy")
    version: str = Field(default="1.0.0")
    environment: str
    groq_model: str
    embedding_model: str
    total_documents: int = Field(
        default=0,
        description="Documentos indexados en el vector store",
    )
