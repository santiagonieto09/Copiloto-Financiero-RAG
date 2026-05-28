"""
Rutas de la API REST del Copiloto Financiero.

Endpoints:
- POST /api/v1/chat          → Chat con RAG (documentos indexados)
- POST /api/v1/chat/direct   → Chat directo sin RAG
- POST /api/v1/documents/upload → Subir documento para indexar
- POST /api/v1/documents/ingest-directory → Indexar directorio completo
- GET  /api/v1/documents/stats → Estadísticas de la colección
- DELETE /api/v1/documents    → Limpiar colección
- DELETE /api/v1/sessions/{session_id} → Limpiar sesión de chat
- GET  /api/v1/health         → Health check
"""

import os
import shutil
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, status

from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentUploadResponse,
    DocumentType,
    CollectionStats,
    HealthResponse,
)
from app.services.chat_service import get_chat_service
from app.services.document_service import get_document_service
from app.core.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Copiloto Financiero"])


# ============================================
# Chat Endpoints
# ============================================

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat con RAG",
    description="Envía una pregunta y obtiene respuesta basada en documentos financieros indexados.",
)
async def chat_with_rag(request: ChatRequest):
    """
    Endpoint principal del copiloto financiero.
    Usa RAG para responder con contexto de documentos.
    """
    try:
        chat_service = get_chat_service()
        doc_service = get_document_service()

        # Verificar si hay documentos indexados
        stats = doc_service.get_collection_stats()
        if stats["total_documents"] == 0:
            # Fallback a chat sin RAG si no hay documentos
            logger.warning("No hay documentos indexados. Usando chat directo.")
            response = await chat_service.chat_without_rag(
                question=request.question,
                session_id=request.session_id,
            )
            response.answer = (
                "⚠️ **No hay documentos indexados.** Respondo con conocimiento general.\n\n"
                + response.answer
                + "\n\n---\n*Sube documentos financieros para obtener respuestas más precisas y contextualizadas.*"
            )
            return response

        response = await chat_service.chat(
            question=request.question,
            session_id=request.session_id,
        )
        return response

    except Exception as e:
        logger.error(f"Error en /chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error procesando la solicitud: {str(e)}",
        )


@router.post(
    "/chat/direct",
    response_model=ChatResponse,
    summary="Chat directo (sin RAG)",
    description="Chat directo con el LLM sin consultar documentos.",
)
async def chat_direct(request: ChatRequest):
    """Chat directo con Groq sin búsqueda en documentos."""
    try:
        chat_service = get_chat_service()
        response = await chat_service.chat_without_rag(
            question=request.question,
            session_id=request.session_id,
        )
        return response
    except Exception as e:
        logger.error(f"Error en /chat/direct: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error procesando la solicitud: {str(e)}",
        )


# ============================================
# Document Endpoints
# ============================================

@router.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    summary="Subir documento",
    description="Sube un documento financiero (PDF, TXT, DOCX) para indexar en el sistema RAG.",
)
async def upload_document(file: UploadFile = File(...)):
    """
    Sube y procesa un documento para el sistema RAG.
    El documento se divide en chunks y se almacena en ChromaDB.
    """
    # Validar extensión
    allowed_extensions = {".pdf", ".txt", ".docx"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Formato no soportado: {file_ext}. "
                f"Formatos permitidos: {', '.join(allowed_extensions)}"
            ),
        )

    # Guardar archivo temporal y procesar
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_ext,
            prefix=f"{Path(file.filename).stem}_",
        ) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        doc_service = get_document_service()
        result = doc_service.ingest_document(tmp_path)

        return DocumentUploadResponse(
            filename=file.filename,
            chunks_created=result["chunks_created"],
            doc_type=DocumentType(result["doc_type"]),
            message=f"Documento '{file.filename}' procesado exitosamente. "
                    f"{result['chunks_created']} chunks indexados.",
        )

    except Exception as e:
        logger.error(f"Error subiendo documento: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error procesando el documento: {str(e)}",
        )
    finally:
        # Limpiar archivo temporal
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.post(
    "/documents/ingest-directory",
    summary="Indexar directorio",
    description="Indexa todos los documentos soportados de un directorio.",
)
async def ingest_directory(directory_path: str = "data/sample_docs"):
    """Indexa todos los documentos de un directorio."""
    try:
        doc_service = get_document_service()
        results = doc_service.ingest_directory(directory_path)

        total_chunks = sum(r.get("chunks_created", 0) for r in results)
        errors = [r for r in results if "error" in r]

        return {
            "processed_files": len(results),
            "total_chunks_created": total_chunks,
            "errors": len(errors),
            "details": results,
        }

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error indexando directorio: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error indexando directorio: {str(e)}",
        )


@router.get(
    "/documents/stats",
    response_model=CollectionStats,
    summary="Estadísticas de documentos",
)
async def get_document_stats():
    """Retorna estadísticas de la colección de documentos indexados."""
    try:
        doc_service = get_document_service()
        stats = doc_service.get_collection_stats()
        return CollectionStats(**stats)
    except Exception as e:
        logger.error(f"Error obteniendo stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.delete(
    "/documents",
    summary="Limpiar colección",
    description="Elimina todos los documentos indexados.",
)
async def clear_documents():
    """Limpia toda la colección de documentos."""
    try:
        doc_service = get_document_service()
        doc_service.clear_collection()
        return {"message": "Colección limpiada exitosamente"}
    except Exception as e:
        logger.error(f"Error limpiando colección: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# ============================================
# Session Endpoints
# ============================================

@router.delete(
    "/sessions/{session_id}",
    summary="Limpiar sesión de chat",
)
async def clear_session(session_id: str):
    """Elimina el historial de una sesión de conversación."""
    chat_service = get_chat_service()
    deleted = chat_service.clear_session(session_id)
    if deleted:
        return {"message": f"Sesión '{session_id}' eliminada"}
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Sesión '{session_id}' no encontrada",
    )


@router.get(
    "/sessions",
    summary="Listar sesiones activas",
)
async def list_sessions():
    """Lista todas las sesiones de chat activas."""
    chat_service = get_chat_service()
    return {"sessions": chat_service.list_sessions()}


# ============================================
# Health Check
# ============================================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Verifica el estado del servicio.",
)
async def health_check():
    """Health check del copiloto financiero."""
    settings = get_settings()
    try:
        doc_service = get_document_service()
        stats = doc_service.get_collection_stats()
        total_docs = stats["total_documents"]
    except Exception:
        total_docs = 0

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        environment=settings.app_env,
        groq_model=settings.groq_model,
        embedding_model=settings.embedding_model,
        total_documents=total_docs,
    )
