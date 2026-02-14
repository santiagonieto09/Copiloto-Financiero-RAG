"""
Copiloto Financiero RAG - Punto de entrada principal.

Un asistente de IA especializado en servicios financieros y modernización bancaria,
construido con FastAPI + LangChain + Groq + ChromaDB.

Autor: [Tu Nombre]
Licencia: MIT
"""

import logging
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings

# ============================================
# Logging
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================
# Lifespan (startup/shutdown)
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Eventos de inicio y cierre de la aplicación."""
    settings = get_settings()
    logger.info("=" * 60)
    logger.info("🚀 Copiloto Financiero RAG - Iniciando...")
    logger.info(f"   Entorno:    {settings.app_env}")
    logger.info(f"   LLM:        {settings.groq_model}")
    logger.info(f"   Embeddings: {settings.embedding_model}")
    logger.info(f"   ChromaDB:   {settings.chroma_persist_dir}")
    logger.info("=" * 60)
    yield
    logger.info("Copiloto Financiero RAG - Cerrando...")


# ============================================
# App FastAPI
# ============================================

app = FastAPI(
    title="Copiloto Financiero RAG",
    description=(
        "API de un asistente de IA especializado en servicios financieros y "
        "modernización bancaria. Utiliza RAG (Retrieval-Augmented Generation) "
        "para responder preguntas basándose en documentos financieros indexados.\n\n"
        "**Stack tecnológico:**\n"
        "- 🧠 LLM: Groq (Llama 3.3 70B)\n"
        "- 📚 RAG: LangChain + ChromaDB\n"
        "- 🔤 Embeddings: HuggingFace (all-MiniLM-L6-v2)\n"
        "- ⚡ Framework: FastAPI\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "Copiloto Financiero",
            "description": "Endpoints del copiloto financiero con RAG.",
        }
    ],
)

# ============================================
# CORS Middleware
# ============================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, restringir a dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Registrar rutas
# ============================================

app.include_router(router)


# ============================================
# Root endpoint
# ============================================

@app.get("/", tags=["Root"])
async def root():
    """Endpoint raíz con información del servicio."""
    return {
        "service": "Copiloto Financiero RAG",
        "version": "1.0.0",
        "description": "Asistente de IA para servicios financieros con RAG",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


# ============================================
# Entry Point
# ============================================

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "development",
        log_level="info",
    )
