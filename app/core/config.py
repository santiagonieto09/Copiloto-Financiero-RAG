"""
Configuración centralizada del Copiloto Financiero RAG.
Usa pydantic-settings para validar variables de entorno.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """Configuración de la aplicación cargada desde variables de entorno."""

    # --- Groq LLM ---
    groq_api_key: str = Field(..., description="API Key de Groq")
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Modelo LLM a usar en Groq",
    )

    # --- Embeddings Provider ---
    # Opciones: 'huggingface' (local, usa RAM),
    # 'google' (API, GRATIS en ai.google.dev),
    # 'mistral' (API, tier gratuito en console.mistral.ai)
    embedding_provider: str = Field(
        default="huggingface",
        description="Proveedor de embeddings: huggingface, google, mistral",
    )

    # --- API Keys por provider ---
    google_api_key: str | None = Field(
        default=None,
        description="API Key de Google Gemini (requerido si embedding_provider='google') - Gratis en ai.google.dev",
    )
    mistral_api_key: str | None = Field(
        default=None,
        description="API Key de Mistral AI (requerido si embedding_provider='mistral') - Gratis en console.mistral.ai",
    )

    # --- Embeddings ---
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Modelo de embeddings de HuggingFace (solo si provider='huggingface')",
    )
    google_embedding_model: str = Field(
        default="models/embedding-001",
        description="Modelo de embeddings de Google (solo si provider='google')",
    )
    mistral_embedding_model: str = Field(
        default="mistral-embed",
        description="Modelo de embeddings de Mistral (solo si provider='mistral')",
    )

    # --- ChromaDB ---
    chroma_persist_dir: str = Field(
        default="./data/chroma_db",
        description="Directorio de persistencia de ChromaDB",
    )
    chroma_collection_name: str = Field(
        default="financial_docs",
        description="Nombre de la colección en ChromaDB",
    )

    # --- Servidor ---
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)
    app_env: str = Field(default="development")

    # --- RAG ---
    chunk_size: int = Field(
        default=1000,
        description="Tamaño de chunks para dividir documentos",
    )
    chunk_overlap: int = Field(
        default=200,
        description="Solapamiento entre chunks",
    )
    max_retrieved_docs: int = Field(
        default=5,
        description="Número máximo de documentos a recuperar del vector store",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache()
def get_settings() -> Settings:
    """Singleton de configuración con caché."""
    return Settings()
