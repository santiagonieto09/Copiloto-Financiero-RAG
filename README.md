# 🏦 Copiloto Financiero RAG

> Asistente de IA especializado en servicios financieros y modernización bancaria, construido con RAG (Retrieval-Augmented Generation).

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-orange.svg)](https://langchain.com)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-purple.svg)](https://groq.com)

---

## 📋 Descripción

**FinBot** es un copiloto de IA que asiste a analistas, auditores y profesionales del sector financiero. Responde preguntas basándose en documentos financieros indexados mediante RAG, proporcionando respuestas precisas con citas de fuentes.

### Casos de uso:
- Consultas sobre regulaciones bancarias (Basilea III, AML/KYC)
- Análisis de políticas y procedimientos financieros
- Asistencia en procesos de auditoría y compliance
- Automatización de consultas sobre productos crediticios

---

## 🏗️ Arquitectura

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Cliente    │────▶│  FastAPI      │────▶│  LangChain  │
│  (API/Web)   │◀────│  REST API    │◀────│  RAG Chain  │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                 │
                                    ┌────────────┼────────────┐
                                    ▼            ▼            ▼
                              ┌──────────┐ ┌──────────┐ ┌──────────┐
                              │ ChromaDB │ │  Groq    │ │HuggingFace│
                              │ (Vector  │ │  LLM     │ │Embeddings │
                              │  Store)  │ │(Llama3.3)│ │(MiniLM)   │
                              └──────────┘ └──────────┘ └──────────┘
```

---

## 🛠️ Stack Tecnológico

| Componente | Tecnología | Propósito |
|---|---|---|
| **API** | FastAPI | REST API con documentación automática |
| **LLM** | Groq (Llama 3.3 70B) | Generación de respuestas |
| **Orquestación** | LangChain | Pipeline RAG conversacional |
| **Vector Store** | ChromaDB | Almacenamiento de embeddings |
| **Embeddings** | HuggingFace (all-MiniLM-L6-v2) | Vectorización de documentos |
| **Validación** | Pydantic | Esquemas de datos tipados |

---

## 🚀 Instalación

### Requisitos
- Python 3.10+
- API Key de [Groq](https://console.groq.com/keys) (gratuita)

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/copiloto-financiero-rag.git
cd copiloto-financiero-rag

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env
# Editar .env y agregar tu GROQ_API_KEY

# 5. Ejecutar el servidor
python main.py
```

El servidor iniciará en `http://localhost:8000`. La documentación interactiva estará en `http://localhost:8000/docs`.

---

## 📡 Endpoints de la API

### Chat
| Método | Endpoint | Descripción |
|---|---|---|
| `POST` | `/api/v1/chat` | Chat con RAG (usa documentos indexados) |
| `POST` | `/api/v1/chat/direct` | Chat directo con el LLM |

### Documentos
| Método | Endpoint | Descripción |
|---|---|---|
| `POST` | `/api/v1/documents/upload` | Subir documento (PDF/TXT/DOCX) |
| `POST` | `/api/v1/documents/ingest-directory` | Indexar directorio completo |
| `GET` | `/api/v1/documents/stats` | Estadísticas de la colección |
| `DELETE` | `/api/v1/documents` | Limpiar colección |

### Sesiones
| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/api/v1/sessions` | Listar sesiones activas |
| `DELETE` | `/api/v1/sessions/{id}` | Limpiar sesión de chat |

### Sistema
| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/api/v1/health` | Health check |
| `GET` | `/docs` | Documentación Swagger UI |

---

## 💬 Ejemplo de Uso

### 1. Indexar documentos de ejemplo
```bash
curl -X POST "http://localhost:8000/api/v1/documents/ingest-directory" \
  -H "Content-Type: application/json"
```

### 2. Hacer una pregunta
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "¿Cuáles son los requisitos de capital según Basilea III?",
    "session_id": "mi-sesion-1"
  }'
```

### 3. Subir un documento propio
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@mi_documento.pdf"
```

---

## 📁 Estructura del Proyecto

```
copiloto-financiero-rag/
├── main.py                     # Punto de entrada
├── requirements.txt            # Dependencias
├── .env.example                # Template de variables de entorno
├── .gitignore
├── README.md
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py           # Endpoints REST
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py           # Configuración centralizada
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py          # Esquemas Pydantic
│   └── services/
│       ├── __init__.py
│       ├── chat_service.py     # Servicio de chat con RAG
│       └── document_service.py # Ingestión de documentos
├── data/
│   └── sample_docs/            # Documentos financieros de ejemplo
│       ├── regulaciones_bancarias_basilea.txt
│       ├── politica_antilavado_aml.txt
│       └── procedimiento_credito_hipotecario.txt
└── tests/
    └── __init__.py
```

---

## 🔑 Competencias Demostradas

Este proyecto demuestra experiencia en:

- **LLMs y Prompt Engineering**: System prompts especializados, manejo de contexto conversacional.
- **RAG (Retrieval-Augmented Generation)**: Pipeline completo de ingestión, vectorización y recuperación.
- **Bases de datos vectoriales**: ChromaDB con persistencia y búsqueda por similitud.
- **APIs REST**: FastAPI con documentación automática, validación Pydantic, manejo de errores.
- **Python avanzado**: Patrones de diseño (Singleton, Factory), async/await, type hints.
- **Arquitectura de software**: Separación de responsabilidades, inyección de dependencias, configuración centralizada.
