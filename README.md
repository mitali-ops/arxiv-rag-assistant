# ArXiv Research Assistant



A production-scale RAG (Retrieval-Augmented Generation) system that enables intelligent search and question-answering across 2.8+ million ArXiv research papers.



## System Overview



- **Database Scale**: 2,806,229 research papers from ArXiv

- **Search Performance**: Sub-second to 2-second response times

- **Full-Text Search**: Advanced FTS5 indexing with relevance scoring

- **Multiple Interfaces**: Command-line, interactive, and web-based



## Quick Start



1. Install dependencies: `pip install streamlit`

2. Run web interface: `streamlit run streamlit\_app.py`

3. Run CLI: `python simple\_rag.py "your question here"`



## Features



- **Smart Search**: Full-text search through 2.8M papers

- **Template Responses**: Context-aware answer generation

- **Web Interface**: Professional Streamlit application

- **Production Ready**: Robust error handling and logging



## Performance Metrics



| Metric | Value |

|--------|-------|

| Total Papers | 2,806,229 |

| Unique Authors | 2,205,414 |

| Categories | 176 |

| Database Size | 6.3 GB |

| Average Query Time | 0.5-2.0 seconds |



## Architecture



- `preprocessing.py`: Data cleaning and validation

- `database.py`: SQLite database with FTS5 indexing  

- `simple\_rag.py`: Core RAG system

- `streamlit\_app.py`: Web interface



Built with Python, SQLite, and Streamlit.

