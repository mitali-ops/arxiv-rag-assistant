# streamlit_app.py - Web Interface for ArXiv RAG System
import streamlit as st
import sqlite3
import logging
import time
import re
from pathlib import Path
from typing import List, Dict
import json

# Configure Streamlit page
st.set_page_config(
    page_title="ArXiv Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
DATA_DIR = Path("data")
DATABASE_PATH = DATA_DIR / "arxiv.db"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .paper-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: #f9f9f9;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .query-examples {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .search-result {
        border-left: 4px solid #667eea;
        padding-left: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Database connection with caching
@st.cache_resource
def init_database():
    """Initialize database connection"""
    if not DATABASE_PATH.exists():
        st.error(f"Database not found at {DATABASE_PATH}")
        st.info("Please ensure your database file is in the correct location.")
        st.stop()
    
    conn = sqlite3.connect(str(DATABASE_PATH), check_same_thread=False)
    return conn

# Search functionality
class ArXivSearchWeb:
    def __init__(self, conn):
        self.conn = conn
        self.cursor = conn.cursor()
        self.fts_available = self._check_fts()
    
    def _check_fts(self) -> bool:
        try:
            self.cursor.execute("SELECT COUNT(*) FROM papers_fts LIMIT 1")
            return True
        except sqlite3.OperationalError:
            return False
    
    def _clean_fts_query(self, query: str) -> str:
        cleaned = re.sub(r'[^\w\s]', ' ', query)
        cleaned = ' '.join(cleaned.split())
        return cleaned
    
    def search_papers(self, query: str, limit: int = 10) -> List[Dict]:
        if not query.strip():
            return []
        
        clean_query = query.strip().lower()
        
        try:
            if self.fts_available:
                return self._fts_search(clean_query, limit)
            else:
                return self._like_search(clean_query, limit)
        except Exception:
            return self._like_search(clean_query, limit)
    
    def _fts_search(self, query: str, limit: int) -> List[Dict]:
        fts_query = self._clean_fts_query(query)
        
        if not fts_query.strip():
            return self._like_search(query, limit)
        
        try:
            self.cursor.execute('''
                SELECT p.id, p.arxiv_id, p.title, p.abstract, p.published,
                       bm25(papers_fts) as score
                FROM papers_fts 
                JOIN papers p ON papers_fts.rowid = p.id
                WHERE papers_fts MATCH ?
                ORDER BY score
                LIMIT ?
            ''', (fts_query, limit))
            
            results = []
            for row in self.cursor.fetchall():
                paper = {
                    'id': row[0],
                    'arxiv_id': row[1],
                    'title': row[2],
                    'abstract': row[3],
                    'published': row[4],
                    'score': row[5],
                    'authors': self._get_authors(row[0]),
                    'categories': self._get_categories(row[0])
                }
                results.append(paper)
            
            return results
        except Exception:
            return self._like_search(query, limit)
    
    def _like_search(self, query: str, limit: int) -> List[Dict]:
        self.cursor.execute('''
            SELECT id, arxiv_id, title, abstract, published
            FROM papers 
            WHERE title LIKE ? OR abstract LIKE ?
            ORDER BY published DESC
            LIMIT ?
        ''', (f'%{query}%', f'%{query}%', limit))
        
        results = []
        for row in self.cursor.fetchall():
            paper = {
                'id': row[0],
                'arxiv_id': row[1],
                'title': row[2],
                'abstract': row[3],
                'published': row[4],
                'score': 1.0,
                'authors': self._get_authors(row[0]),
                'categories': self._get_categories(row[0])
            }
            results.append(paper)
        
        return results
    
    def _get_authors(self, paper_id: int) -> List[str]:
        self.cursor.execute('''
            SELECT name FROM authors 
            WHERE paper_id = ? 
            ORDER BY position
            LIMIT 10
        ''', (paper_id,))
        return [row[0] for row in self.cursor.fetchall()]
    
    def _get_categories(self, paper_id: int) -> List[str]:
        self.cursor.execute('''
            SELECT DISTINCT category FROM categories 
            WHERE paper_id = ?
            LIMIT 5
        ''', (paper_id,))
        return [row[0] for row in self.cursor.fetchall()]
    
    def get_stats(self) -> Dict:
        stats = {}
        try:
            self.cursor.execute('SELECT COUNT(*) FROM papers')
            stats['total_papers'] = self.cursor.fetchone()[0]
            
            self.cursor.execute('SELECT COUNT(DISTINCT name) FROM authors')
            stats['unique_authors'] = self.cursor.fetchone()[0]
            
            self.cursor.execute('SELECT COUNT(DISTINCT category) FROM categories')
            stats['unique_categories'] = self.cursor.fetchone()[0]
            
            self.cursor.execute('''
                SELECT category, COUNT(*) as count 
                FROM categories 
                GROUP BY category 
                ORDER BY count DESC 
                LIMIT 10
            ''')
            stats['top_categories'] = self.cursor.fetchall()
        except Exception:
            pass
        
        return stats

# Answer generation
class WebAnswerGenerator:
    def generate_answer(self, query: str, papers: List[Dict]) -> str:
        if not papers:
            return self._no_results_response(query)
        
        query_lower = query.lower()
        query_type = self._classify_query(query_lower)
        
        intro = self._generate_introduction(query, query_type, len(papers))
        conclusion = self._generate_conclusion(query_type, len(papers))
        
        return {
            'intro': intro,
            'conclusion': conclusion,
            'query_type': query_type
        }
    
    def _classify_query(self, query: str) -> str:
        if any(word in query for word in ['what is', 'what are', 'define', 'definition']):
            return 'definition'
        elif any(word in query for word in ['how', 'method', 'approach', 'algorithm']):
            return 'methodology'
        elif any(word in query for word in ['application', 'used', 'use', 'apply']):
            return 'applications'
        elif any(word in query for word in ['recent', 'latest', 'new', 'advance']):
            return 'recent_advances'
        elif any(word in query for word in ['compare', 'difference', 'vs', 'versus']):
            return 'comparison'
        else:
            return 'general'
    
    def _generate_introduction(self, query: str, query_type: str, paper_count: int) -> str:
        introductions = {
            'definition': f"Found {paper_count} papers that define and explain **'{query}'**:",
            'methodology': f"Found {paper_count} papers explaining the methods and approaches for **'{query}'**:",
            'applications': f"Found {paper_count} papers showcasing applications of **'{query}'**:",
            'recent_advances': f"Found {paper_count} papers on recent advances in **'{query}'**:",
            'comparison': f"Found {paper_count} papers that help compare and understand **'{query}'**:",
            'general': f"Found {paper_count} relevant research papers about **'{query}'**:"
        }
        
        return introductions.get(query_type, introductions['general'])
    
    def _generate_conclusion(self, query_type: str, paper_count: int) -> str:
        conclusions = {
            'definition': f"These {paper_count} papers provide comprehensive definitions and theoretical foundations.",
            'methodology': f"These {paper_count} papers detail various approaches and implementation strategies.",
            'applications': f"These {paper_count} papers demonstrate diverse practical applications and use cases.",
            'recent_advances': f"These {paper_count} papers represent cutting-edge research and recent developments.",
            'comparison': f"These {paper_count} papers offer different perspectives and comparative insights.",
            'general': f"These {paper_count} papers provide comprehensive coverage from multiple research perspectives."
        }
        
        return conclusions.get(query_type, conclusions['general'])
    
    def _no_results_response(self, query: str) -> Dict:
        return {
            'intro': f"No papers found matching **'{query}'**",
            'conclusion': "Try different keywords, check spelling, or use more general terms.",
            'query_type': 'no_results'
        }

# Initialize components
@st.cache_resource
def init_components():
    conn = init_database()
    search_engine = ArXivSearchWeb(conn)
    answer_generator = WebAnswerGenerator()
    return search_engine, answer_generator

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔬 ArXiv Research Assistant</h1>
        <p>AI-powered search through 2.8M+ research papers</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize components
    search_engine, answer_generator = init_components()
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Search Settings")
        
        # Search parameters
        max_results = st.slider("Maximum results", 1, 20, 10)
        
        # Database statistics
        st.header("📊 Database Stats")
        
        with st.spinner("Loading statistics..."):
            stats = search_engine.get_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Papers", f"{stats.get('total_papers', 0):,}")
            st.metric("Authors", f"{stats.get('unique_authors', 0):,}")
        with col2:
            st.metric("Categories", f"{stats.get('unique_categories', 0):,}")
            
        # Top categories
        st.subheader("🏷️ Top Categories")
        for category, count in stats.get('top_categories', [])[:5]:
            st.write(f"**{category}**: {count:,}")
        
        # Example queries
        st.header("💡 Example Queries")
        examples = [
            "transformer models",
            "deep learning applications", 
            "neural network architectures",
            "computer vision techniques",
            "natural language processing",
            "reinforcement learning"
        ]
        
        selected_example = st.selectbox(
            "Try an example:",
            [""] + examples,
            index=0
        )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 Search Research Papers")
        
        # Search input
        query = st.text_input(
            "Enter your research question:",
            value=selected_example if selected_example else "",
            placeholder="e.g., What are transformer models in machine learning?",
            help="Use specific technical terms for better results"
        )
        
        # Search button
        if st.button("🔍 Search", type="primary", use_container_width=True):
            if query.strip():
                with st.spinner("Searching through 2.8M papers..."):
                    start_time = time.time()
                    papers = search_engine.search_papers(query, max_results)
                    search_time = time.time() - start_time
                    
                    # Generate answer context
                    answer_context = answer_generator.generate_answer(query, papers)
                
                if papers:
                    # Display results summary
                    st.success(f"Found {len(papers)} papers in {search_time:.2f} seconds")
                    
                    # Answer introduction
                    st.markdown(f"### 💡 {answer_context['intro']}")
                    
                    # Display papers
                    st.markdown("### 📚 Research Papers")
                    
                    for i, paper in enumerate(papers, 1):
                        with st.container():
                            st.markdown(f"""
                            <div class="paper-card">
                                <h4>{i}. {paper['title']}</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Paper details in columns
                            col_a, col_b = st.columns([2, 1])
                            
                            with col_a:
                                # Authors
                                authors = paper.get('authors', [])
                                if authors:
                                    author_str = ', '.join(authors[:4])
                                    if len(authors) > 4:
                                        author_str += f' and {len(authors) - 4} others'
                                    st.write(f"**Authors:** {author_str}")
                                
                                # Abstract preview
                                abstract = paper.get('abstract', '')
                                if abstract:
                                    preview = abstract[:300] + "..." if len(abstract) > 300 else abstract
                                    st.write(f"**Abstract:** {preview}")
                            
                            with col_b:
                                # Metadata
                                if paper.get('arxiv_id'):
                                    st.write(f"**ArXiv ID:** {paper['arxiv_id']}")
                                
                                if paper.get('published'):
                                    year = paper['published'][:4]
                                    st.write(f"**Published:** {year}")
                                
                                # Categories
                                categories = paper.get('categories', [])
                                if categories:
                                    st.write(f"**Categories:** {', '.join(categories[:3])}")
                                
                                # Relevance score
                                if 'score' in paper and paper['score'] != 1.0:
                                    st.metric("Relevance", f"{abs(paper['score']):.2f}")
                            
                            # Expandable full abstract
                            if abstract and len(abstract) > 300:
                                with st.expander("Read full abstract"):
                                    st.write(abstract)
                            
                            st.divider()
                    
                    # Conclusion
                    st.markdown(f"### 📋 Summary")
                    st.info(answer_context['conclusion'])
                    
                else:
                    st.warning("No papers found matching your query.")
                    st.markdown("""
                    **Suggestions:**
                    - Try using more general terms
                    - Check spelling and terminology
                    - Use broader categories or synonyms
                    - Try different phrasings of your question
                    """)
            else:
                st.warning("Please enter a search query.")
    
    with col2:
        st.header("📖 How to Use")
        
        st.markdown("""
        **Query Types:**
        - **Definitions**: "What are neural networks?"
        - **Methods**: "How do transformers work?"
        - **Applications**: "Uses of deep learning"
        - **Recent work**: "Latest advances in NLP"
        - **Comparisons**: "CNN vs RNN"
        
        **Search Tips:**
        - Use specific technical terms
        - Ask complete questions
        - Include domain context
        - Try different phrasings
        
        **Database:**
        - 2.8M+ ArXiv papers
        - Full-text search capability
        - Author and category filtering
        - Publication date sorting
        """)
        
        # Search history (session state)
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        
        if st.session_state.search_history:
            st.subheader("🕒 Recent Searches")
            for recent_query in st.session_state.search_history[-5:]:
                if st.button(f"🔄 {recent_query}", key=f"history_{recent_query}"):
                    st.experimental_set_query_params(q=recent_query)
                    st.experimental_rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit • Database: 2.8M+ ArXiv papers • 
        <a href='https://github.com/yourusername/arxiv-rag'>View Source</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()