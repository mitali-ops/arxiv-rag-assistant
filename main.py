# main.py - Fixed ArXiv RAG System (Windows Compatible)
import sqlite3
import logging
import sys
import time
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings
import re

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Try to import transformers
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available. Install with: pip install transformers torch")

# Configuration
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_PATH = DATA_DIR / "arxiv.db"

# Set up logging (Windows compatible)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('arxiv_rag.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ArXivSearch:
    """Enhanced search engine for ArXiv papers"""
    
    def __init__(self, db_path: Path = DATABASE_PATH):
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.fts_available = self._check_fts()
        
        logger.info(f"Search engine initialized. FTS available: {self.fts_available}")
    
    def _check_fts(self) -> bool:
        """Check if FTS is available"""
        try:
            self.cursor.execute("SELECT COUNT(*) FROM papers_fts LIMIT 1")
            return True
        except sqlite3.OperationalError:
            return False
    
    def _clean_fts_query(self, query: str) -> str:
        """Clean query for FTS5 compatibility"""
        # Remove special characters that cause FTS issues
        cleaned = re.sub(r'[^\w\s]', ' ', query)
        # Remove extra spaces
        cleaned = ' '.join(cleaned.split())
        return cleaned
    
    def search_papers(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for papers using the best available method"""
        if not query.strip():
            return []
        
        clean_query = query.strip().lower()
        
        try:
            if self.fts_available:
                return self._fts_search(clean_query, limit)
            else:
                return self._like_search(clean_query, limit)
        except Exception as e:
            logger.error(f"Search error: {e}")
            # Always fallback to LIKE search if FTS fails
            return self._like_search(clean_query, limit)
    
    def _fts_search(self, query: str, limit: int) -> List[Dict]:
        """Full-text search using FTS5"""
        try:
            # Clean query for FTS
            fts_query = self._clean_fts_query(query)
            
            if not fts_query.strip():
                return self._like_search(query, limit)
            
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
            
        except Exception as e:
            logger.warning(f"FTS search failed, using LIKE search: {e}")
            return self._like_search(query, limit)
    
    def _like_search(self, query: str, limit: int) -> List[Dict]:
        """Fallback LIKE-based search"""
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
                'score': 1.0,  # Default score
                'authors': self._get_authors(row[0]),
                'categories': self._get_categories(row[0])
            }
            results.append(paper)
        
        return results
    
    def _get_authors(self, paper_id: int) -> List[str]:
        """Get authors for a paper"""
        self.cursor.execute('''
            SELECT name FROM authors 
            WHERE paper_id = ? 
            ORDER BY position
            LIMIT 10
        ''', (paper_id,))
        return [row[0] for row in self.cursor.fetchall()]
    
    def _get_categories(self, paper_id: int) -> List[str]:
        """Get categories for a paper"""
        self.cursor.execute('''
            SELECT DISTINCT category FROM categories 
            WHERE paper_id = ?
            LIMIT 5
        ''', (paper_id,))
        return [row[0] for row in self.cursor.fetchall()]
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        stats = {}
        try:
            self.cursor.execute('SELECT COUNT(*) FROM papers')
            stats['total_papers'] = self.cursor.fetchone()[0]
            
            self.cursor.execute('SELECT COUNT(DISTINCT name) FROM authors')
            stats['unique_authors'] = self.cursor.fetchone()[0]
            
            self.cursor.execute('SELECT COUNT(DISTINCT category) FROM categories')
            stats['unique_categories'] = self.cursor.fetchone()[0]
            
            # Top categories
            self.cursor.execute('''
                SELECT category, COUNT(*) as count 
                FROM categories 
                GROUP BY category 
                ORDER BY count DESC 
                LIMIT 5
            ''')
            stats['top_categories'] = self.cursor.fetchall()
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
        
        return stats
    
    def close(self):
        """Close database connection"""
        self.conn.close()

class SimpleRAGModel:
    """Simple RAG model that can work with or without transformers"""
    
    def __init__(self):
        self.device = "cpu"  # Default to CPU for reliability
        self.model = None
        self.tokenizer = None
        
        if TRANSFORMERS_AVAILABLE:
            self._load_model()
        else:
            logger.info("Transformers not available. Using template-based responses.")
    
    def _load_model(self):
        """Load a small, reliable model"""
        try:
            model_name = "gpt2"  # Start with most reliable model
            logger.info(f"Loading model: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Model loading failed: {e}")
            self.model = None
            self.tokenizer = None
    
    def generate_answer(self, query: str, papers: List[Dict]) -> str:
        """Generate answer using papers as context"""
        if not papers:
            return "I couldn't find any relevant papers to answer your question. Please try rephrasing or using different keywords."
        
        # Create context from papers
        context = self._create_context(papers)
        
        if self.model and self.tokenizer:
            return self._generate_with_model(query, context)
        else:
            return self._generate_template_response(query, papers)
    
    def _create_context(self, papers: List[Dict]) -> str:
        """Create context string from papers"""
        context_parts = ["Based on recent research papers:\n"]
        
        for i, paper in enumerate(papers[:3], 1):  # Use top 3 papers
            title = paper.get('title', 'Unknown Title')
            abstract = paper.get('abstract', '')
            
            # Truncate long abstracts
            if len(abstract) > 300:
                abstract = abstract[:300] + "..."
            
            authors = paper.get('authors', [])
            author_str = ', '.join(authors[:3])
            if len(authors) > 3:
                author_str += ' et al.'
            
            context_parts.append(f"\nPaper {i}: {title}")
            if author_str:
                context_parts.append(f"Authors: {author_str}")
            context_parts.append(f"Abstract: {abstract}\n")
        
        return ''.join(context_parts)
    
    def _generate_with_model(self, query: str, context: str) -> str:
        """Generate answer using language model"""
        try:
            prompt = f"{context}\nQuestion: {query}\nAnswer based on the papers above:"
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = response[len(prompt):].strip()
            
            if len(answer) < 20:
                return self._generate_template_response(query, [])
            
            return answer
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return self._generate_template_response(query, [])
    
    def _generate_template_response(self, query: str, papers: List[Dict]) -> str:
        """Generate template-based response when model is unavailable"""
        if not papers:
            return "I couldn't find relevant papers for your question. Please try different keywords or be more specific."
        
        response_parts = [
            f"Based on {len(papers)} relevant research papers, here's what I found:\n"
        ]
        
        for i, paper in enumerate(papers[:3], 1):
            title = paper.get('title', 'Unknown Title')
            authors = paper.get('authors', [])
            categories = paper.get('categories', [])
            
            author_str = ', '.join(authors[:2])
            if len(authors) > 2:
                author_str += ' et al.'
            
            response_parts.append(f"{i}. **{title}**")
            if author_str:
                response_parts.append(f"   Authors: {author_str}")
            if categories:
                response_parts.append(f"   Categories: {', '.join(categories[:2])}")
            
            # Add a brief summary from abstract
            abstract = paper.get('abstract', '')
            if abstract:
                summary = abstract[:200] + "..." if len(abstract) > 200 else abstract
                response_parts.append(f"   Summary: {summary}")
            
            response_parts.append("")  # Empty line
        
        response_parts.append("For detailed information, please refer to the complete papers above.")
        
        return '\n'.join(response_parts)

class ArXivRAGSystem:
    """Complete RAG system for ArXiv research"""
    
    def __init__(self):
        logger.info("Initializing ArXiv RAG System...")
        
        try:
            self.search_engine = ArXivSearch()
            self.rag_model = SimpleRAGModel()
            
            # Get system stats
            stats = self.search_engine.get_stats()
            logger.info(f"System initialized with {stats.get('total_papers', 0):,} papers")
            
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise
    
    def ask_question(self, query: str) -> Dict:
        """Main method to ask questions"""
        logger.info(f"Processing query: '{query}'")
        start_time = time.time()
        
        try:
            # Search for relevant papers
            papers = self.search_engine.search_papers(query, limit=5)
            
            if not papers:
                return {
                    "query": query,
                    "papers_found": 0,
                    "answer": "No relevant papers found. Try different keywords or check spelling.",
                    "papers": [],
                    "processing_time": time.time() - start_time
                }
            
            logger.info(f"Found {len(papers)} papers")
            
            # Generate answer
            answer = self.rag_model.generate_answer(query, papers)
            
            processing_time = time.time() - start_time
            logger.info(f"Answer generated in {processing_time:.2f}s")
            
            return {
                "query": query,
                "papers_found": len(papers),
                "answer": answer,
                "papers": papers,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "query": query,
                "papers_found": 0,
                "answer": f"An error occurred: {str(e)}",
                "papers": [],
                "processing_time": time.time() - start_time
            }
    
    def interactive_session(self):
        """Start interactive Q&A session"""
        print("\n" + "="*60)
        print("ArXiv Research Assistant")
        print("Ask questions about research papers!")
        print("\nCommands:")
        print("  'quit' or 'exit' - Exit")
        print("  'help' - Show help")
        print("  'stats' - Database statistics")
        print("="*60)
        
        while True:
            try:
                query = input("\nYour question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Thank you for using ArXiv Research Assistant!")
                    break
                
                if query.lower() == 'help':
                    self._show_help()
                    continue
                
                if query.lower() == 'stats':
                    self._show_stats()
                    continue
                
                if not query:
                    print("Please enter a question.")
                    continue
                
                # Process question
                print("\nSearching...")
                result = self.ask_question(query)
                
                # Display results
                self._display_results(result)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _show_help(self):
        """Show help information"""
        help_text = """
ArXiv Research Assistant Help

EXAMPLE QUESTIONS:
• "What are transformer models in machine learning?"
• "Recent advances in computer vision"
• "How do neural networks work?"
• "Applications of deep learning in NLP"
• "Graph neural networks for recommendation systems"

TIPS:
• Use specific technical terms for better results
• Ask about concepts, methods, applications, or recent advances
• Include domain keywords (deep learning, NLP, computer vision, etc.)
• Questions work better than single keywords

FEATURES:
• Searches through 2.8M+ research papers
• Provides comprehensive answers based on real papers
• Shows source papers for further reading
        """
        print(help_text)
    
    def _show_stats(self):
        """Show system statistics"""
        try:
            stats = self.search_engine.get_stats()
            
            print(f"\nDatabase Statistics:")
            print(f"  Total Papers: {stats.get('total_papers', 0):,}")
            print(f"  Unique Authors: {stats.get('unique_authors', 0):,}")
            print(f"  Categories: {stats.get('unique_categories', 0)}")
            
            print(f"\nTop Categories:")
            for category, count in stats.get('top_categories', [])[:5]:
                print(f"  • {category}: {count:,} papers")
                
        except Exception as e:
            print(f"Error getting statistics: {e}")
    
    def _display_results(self, result: Dict):
        """Display formatted results"""
        print(f"\nAnswer:")
        print("-" * 50)
        print(result['answer'])
        
        if result['papers_found'] > 0:
            print(f"\nBased on {result['papers_found']} papers (in {result['processing_time']:.2f}s):")
            print("-" * 50)
            
            for i, paper in enumerate(result['papers'][:3], 1):
                title = paper.get('title', 'Unknown')
                if len(title) > 80:
                    title = title[:77] + "..."
                
                print(f"{i}. {title}")
                
                authors = paper.get('authors', [])
                if authors:
                    author_str = ', '.join(authors[:3])
                    if len(authors) > 3:
                        author_str += ' et al.'
                    print(f"   Authors: {author_str}")
                
                categories = paper.get('categories', [])
                if categories:
                    print(f"   Categories: {', '.join(categories[:3])}")
                
                print()
    
    def close(self):
        """Clean up resources"""
        self.search_engine.close()

def main():
    """Main function"""
    if not DATABASE_PATH.exists():
        print(f"Database not found: {DATABASE_PATH}")
        print("Please run database_recovery.py first!")
        return
    
    try:
        rag_system = ArXivRAGSystem()
        
        # Check for command line arguments
        if len(sys.argv) > 1:
            # Single query mode
            query = " ".join(sys.argv[1:])
            result = rag_system.ask_question(query)
            
            print(f"Query: {query}")
            print(f"Answer: {result['answer']}")
            print(f"\nFound {result['papers_found']} papers in {result['processing_time']:.2f}s")
        else:
            # Interactive mode
            rag_system.interactive_session()
    
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Fatal error: {e}")
        logger.error(f"Fatal error: {e}")
    finally:
        try:
            rag_system.close()
        except:
            pass

if __name__ == "__main__":
    main()