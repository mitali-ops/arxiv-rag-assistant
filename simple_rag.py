# simple_rag.py - Template-Based RAG System (No LLM Generation)
import sqlite3
import logging
import sys
import time
import re
from pathlib import Path
from typing import List, Dict, Optional

# Configuration
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_PATH = DATA_DIR / "arxiv.db"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ArXivSearch:
    """Search engine for ArXiv papers"""
    
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
        cleaned = re.sub(r'[^\w\s]', ' ', query)
        cleaned = ' '.join(cleaned.split())
        return cleaned
    
    def search_papers(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for papers"""
        if not query.strip():
            return []
        
        clean_query = query.strip().lower()
        
        try:
            if self.fts_available:
                return self._fts_search(clean_query, limit)
            else:
                return self._like_search(clean_query, limit)
        except Exception as e:
            logger.warning(f"FTS search failed, using LIKE search: {e}")
            return self._like_search(clean_query, limit)
    
    def _fts_search(self, query: str, limit: int) -> List[Dict]:
        """Full-text search using FTS5"""
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
                'score': 1.0,
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
            LIMIT 5
        ''', (paper_id,))
        return [row[0] for row in self.cursor.fetchall()]
    
    def _get_categories(self, paper_id: int) -> List[str]:
        """Get categories for a paper"""
        self.cursor.execute('''
            SELECT DISTINCT category FROM categories 
            WHERE paper_id = ?
            LIMIT 3
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

class TemplateAnswerGenerator:
    """Generate structured answers using templates"""
    
    def generate_answer(self, query: str, papers: List[Dict]) -> str:
        """Generate answer using paper templates"""
        if not papers:
            return self._no_results_response(query)
        
        # Determine query type for better responses
        query_lower = query.lower()
        query_type = self._classify_query(query_lower)
        
        # Generate structured response
        response_parts = []
        
        # Add contextual introduction
        intro = self._generate_introduction(query, query_type, len(papers))
        response_parts.append(intro)
        
        # Add paper summaries
        for i, paper in enumerate(papers, 1):
            paper_summary = self._format_paper_summary(i, paper, query_type)
            response_parts.append(paper_summary)
        
        # Add conclusion
        conclusion = self._generate_conclusion(query_type, len(papers))
        response_parts.append(conclusion)
        
        return '\n'.join(response_parts)
    
    def _classify_query(self, query: str) -> str:
        """Classify the type of query to generate appropriate responses"""
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
        """Generate contextual introduction"""
        introductions = {
            'definition': f"Based on {paper_count} research papers, here's what the literature tells us about '{query}':",
            'methodology': f"I found {paper_count} papers that explain the methods and approaches for '{query}':",
            'applications': f"Here are {paper_count} papers showing applications and uses of '{query}':",
            'recent_advances': f"Recent research shows {paper_count} papers on advances in '{query}':",
            'comparison': f"Found {paper_count} papers that help understand '{query}':",
            'general': f"Based on {paper_count} relevant research papers about '{query}':"
        }
        
        return introductions.get(query_type, introductions['general']) + "\n"
    
    def _format_paper_summary(self, index: int, paper: Dict, query_type: str) -> str:
        """Format individual paper summary"""
        title = paper.get('title', 'Unknown Title')
        authors = paper.get('authors', [])
        categories = paper.get('categories', [])
        abstract = paper.get('abstract', '')
        arxiv_id = paper.get('arxiv_id', '')
        published = paper.get('published', '')
        
        # Format authors
        if authors:
            if len(authors) <= 3:
                author_str = ', '.join(authors)
            else:
                author_str = ', '.join(authors[:3]) + ' et al.'
        else:
            author_str = 'Unknown authors'
        
        # Format publication info
        pub_info = []
        if published and len(published) >= 4:
            year = published[:4]
            pub_info.append(f"Published: {year}")
        if arxiv_id:
            pub_info.append(f"ArXiv: {arxiv_id}")
        
        pub_str = ' | '.join(pub_info) if pub_info else ''
        
        # Extract key insight from abstract
        key_insight = self._extract_key_insight(abstract, query_type)
        
        # Build summary
        summary_parts = [f"\n{index}. **{title}**"]
        summary_parts.append(f"   Authors: {author_str}")
        
        if categories:
            summary_parts.append(f"   Categories: {', '.join(categories)}")
        
        if pub_str:
            summary_parts.append(f"   {pub_str}")
        
        if key_insight:
            summary_parts.append(f"   Key contribution: {key_insight}")
        
        return '\n'.join(summary_parts)
    
    def _extract_key_insight(self, abstract: str, query_type: str) -> str:
        """Extract relevant insight from abstract based on query type"""
        if not abstract:
            return ""
        
        # Split into sentences
        sentences = [s.strip() for s in abstract.split('.') if s.strip()]
        
        if not sentences:
            return abstract[:200] + "..." if len(abstract) > 200 else abstract
        
        # For definition queries, prefer first sentence
        if query_type == 'definition':
            return sentences[0] + "." if sentences else ""
        
        # For methodology, look for method-related sentences
        if query_type == 'methodology':
            method_keywords = ['method', 'approach', 'algorithm', 'technique', 'propose', 'develop']
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in method_keywords):
                    return sentence + "."
        
        # For applications, look for application-related content
        if query_type == 'applications':
            app_keywords = ['application', 'applied', 'use', 'used', 'demonstrate', 'show']
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in app_keywords):
                    return sentence + "."
        
        # Default: return first sentence or truncated abstract
        first_sentence = sentences[0] + "." if sentences else abstract[:200]
        return first_sentence if len(first_sentence) <= 200 else first_sentence[:200] + "..."
    
    def _generate_conclusion(self, query_type: str, paper_count: int) -> str:
        """Generate appropriate conclusion"""
        conclusions = {
            'definition': f"\nThese {paper_count} papers provide comprehensive definitions and conceptual frameworks for understanding the topic.",
            'methodology': f"\nThese {paper_count} papers detail various methodological approaches and implementation strategies.",
            'applications': f"\nThese {paper_count} papers showcase diverse applications and practical implementations.",
            'recent_advances': f"\nThese {paper_count} papers represent recent developments and cutting-edge research in the field.",
            'comparison': f"\nThese {paper_count} papers provide different perspectives and comparative insights.",
            'general': f"\nThese {paper_count} papers offer comprehensive coverage of the topic from multiple perspectives."
        }
        
        base_conclusion = conclusions.get(query_type, conclusions['general'])
        base_conclusion += "\n\nFor detailed technical information and complete methodologies, please refer to the full papers."
        
        return base_conclusion
    
    def _no_results_response(self, query: str) -> str:
        """Generate response when no papers are found"""
        suggestions = [
            "Try using more general terms (e.g., 'machine learning' instead of specific model names)",
            "Check spelling and use standard academic terminology", 
            "Use broader categories (e.g., 'neural networks' instead of specific architectures)",
            "Try searching with different synonyms or related terms"
        ]
        
        response = f"I couldn't find papers specifically matching '{query}' in the database.\n\n"
        response += "Suggestions to improve your search:\n"
        for i, suggestion in enumerate(suggestions, 1):
            response += f"{i}. {suggestion}\n"
        
        return response

class SimpleRAGSystem:
    """Simple template-based RAG system"""
    
    def __init__(self):
        logger.info("Initializing Simple RAG System...")
        
        try:
            self.search_engine = ArXivSearch()
            self.answer_generator = TemplateAnswerGenerator()
            
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
            
            logger.info(f"Found {len(papers)} papers")
            
            # Generate answer using templates
            answer = self.answer_generator.generate_answer(query, papers)
            
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
                "answer": f"An error occurred while processing your question: {str(e)}",
                "papers": [],
                "processing_time": time.time() - start_time
            }
    
    def interactive_session(self):
        """Start interactive Q&A session"""
        print("\n" + "="*60)
        print("Simple ArXiv Research Assistant")
        print("Ask questions about research papers!")
        print("\nCommands:")
        print("  'quit' or 'exit' - Exit")
        print("  'help' - Show help")
        print("  'stats' - Database statistics")
        print("  'examples' - Show example queries")
        print("="*60)
        
        while True:
            try:
                query = input("\nYour question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Thank you for using the ArXiv Research Assistant!")
                    break
                
                if query.lower() == 'help':
                    self._show_help()
                    continue
                
                if query.lower() == 'stats':
                    self._show_stats()
                    continue
                    
                if query.lower() == 'examples':
                    self._show_examples()
                    continue
                
                if not query:
                    print("Please enter a question.")
                    continue
                
                # Process question
                print("\nSearching...")
                result = self.ask_question(query)
                
                # Display results
                print(f"\nProcessed in {result['processing_time']:.2f} seconds")
                print("="*60)
                print(result['answer'])
                print("="*60)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _show_help(self):
        """Show help information"""
        help_text = """
HELP - Simple ArXiv Research Assistant

QUERY TYPES:
• Definition: "What are transformers?" "Define neural networks"
• Methods: "How do GANs work?" "Explain attention mechanism"  
• Applications: "Applications of deep learning" "Uses of BERT"
• Recent work: "Recent advances in NLP" "Latest computer vision"
• Comparisons: "Compare CNN vs transformer" "RNN vs LSTM"

SEARCH TIPS:
• Use clear, specific technical terms
• Ask complete questions rather than single keywords
• Include domain context (e.g., "in machine learning", "for NLP")
• Try different phrasings if you don't get good results

DATABASE:
• 2.8M+ research papers from ArXiv
• Full-text search across titles and abstracts
• Papers include authors, categories, and publication dates
        """
        print(help_text)
    
    def _show_examples(self):
        """Show example queries"""
        examples = [
            "What are transformer models in machine learning?",
            "How do convolutional neural networks work?", 
            "Applications of BERT in natural language processing",
            "Recent advances in computer vision",
            "Compare RNNs and transformers for sequence modeling",
            "What is attention mechanism in deep learning?",
            "Graph neural networks for recommendation systems",
            "Explain generative adversarial networks",
            "Deep reinforcement learning applications",
            "Transfer learning in computer vision"
        ]
        
        print("\nEXAMPLE QUERIES:")
        for i, example in enumerate(examples, 1):
            print(f"{i:2d}. {example}")
        print()
    
    def _show_stats(self):
        """Show system statistics"""
        try:
            stats = self.search_engine.get_stats()
            
            print(f"\nDATABASE STATISTICS:")
            print(f"  Total Papers: {stats.get('total_papers', 0):,}")
            print(f"  Unique Authors: {stats.get('unique_authors', 0):,}")
            print(f"  Categories: {stats.get('unique_categories', 0)}")
            
            print(f"\nTOP CATEGORIES:")
            for category, count in stats.get('top_categories', []):
                print(f"  • {category}: {count:,} papers")
        except Exception as e:
            print(f"Error getting statistics: {e}")
    
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
        rag_system = SimpleRAGSystem()
        
        # Check for command line arguments
        if len(sys.argv) > 1:
            # Single query mode
            query = " ".join(sys.argv[1:])
            result = rag_system.ask_question(query)
            
            print(f"Query: {query}")
            print(f"Found {result['papers_found']} papers in {result['processing_time']:.2f}s")
            print("="*60)
            print(result['answer'])
            print("="*60)
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