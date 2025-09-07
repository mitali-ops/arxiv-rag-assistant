# database.py - Complete Database Setup for ArXiv RAG System
import sqlite3
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

# Configuration
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_PATH = DATA_DIR / "arxiv.db"
PREPROCESSED_DATA_PATH = DATA_DIR / "preprocessed_arxiv_dataset.json"

# Create data directory if it doesn't exist
DATA_DIR.mkdir(exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ArXivDatabase:
    def __init__(self, db_path: Path = DATABASE_PATH):
        self.db_path = db_path
        logger.info(f"Initializing database at: {db_path}")
        
        # Remove existing database to start fresh
        if db_path.exists():
            logger.info("Removing existing database...")
            db_path.unlink()
        
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        # Optimize SQLite settings for bulk inserts
        self.cursor.execute("PRAGMA foreign_keys = ON;")
        self.cursor.execute("PRAGMA journal_mode = WAL;")
        self.cursor.execute("PRAGMA synchronous = NORMAL;")
        self.cursor.execute("PRAGMA cache_size = 10000;")
        self.cursor.execute("PRAGMA temp_store = MEMORY;")
        
        logger.info("Database connection established with optimized settings")

    def create_tables(self):
        """Create all necessary database tables with proper indexes"""
        logger.info("Creating database tables...")
        
        # Papers table - main table for research papers
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                arxiv_id TEXT UNIQUE,
                title TEXT NOT NULL,
                abstract TEXT NOT NULL,
                published TEXT,
                updated TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Authors table - normalized author storage
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS authors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                position INTEGER DEFAULT 0,
                FOREIGN KEY (paper_id) REFERENCES papers (id) ON DELETE CASCADE
            )
        ''')

        # Categories table - paper categories/subjects
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER NOT NULL,
                category TEXT NOT NULL,
                FOREIGN KEY (paper_id) REFERENCES papers (id) ON DELETE CASCADE
            )
        ''')

        # Create indexes for better search performance
        logger.info("Creating database indexes...")
        
        # Primary search indexes
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_papers_title ON papers(title);')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_papers_arxiv_id ON papers(arxiv_id);')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_papers_published ON papers(published);')
        
        # Foreign key indexes
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_authors_paper_id ON authors(paper_id);')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_authors_name ON authors(name);')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_categories_paper_id ON categories(paper_id);')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_categories_category ON categories(category);')

        # Create Full-Text Search virtual table
        logger.info("Setting up Full-Text Search...")
        try:
            self.cursor.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
                    title, abstract, 
                    content='papers', 
                    content_rowid='id',
                    tokenize='porter'
                );
            ''')
            logger.info("✅ FTS5 virtual table created successfully")
        except sqlite3.OperationalError as e:
            logger.warning(f"Could not create FTS table: {e}")
            logger.warning("Full-text search will use LIKE queries instead")

        self.conn.commit()
        logger.info("✅ Database schema created successfully")

    def insert_paper(self, paper_data: Dict) -> Optional[int]:
        """Insert a single paper with all related data"""
        try:
            # Insert paper
            self.cursor.execute('''
                INSERT OR IGNORE INTO papers (arxiv_id, title, abstract, published, updated) 
                VALUES (?, ?, ?, ?, ?)
            ''', (
                paper_data.get('arxiv_id', ''),
                paper_data['title'],
                paper_data['abstract'],
                paper_data.get('published', ''),
                paper_data.get('updated', '')
            ))
            
            if self.cursor.rowcount == 0:
                # Paper already exists, get its ID
                self.cursor.execute('SELECT id FROM papers WHERE arxiv_id = ? OR (title = ? AND abstract = ?)', 
                                  (paper_data.get('arxiv_id', ''), paper_data['title'], paper_data['abstract']))
                result = self.cursor.fetchone()
                return result[0] if result else None
            
            paper_id = self.cursor.lastrowid

            # Insert authors
            for i, author in enumerate(paper_data.get('authors', [])):
                if author and author.strip():  # Skip empty authors
                    self.cursor.execute('''
                        INSERT INTO authors (paper_id, name, position) VALUES (?, ?, ?)
                    ''', (paper_id, author.strip(), i))

            # Insert categories
            for category in paper_data.get('categories', []):
                if category and category.strip():  # Skip empty categories
                    self.cursor.execute('''
                        INSERT INTO categories (paper_id, category) VALUES (?, ?)
                    ''', (paper_id, category.strip()))

            return paper_id

        except Exception as e:
            logger.error(f"Error inserting paper '{paper_data.get('title', 'Unknown')}': {e}")
            return None

    def update_fts_table(self):
        """Update the FTS table with current papers"""
        try:
            logger.info("Updating Full-Text Search index...")
            self.cursor.execute('DELETE FROM papers_fts')
            self.cursor.execute('''
                INSERT INTO papers_fts(rowid, title, abstract)
                SELECT id, title, abstract FROM papers
            ''')
            self.conn.commit()
            logger.info("✅ FTS index updated successfully")
        except sqlite3.OperationalError:
            logger.warning("FTS table update failed - FTS not available")

    def get_stats(self) -> Dict:
        """Get comprehensive database statistics"""
        stats = {}
        
        try:
            # Basic counts
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
                LIMIT 10
            ''')
            stats['top_categories'] = self.cursor.fetchall()
            
            # Publication year distribution (extract year from published date)
            self.cursor.execute('''
                SELECT substr(published, 1, 4) as year, COUNT(*) as count
                FROM papers 
                WHERE published != '' AND length(published) >= 4
                GROUP BY year 
                ORDER BY year DESC 
                LIMIT 10
            ''')
            stats['papers_by_year'] = self.cursor.fetchall()
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            
        return stats

    def close(self):
        """Close database connection"""
        try:
            self.conn.close()
            logger.info("Database connection closed")
        except:
            pass

def load_and_populate_database():
    """Main function to load preprocessed data and populate database"""
    
    # Check if preprocessed data exists
    if not PREPROCESSED_DATA_PATH.exists():
        logger.error(f"❌ Preprocessed data not found: {PREPROCESSED_DATA_PATH}")
        logger.info("Please run preprocessing.py first to create the preprocessed data")
        return False

    # Check file size
    file_size_mb = PREPROCESSED_DATA_PATH.stat().st_size / (1024 * 1024)
    logger.info(f"Preprocessed data file size: {file_size_mb:.1f} MB")

    # Initialize database
    db = ArXivDatabase()
    db.create_tables()

    try:
        # Load preprocessed data
        logger.info("Loading preprocessed data...")
        start_time = time.time()
        
        with open(PREPROCESSED_DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        load_time = time.time() - start_time
        logger.info(f"✅ Loaded {len(data):,} papers in {load_time:.1f} seconds")

        # Populate database in batches
        logger.info("Populating database...")
        batch_size = 1000
        inserted_count = 0
        failed_count = 0
        
        start_time = time.time()
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batch_start_time = time.time()
            
            # Process batch
            for paper in batch:
                paper_id = db.insert_paper(paper)
                if paper_id:
                    inserted_count += 1
                else:
                    failed_count += 1
            
            # Commit batch
            db.conn.commit()
            
            # Progress reporting
            batch_time = time.time() - batch_start_time
            total_processed = i + len(batch)
            progress = (total_processed / len(data)) * 100
            
            if (i // batch_size + 1) % 10 == 0:  # Report every 10 batches
                logger.info(f"Progress: {progress:.1f}% ({total_processed:,}/{len(data):,}) - "
                          f"Inserted: {inserted_count:,}, Failed: {failed_count}, "
                          f"Batch time: {batch_time:.2f}s")

        # Update FTS table
        db.update_fts_table()

        # Final statistics
        total_time = time.time() - start_time
        stats = db.get_stats()
        
        # Display results
        logger.info("=" * 60)
        logger.info("🎉 DATABASE POPULATION COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"📊 STATISTICS:")
        logger.info(f"   Total papers in database: {stats['total_papers']:,}")
        logger.info(f"   Successfully inserted: {inserted_count:,}")
        logger.info(f"   Failed insertions: {failed_count:,}")
        logger.info(f"   Unique authors: {stats['unique_authors']:,}")
        logger.info(f"   Unique categories: {stats['unique_categories']:,}")
        logger.info(f"   Processing time: {total_time:.1f} seconds")
        logger.info(f"   Database file size: {DATABASE_PATH.stat().st_size / (1024*1024):.1f} MB")
        
        logger.info(f"\n🔥 TOP CATEGORIES:")
        for category, count in stats['top_categories'][:5]:
            logger.info(f"   {category}: {count:,} papers")
        
        if stats.get('papers_by_year'):
            logger.info(f"\n📅 RECENT YEARS:")
            for year, count in stats['papers_by_year'][:5]:
                logger.info(f"   {year}: {count:,} papers")
        
        logger.info("=" * 60)
        logger.info("✅ Ready to run the RAG system!")
        logger.info("Next steps:")
        logger.info("1. python main.py                  # CLI interface")
        logger.info("2. streamlit run web_app.py       # Web interface")
        logger.info("=" * 60)
        
        return True

    except FileNotFoundError:
        logger.error("❌ Preprocessed data file not found")
        return False
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in preprocessed data: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error during database population: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()

if __name__ == '__main__':
    print("🚀 Starting ArXiv Database Setup...")
    print(f"Database location: {DATABASE_PATH}")
    print(f"Data source: {PREPROCESSED_DATA_PATH}")
    
    success = load_and_populate_database()
    
    if success:
        print("🎉 Database setup completed successfully!")
    else:
        print("❌ Database setup failed. Check the logs above.")
        exit(1)