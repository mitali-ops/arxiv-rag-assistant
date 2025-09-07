# database_recovery.py - Fix and complete the database setup
import sqlite3
import logging
import time
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_PATH = DATA_DIR / "arxiv.db"
BACKUP_PATH = DATA_DIR / "arxiv_backup.db"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def recover_and_complete_database():
    """Recover the database and complete the setup without FTS initially"""
    
    if not DATABASE_PATH.exists():
        logger.error("❌ Database file not found!")
        return False
    
    try:
        logger.info("🔧 Starting database recovery...")
        
        # Connect to the existing database
        conn = sqlite3.connect(str(DATABASE_PATH))
        cursor = conn.cursor()
        
        # Check if the main data is intact
        logger.info("Checking database integrity...")
        
        try:
            cursor.execute('SELECT COUNT(*) FROM papers')
            paper_count = cursor.fetchone()[0]
            logger.info(f"✅ Found {paper_count:,} papers in database")
            
            cursor.execute('SELECT COUNT(*) FROM authors')
            author_count = cursor.fetchone()[0]
            logger.info(f"✅ Found {author_count:,} author records")
            
            cursor.execute('SELECT COUNT(*) FROM categories')
            category_count = cursor.fetchone()[0]
            logger.info(f"✅ Found {category_count:,} category records")
            
        except Exception as e:
            logger.error(f"❌ Main tables corrupted: {e}")
            return False
        
        # Try to fix the FTS table issue
        logger.info("🔧 Fixing FTS table...")
        
        try:
            # Drop the corrupted FTS table
            cursor.execute('DROP TABLE IF EXISTS papers_fts')
            conn.commit()
            logger.info("✅ Dropped corrupted FTS table")
            
            # Recreate FTS table (simpler version)
            cursor.execute('''
                CREATE VIRTUAL TABLE papers_fts USING fts5(
                    title, abstract, 
                    content='papers', 
                    content_rowid='id'
                );
            ''')
            logger.info("✅ Created new FTS table")
            
            # Populate FTS table in smaller chunks to avoid memory issues
            logger.info("📝 Populating FTS table in chunks...")
            
            cursor.execute('SELECT COUNT(*) FROM papers')
            total_papers = cursor.fetchone()[0]
            
            chunk_size = 10000  # Smaller chunks
            processed = 0
            
            for offset in range(0, total_papers, chunk_size):
                start_time = time.time()
                
                cursor.execute('''
                    INSERT INTO papers_fts(rowid, title, abstract)
                    SELECT id, title, abstract FROM papers 
                    LIMIT ? OFFSET ?
                ''', (chunk_size, offset))
                
                processed += cursor.rowcount
                conn.commit()
                
                chunk_time = time.time() - start_time
                progress = (processed / total_papers) * 100
                
                if offset % (chunk_size * 10) == 0:  # Log every 100k records
                    logger.info(f"FTS Progress: {progress:.1f}% ({processed:,}/{total_papers:,}) - {chunk_time:.2f}s")
            
            logger.info("✅ FTS table populated successfully")
            
        except Exception as e:
            logger.warning(f"⚠️  FTS creation failed: {e}")
            logger.info("Database will work without FTS (using LIKE queries instead)")
        
        # Get final statistics
        logger.info("📊 Generating final statistics...")
        
        # Basic counts
        cursor.execute('SELECT COUNT(*) FROM papers')
        total_papers = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT name) FROM authors')
        unique_authors = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT category) FROM categories')
        unique_categories = cursor.fetchone()[0]
        
        # Top categories
        cursor.execute('''
            SELECT category, COUNT(*) as count 
            FROM categories 
            GROUP BY category 
            ORDER BY count DESC 
            LIMIT 10
        ''')
        top_categories = cursor.fetchall()
        
        # Recent years
        cursor.execute('''
            SELECT substr(published, 1, 4) as year, COUNT(*) as count
            FROM papers 
            WHERE published != '' AND length(published) >= 4
            GROUP BY year 
            ORDER BY year DESC 
            LIMIT 5
        ''')
        recent_years = cursor.fetchall()
        
        # File size
        db_size_mb = DATABASE_PATH.stat().st_size / (1024 * 1024)
        
        # Display final results
        logger.info("=" * 60)
        logger.info("🎉 DATABASE RECOVERY COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"📊 FINAL STATISTICS:")
        logger.info(f"   Total papers: {total_papers:,}")
        logger.info(f"   Unique authors: {unique_authors:,}")
        logger.info(f"   Unique categories: {unique_categories:,}")
        logger.info(f"   Database size: {db_size_mb:.1f} MB")
        
        logger.info(f"\n🔥 TOP CATEGORIES:")
        for category, count in top_categories[:5]:
            logger.info(f"   {category}: {count:,} papers")
        
        if recent_years:
            logger.info(f"\n📅 RECENT YEARS:")
            for year, count in recent_years:
                if year and year.isdigit():
                    logger.info(f"   {year}: {count:,} papers")
        
        # Test a simple query
        logger.info(f"\n🧪 TESTING DATABASE:")
        cursor.execute("SELECT title FROM papers WHERE title LIKE '%transformer%' LIMIT 3")
        test_results = cursor.fetchall()
        
        if test_results:
            logger.info("✅ Sample search results:")
            for i, (title,) in enumerate(test_results, 1):
                logger.info(f"   {i}. {title[:80]}...")
        else:
            logger.info("✅ Database queries working (no 'transformer' papers found)")
        
        logger.info("=" * 60)
        logger.info("✅ DATABASE IS READY TO USE!")
        logger.info("Next steps:")
        logger.info("1. python main.py                  # CLI interface")
        logger.info("2. streamlit run web_app.py       # Web interface")
        logger.info("=" * 60)
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Recovery failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_simple_test_script():
    """Create a simple test script to verify the database works"""
    
    test_script = '''
# test_database.py - Simple database test
import sqlite3
from pathlib import Path

DATABASE_PATH = Path("data/arxiv.db")

def test_database():
    if not DATABASE_PATH.exists():
        print("❌ Database not found!")
        return
    
    conn = sqlite3.connect(str(DATABASE_PATH))
    cursor = conn.cursor()
    
    try:
        # Test basic queries
        cursor.execute('SELECT COUNT(*) FROM papers')
        paper_count = cursor.fetchone()[0]
        print(f"✅ Papers in database: {paper_count:,}")
        
        # Test search
        cursor.execute("SELECT title FROM papers WHERE title LIKE '%neural%' LIMIT 5")
        results = cursor.fetchall()
        
        print("\\n🔍 Sample search results for 'neural':")
        for i, (title,) in enumerate(results, 1):
            print(f"   {i}. {title[:70]}...")
            
        print("\\n✅ Database is working correctly!")
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    test_database()
'''
    
    with open("test_database.py", "w") as f:
        f.write(test_script)
    
    logger.info("✅ Created test_database.py for verification")

if __name__ == '__main__':
    logger.info("🚀 Starting database recovery...")
    
    success = recover_and_complete_database()
    
    if success:
        create_simple_test_script()
        print("\n🎉 Database recovery completed successfully!")
        print("📝 Run 'python test_database.py' to verify everything works")
    else:
        print("\n❌ Database recovery failed.")
        print("💡 Try running with more disk space or on a different drive")