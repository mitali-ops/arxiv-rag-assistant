import json
import re
from typing import List, Dict
from pathlib import Path
import logging
from config import DATA_DIR, PREPROCESSED_DATA_PATH

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\-.,;:!?()]', '', text)
    return text

def extract_paper_info(paper: Dict) -> Dict:
    """Extract and validate paper information"""
    try:
        # Handle different possible data structures
        authors = paper.get('authors', [])
        if isinstance(authors, str):
            authors = [authors]
        elif isinstance(authors, list) and authors and isinstance(authors[0], dict):
            authors = [author.get('name', str(author)) for author in authors]
        
        categories = paper.get('categories', [])
        if isinstance(categories, str):
            categories = categories.split()
        
        return {
            'title': clean_text(paper.get('title', '')),
            'abstract': clean_text(paper.get('abstract', paper.get('summary', ''))),
            'authors': [clean_text(author) for author in authors if author],
            'categories': categories,
            'arxiv_id': paper.get('id', paper.get('arxiv_id', '')),
            'published': paper.get('published', paper.get('created', '')),
            'updated': paper.get('updated', paper.get('modified', ''))
        }
    except Exception as e:
        logger.error(f"Error extracting paper info: {e}")
        return {}

def preprocess_data(data: List[Dict]) -> List[Dict]:
    """Preprocess ArXiv data with proper cleaning and validation"""
    processed_data = []
    skipped = 0
    
    for i, paper in enumerate(data):
        try:
            processed_paper = extract_paper_info(paper)
            
            # Skip papers with missing essential fields
            if not processed_paper.get('title') or not processed_paper.get('abstract'):
                skipped += 1
                continue
                
            # Skip papers with very short abstracts (likely incomplete)
            if len(processed_paper['abstract']) < 100:
                skipped += 1
                continue
                
            processed_data.append(processed_paper)
            
            if (i + 1) % 1000 == 0:
                logger.info(f"Processed {i + 1} papers, kept {len(processed_data)}, skipped {skipped}")
            
        except Exception as e:
            logger.error(f"Error processing paper {i}: {e}")
            skipped += 1
            continue
    
    logger.info(f"Final stats: Processed {len(processed_data)} papers, skipped {skipped}")
    return processed_data

def main():
    """Main preprocessing function"""
    logger.info("Starting ArXiv data preprocessing...")
    
    input_file = DATA_DIR / "arxiv_dataset.json"
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        logger.info("Please download the ArXiv dataset and place it in the data directory")
        return
    
    try:
        data = []
        logger.info(f"Loading data from {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            # Handle both JSON lines and regular JSON formats
            content = f.read().strip()
            if content.startswith('['):
                # Regular JSON array
                data = json.loads(content)
            else:
                # JSON lines format
                f.seek(0)
                for line_num, line in enumerate(f):
                    try:
                        data.append(json.loads(line))
                        if (line_num + 1) % 1000 == 0:
                            logger.info(f"Loaded {line_num + 1} lines...")
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON on line {line_num + 1}")
                        continue
        
        logger.info(f"Loaded {len(data)} papers total")
        
        # Preprocess the data
        preprocessed_data = preprocess_data(data)
        logger.info(f"Successfully preprocessed {len(preprocessed_data)} papers")
        
        # Save preprocessed data
        with open(PREPROCESSED_DATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(preprocessed_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Preprocessing completed! Saved to {PREPROCESSED_DATA_PATH}")
        
    except FileNotFoundError:
        logger.error("❌ Error: 'arxiv_dataset.json' file not found in data directory")
    except Exception as e:
        logger.error(f"❌ An error occurred: {e}")

if __name__ == '__main__':
    main()