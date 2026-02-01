"""
Data collection and preprocessing module for Hybrid RAG System
"""
import json
import random
import time
from typing import List, Dict, Optional
from tqdm import tqdm
import wikipediaapi
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class WikipediaDataCollector:
    """Collect Wikipedia articles and preprocess them"""
    
    def __init__(self, language='en'):
        self.wiki = wikipediaapi.Wikipedia(language=language)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_random_wikipedia_urls(self, count: int = 300) -> List[str]:
        """Get random Wikipedia URLs by fetching from main page and generating URLs"""
        urls = []
        topics = [
            'Science', 'Technology', 'History', 'Geography', 'Politics', 'Culture',
            'Literature', 'Art', 'Music', 'Sports', 'Biology', 'Physics', 'Chemistry',
            'Mathematics', 'Philosophy', 'Religion', 'Medicine', 'Business', 'Economics',
            'Law', 'Education', 'Environment', 'Space', 'Architecture', 'Transportation'
        ]
        
        # Generate Wikipedia URLs from common articles
        common_articles = [
            'Artificial_intelligence', 'Machine_learning', 'Deep_learning', 'Natural_language_processing',
            'Computer_science', 'Data_science', 'Python', 'Java', 'C++', 'JavaScript',
            'Internet', 'World_Wide_Web', 'Database', 'Algorithm', 'Data_structure',
            'Quantum_computing', 'Cryptography', 'Cybersecurity', 'Blockchain', 'Cloud_computing',
            'Ancient_Rome', 'Ancient_Greece', 'Medieval_Europe', 'Renaissance', 'Industrial_Revolution',
            'World_War_I', 'World_War_II', 'Cold_War', 'American_Civil_War', 'French_Revolution',
            'United_States', 'United_Kingdom', 'France', 'Germany', 'Japan', 'China', 'India',
            'Brazil', 'Australia', 'Canada', 'Russia', 'Mexico', 'South_Korea', 'Spain',
            'William_Shakespeare', 'Jane_Austen', 'Charles_Dickens', 'Mark_Twain', 'Leo_Tolstoy',
            'Pablo_Picasso', 'Vincent_van_Gogh', 'Michelangelo', 'Leonardo_da_Vinci', 'Andy_Warhol',
            'Albert_Einstein', 'Isaac_Newton', 'Stephen_Hawking', 'Marie_Curie', 'Galileo_Galilei',
            'Solar_System', 'Moon', 'Milky_Way', 'Black_hole', 'Big_Bang', 'Evolution', 'DNA',
            'Heart', 'Brain', 'Photosynthesis', 'Gravity', 'Electricity', 'Magnetism', 'Light',
            'Olympics', 'Football', 'Basketball', 'Tennis', 'Cricket', 'Baseball', 'Ice_hockey'
        ]
        
        base_url = 'https://en.wikipedia.org/wiki/'
        urls = [base_url + article for article in common_articles[:count]]
        
        # If we need more, generate random variations
        while len(urls) < count:
            topic = random.choice(topics)
            urls.append(base_url + topic)
        
        return list(set(urls[:count]))
    
    def fetch_article_content(self, url: str, timeout: int = 10) -> Optional[str]:
        """Fetch article content from Wikipedia URL"""
        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(['script', 'style']):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text if len(text.split()) >= 200 else None
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return None
    
    def chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[Dict]:
        """Split text into overlapping chunks"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            tokens = sentence.split()
            current_length += len(tokens)
            current_chunk.append(sentence)
            
            if current_length >= chunk_size:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'token_count': len(chunk_text.split())
                })
                
                # Keep last sentences for overlap
                overlap_tokens = 0
                overlap_sentences = []
                for sent in reversed(current_chunk):
                    overlap_tokens += len(sent.split())
                    overlap_sentences.insert(0, sent)
                    if overlap_tokens >= overlap:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text.split()) >= 50:  # Minimum chunk size
                chunks.append({
                    'text': chunk_text,
                    'token_count': len(chunk_text.split())
                })
        
        return chunks


class DataPreprocessor:
    """Preprocess and organize Wikipedia data"""
    
    @staticmethod
    def collect_wikipedia_corpus(fixed_urls: List[str], random_count: int = 300) -> Dict:
        """Collect Wikipedia articles from URLs"""
        collector = WikipediaDataCollector()
        corpus = []
        chunk_id = 0
        
        all_urls = fixed_urls + collector.get_random_wikipedia_urls(random_count)
        
        print(f"Collecting {len(all_urls)} Wikipedia articles...")
        
        for url in tqdm(all_urls, desc="Fetching articles"):
            content = collector.fetch_article_content(url)
            
            if content and len(content.split()) >= 200:
                # Extract title from URL
                title = url.split('/wiki/')[-1].replace('_', ' ')
                
                # Chunk the content
                chunks = collector.chunk_text(content)
                
                for chunk_data in chunks:
                    corpus.append({
                        'chunk_id': f'chunk_{chunk_id}',
                        'url': url,
                        'title': title,
                        'text': chunk_data['text'],
                        'token_count': chunk_data['token_count']
                    })
                    chunk_id += 1
                
                time.sleep(0.1)  # Rate limiting
        
        return {
            'total_chunks': len(corpus),
            'total_urls': len(set([c['url'] for c in corpus])),
            'chunks': corpus
        }
    
    @staticmethod
    def save_corpus(corpus: Dict, filepath: str) -> None:
        """Save corpus to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(corpus, f, indent=2, ensure_ascii=False)
        print(f"Corpus saved to {filepath}")
    
    @staticmethod
    def load_corpus(filepath: str) -> Dict:
        """Load corpus from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)


def create_fixed_urls_file(filepath: str = 'fixed_urls.json', count: int = 200) -> List[str]:
    """Create a fixed list of Wikipedia URLs"""
    collector = WikipediaDataCollector()
    urls = collector.get_random_wikipedia_urls(count)
    
    with open(filepath, 'w') as f:
        json.dump({'urls': urls, 'count': len(urls)}, f, indent=2)
    
    print(f"Fixed URLs saved to {filepath}")
    return urls


if __name__ == "__main__":
    # Example usage
    fixed_urls = create_fixed_urls_file('fixed_urls.json', 200)
    print(f"Created fixed URLs: {len(fixed_urls)}")
    
    # Collect corpus
    preprocessor = DataPreprocessor()
    corpus = preprocessor.collect_wikipedia_corpus(fixed_urls, random_count=300)
    preprocessor.save_corpus(corpus, 'wikipedia_corpus.json')
    print(f"Corpus created with {corpus['total_chunks']} chunks from {corpus['total_urls']} URLs")
