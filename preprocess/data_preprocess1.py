import os
import json
import re
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
import jieba
import pdfplumber
from chunking.recursive_token_chunker import RecursiveTokenChunker
from typing import Set, Dict, List

class EnhancedDocumentPreprocessor:
    def __init__(self, reference_path: str):
        self.reference_path = Path(reference_path)
        
        # Load custom dictionary and stopwords
        jieba.load_userdict("./model/custom_dict.txt")
        self.custom_terms = self._load_custom_terms()
        self.stopwords = self._load_stopwords()
        self.abbreviations = self._load_abbreviations()
        
        self.chunker = RecursiveTokenChunker(chunk_size=20, chunk_overlap=12)
    def _load_stopwords(self) -> Set[str]:
        """Load Chinese stopwords"""
        stopwords = set()
        stopwords_path = Path("./model/stopwords.txt")
        if stopwords_path.exists():
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                stopwords = {line.strip() for line in f if line.strip()}
        return stopwords

    def _load_abbreviations(self) -> Dict[str, str]:
        """Load abbreviations mapping"""
        abbreviations = {}
        abbrev_path = Path("./model/abbreviations.json")
        if abbrev_path.exists():
            with open(abbrev_path, 'r', encoding='utf-8') as f:
                abbreviations = json.load(f)
        return abbreviations
        
    def _load_custom_terms(self) -> Dict[str, float]:
        """Load terms from custom dictionary with weights"""
        terms = {}
        try:
            with open("./model/custom_dict.txt", 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        parts = line.strip().split()
                        term = parts[0]
                        weight = float(parts[1]) if len(parts) > 1 else 1.0
                        terms[term] = weight
        except Exception as e:
            print(f"Error loading custom dictionary: {e}")
            terms = {}
        return terms

    def _read_pdf(self, pdf_path: Path) -> str:
        """Read PDF with improved extraction"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                texts = []
                for page in pdf.pages:
                    text = page.extract_text(x_tolerance=2, y_tolerance=2)
                    if text:
                        texts.append(text)
                return '\n'.join(texts)
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
            return ""
        
    def remove_consecutive_duplicates(self, text: str, min_length: int = 2) -> str:
        """Remove consecutive duplicate words/phrases"""
        words = text.split()
        if not words:
            return text
            
        result = [words[0]]
        for word in words[1:]:
            if len(word) >= min_length and word != result[-1]:
                result.append(word)
            elif len(word) < min_length:  # Always keep short words/particles
                result.append(word)
        return ' '.join(result)
    
    def remove_nearby_duplicates(self, text: str, window_size: int = 10) -> str:
        """Remove duplicates within a sliding window while preserving structure"""
        words = text.split()
        if not words:
            return text
            
        result = []
        window = []
        
        for word in words:
            # Always keep short words and special characters
            if len(word) < 2 or word in ['的', '了', '和', '與', '及']:
                result.append(word)
                continue
                
            # Check if word appears too frequently in recent window
            if word in window and window.count(word) >= 2:
                continue
                
            result.append(word)
            window.append(word)
            if len(window) > window_size:
                window.pop(0)
                
        return ' '.join(result)
    
    def normalize_chinese_numbers(self, text: str) -> str:
        """Normalize Chinese numbers and quantities"""
        # Map for Chinese numbers to Arabic numbers
        chinese_numbers = {
            '零': '0', '一': '1', '二': '2', '三': '3', '四': '4',
            '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
            '十': '10', '百': '100', '千': '1000', '萬': '10000'
        }
        
        # Replace Chinese numbers with Arabic numbers
        for cn, an in chinese_numbers.items():
            text = text.replace(cn, an)
        
        return text
    
    def clean_punctuation_spacing(self, text: str) -> str:
        """Clean up spaces around punctuation for Chinese text"""
        # Remove spaces around Chinese punctuation
        text = re.sub(r'\s*([，。！？；：、）】」』])\s*', r'\1', text)
        text = re.sub(r'\s*([（【「『])\s*', r'\1', text)
        
        # Ensure English punctuation has proper spacing
        text = re.sub(r'\s*([,.!?;:])\s*', r'\1 ', text)
        text = re.sub(r'\s*([\[\({])\s*', r'\1', text)
        text = re.sub(r'\s*([\])}])\s*', r'\1 ', text)
        
        return text

    def _preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing pipeline"""
        if not isinstance(text, str):
            text = str(text)
        
        # Basic cleaning
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize text
        text = self.normalize_chinese_numbers(text)
        
        # Remove repeated content
        text = self.remove_consecutive_duplicates(text)
        text = self.remove_nearby_duplicates(text)
        
        # Handle numbers and special characters
        text = re.sub(r'(\d+\.?\d*)', r' \1 ', text)  # Separate numbers
        text = re.sub(r'[^\w\s\u4e00-\u9fff.,%()，。、：；？！（）]', ' ', text)  # Keep necessary punctuation
        
        # Final cleaning
        text = self.clean_punctuation_spacing(text)
        text = ' '.join(text.split())  # Normalize spaces
        
        return text
    
    def _tokenize_for_tfidf(self, text: str) -> List[str]:
        """Tokenize text for TF-IDF"""
        text = self._preprocess_text(text)
        tokens = jieba.cut_for_search(text)
        return [t for t in tokens if t not in self.stopwords]

    def _load_single_corpus(self, category: str) -> Dict[int, str]:
        """Load corpus for a single category with caching"""
        category_path = self.reference_path / category
        corpus_dict = {}
        
        # Define the cache path
        cache_path = category_path / 'chunk_corpus_cache.json'

        # Load from cache if it exists
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    corpus_dict = json.load(f)
                    print(f"Loaded {category} corpus from cache.")
                    return {k: self._preprocess_text(v) for k, v in corpus_dict.items()}
            except Exception as e:
                print(f"Error loading cache for {category}: {e}")

        # If cache does not exist, load the corpus
        if category == 'faq':
            faq_path = category_path / 'pid_map_content.json'
            try:
                with open(faq_path, 'r', encoding='utf-8') as f:
                    corpus_dict = {int(k): str(v) for k, v in json.load(f).items()}
            except Exception as e:
                print(f"Error loading FAQ corpus: {e}")
                return {}
        else:
            for file in tqdm(list(category_path.glob('*.pdf')), desc=f'Loading {category}'):
                try:
                    doc_id = int(file.stem)
                    text = self._read_pdf(file)
                    if text:
                        corpus_dict[doc_id] = text
                except ValueError as e:
                    print(f"Error processing {file}: {e}")
                    continue
        
        # Preprocess the loaded corpus
        processed_corpus = {k: self._preprocess_text(v) for k, v in corpus_dict.items()}

        # Chunking: assumption - there are at most 1 << 16 = 65536 chunks
        chunked_corpus = dict()
        for doc_id, text in processed_corpus.items():
            tokens = self.chunker.split_text(text)
            for i, token in enumerate(tokens):
                chunked_corpus[(doc_id << 16) | i] = token
        sorted_chunked_corpus = {x: i for x, i in sorted(chunked_corpus.items())}
        # Save the processed corpus to cache
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(sorted_chunked_corpus, f, ensure_ascii=False, indent=4)
            print(f"Saved {category} corpus to cache.")
        except Exception as e:
            print(f"Error saving cache for {category}: {e}")

        return sorted_chunked_corpus

    