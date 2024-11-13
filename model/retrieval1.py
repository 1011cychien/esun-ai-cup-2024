import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from ..preprocess.data_preprocess1 import EnhancedDocumentPreprocessor
import torch
from tqdm import tqdm
from typing import List, Dict
from pathlib import Path

class SearchResult:
    doc_id: int
    sparse_score: float
    dense_score: float
    combined_score: float

class EnhancedDocumentRetriever:
    def __init__(self, preprocessor: EnhancedDocumentPreprocessor, device: str = "cuda"):
        self.preprocessor = preprocessor
        self.device = device
        
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('BAAI/bge-m3', device=self.device)
        
        self.corpus_data = {}
        self._initialize_corpus()

    def get_embeddings_with_memory_optimization(self, texts, max_batch_size=8):
            """Calculate embeddings with memory-optimized approach"""
            all_embeddings = []
            
            # Calculate total batches for progress tracking
            total_batches = (len(texts) + max_batch_size - 1) // max_batch_size
            
            for i in tqdm(range(0, len(texts), max_batch_size), total=total_batches, 
                        desc="Calculating embeddings"):
                # Process smaller batches
                batch_texts = texts[i:i + max_batch_size]
                try:
                    batch_embeddings = self.embedding_model.encode(
                        batch_texts,
                        batch_size=max_batch_size,
                        show_progress_bar=False,  # Disable inner progress bar
                        convert_to_tensor=True,
                        normalize_embeddings=True
                    )
                    all_embeddings.append(batch_embeddings)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        # If OOM occurs, try with even smaller batch
                        print(f"OOM error, trying with smaller batch...")
                        torch.cuda.empty_cache()
                        batch_embeddings = self.embedding_model.encode(
                            batch_texts,
                            batch_size=1,
                            show_progress_bar=False,
                            convert_to_tensor=True,
                            normalize_embeddings=True
                        )
                        all_embeddings.append(batch_embeddings)
                    else:
                        raise e
                        
                # Clear GPU cache after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Concatenate all embeddings
            return torch.cat(all_embeddings, dim=0)

    # Modified _initialize_corpus method:
    def _initialize_corpus(self):
        """Initialize all corpora with enhanced vector representations"""
        self.corpus_data = {}
        
        for category in ['insurance', 'finance', 'faq']:
            corpus_texts = self.preprocessor._load_single_corpus(category)

            if corpus_texts:
                self.corpus_data[category] = {
                    'texts': corpus_texts,
                    'doc_ids': list(corpus_texts.keys())
                }
                
                # Calculate and cache embeddings
                cache_path = Path(f'embeddings_{category}.pt')
                if cache_path.exists():
                    print(f"Loading cached embeddings for {category}...")
                    embeddings = torch.load(cache_path)
                else:
                    print(f"Calculating embeddings for {category}...")
                    texts = list(corpus_texts.values())
                    try:
                        # Use the memory-optimized embedding calculation
                        embeddings = self.get_embeddings_with_memory_optimization(texts)
                        torch.save(embeddings, cache_path)
                    except Exception as e:
                        print(f"Error calculating embeddings for {category}: {e}")
                        continue
                
                # Store embeddings and create FAISS index
                self.corpus_data[category]['embeddings'] = embeddings.cpu().numpy()
                

    def _get_hybrid_scores(
        self, 
        query: str, 
        corpus_data: Dict, 
        source_ids: List[int], 
        category: str,
        top_k: int = 9  # Limit to 9 choices
    ) -> List[SearchResult]:
        """Get hybrid scores combining sparse and dense retrieval with reranking"""
        # Filter corpus to only include source_ids
        valid_indices = [i for i, doc_id in enumerate(corpus_data['doc_ids']) 
                        if (int(doc_id) >> 16) in source_ids]

        if not valid_indices:
            return []
            
        filtered_embeddings = corpus_data['embeddings'][valid_indices]
        try:
            filtered_texts = [corpus_data['texts'][int(corpus_data['doc_ids'][i])] 
                            for i in valid_indices]
        except:
            filtered_texts = [corpus_data['texts'][str(corpus_data['doc_ids'][i])] 
                            for i in valid_indices]
        filtered_doc_ids = [corpus_data['doc_ids'][i] for i in valid_indices]
        # 1. Dense retrieval (embedding similarity)
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True
        ).cpu().numpy()
        
        # Calculate cosine similarities
        dense_scores = np.dot(filtered_embeddings, query_embedding)
        
        # 2. Sparse retrieval (BM25)
        tokenized_corpus = [self.preprocessor._tokenize_for_tfidf(text) for text in filtered_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        query_tokens = self.preprocessor._tokenize_for_tfidf(query)
        sparse_scores = bm25.get_scores(query_tokens)
        
        # Normalize scores
        def normalize_scores(scores):
            if len(scores) == 0:
                return scores
            min_score = min(scores)
            max_score = max(scores)
            if max_score == min_score:
                return [0.5] * len(scores)
            return [(s - min_score) / (max_score - min_score) for s in scores]
        dense_scores_norm = normalize_scores(dense_scores.tolist())
        sparse_scores_norm = normalize_scores(sparse_scores.tolist())

        
        # Combine scores with weights based on category
        if category == "insurance":
            weights = {'sparse': 0.3, 'dense': 0.7}
        else:
            weights = {'sparse': 0.0, 'dense': 1.0}
        
        # Calculate initial combined scores
        initial_scores = [
            weights['sparse'] * s + weights['dense'] * d
            for s, d in zip(sparse_scores_norm, dense_scores_norm)
        ]

        initial_scores_norm = normalize_scores(initial_scores)
        
        # Get top k candidates for reranking
        top_k = min(top_k, len(initial_scores))
        top_k = int(top_k)
        top_indices = np.argsort(initial_scores)[-top_k:]
        
        # Create final results
        results = []
        for idx, original_idx in enumerate(top_indices):
            doc_id = filtered_doc_ids[original_idx]
            results.append(SearchResult(
                doc_id=str(doc_id),
                sparse_score=sparse_scores_norm[original_idx],
                dense_score=dense_scores_norm[original_idx],
                combined_score=initial_scores_norm[original_idx]  # Use rerank score as final score
            ))
        
        return sorted(results, key=lambda x: x.combined_score, reverse=True)
    def retrieve(self, query: str, source: List[int], category: str) -> int:
        """Retrieve best matching document using hybrid approach"""
        if category not in self.corpus_data:
            print(f"Warning: Unknown category '{category}', returning first source document")
            return source[0]
        
        if not source:
            print("Warning: Empty source list")
            return 0
        
        # try:
        results = self._get_hybrid_scores(query, self.corpus_data[category], source, category)
        if results:
            return int(results[0].doc_id) >> 16
        
        return source[0]