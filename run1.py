import os
import json
import argparse
from preprocess.data_preprocess1 import EnhancedDocumentPreprocessor
from model.retrieval1 import EnhancedDocumentRetriever
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--question_path', default=".\data\dataset\preliminary\questions_example.json")
    parser.add_argument('--source_path', default=".\data\reference")
    parser.add_argument('--output_path', default=".\data\dataset\preliminary\pred_retrieve.json")
    args = parser.parse_args()
    
    try:
        # Initialize preprocessor and retriever
        print("Initializing document preprocessor and retriever...")
        preprocessor = EnhancedDocumentPreprocessor(args.source_path)
        retriever = EnhancedDocumentRetriever(preprocessor)
        
        # Load questions
        print("Loading questions...")
        with open(args.question_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)['questions']
        
        # Process questions
        print("Processing questions...")
        answers = []
        for q in tqdm(questions, desc='Processing questions'):
            try:
                retrieved = retriever.retrieve(q['query'], q['source'], q['category'])
                answers.append({"qid": q['qid'], "retrieve": int(retrieved)})
            except Exception as e:
                print(f"Error processing qid {q['qid']}: {e}")
                answers.append({"qid": q['qid'], "retrieve": q['source'][0]})
        
        # Save results
        print("Saving results...")
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump({"answers": answers}, f, ensure_ascii=False, indent=2)
            
        print("Processing completed successfully!")
        
    except Exception as e:
        print(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()