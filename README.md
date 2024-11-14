# esun-ai-cup-2024
Source codes for competition: AI CUP 2024 玉山人工智慧公開挑戰賽－RAG與LLM在金融問答的應用



## Dataset


```
esun-ai-cup-2024
├── run1.py
├── model
├─  model2
├── preprocess
├─  preprocess2
└── data 
    ├── dataset/preliminary
    │    └── questions_example.json
    └── reference
        ├── faq
        ├── finance
        └── insurance
```

## Version 1

Run the following code under the `esun-ai-cup-2024` folder:

```bash
    python run1.py
```

Expected output:

```
esun-ai-cup-2024
├── embeddings_insurance.pt 
├── embeddings_finance.pt 
├── embeddings_faq.pt
└── data 
    ├── dataset/preliminary
    │    └── pred_retrieve.json
    └── reference
        ├── faq
        │   └── chunk_corpus_cache.json
        ├── finance
        │   └── chunk_corpus_cache.json         
        └── insurance
            └── chunk_corpus_cache.json
```


## Dependencies

Run the following code under the `esun-ai-cup-2024` folder to install required dependencies:

```bash
   pip install -r requirements.txt
```