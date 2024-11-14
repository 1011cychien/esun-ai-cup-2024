# esun-ai-cup-2024
TEAM\_6124's ource codes for competition: AI CUP 2024 玉山人工智慧公開挑戰賽－RAG與LLM在金融問答的應用



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

## Two Architectures

We have two main working flows in this project. For convention, those are simply named version 1 and 2. After running both models, we use a blended strategy to combine the predictions and try to output an optimized result.

### Version 1

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


### Version 2

Please refer to the README's in [preprocess2/](https://github.com/1011cychien/esun-ai-cup-2024/tree/main/preprocess2/) and [model2/](https://github.com/1011cychien/esun-ai-cup-2024/tree/main/model2).

### Blended Strategy

After several experiments, here are our findings.

* For faq, both models perform well.
* For insurance, version 1 significantly outplays version 2.
* For finance, version 2 perform slightly better than version 1. So we utilize a blended strategy for this category.

Hence, we implemented a simple script that takes two jsons as input and produce a json for submission.

#### Usage

Prerequisites: running both models to generate two predictions

```bash
mv <version-1-output> predictions1.json
mv <version-2-output> predictions2.json
python main.py
```

Then, it writes the blended output to `final_retrieve.json`.

## Dependencies

Run the following code under the `esun-ai-cup-2024` folder to install required dependencies:

```bash
   pip install -r requirements.txt
```
