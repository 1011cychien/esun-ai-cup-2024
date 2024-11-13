## PreRequest

- pip install required packages
- cd to folder *esun-ai-cup-2024*
- The reference folder is in *./data*
- The file *questions_example.json* is in *./data/dataset/preliminary*
- The file *ground_truths_example.json* is in *./data/dataset/preliminary*
- The file *questions_preliminary.json* is in *./data/dataset/realTest*
- The file *llama_finance2.json* is in *./preprocess2/data/llamaparse*


## How to run

```bash
python ./model2/runfaq.py
python ./model2/runfinance.py
```

## expected result

There's a file *./model2/faqonly.json* and a file *./model2/financeonly.json*