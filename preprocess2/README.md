## PreRequest

- pip install required packages
- cd to folder *esun-ai-cup-2024*
- The reference folder is in *.\data*
- go to https://docs.cloud.llamaindex.ai/llamacloud/getting_started/api_key to get an api key \
write the following in the file *.\\.env*
```
LLAMA_CLOUD_API_KEY={your key}
```


## How to run

Since llamaparse has limit for each day, so it takes 5 days for the preprocess2.

### Day 1

```bash
python .\preprocess2\stage1.py
(then manually input "0, 207")
```

### Day 2

```bash
python .\preprocess2\stage1.py
(then manually input "207, 414")
```

### Day 3

```bash
python .\preprocess2\stage1.py
(then manually input "414, 621")
```

### Day 4

```bash
python .\preprocess2\stage1.py
(then manually input "621, 828")
```

### Day 5

```bash
python .\preprocess2\stage1.py
(then manually input "828, 1035")
```

```bash
python .\preprocess2\stage2.py
```

## expected result

There's a file *.\preprocess2\data\llamaparse\llama_finance2.json*