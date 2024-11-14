import json
import random

results = []

# Load predictions from two sources
with open('predictions1.json') as f:
    predictions1 = json.load(f)['answers']

with open('predictions2_finance.json') as f:
    predictions2_finance = json.load(f)['answers']

with open('predictions2_faq.json') as f:
    predictions2_faq = json.load(f)['answers']

# Take insurance from model 1
results = predictions1[0:300]

# Take finance result from model 1 with 20% probability and 80% from model 2
for i in range(300):
    rand_num = random.randint(0, 4)
    if rand_num == 0:
        results.append(predicsions1[300 + i])
    else
        results.append(predictions2_finance[i])

# Take faq result from model 1 with 50% probability and 50% from model 2
for i in range(300):
    rand_num = random.randint(0, 1)
    if rand_num == 0:
        results.append(predicsions1[600 + i])
    else
        results.append(predictions2_faq[i])

with open('final_retrieve.json', 'w') as f:
    json.dump({"answers": results}, f, indent=4)
