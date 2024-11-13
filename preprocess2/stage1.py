# bring in our LLAMA_CLOUD_API_KEY
from dotenv import load_dotenv
import json
load_dotenv()

# bring in deps
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

from pathlib import Path

# set up parser
parser = LlamaParse(
    result_type="text",  # "markdown" and "text" are available
    language="ch_tra"
)

financePath = Path('data') / 'reference' / 'finance'

# use SimpleDirectoryReader to parse our file

file_extractor = {".pdf": parser}

fintechfinal = Path('preprocess2')
llamaparse = fintechfinal / 'data' / 'llamaparse'

def save_i(i):
    documents = SimpleDirectoryReader(input_files=[financePath / f'{i}.pdf'], file_extractor=file_extractor).load_data()
    
    try:
        with open(llamaparse / 'llama_finance.json', 'r', encoding='utf-8') as fff:
            llama_finance = json.load(fff)
    except:
        llama_finance = {}
    llama_finance[str(i)] = ''.join([x.text for x in documents]).replace(' ', '').replace(',', '')
    with open(llamaparse / 'llama_finance.json', 'w', encoding='utf-8') as fff:
        json.dump(llama_finance, fff, ensure_ascii=False, indent=4)

from tqdm import tqdm

s,t = input("start, end = ").split(',')
s = int(s)
t = int(t)

for i in tqdm(range(s, t)):
    save_i(i)
