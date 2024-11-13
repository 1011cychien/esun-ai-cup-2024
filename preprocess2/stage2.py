from pathlib import Path
import json
import pandas as pd
import re

fintechfinal = Path('preprocess2')

llamaparse = fintechfinal / 'data' / 'llamaparse'

with open(llamaparse / 'llama_finance.json', 'r', encoding='utf-8') as fff:
    llama_finance = json.load(fff)

def remove_long_numeric_substrings_with_dot(text):
    # This pattern matches substrings with at least one dot, only digits and dots, and length >= 3
    pattern = r'(?=[\d.]{3,})(\d+(\.\d+)+)'
    result = re.sub(pattern, '', text)
    return result

def remove_large_numbers(text):
    # This regex matches numbers, both integers and decimals
    pattern = r'\d+(\.\d+)?'
    
    # Function to check if the matched number is 1000 or greater
    def replace_if_large(match):
        number = float(match.group())
        return '' if number >= 2300 else match.group()

    result = re.sub(pattern, replace_if_large, text)
    return result

def remove_consecutive_dashes(text, min_count=2):
    result = re.sub(rf'-{{{min_count},}}', '', text)
    return result

def replace_consecutive_newlines(text):
    # Replace two or more consecutive newlines with a single newline
    result = re.sub(r'\n{2,}', '\n', text)
    return result

def allinone(text):
    text = remove_long_numeric_substrings_with_dot(text)
    text = remove_large_numbers(text)
    text = text.replace('()', '')
    
    text = remove_consecutive_dashes(text)
    text = replace_consecutive_newlines(text)
    text = text.replace('$', '')
    text = text.replace('%', '')
    
    text = remove_long_numeric_substrings_with_dot(text)
    text = remove_large_numbers(text)
    return text

llama_finance2 = {i: allinone(llama_finance[i]) for i in llama_finance}

with open(llamaparse / 'llama_finance2.json', 'w', encoding='utf-8') as asd:
    json.dump(llama_finance2, asd, ensure_ascii=False, indent=4)