from gpt_matcher import gpt_match
from self_learning_cleaner import clean, learn
from suggestions import get_suggestions

def is_confident(lines):
    return not any("уточнения" in l for l in lines) and len(lines)<=5

def process(text, price_list, complex_list):
    c=clean(text)
    if c: text="\n".join(c)

    if len(text)<4:
        return get_suggestions(text, price_list), None

    result,usage=gpt_match(text,price_list,complex_list)
    lines=result.split("\n")

    if is_confident(lines):
        return lines, usage
    else:
        return lines, usage

def confirm(text, final):
    learn(text, final)
