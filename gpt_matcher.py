from openai import OpenAI

client = OpenAI()
MODEL = "gpt-4.1-nano"

def build_catalog(price_list, complex_list):
    lines = []
    for i in price_list:
        lines.append(i.name)
    for i in complex_list:
        if hasattr(i,"composition") and i.composition:
            lines.append(f"{i.name}:")
            for t in i.composition.split(","):
                lines.append(f"- {t.strip()}")
        else:
            lines.append(i.name)
    return "\n".join(lines)

def call_gpt(text, catalog):
    r = client.responses.create(
        model=MODEL,
        temperature=0,
        max_output_tokens=300,
        input=[
            {"role":"system","content":"Ты ассистент лаборатории"},
            {"role":"user","content":text+"\n"+catalog}
        ]
    )
    return r.output_text, r.usage

def gpt_match(text, price_list, complex_list):
    catalog = build_catalog(price_list, complex_list)
    return call_gpt(text, catalog)
