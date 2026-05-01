import json, os
FILE="cleaner_memory.json"

def load():
    return json.load(open(FILE)) if os.path.exists(FILE) else {}

def save(m):
    json.dump(m,open(FILE,"w"),ensure_ascii=False,indent=2)

def clean(text):
    m=load()
    res=[]
    for k,v in m.items():
        if k in text.lower():
            res.append(v)
    return res

def learn(text,final):
    m=load()
    for w in text.split():
        if len(w)>4:
            m[w]=final[0] if final else w
    save(m)
