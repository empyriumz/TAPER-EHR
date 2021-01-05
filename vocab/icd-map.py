import re
import pickle 

def code_map(map = 'icd'):
    file_path = map + "_map.txt"
    text = str(open(file_path, 'rb').read())
    text = text.split('\\n')
    text = text[3:]
    text = [t.strip(' ') for t in text]
    text = [re.sub(' +', ' ', t) for t in text]

    name_icd9 = {}
    tok_icd9 = {}
    first = True
    key = None
    for i, t in enumerate(text):
        if len(t) == 0:
            t = text[i+1]
            t = t.split(' ')
            key = int(t[0])
            name_icd9[key] = ' '.join(t[1:])
            tok_icd9[key] = []
            first = True
        if (not first):
            t = t.split(' ')
            # exception for procedure codes start with 0
            if map == 'proc':
                t = [tx[1:] if tx.startswith('0') else tx for tx in t]
            # for icd codes we don't have to do this, because the original data is already string
            tok_icd9[key].extend(t)
        else:
            first = False
            
    t = {}
    for k, v in tok_icd9.items():
        for vv in v:
            t[vv] = int(k)
    tok_icd9 = t

    # incremental tokens 1, 2, 3..
    t = {}
    for k, v in name_icd9.items():
        if (k not in t.keys()):
            t[k] = len(t)
            
    # retokenize
    for k, v in tok_icd9.items():
        tok_icd9[k] = t[v]

    fil_name = map + '-map.pkl'
    with open(fil_name, 'wb') as handle:
        pickle.dump(tok_icd9, handle, protocol=pickle.HIGHEST_PROTOCOL)

code_map('icd')
code_map('proc')
