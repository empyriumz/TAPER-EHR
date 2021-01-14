import re
import pickle 

def code_map(map = 'icd'):
    file_path = map + "_map.txt"
    text = str(open(file_path, 'rb').read())
    text = text.split('\\n')
    text = text[3:]
    text = [t.strip(' ') for t in text]
    text = [re.sub(' +', ' ', t) for t in text]
    name_icd = {}
    tok_icd = {}
    first = True
    key = -1
    for t in text:
        if len(t) == 0:
            first = True
            key += 1
            continue
        else:
            t = t.split(' ')
            if first:               
                name_icd[key] = ' '.join(t[1:])
                tok_icd[key] = []
                first = False
            else:
                if map == 'proc':
                    t = [tx[1:] if tx.startswith('0') else tx for tx in t]
                # for icd codes we don't have to do this, because the original data is already string
                tok_icd[key].extend(t)

    t = {}
    for k, v in tok_icd.items():
        for vv in v:
            t[vv] = int(k)
    tok_icd = t

    last_key = list(tok_icd.keys())[-1]
    tok_icd[last_key[:-1]] = tok_icd[last_key]
    del tok_icd[last_key]
    
    fil_name = map + '-name.pkl'
    with open(fil_name, 'wb') as handle:
        pickle.dump(name_icd, handle, protocol=pickle.HIGHEST_PROTOCOL)

    fil_name = map + '-map.pkl'
    with open(fil_name, 'wb') as handle:
        pickle.dump(tok_icd, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
code_map('icd')
code_map('proc')
