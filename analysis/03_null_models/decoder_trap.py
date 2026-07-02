"""
THE TRAP every false solution falls into.
Build a 'decoder': map Voynich glyphs -> Latin letters by frequency rank, then
score the output by how Latin-like its character trigrams are (model trained on
real Cicero). Run it on (a) REAL Voynich and (b) MEANINGLESS null text.
If both score about the same as each other -- and not far below real Latin --
then 'it comes out looking like Latin' proves NOTHING. That is the whole game.
"""
import json, re, math, random
from collections import Counter, defaultdict
random.seed(1989)

C=json.load(open('data/corpus/corpus.json'))
voy=[w for f in C['folios'].values() for ln in f for w in ln['words'] if '?' not in w]

# Latin trigram log-prob model (character level, with add-1 smoothing)
lat=re.sub(r'[^a-z]',' ',open('data/controls/latin.txt',encoding='utf-8',errors='ignore').read().lower())
def trigram_model(text):
    tri=Counter(); bi=Counter()
    for i in range(len(text)-2):
        tri[text[i:i+3]]+=1; bi[text[i:i+2]]+=1
    V=26
    def lp(a,b,c):
        return math.log((tri[a+b+c]+1)/(bi[a+b]+V))
    return lp
LP=trigram_model(lat)
def latin_score(text):
    text=re.sub(r'[^a-z]',' ',text.lower())
    s=n=0
    for i in range(len(text)-2):
        s+=LP(text[i],text[i+1],text[i+2]); n+=1
    return s/max(1,n)   # avg log-prob per trigram; higher (less negative)=more Latin-like

# baseline: how Latin scores itself, and how random letters score
real_latin_score = latin_score(lat[:20000])
rand_text=''.join(random.choice('abcdefghijklmnopqrstuvwxyz ') for _ in range(20000))
random_score = latin_score(rand_text)

# greedy frequency-rank substitution: kth most common Voynich glyph -> kth most common Latin letter
def freq_map_decode(words):
    text=' '.join(words)
    vfreq=[g for g,_ in Counter(c for c in text if c!=' ').most_common()]
    lfreq=[g for g,_ in Counter(c for c in lat if c.isalpha()).most_common()]
    m={v:lfreq[i] if i<len(lfreq) else 'x' for i,v in enumerate(vfreq)}
    m[' ']=' '
    return ''.join(m.get(c,' ') for c in text)

# null text: order-2 char Markov trained on Voynich (meaningless by construction)
def char_markov(ws,k=2,n=8000):
    md=defaultdict(Counter)
    for w in ws:
        s='^'*k+w+'$'
        for i in range(k,len(s)): md[s[i-k:i]][s[i]]+=1
    out=[]
    for _ in range(n):
        s='^'*k
        while True:
            nx=md[s[-k:]]
            if not nx: break
            ch=random.choices(list(nx),weights=list(nx.values()))[0]
            if ch=='$': break
            s+=ch
        out.append(s.strip('^'))
    return out
null_words=char_markov(voy)

real_dec   = freq_map_decode(voy[:8000])
null_dec   = freq_map_decode(null_words)

print("="*70)
print("DOES A FREQUENCY-DECODER TELL REAL VOYNICH FROM GIBBERISH?")
print("="*70)
print(f"  Latin-likeness score (avg log-prob/trigram; higher = more Latin):")
print(f"    real Latin (ceiling) : {real_latin_score:7.3f}")
print(f"    random letters (floor): {random_score:7.3f}")
print(f"    -------------------------------------------------")
print(f"    REAL Voynich decoded : {latin_score(real_dec):7.3f}")
print(f"    NULL gibberish decoded:{latin_score(null_dec):7.3f}")
print()
print(f"  real Voynich sample : {real_dec[:80]!r}")
print(f"  null gibberish sample: {null_dec[:80]!r}")
print()
gap_real = latin_score(real_dec)-random_score
gap_null = latin_score(null_dec)-random_score
print(f"  lift over random:  real={gap_real:+.3f}   null={gap_null:+.3f}")
print("  --> If real and null land in the same neighborhood, a decoding that")
print("      'reads a bit like Latin' is the EXPECTED output of frequency matching")
print("      applied to ANY language-shaped token stream. It is not evidence.")
