"""Build page-bounded ~120-word passages from KEPT pages of ACCEPTED docs; check exact/near duplicates.
Usage: build_passages.py docs_summary.json pdf_dir out.jsonl page_scores1.jsonl ..."""
import sys, json, re, unicodedata, hashlib, os, collections, pypdf
summ=json.load(open(sys.argv[1])); pdfdir=sys.argv[2]; out=sys.argv[3]
keep=collections.defaultdict(set)
for f in sys.argv[4:]:
    for l in open(f,encoding='utf-8'):
        r=json.loads(l)
        if r['keep']: keep[r['sha']].add(r['page'])
acc=[d for d in summ if d['status']=='ACCEPT']
P=120; n=0; seen={}; dup=0; shingles=collections.Counter(); recs=[]
def key(t): return re.sub(r'[^ء-ي]','',t)
with open(out,'w',encoding='utf-8') as o:
    for d in acc:
        r=pypdf.PdfReader(os.path.join(pdfdir,d['file']))
        for pg in sorted(keep[d['sha']]):
            t=unicodedata.normalize('NFKC',r.pages[pg-1].extract_text() or '').replace('ـ','')
            w=t.split(); k=max(1,round(len(w)/P)); size=-(-len(w)//k)
            for j in range(k):
                seg=' '.join(w[j*size:(j+1)*size])
                if not seg: continue
                h=hashlib.md5(key(seg).encode()).hexdigest()
                isdup = h in seen
                if isdup: dup+=1
                seen.setdefault(h,(d['file'],pg))
                n+=1
                o.write(json.dumps(dict(passage_id=f"{d['file'][6:-4]}_p{pg:04d}_{j}",file=d['file'],pdf_page=pg,words=len(seg.split()),exact_dup_of=None if not isdup else seen[h],text=seg),ensure_ascii=False)+'\n')
print('passages',n,'exact_dups',dup)
