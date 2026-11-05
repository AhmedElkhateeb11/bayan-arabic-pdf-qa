"""Page-level screening v2 (pypdf, no OCR). Text is NFKC-normalised first (presentation forms -> letters),
tatweel removed. Adds a word-order check: common Arabic collocations seen forward vs reversed.
Usage: python3 pagescore2.py out.jsonl file1.pdf ..."""
import sys, re, json, hashlib, os, unicodedata, pypdf
from wordfreq import zipf_frequency as z
DIA=re.compile(r'[ً-ْٰ]'); ARW=re.compile(r'^[ء-ي]+$')
AR=re.compile(r'[ء-ي]'); LAT=re.compile(r'[A-Za-z]')
SWAP=re.compile(r'(?:^|\s)(?:اال|األ|اإل|اآل)')
BIG=["منظمة الصحة","الصحة العالمية","على سبيل","سبيل المثال","في حين","بالاضافة الى","من خلال","على الاقل","في مجال","حقوق الانسان",
"الملكية الفكرية","التنمية المستدامة","الامم المتحدة","بما في","في ذلك","الى ان","على ان","فضلا عن","من اجل","على الرغم","الرغم من",
"لا سيما","اكثر من","يمكن ان","ينبغي ان","يجب ان","في هذا","هذا الصدد","العلامات التجارية","البلدان النامية","شرق المتوسط",
"الدول الاعضاء","الذكاء الاصطناعي","على المستوى","المستوى الوطني","في اطار","الى جانب","فيما يتعلق","الرعاية الصحية","الخدمات الصحية",
"في المائة","ما اذا","اي شخص","عدد من","العديد من","بشكل خاص","في العالم","البلدان المنخفضة","الدخل المنخفض","النظم الصحية"]
REV=[" ".join(b.split()[::-1]) for b in BIG]
def norm(t):
    t=unicodedata.normalize('NFKC',t).replace('ـ','')
    return t
def simp(t): return re.sub('[أإآ]','ا',DIA.sub('',t))
cache={}
def known(w):
    if w not in cache: cache[w]=z(w,'ar')>0
    return cache[w]
out=open(sys.argv[1],'a',encoding='utf-8')
for f in sys.argv[2:]:
    try: r=pypdf.PdfReader(f)
    except Exception as e: print('FAIL',f,e); continue
    h=hashlib.sha256(open(f,'rb').read()).hexdigest()
    for i,p in enumerate(r.pages):
        try: t=norm(p.extract_text() or '')
        except Exception: t=''
        toks=[DIA.sub('',x).strip('.,:;()«»"،؛؟!-–[]{}/\'*•') for x in t.split()]
        ar=[w for w in toks if ARW.match(w)]
        n=len(ar); oov=sum(not known(w) for w in ar)
        nch=len(re.sub(r'[\s\d\W_]','',t)) or 1
        s=' '+re.sub(r'\s+',' ',simp(t))+' '
        fw=sum(s.count(' '+b+' ') for b in BIG); rv=sum(s.count(' '+b+' ') for b in REV)
        rec=dict(file=os.path.basename(f), sha=h, page=i+1, words=len(t.split()), ar_tokens=n,
                 oov=round(oov/n,4) if n else None, ar_share=round(len(AR.findall(t))/nch,3),
                 lat_share=round(len(LAT.findall(t))/nch,3), fffd=t.count('�'), swaps=len(SWAP.findall(t)), fwd=fw, rev=rv)
        rec['reversed']= rv>=2 and rv>fw
        rec['keep']= bool(n>=40 and rec['ar_share']>=0.6 and rec['oov']<=0.05 and rec['fffd']<=3 and rec['swaps']<=2 and not rec['reversed'])
        out.write(json.dumps(rec,ensure_ascii=False)+'\n')
    print('done',os.path.basename(f),len(r.pages),flush=True)
