"""Compare pypdf words with PyMuPDF words clipped to the visible page. Ratio >> 1 means pypdf is reading
text that is outside the page box or hidden. Also counts repeated pages (same Arabic letters).
Usage: hiddencheck.py file.pdf ..."""
import sys, re, unicodedata, hashlib, pypdf, pymupdf
for f in sys.argv[1:]:
    r=pypdf.PdfReader(f); d=pymupdf.open(f)
    a=b=0; hs={}; rep=0; inv=0; tot=0
    for i in range(len(r.pages)):
        t=unicodedata.normalize('NFKC',r.pages[i].extract_text() or ''); a+=len(t.split())
        b+=len(d[i].get_text(clip=d[i].rect).split())
        k=hashlib.md5(re.sub(r'[^ء-ي]','',t).encode()).hexdigest()
        if len(t)>200:
            if k in hs: rep+=1
            hs[k]=1
        for s in d[i].get_texttrace():
            n=len(s['chars']); tot+=n
            if s.get('type')==3 or s.get('opacity',1)==0: inv+=n
    print(f"{f.split('/')[-1][:40]:40s} pypdf/pymupdf words {a/max(1,b):.2f}  repeated_pages {rep}  invisible_chars {inv/max(1,tot):.1%}",flush=True)
