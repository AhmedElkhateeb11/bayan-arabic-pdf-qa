"""Doc-level decision on page scores. Accept doc if kept-page damaged-token rate <= 3% and word order is not reversed
(reversed collocations <= 15% of all collocation hits). Passages = page-bounded ~120-word chunks of kept pages.
Usage: summarize2.py pages1.jsonl [pages2.jsonl ...]  -> prints table + writes docs_summary.json next to first file"""
import sys, json, collections, os
P=120
rows=[json.loads(l) for f in sys.argv[1:] for l in open(f,encoding='utf-8')]
by=collections.OrderedDict()
for r in rows: by.setdefault(r['sha'],[]).append(r)
res=[]; tot=collections.Counter()
for sha,ps in by.items():
    ps=list({p['page']:p for p in ps}.values())
    keep=[p for p in ps if p['keep']]
    kw=sum(p['words'] for p in keep); pas=sum(max(1,round(p['words']/P)) for p in keep)
    toks=sum(p['ar_tokens'] for p in keep); bad=sum(round((p['oov'] or 0)*p['ar_tokens']) for p in keep)
    fw=sum(p.get('fwd',0) for p in ps); rv=sum(p.get('rev',0) for p in ps)
    oovk=bad/max(1,toks); revr=rv/max(1,fw+rv)
    status='ACCEPT' if (keep and oovk<=0.03 and revr<=0.15) else 'REJECT'
    d=dict(file=ps[0]['file'],sha=sha,pages=len(ps),kept=len(keep),words=kw,passages=pas,oov_kept=round(oovk,4),rev_ratio=round(revr,3),status=status)
    res.append(d)
    if status=='ACCEPT': tot.update(files=1,pages=len(ps),kept=len(keep),words=kw,passages=pas)
for d in res: print(f"{d['status']:6s} {d['file'][:44]:44s} pg {d['pages']:4d} kept {d['kept']:4d} words {d['words']:7d} pass {d['passages']:5d} oov {d['oov_kept']:.2%} rev {d['rev_ratio']:.2f}")
print('ACCEPTED TOTAL',dict(tot))
json.dump(res,open(os.path.join(os.path.dirname(sys.argv[1]),'docs_summary.json'),'w'),ensure_ascii=False,indent=1)
