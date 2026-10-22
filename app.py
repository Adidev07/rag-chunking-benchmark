"""Measure how chunk size and overlap affect retrieval quality and context volume."""
from kit import *
TITLE='RAG Chunking and Retrieval Benchmark'
LIVE=False

def benchmark(documents,queries,size,overlap):
    rows=chunks(documents,size,overlap);rr=[];recall=[];wordcounts=[];examples=[]
    for q in queries:
        hits=bm25(q['query'],rows,3);ranks=[i for i,r in enumerate(hits,1) if r['source']==q['source']];rr.append(1/min(ranks) if ranks else 0);recall.append(bool(ranks));wordcounts.append(sum(len(h['text'].split()) for h in hits));examples.append(dict(query=q['query'],expected=q['source'],hits=hits))
    return dict(size=size,overlap=overlap,chunks=len(rows),recall_at_3=float(np.mean(recall)),mrr=float(np.mean(rr)),mean_context_words=float(np.mean(wordcounts)),queries=examples)
def run(d,a):
    configs=[benchmark(d['documents'],d['queries'],size,overlap) for size in (12,24,48) for overlap in (0,size//4)]
    return dict(configurations=configs,selection_note='Development comparison only; no held-out final test or generated answer claims.'),None

if __name__=="__main__":main(run)
