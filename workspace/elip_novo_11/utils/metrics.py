import math

def recall_at_k(ranks, k):
    hits = sum(1 for r in ranks if r <= k)
    return hits / len(ranks)

def mrr(ranks):
    return sum(1.0/r for r in ranks) / len(ranks)

def ndcg_at_10(ranks):
    def dcg(rank):
        if rank > 10:
            return 0.0
        return 1.0 / math.log2(rank + 1)
    return sum(dcg(r) for r in ranks) / len(ranks)
