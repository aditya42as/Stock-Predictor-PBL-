from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def score_texts(texts):
    results = []
    for t in texts:
        if not isinstance(t, str):
            t = str(t)
        scores = analyzer.polarity_scores(t)
        results.append({"text": t, **scores})
    return results

def aggregate_scores(scores, weight_recency=True):
    if not scores:
        return {"compound_mean": 0, "pos_mean": 0, "neg_mean": 0, "label": "neutral"}

    compounds = [s["compound"] for s in scores]
    pos = [s["pos"] for s in scores]
    neg = [s["neg"] for s in scores]

    if weight_recency:
        n = len(scores)
        weights = [(i + 1) / n for i in range(n)]

        def w_avg(arr):
            return sum(a * w for a, w in zip(arr, weights)) / sum(weights)

        c = w_avg(compounds)
        p = w_avg(pos)
        n_ = w_avg(neg)
    else:
        c = sum(compounds) / len(compounds)
        p = sum(pos) / len(pos)
        n_ = sum(neg) / len(neg)

    if c >= 0.05:
        label = "positive"
    elif c <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return {
        "compound_mean": float(c),
        "pos_mean": float(p),
        "neg_mean": float(n_),
        "label": label
    }
