"""Match-balanced categorical scores and validation-only temperature fitting."""
import numpy as np
from .features import probabilities


def match_weights(groups):
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1. / counts[inverse]
    return weights / weights.sum()


def per_sample(prob, labels):
    p = np.asarray(prob, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 4 or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1):
        raise ValueError('Expected finite normalized four-candidate probabilities')
    y = np.asarray(labels)
    ties = np.isclose(p, p.max(1, keepdims=True), rtol=0, atol=1e-12)
    accuracy = ties[np.arange(len(p)), y] / ties.sum(1)
    target = np.eye(4)[y]
    return {'accuracy': accuracy, 'argmax_accuracy': (p.argmax(1) == y).astype(float),
            'nll': -np.log(np.maximum(p[np.arange(len(p)), y], 1e-15)),
            'brier': np.square(p-target).sum(1), 'confidence': p.max(1),
            'entropy': -(p*np.log(np.maximum(p, 1e-300))).sum(1)}


def metrics(prob, labels, groups):
    scores = per_sample(prob, labels)
    weights = match_weights(groups)
    result = {k: float(weights @ value) for k, value in scores.items()}
    bins = []
    ece = 0.
    indices = np.minimum((scores['confidence']*10).astype(int), 9)
    for index in range(10):
        mask = indices == index
        mass = float(weights[mask].sum())
        confidence = float(weights[mask] @ scores['confidence'][mask] / mass) if mass else None
        accuracy = float(weights[mask] @ scores['accuracy'][mask] / mass) if mass else None
        if mass:
            ece += mass * abs(confidence-accuracy)
        bins.append({'lower': index/10, 'upper': (index+1)/10, 'mass': mass,
                     'confidence': confidence, 'accuracy': accuracy, 'samples': int(mask.sum())})
    result.update(ece=ece, reliability=bins, samples=len(labels), matches=len(np.unique(groups)))
    return result


def fit_temperature(logits, labels, groups):
    # Deterministic one-dimensional search; validation data only. Include T=1.
    weights = match_weights(groups)
    best = (float('inf'), 1.)
    lower, upper = -3., 3.
    for _ in range(4):
        for log_temp in np.unique(np.r_[np.linspace(lower, upper, 81), 0.]):
            temp = float(np.exp(log_temp))
            p = probabilities(logits, temp)
            loss = float(weights @ -np.log(np.maximum(p[np.arange(len(p)), labels], 1e-15)))
            if loss < best[0]:
                best = loss, temp
        center = np.log(best[1]); width = (upper-lower)/40
        lower, upper = center-width, center+width
    return best[1]


def paired_nll_interval(prob, reference, labels, groups, seed=1729, repeats=1000):
    difference = per_sample(prob, labels)['nll'] - per_sample(reference, labels)['nll']
    means = np.array([difference[groups == g].mean() for g in np.unique(groups)])
    rng = np.random.default_rng(seed)
    samples = means[rng.integers(len(means), size=(repeats, len(means)))].mean(1)
    return {'mean': float(means.mean()), 'ci95': list(map(float, np.quantile(samples, [.025, .975]))),
            'unit': 'match', 'replicates': repeats}
