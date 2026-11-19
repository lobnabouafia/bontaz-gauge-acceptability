"""Reproduce the main numbers of the paper from the shared CSV files (CPU, a few seconds).

    python analysis/paper_numbers.py
"""
import os
import numpy as np, pandas as pd
from scipy.stats import beta as _beta
from sklearn.metrics import roc_auc_score

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
TAU = {'recipe': 8.6561, 'standard': 2.0330}          # fixed before scoring (96th percentile of calibration scores)
B = pd.read_csv(os.path.join(D, 'model_scores.csv'))
H = pd.read_csv(os.path.join(D, 'operator_decisions.csv'))
rng = np.random.default_rng(150360)

def cp(k, n):
    lo = 0.0 if k == 0 else _beta.ppf(.025, k, n - k + 1)
    hi = 1.0 if k == n else _beta.ppf(.975, k + 1, n - k)
    return lo, hi

def auc_ci(y, s, n=2000):
    y, s = np.asarray(y), np.asarray(s); i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    b = [roc_auc_score(y[j], s[j]) for j in (np.r_[rng.choice(i0, len(i0)), rng.choice(i1, len(i1))] for _ in range(n))]
    return roc_auc_score(y, s), *np.percentile(b, [2.5, 97.5])

SETS = {'held-out test': ['test_ok', 'test_nok'], 'gauge set': ['msa_ok', 'msa_nok'], 'matched set': ['matched_ok', 'msa_nok']}
print('MODELS (Tables 3 and 5, Sections 5.1-5.4)')
for model in ('recipe', 'standard'):
    print(f'\n  {model} model, threshold {TAU[model]}')
    for name, sets in SETS.items():
        d = B[(B.model == model) & B.set.isin(sets)]
        a = auc_ci(d.y, d.score); fa = int((d[d.y == 0].score > TAU[model]).sum()); ms = int((d[d.y == 1].score <= TAU[model]).sum())
        n0, n1 = int((d.y == 0).sum()), int((d.y == 1).sum()); la, lb = cp(fa, n0), cp(ms, n1)
        print(f'    {name:13s} AUROC {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}] | alpha {fa}/{n0} = {fa / n0:.1%} [{la[0]:.1%}, {la[1]:.1%}]'
              f' | beta {ms}/{n1} = {ms / n1:.1%} [{lb[0]:.1%}, {lb[1]:.1%}] | E {(n0 - fa + n1 - ms) / (n0 + n1):.1%}'
              f' | distance alone {roc_auc_score(d.y, d.nn):.3f}')
    s2 = B[(B.model == model) & (B.set == 'msa_ok')].set_index('part').score
    s1 = B[(B.model == model) & (B.set == 'matched_ok')].set_index('part').score
    print(f'    same decision on both photographs of a good part: {int(((s2 > TAU[model]) == (s1 > TAU[model])).sum())}/25')

print('\nOPERATORS (Tables 3 and 4)')
H['err'] = (H.decision != H.truth).astype(int)
for op, g in H.groupby('operator'):
    ok, nok = g[g.truth == 'OK'], g[g.truth == 'NOK']
    within = g.groupby('part').decision.nunique().eq(1).mean()
    vsref = g.groupby('part').apply(lambda x: (x.decision == x.truth).all()).mean()
    print(f'  operator {op}: alpha {ok.err.mean():.1%} | beta {nok.err.mean():.1%} | E {1 - g.err.mean():.1%} | within {within:.0%} | against reference {vsref:.0%}')
print('  all operators: alpha {:.1%} | beta {:.1%} | E {:.1%}'.format(H[H.truth == 'OK'].err.mean(), H[H.truth == 'NOK'].err.mean(), 1 - H.err.mean()))
print('  errors by trial:', H.groupby('trial').err.sum().tolist())
print(H.groupby(['difficulty', 'truth']).err.agg(['sum', 'count', 'mean']).round(3).to_string())
