"""Final journal figures (Figures 1-4) from bontaz_round5.csv and msa_human_decisions.csv."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from sklearn.metrics import roc_auc_score, roc_curve

OK_C, NOK_C, ALT_C, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#8a8984'
INK, INK2 = '#0b0b0b', '#52514e'
TAU = 8.6561
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, '..', 'data'); OUT = os.path.join(HERE, '..', 'figures')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': '#b5b4ae', 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': 0.8, 'legend.frameon': False, 'savefig.dpi': 300})
B = pd.read_csv(os.path.join(D, 'model_scores.csv')); R = B[B.model == 'recipe'].copy()
H = pd.read_csv(os.path.join(D, 'operator_decisions.csv')); H['p'] = H.part.str.extract(r'(\d+)').astype(int)
BORDER = set(range(16, 21)) | set(range(31, 36)) | set(range(36, 41)) | set(range(46, 51))
rng = np.random.default_rng(150360)

def save(fig, name):
    for ext in ('png', 'pdf'): fig.savefig(os.path.join(OUT, f'{name}.{ext}'), bbox_inches='tight')
    plt.close(fig)

def panel(ax, s): ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=11, fontweight='bold', ha='right', va='bottom')

# ------------------------------------------------------------------ Figure 1
def box(ax, x, y, w, h, title, body, edge, face):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.2', lw=1.2, ec=edge, fc=face))
    ax.text(x + w / 2, y + h - 3.2, title, ha='center', va='top', fontsize=9.5, fontweight='bold', color=INK)
    ax.text(x + w / 2, y + h - 8.2, body, ha='center', va='top', fontsize=8.2, color=INK2, linespacing=1.45)
def arrow(ax, a, b):
    ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='-|>', color='#6b6a65', lw=1.1, shrinkA=0, shrinkB=0))

fig = plt.figure(figsize=(7.2, 3.4))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 58); ax.axis('off')
W, Hh = 29, 21
N, BL, OR = ('#ffffff', '#5b5a56'), ('#eef4fc', OK_C), ('#fdf0ea', NOK_C)
for x, t in ((3, 'Data'), (36, 'Model'), (69, 'Evaluation')): ax.text(x, 53, t, fontsize=10, fontweight='bold', color=INK)
box(ax, 3, 27, W, Hh, 'Photo archive', '1,126 photographs\n923 OK · 203 NOK', N[1], N[0])
box(ax, 3, 2, W, Hh, 'Cleaning', 'copies removed\nstudy parts kept out\n883 photographs', N[1], N[0])
box(ax, 36, 27, W, Hh, 'Fit on good parts', 'PatchCore-type detector\n343 OK photographs', BL[1], BL[0])
box(ax, 36, 2, W, Hh, 'Fix the threshold', '96th percentile of\n124 calibration scores\nprotocol written first', BL[1], BL[0])
box(ax, 69, 27, W, Hh, 'Held-out test', '214 OK · 105 NOK\nAUROC, α, β', N[1], N[0])
box(ax, 69, 2, W, Hh, 'AIAG gauge study', '50 parts · 3 operators\n× 3 trials\nmodel: one photo per part', OR[1], OR[0])
arrow(ax, (17.5, 27), (17.5, 23)); arrow(ax, (50.5, 27), (50.5, 23))
arrow(ax, (32, 12.5), (36, 37.5)); arrow(ax, (65, 12.5), (69, 37.5)); arrow(ax, (65, 12.5), (69, 12.5))
save(fig, 'fig1_protocol')

# ------------------------------------------------------------------ Figure 2
T = R[R.set.isin(['test_ok', 'test_nok'])]
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.32))
bins = np.linspace(T.score.min() - .02, T.score.max() + .02, 36)
a.hist(T[T.y == 0].score, bins, color=OK_C, alpha=.75, label='OK (214)', ec='white', lw=.6)
a.hist(T[T.y == 1].score, bins, color=NOK_C, alpha=.75, label='NOK (105)', ec='white', lw=.6)
a.axvline(TAU, color=INK, lw=1.2, ls='--'); a.text(TAU, a.get_ylim()[1] * .97, '  threshold τ', va='top', fontsize=8, color=INK)
a.set_xlabel('Anomaly score'); a.set_ylabel('Photographs'); a.legend(loc='upper right', bbox_to_anchor=(1, .86)); panel(a, 'a')
fpr, tpr, _ = roc_curve(T.y, T.score); auc = roc_auc_score(T.y, T.score)
b.plot(fpr, tpr, color=INK, lw=1.8); b.plot([0, 1], [0, 1], color='#c9c8c2', lw=.8, ls=':')
fa, tp = (T[T.y == 0].score > TAU).mean(), (T[T.y == 1].score > TAU).mean()
b.plot(fa, tp, 'o', ms=8, mfc=NOK_C, mec='white', mew=1.5, zorder=3)
b.annotate(f'threshold\nα = {fa:.1%}, β = {1 - tp:.1%}', (fa, tp), xytext=(.25, .55), fontsize=8, color=INK,
           arrowprops=dict(arrowstyle='-', color=GREY, lw=.8))
b.text(.95, .06, f'AUROC = {auc:.3f}', ha='right', fontsize=9, color=INK)
b.set_xlabel('False-alarm rate'); b.set_ylabel('Detection rate (1 − β)'); b.set_xlim(-.01, 1); b.set_ylim(0, 1.01); panel(b, 'b')
save(fig, 'fig2_test')

# ------------------------------------------------------------------ Figure 3
nok = R[R.set == 'msa_nok'].set_index('part').score; okt = R[R.set == 'matched_ok'].set_index('part').score
oks = R[R.set == 'msa_ok'].set_index('part').score
fig, (a, b) = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True, gridspec_kw=dict(height_ratios=[1.6, 1], hspace=.18))
for p in oks.index: a.plot([p, p], [oks[p], okt[p]], color='#c9c8c2', lw=1, zorder=1)
def pts(s, c, m, filled, lab):
    bd = [p in BORDER for p in s.index]
    a.scatter(s.index, s.values, s=34, marker=m, c=[c if filled else 'white' for _ in s], edgecolors=c, linewidths=1.3, zorder=3, label=lab)
    sb = s[[p in BORDER for p in s.index]]
    a.scatter(sb.index, sb.values, s=110, marker=m, facecolors='none', edgecolors=INK, linewidths=.8, zorder=2)
pts(nok, NOK_C, '^', True, 'NOK, taken together'); pts(okt, OK_C, 'o', True, 'OK, taken together (matched set)')
pts(oks, OK_C, 'o', False, 'OK, taken separately (gauge set)')
a.axhline(TAU, color=INK, lw=1.1, ls='--'); a.text(50.6, TAU, 'τ', va='center', fontsize=9, color=INK)
a.scatter([], [], s=110, marker='o', facecolors='none', edgecolors=INK, linewidths=.8, label='borderline part (outer ring)')
a.set_ylabel('Anomaly score'); a.legend(loc='lower center', bbox_to_anchor=(.5, 1.02), fontsize=7.6, ncol=2); panel(a, 'a')
E = H.assign(e=(H.decision != H.truth).astype(int)).groupby('p').agg(e=('e', 'sum'), truth=('truth', 'first'))
b.bar(E.index, E.e, width=.75, color=[NOK_C if t == 'NOK' else OK_C for t in E.truth], edgecolor='white', lw=.5)
b.set_ylabel('Operator errors\n(of 9 decisions)'); b.set_ylim(0, 9); b.set_yticks([0, 3, 6, 9]); b.set_xlabel('Part number')
b.set_xlim(.3, 50.7); b.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]); panel(b, 'b')
b.legend(handles=[Patch(color=NOK_C, label='NOK part'), Patch(color=OK_C, label='OK part')], loc='upper left', fontsize=7.6, ncol=2)
save(fig, 'fig3_gauge_study')

# ------------------------------------------------------------------ Figure 4
S = R.copy()
grp = {'test_ok': ('Held-out OK', '#a9c8ef', 'o', 10), 'test_nok': ('Held-out NOK', '#f4b89f', '^', 10),
       'msa_ok': ('Study OK, taken separately', OK_C, 'o', 30), 'matched_ok': ('Study OK, taken together', ALT_C, 's', 30),
       'msa_nok': ('Study NOK', NOK_C, '^', 34)}
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=.3))
for k, (lab, c, m, s) in grp.items():
    d = S[S.set == k]; a.scatter(d.nn, d.score, s=s, c=c, marker=m, edgecolors='white', linewidths=.4, label=lab, zorder=3 if s > 10 else 2)
a.axhline(TAU, color=INK, lw=1, ls='--'); a.set_xlabel('Distance to nearest training photograph'); a.set_ylabel('Anomaly score')
a.legend(fontsize=7, loc='upper left', handletextpad=.3); panel(a, 'a')
def boot(y, s, n=2000):
    y, s = np.asarray(y), np.asarray(s); i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    v = [roc_auc_score(y[j], s[j]) for j in (np.r_[rng.choice(i0, len(i0)), rng.choice(i1, len(i1))] for _ in range(n))]
    return roc_auc_score(y, s), *np.percentile(v, [2.5, 97.5])
sets = [('Held-out\ntest', ['test_ok', 'test_nok']), ('Gauge\nset', ['msa_ok', 'msa_nok']), ('Matched\nset', ['matched_ok', 'msa_nok'])]
regions = [('Whole image', 'score', INK), ('Part region', 'in1.0', NOK_C), ('Background', 'out1.0', OK_C)]
for i, (lab, ss) in enumerate(sets):
    d = S[S.set.isin(ss)].dropna(subset=['in1.0'])
    for j, (rl, col, c) in enumerate(regions):
        m, lo, hi = boot(d.y, d[col]); x = i + (j - 1) * .22
        b.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt='o', ms=6, color=c, mec='white', mew=1, elinewidth=1.3, capsize=0, label=rl if i == 0 else None)
b.axhline(.5, color='#c9c8c2', lw=.8, ls=':'); b.text(-.42, .5, 'chance', fontsize=7.5, color=INK2, va='bottom', ha='left')
b.set_xticks(range(3)); b.set_xticklabels([s[0] for s in sets]); b.set_ylabel('AUROC'); b.set_ylim(0, 1.03)
b.legend(fontsize=7.5, loc='center', bbox_to_anchor=(.45, .5)); panel(b, 'b')
save(fig, 'fig4_mechanism')
print('done')
