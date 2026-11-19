# =============================================================================
# 17  ROUND-4 CHECKS (Colab, GPU, ~45 min). Same as script 16, but with the FROZEN image pipeline
#     (torchvision Resize, as in script 11), so every diagnostic comes from the code that produced the reported scores.
#     A reproduction check against model_v3/scores_v3.csv stops the run if scores differ by more than 0.005.
#     Additions: part regions of 1.0x, 1.5x and 2.0x the fixed disc radius (stem and tab included in the larger ones),
#     area share of each region (chance baseline for 'strongest anomaly on the part'), the same rate for GOOD images,
#     and a NEGATIVE CONTROL on MVTec AD: the defective test images brightened by 25% (an artificial acquisition shortcut).
#
#  BONTAZ (frozen final model, nothing tuned):
#   [A] EXIF metadata of every photograph (capture time, device, software)
#   [B] archive photographs with serial numbers next to the gauge-set good photographs (t790-t870), per set
#   [C] distance-adjusted AUROC on the held-out test (diagnostic 3, revised): does the score separate the classes
#       at a given distance from the training data?
#   [D] part disc: does its size differ by class? same split with a FIXED radius (defect cannot enlarge it)
#   [E] Figure 5 redrawn: heat maps only, perceptually uniform colour map, outline of the part disc
#  POSITIVE CONTROL on public data (MVTec AD metal_nut and screw; constant acquisition by design):
#   same recipe as the case study (ResNeXt-50 layer2+3, 3x3 pooling, PCA 256, coreset 12,000, seed 150360),
#   80% of train/good -> memory, 20% -> calibration, tau = Q0.96 of calibration; mean and max scores;
#   diagnostics 3 (distance-adjusted AUROC) and 4 (object vs background, object mask from image geometry
#   by thresholding against the background, independent of the defect). Rules written to protocol_round4.json first.
# Outputs: MyDrive/dataset_150360_final/round3/  report_round3.txt, *.csv, r3_fig5_maps.png, r3_public_maps.png
# =============================================================================
import os, io, json, time, math, shutil, hashlib, re, random
import numpy as np, pandas as pd, cv2, torch, torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ExifTags
from scipy import ndimage
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

def _mount():
    try:
        from google.colab import drive
    except ImportError:
        return
    if os.path.ismount('/content/drive') and os.path.isdir('/content/drive/MyDrive'): return
    if os.path.isdir('/content/drive') and os.listdir('/content/drive'):
        shutil.move('/content/drive', f'/content/drive_leftover_{int(time.time())}')
    drive.mount('/content/drive')
_mount()

MY = '/content/drive/MyDrive'
ROOT, FIN = os.path.join(MY, 'dataset_150360_clean'), os.path.join(MY, 'dataset_150360_final')
OLD = os.path.join(MY, 'Dossier sans titre', 'dataset_150360')
OUT = os.path.join(FIN, 'round4'); os.makedirs(OUT, exist_ok=True)
PUB = [p for p in (os.path.join(MY, 'AIAG_STUDY', 'data', 'mvtec'), os.path.join(MY, 'AIAG_STUDY', 'mvtec'), os.path.join(MY, 'mvtec')) if os.path.isdir(p)]
OK_PARTS = set(range(6, 11)) | set(range(26, 31)) | set(range(36, 51))
SEED = 150360; DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
LOG = io.StringIO()
def log(*a): print(*a); print(*a, file=LOG)
IMG = ('.png', '.jpg', '.jpeg', '.bmp')
num = lambda s: int(''.join(c for c in os.path.splitext(os.path.basename(str(s)))[0] if c.isdigit()) or -1)
rng = np.random.default_rng(SEED)
def auc_ci(y, s, B=2000):
    y, s = np.asarray(y), np.asarray(s); i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]; b = []
    for _ in range(B):
        j = np.r_[rng.choice(i0, len(i0)), rng.choice(i1, len(i1))]; b.append(roc_auc_score(y[j], s[j]))
    return roc_auc_score(y, s), *np.percentile(b, [2.5, 97.5])
def adj_auc(y, score, dist, fit_mask):
    """AUROC of the score after removing its linear dependence on distance (line fitted on good photos only)."""
    b = np.polyfit(np.asarray(dist)[fit_mask], np.asarray(score)[fit_mask], 1)
    return auc_ci(y, np.asarray(score) - np.polyval(b, dist))

# ----------------------------------------------------------------------------- shared feature extractor
CFG = json.load(open(os.path.join(ROOT, 'metadata', 'sealed_model_v2.json')))['config']
net = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2).eval().to(DEV)
for p_ in net.parameters(): p_.requires_grad_(False)
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
@torch.no_grad()
def feats(img, size):
    fs = {}
    h1 = net.layer2.register_forward_hook(lambda a, b, o: fs.__setitem__('a', o))
    h2 = net.layer3.register_forward_hook(lambda a, b, o: fs.__setitem__('b', o))
    pil = img if isinstance(img, Image.Image) else Image.open(img)
    x = NORM(transforms.ToTensor()(transforms.Resize((size[0], size[1]))(pil.convert('RGB'))))   # identical to script 11
    net(x.unsqueeze(0).to(DEV)); h1.remove(); h2.remove()
    fa = F.avg_pool2d(fs['a'], 3, 1, 1); fb = F.interpolate(F.avg_pool2d(fs['b'], 3, 1, 1), size=fa.shape[-2:], mode='bilinear', align_corners=False)
    f = torch.cat([fa, fb], 1); return f[0].permute(1, 2, 0).reshape(-1, f.shape[1]).float(), fa.shape[-2:]
class Model:
    def __init__(self, mem, W, mu, size): self.MEM, self.W, self.MU, self.size = mem.to(DEV), W.to(DEV), mu.to(DEV), size
    @torch.no_grad()
    def run(self, path):
        f, (gh, gw) = feats(path, self.size); z = (f - self.MU) @ self.W
        d = torch.cdist(z, self.MEM).min(1).values.reshape(gh, gw)
        return d.cpu().numpy(), z.mean(0).cpu().numpy()
def fit(paths, size, pca=256, nbank=12000, per=500):
    g = torch.Generator().manual_seed(SEED); pool = []
    for p in paths:
        f, _ = feats(p, size); pool.append(f[torch.randperm(f.shape[0], generator=g)[:per].to(DEV)].half())
    P = torch.cat(pool).float(); mu = P.mean(0); C = (P - mu).T @ (P - mu) / (P.shape[0] - 1)
    ev, V = torch.linalg.eigh(C); W = V[:, torch.argsort(ev, descending=True)[:pca]]; Z = (P - mu) @ W; del P
    n = min(nbank, Z.shape[0]); gg = torch.Generator().manual_seed(SEED)
    sel = [int(torch.randint(0, Z.shape[0], (1,), generator=gg))]; dmin = torch.cdist(Z, Z[sel[0]:sel[0] + 1]).squeeze(1)
    for _ in range(n - 1):
        j = int(torch.argmax(dmin)); sel.append(j); dmin = torch.minimum(dmin, torch.cdist(Z, Z[j:j + 1]).squeeze(1))
    return Model(Z[torch.tensor(sel, device=DEV)], W, mu, size)

json.dump(dict(written=time.strftime('%Y-%m-%d %H:%M'), bontaz='frozen final model and frozen image pipeline; diagnostics only',
               negative_control='MVTec defective test images brightened x1.25 (PIL ImageEnhance.Brightness); same model and threshold',
               regions='fixed disc radius = median good-photo radius; scales 1.0, 1.5, 2.0',
               public=dict(categories=['metal_nut', 'screw'], recipe='case-study recipe, input 360x360', split='train/good 80% memory, 20% calibration (seed 150360)',
                           tau='Q0.96 calibration', scores=['mean', 'max'], object_mask='threshold of |grey - border median| (Otsu), largest component, holes filled')),
          open(os.path.join(OUT, 'protocol_round3.json'), 'w'), indent=1)

# ============================================================================= BONTAZ
B = torch.load(os.path.join(FIN, 'model_v3', 'bank_v3.pt'), map_location='cpu')
BM = Model(B['mem'].float(), B['W'].float(), B['mu'].float(), (CFG['img_h'], CFG['img_w']))
TAU = float(json.load(open(os.path.join(FIN, 'model_v3', 'model_v3.json')))['tau'])
K = pd.read_csv(os.path.join(FIN, 'manifest_v3.csv'), dtype=str); K['path'] = [os.path.join(FIN, f) for f in K.file_v3]
matched = sorted(os.path.join(OLD, 'ts_ok', f) for f in os.listdir(os.path.join(OLD, 'ts_ok')) if f.lower().endswith(IMG) and num(f) in OK_PARTS)

# [A] EXIF
def exif(p):
    try:
        ex = Image.open(p).getexif(); sub = ex.get_ifd(0x8769) if hasattr(ex, 'get_ifd') else {}
        return dict(dt_original=str(sub.get(36867) or ''), dt=str(ex.get(306) or ''), make=str(ex.get(271) or ''),
                    model=str(ex.get(272) or ''), software=str(ex.get(305) or ''), n_tags=len(ex) + len(sub))
    except Exception as e:
        return dict(dt_original='', dt='', make='', model='', software='', n_tags=-1)
E = pd.DataFrame([dict(set=r.split_v3, file=r.original_file, **exif(r.path)) for r in K.itertuples()] +
                 [dict(set='matched_ok', file=os.path.basename(p), **exif(p)) for p in matched])
E.to_csv(os.path.join(OUT, 'exif.csv'), index=False)
log(f"[A] EXIF over {len(E)} photographs: any tag {int((E.n_tags > 0).sum())}, capture time {int((E.dt_original != '').sum())}, "
    f"date/time {int((E.dt != '').sum())}, device model {int((E.model != '').sum())}, software {int((E.software != '').sum())}")
if (E.model != '').any(): log("    devices: " + str(E[E.model != ''].groupby(['make', 'model']).size().to_dict()))
if (E.dt_original != '').any(): log("    capture dates by set:\n" + E[E.dt_original != ''].assign(day=E.dt_original.str[:10]).groupby(['set', 'day']).size().to_string())

# [B] serial neighbours of the gauge-set good photographs (archive t810-t846, t590)
K['serial'] = [num(f) if os.path.basename(str(f)).lower().startswith('t') and not os.path.basename(str(f)).lower().startswith('ts') else -1 for f in K.original_file]
nb = K[(K.serial >= 790) & (K.serial <= 870) & (~K.split_v3.str.startswith('msa'))]
log(f"\n[B] development photographs with serial numbers t790-t870 (neighbours of the gauge-set good photographs t810-t846): {len(nb)}")
log("    by set: " + str(nb.split_v3.value_counts().to_dict()))

# [C]-[D] score test + study photographs; distances to training
rows, maps = [], {}
test = K[K.split_v3.isin(['test_ok', 'test_nok', 'msa_ok', 'msa_nok'])]
items = [(r.split_v3, r.path, int(r.label_v3 == 'NOK'), num(r.file_v3) if r.split_v3.startswith('msa') else -1) for r in test.itertuples()] + \
        [('matched_ok', p, 0, num(p)) for p in matched]
def disc(pil, shape, r_fixed=None):
    rgb = np.asarray(pil.convert('RGB')); H0, W0 = rgb.shape[:2]; s = 800 / max(H0, W0)
    small = cv2.resize(rgb, (round(W0 * s), round(H0 * s)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV); h, sat, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = cv2.morphologyEx(((h >= 3) & (h <= 25) & (sat >= 60) & (v >= 60)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    if n < 2: return None, None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); r = 1.6 * math.sqrt(st[i, 4]) / small.shape[0] if r_fixed is None else r_fixed
    gh, gw = shape; yy, xx = np.mgrid[:gh, :gw]; cy, cx = cen[i, 1] / small.shape[0], cen[i, 0] / small.shape[1]
    mk = np.hypot((yy + .5) / gh - cy, ((xx + .5) / gw - cx) * small.shape[1] / small.shape[0]) <= r
    return mk, (cx, cy, r)
for s, p, y, part in items:
    d, g = BM.run(p); pil = Image.open(p); mk, geo = disc(pil, d.shape)
    rows.append(dict(set=s, y=y, part=part, path=p, mean=float(d.mean()), g=g, mk=mk, geo=geo, d=d, pil_size=pil.size))
R = pd.DataFrame(rows)
_ref = pd.read_csv(os.path.join(FIN, 'model_v3', 'scores_v3.csv'))
_ref['path'] = [os.path.join(FIN, f) for f in _ref.file_v3]
_m = R.merge(_ref[['path', 'score']], on='path', how='inner')
_dmax = float((_m['mean'] - _m.score).abs().max())
log(f"reproduction check: {len(_m)} photographs compared with the frozen scores, max |difference| = {_dmax:.5f}")
if _dmax > 0.005: raise SystemExit("scores differ from the frozen run - stop and tell Claude")
G = np.stack([BM.run(p)[1] for p in K[K.split_v3 == 'train_ok'].path])
R['nn'] = [float(np.linalg.norm(G - g, axis=1).min()) for g in R.g]
log("\n[C] distance-adjusted AUROC (score minus its linear trend on distance, fitted on good photographs):")
for nm, ss in (('held-out test', ['test_ok', 'test_nok']), ('gauge set', ['msa_ok', 'msa_nok']), ('matched set', ['matched_ok', 'msa_nok'])):
    D = R[R.set.isin(ss)]
    a = auc_ci(D.y, D['mean']); b = adj_auc(D.y.values, D['mean'].values, D.nn.values, D.y.values == 0); c = roc_auc_score(D.y, D.nn)
    log(f"  {nm:14s} AUROC {a[0]:.3f} | distance alone {c:.3f} | distance-adjusted {b[0]:.3f} [{b[1]:.3f}, {b[2]:.3f}] | "
        f"median distance OK {D[D.y == 0].nn.median():.2f} (IQR {D[D.y == 0].nn.quantile(.25):.2f}-{D[D.y == 0].nn.quantile(.75):.2f}), "
        f"NOK {D[D.y == 1].nn.median():.2f} (IQR {D[D.y == 1].nn.quantile(.25):.2f}-{D[D.y == 1].nn.quantile(.75):.2f})")
R['disc_r'] = [g[2] if g else np.nan for g in R.geo]
rfix = float(R[(R.y == 0) & R.set.isin(['test_ok'])].disc_r.median())
log(f"\n[D] part disc radius (fraction of image height), median by class: test OK {R[(R.set == 'test_ok')].disc_r.median():.3f}, "
    f"test NOK {R[(R.set == 'test_nok')].disc_r.median():.3f} (Mann-Whitney AUROC of radius {roc_auc_score(R[R.set.isin(['test_ok','test_nok'])].dropna(subset=['disc_r']).y, R[R.set.isin(['test_ok','test_nok'])].dropna(subset=['disc_r']).disc_r):.3f})")
for scale in (1.0, 1.5, 2.0):
    for nm, ss in (('held-out test', ['test_ok', 'test_nok']), ('matched set', ['matched_ok', 'msa_nok'])):
        rr = []
        for r in R[R.set.isin(ss)].itertuples():
            mk, _ = disc(Image.open(r.path), r.d.shape, r_fixed=rfix * scale)
            if mk is None or not mk.any() or mk.all(): continue
            rr.append(dict(y=r.y, inside=float(r.d[mk].mean()), outside=float(r.d[~mk].mean()), peak=bool(mk.flat[int(r.d.argmax())]), area=float(mk.mean())))
        Q = pd.DataFrame(rr)
        log(f"  radius {scale:.1f}x {rfix:.3f}, {nm:13s} n={len(Q)}: area share {Q.area.mean():.1%} | part {roc_auc_score(Q.y, Q.inside):.3f} | background {roc_auc_score(Q.y, Q.outside):.3f} | "
            f"strongest patch on part: defects {Q[Q.y == 1].peak.mean():.0%} ({int(Q[Q.y == 1].peak.sum())}/{int((Q.y == 1).sum())}), good {Q[Q.y == 0].peak.mean():.0%} ({int(Q[Q.y == 0].peak.sum())}/{int((Q.y == 0).sum())})")

# [E] Figure 5 redrawn (no photograph): parts 6, 29, 46 (gauge, matched) and defects 1, 16, 31
pick = [('msa_ok', 6), ('msa_ok', 29), ('msa_ok', 46), ('matched_ok', 6), ('matched_ok', 29), ('matched_ok', 46), ('msa_nok', 1), ('msa_nok', 16), ('msa_nok', 31)]
sub = [R[(R.set == s) & (R.part == p)].iloc[0] for s, p in pick]
allv = np.concatenate([r.d.ravel() for r in sub]); LO, HI = np.percentile(allv, 50), np.percentile(allv, 99.7)
fig, axs = plt.subplots(3, 3, figsize=(5.6, 6.8), gridspec_kw=dict(hspace=.25, wspace=.06))
lab = ['OK, occasion 2\n(gauge set)', 'OK, occasion 1\n(matched set)', 'NOK, occasion 1']
for k, r in enumerate(sub):
    a = axs[k // 3, k % 3]; w, h = r.pil_size; hm = cv2.resize(r.d, (240, 320), interpolation=cv2.INTER_CUBIC)
    im = a.imshow(np.clip((hm - LO) / (HI - LO), 0, 1), cmap='inferno', vmin=0, vmax=1)
    if r.geo: a.add_patch(plt.Circle((r.geo[0] * 240, r.geo[1] * 320), r.geo[2] * 320, fill=False, ec='white', lw=1, ls='--'))
    a.set_xticks([]); a.set_yticks([]); a.set_title(f"part {r.part}   score {r['mean']:.2f}", fontsize=7)
    if k % 3 == 0: a.set_ylabel(lab[k // 3], fontsize=7.5)
cb = fig.add_axes([.25, .06, .5, .012]); fig.colorbar(im, cax=cb, orientation='horizontal', ticks=[0, 1]); cb.set_xticklabels(['low', 'high patch distance'], fontsize=7)
fig.savefig(os.path.join(OUT, 'r4_fig5_maps.png'), dpi=300, bbox_inches='tight'); plt.close(fig)
R.drop(columns=['g', 'mk', 'geo', 'd', 'path']).to_csv(os.path.join(OUT, 'bontaz_round4.csv'), index=False)

# ============================================================================= POSITIVE CONTROL (public)
def obj_mask(path, shape):
    a = np.asarray(Image.open(path).convert('L').resize((256, 256)), dtype=np.float32)
    border = np.r_[a[0], a[-1], a[:, 0], a[:, -1]]; diff = np.abs(a - np.median(border)).astype(np.uint8)
    _, m = cv2.threshold(diff, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)); n, lab_, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2: return None
    m = ndimage.binary_fill_holes(lab_ == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))
    return cv2.resize(m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
pub_rows, pub_tiles = [], []
if not PUB: log("\n[public] MVTec folder not found in Drive - positive control skipped")
for cat in ('metal_nut', 'screw'):
    base = next((os.path.join(p, cat) for p in PUB if os.path.isdir(os.path.join(p, cat))), None)
    if base is None: log(f"\n[public] {cat}: not found"); continue
    tr = sorted(os.path.join(base, 'train', 'good', f) for f in os.listdir(os.path.join(base, 'train', 'good')) if f.lower().endswith(IMG))
    random.Random(SEED).shuffle(tr); cut = int(.8 * len(tr)); bank, cal = tr[:cut], tr[cut:]
    M = fit(bank, (360, 360)); Gp = np.stack([M.run(p)[1] for p in bank])
    cs = [M.run(p)[0] for p in cal]; tmean, tmax = np.quantile([d.mean() for d in cs], .96), np.quantile([d.max() for d in cs], .96)
    te = []
    for dt in sorted(os.listdir(os.path.join(base, 'test'))):
        for f in sorted(os.listdir(os.path.join(base, 'test', dt))):
            if f.lower().endswith(IMG): te.append((os.path.join(base, 'test', dt, f), int(dt != 'good'), dt))
    for p, y, dt in te:
        d, g = M.run(p); mk = obj_mask(p, d.shape)
        rec = dict(category=cat, defect=dt, y=y, mean=float(d.mean()), max=float(d.max()), nn=float(np.linalg.norm(Gp - g, axis=1).min()))
        if mk is not None and mk.any() and not mk.all():
            rec.update(inside=float(d[mk].mean()), outside=float(d[~mk].mean()), peak=bool(mk.flat[int(d.argmax())]), obj_frac=float(mk.mean()))
        pub_rows.append(rec)
        if len([t for t in pub_tiles if t[0] == cat]) < 4 and (y == 0 or dt in ('scratch', 'bent', 'color', 'flip', 'thread_side', 'manipulated_front')):
            if sum(1 for t in pub_tiles if t[0] == cat and t[2] == y) < 2: pub_tiles.append((cat, p, y, d, mk))
    P = pd.DataFrame([r for r in pub_rows if r['category'] == cat])
    log(f"\n[public] {cat}: memory {len(bank)}, calibration {len(cal)}, test {int((P.y == 0).sum())} good + {int(P.y.sum())} defective")
    for sc, t in (('mean', tmean), ('max', tmax)):
        a = auc_ci(P.y, P[sc]); fa, ms = int((P[P.y == 0][sc] > t).sum()), int((P[P.y == 1][sc] <= t).sum())
        log(f"  {sc:4s}: AUROC {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}] | false alarms {fa}/{int((P.y == 0).sum())} | misses {ms}/{int(P.y.sum())}")
    b = adj_auc(P.y.values, P['mean'].values, P.nn.values, P.y.values == 0)
    log(f"  diagnostic 3: AUROC of distance alone {roc_auc_score(P.y, P.nn):.3f} | distance-adjusted AUROC (mean score) {b[0]:.3f} [{b[1]:.3f}, {b[2]:.3f}]")
    Q = P.dropna(subset=['inside'])
    log(f"  diagnostic 4 chance baseline: object area share {Q.obj_frac.mean():.1%} | strongest patch on object, good images {Q[Q.y == 0].peak.mean():.0%} ({int(Q[Q.y == 0].peak.sum())}/{int((Q.y == 0).sum())}), defects {Q[Q.y == 1].peak.mean():.0%} ({int(Q[Q.y == 1].peak.sum())}/{int((Q.y == 1).sum())})")
    # NEGATIVE CONTROL: brighten the defective test images by 25% (an acquisition difference that lines up with class)
    from PIL import ImageEnhance
    nc = []
    for p, y, dt in te:
        img = Image.open(p).convert('RGB'); img = ImageEnhance.Brightness(img).enhance(1.25) if y == 1 else img
        d, g = M.run(img); mk = obj_mask(p, d.shape)
        rec = dict(y=y, mean=float(d.mean()), nn=float(np.linalg.norm(Gp - g, axis=1).min()))
        if mk is not None and mk.any() and not mk.all(): rec.update(inside=float(d[mk].mean()), outside=float(d[~mk].mean()), peak=bool(mk.flat[int(d.argmax())]))
        nc.append(rec)
    N_ = pd.DataFrame(nc); an = auc_ci(N_.y, N_['mean']); bn = adj_auc(N_.y.values, N_['mean'].values, N_.nn.values, N_.y.values == 0); Qn = N_.dropna(subset=['inside'])
    log(f"  NEGATIVE CONTROL (defective images brightened x1.25): AUROC {an[0]:.3f} | distance alone {roc_auc_score(N_.y, N_.nn):.3f} | distance-adjusted {bn[0]:.3f} [{bn[1]:.3f}, {bn[2]:.3f}] | "
        f"object {roc_auc_score(Qn.y, Qn.inside):.3f} | background {roc_auc_score(Qn.y, Qn.outside):.3f} | strongest patch on object, defects {Qn[Qn.y == 1].peak.mean():.0%}")
    log(f"  diagnostic 4 (object mask from geometry, n={len(Q)}): whole {roc_auc_score(Q.y, Q['mean']):.3f} | object {roc_auc_score(Q.y, Q.inside):.3f} | "
        f"background {roc_auc_score(Q.y, Q.outside):.3f} | peak on object, defects {Q[Q.y == 1].peak.mean():.0%} | object area by class {Q[Q.y == 0].obj_frac.median():.2f} / {Q[Q.y == 1].obj_frac.median():.2f}")
    log("  diagnostics 1-2: acquisition is constant in this dataset, so the AUROC above is already a matched comparison; no acquisition order is recorded.")
pd.DataFrame(pub_rows).to_csv(os.path.join(OUT, 'public_round4.csv'), index=False)
if pub_tiles:
    fig, axs = plt.subplots(len(pub_tiles) // 4 + (len(pub_tiles) % 4 > 0), 4, figsize=(7.2, 1.9 * (len(pub_tiles) // 4 + 1)), squeeze=False)
    for k, (cat, p, y, d, mk) in enumerate(pub_tiles):
        a = axs[k // 4, k % 4]; im = np.asarray(Image.open(p).convert('RGB').resize((256, 256))) / 255
        hm = cv2.resize(d, (256, 256), interpolation=cv2.INTER_CUBIC); a.imshow(im); a.imshow(hm, cmap='inferno', alpha=.5)
        if mk is not None: a.contour(cv2.resize(mk.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST), levels=[.5], colors='white', linewidths=.6)
        a.set_xticks([]); a.set_yticks([]); a.set_title(f"{cat}, {'defective' if y else 'good'}", fontsize=7)
    for a in axs.ravel()[len(pub_tiles):]: a.axis('off')
    fig.savefig(os.path.join(OUT, 'r4_public_maps.png'), dpi=250, bbox_inches='tight'); plt.close(fig)
open(os.path.join(OUT, 'report_round4.txt'), 'w').write(LOG.getvalue())
print(f"\nsaved -> {OUT}\nSend the printout (or report_round4.txt) to Claude.")
