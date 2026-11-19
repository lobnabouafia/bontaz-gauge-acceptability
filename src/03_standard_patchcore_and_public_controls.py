# =============================================================================
# 18  SECOND DETECTOR + WIDER CONTROLS (Colab, GPU, ~60 min). Nothing here changes the reported model.
#
#  [1] STANDARD PATCHCORE on the BONTAZ data (Roth et al. 2022 settings): Wide-ResNet-50-2 (ImageNet), layers 2+3,
#      3x3 neighbourhood averaging, greedy k-centre coreset of 1% of all training patches (selected on a random
#      128-d projection), nearest-neighbour distance in the full feature space, image score = MAXIMUM patch distance.
#      Same images, same field of view (360x480), same threshold rule (Q0.96 of calibration scores).
#      Scored on the held-out test, the gauge set and the matched set; AIAG rates; region split; distance alone.
#  [2] PAIRED COMPARISONS for both detectors: bootstrap (2,000, stratified by class) of
#      AUROC(model) - AUROC(image-level distance) and AUROC(part) - AUROC(background).
#  [3] ORDER ADJUSTMENT for the standard detector (drift refitted in every bootstrap replicate).
#  [4] PUBLIC CONTROLS, wider: MVTec AD metal_nut + screw x 3 seeds x 4 artificial shortcuts x 2 strengths
#      (brightness, illumination gradient, zoom, blur), applied to defective test images only.
#      For every condition: model AUROC, distance alone, object/background, and the two paired differences.
#      Standard PatchCore on the clean public test sets, for comparison with published figures.
#  All rules are written to protocol_round5.json before anything is scored.
# Outputs: MyDrive/dataset_150360_final/round5/  report_round5.txt, bontaz_round5.csv, public_round5.csv
# =============================================================================
import os, io, re, json, time, math, shutil, random
import numpy as np, pandas as pd, cv2, torch, torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage
from sklearn.metrics import roc_auc_score

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
OUT = os.path.join(FIN, 'round5'); os.makedirs(OUT, exist_ok=True)
PUB = next((p for p in (os.path.join(MY, 'AIAG_STUDY', 'data', 'mvtec'), os.path.join(MY, 'AIAG_STUDY', 'mvtec'), os.path.join(MY, 'mvtec')) if os.path.isdir(p)), None)
OK_PARTS = set(range(6, 11)) | set(range(26, 31)) | set(range(36, 51))
SEED, Q, NBOOT = 150360, 0.96, 2000
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
LOG = io.StringIO()
def log(*a): print(*a); print(*a, file=LOG); open(os.path.join(OUT, 'report_round5.txt'), 'w').write(LOG.getvalue())
IMG = ('.png', '.jpg', '.jpeg', '.bmp')
num = lambda s: int(''.join(c for c in os.path.splitext(os.path.basename(str(s)))[0] if c.isdigit()) or -1)
if DEV == 'cpu': print("WARNING: no GPU - running in CPU mode (slower; all steps are cached in Drive, so a re-run resumes).")
HDT = torch.float16 if DEV == 'cuda' else torch.float32
torch.set_num_threads(max(1, os.cpu_count() or 1))

SHORTCUTS = [('brightness', 1.15), ('brightness', 1.30), ('gradient', 0.3), ('gradient', 0.6),
             ('zoom', 1.05), ('zoom', 1.10), ('blur', 1.0), ('blur', 2.0)]
PUB_SEEDS = [150360, 1, 2]
json.dump(dict(written=time.strftime('%Y-%m-%d %H:%M'),
               standard_patchcore=dict(backbone='wide_resnet50_2 IMAGENET1K_V1', layers=[2, 3], pooling='3x3 avg', coreset='1% greedy k-centre on 128-d random projection',
                                       score='max patch NN distance', input='360x480 (BONTAZ), 360x360 (MVTec)', threshold=f'Q{Q} calibration'),
               paired_tests=f'stratified bootstrap, {NBOOT} replicates, difference of AUROCs on the same images',
               flags=dict(D3='lower 95% bound of AUROC(model)-AUROC(distance) <= 0', D4='lower 95% bound of AUROC(part)-AUROC(background) <= 0'),
               public=dict(categories=['metal_nut', 'screw'], seeds=PUB_SEEDS, shortcuts=SHORTCUTS, applied_to='defective test images only')),
          open(os.path.join(OUT, 'protocol_round5.json'), 'w'), indent=1)

# ----------------------------------------------------------------------------- statistics
rng = np.random.default_rng(SEED)
def boot_idx(y):
    y = np.asarray(y); i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    return [np.r_[rng.choice(i0, len(i0)), rng.choice(i1, len(i1))] for _ in range(NBOOT)]
def auc_ci(y, s, idx=None):
    y, s = np.asarray(y), np.asarray(s); idx = idx or boot_idx(y)
    b = [roc_auc_score(y[j], s[j]) for j in idx]; return roc_auc_score(y, s), *np.percentile(b, [2.5, 97.5])
def paired(y, a, b, idx=None):
    """AUROC(a) - AUROC(b) on the same images, with a stratified bootstrap interval."""
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b); idx = idx or boot_idx(y)
    d = [roc_auc_score(y[j], a[j]) - roc_auc_score(y[j], b[j]) for j in idx]
    return roc_auc_score(y, a) - roc_auc_score(y, b), *np.percentile(d, [2.5, 97.5])
def cp(k, n, a=0.05):
    from scipy.stats import beta
    return (0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1), 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k))
f3 = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"

# ----------------------------------------------------------------------------- feature extractors
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
NETS = {}
def net(kind):
    if kind not in NETS:
        m = (models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2) if kind == 'recipe'
             else models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)).eval().to(DEV)
        for p_ in m.parameters(): p_.requires_grad_(False)
        NETS[kind] = m
    return NETS[kind]
@torch.no_grad()
def feats(img, size, kind):
    fs = {}; m = net(kind)
    h1 = m.layer2.register_forward_hook(lambda a, b, o: fs.__setitem__('a', o))
    h2 = m.layer3.register_forward_hook(lambda a, b, o: fs.__setitem__('b', o))
    pil = img if isinstance(img, Image.Image) else Image.open(img)
    x = NORM(transforms.ToTensor()(transforms.Resize((size[0], size[1]))(pil.convert('RGB'))))
    m(x.unsqueeze(0).to(DEV)); h1.remove(); h2.remove()
    fa = F.avg_pool2d(fs['a'], 3, 1, 1); fb = F.interpolate(F.avg_pool2d(fs['b'], 3, 1, 1), size=fa.shape[-2:], mode='bilinear', align_corners=False)
    f = torch.cat([fa, fb], 1); return f[0].permute(1, 2, 0).reshape(-1, f.shape[1]).float(), fa.shape[-2:]
class Model:
    """kind='recipe': PCA projection + mean score (the paper's model). kind='std': full features + max score."""
    def __init__(self, kind, size, mem, W=None, mu=None):
        self.kind, self.size, self.MEM = kind, size, mem.to(DEV)
        self.W = None if W is None else W.to(DEV); self.mu = None if mu is None else mu.to(DEV)
    @torch.no_grad()
    def run(self, img):
        f, (gh, gw) = feats(img, self.size, self.kind)
        z = f if self.W is None else (f - self.mu) @ self.W
        d = torch.cat([torch.cdist(z[i:i + 4096], self.MEM).min(1).values for i in range(0, z.shape[0], 4096)]).reshape(gh, gw)
        d = d.cpu().numpy(); s = float(d.max()) if self.kind == 'std' else float(d.mean())
        return s, d, z.mean(0).cpu().numpy()
def greedy(Z, n, seed):
    # greedy k-centre coreset on squared distances (same selection as with distances; ~25x faster on CPU)
    g = torch.Generator().manual_seed(seed); j = int(torch.randint(0, Z.shape[0], (1,), generator=g)); sel = [j]
    sq = (Z * Z).sum(1); dmin = (sq - 2 * (Z @ Z[j]) + sq[j]).clamp_min_(0)
    for _ in range(n - 1):
        j = int(torch.argmax(dmin)); sel.append(j); torch.minimum(dmin, (sq - 2 * (Z @ Z[j]) + sq[j]).clamp_min_(0), out=dmin)
    return torch.tensor(sel, device=Z.device)
def fit_std(paths, size, seed, ratio=0.01):
    P = torch.cat([feats(p, size, 'std')[0].to(HDT) for p in paths])
    g = torch.Generator(device='cpu').manual_seed(seed); R = torch.randn(P.shape[1], 128, generator=g).to(DEV).to(HDT)
    sel = greedy((P @ R).float(), max(1, int(ratio * P.shape[0])), seed)
    mem = P[sel].float(); del P
    if DEV == 'cuda': torch.cuda.empty_cache()
    return Model('std', size, mem)
def fit_recipe(paths, size, seed, pca=256, nbank=12000, per=500):
    g = torch.Generator().manual_seed(seed); pool = []
    for p in paths:
        f, _ = feats(p, size, 'recipe'); pool.append(f[torch.randperm(f.shape[0], generator=g)[:per].to(DEV)].to(HDT))
    P = torch.cat(pool).float(); mu = P.mean(0); C = (P - mu).T @ (P - mu) / (P.shape[0] - 1)
    ev, V = torch.linalg.eigh(C); W = V[:, torch.argsort(ev, descending=True)[:pca]]; Z = (P - mu) @ W; del P
    return Model('recipe', size, Z[greedy(Z, min(nbank, Z.shape[0]), seed)], W, mu)

# ----------------------------------------------------------------------------- part region (BONTAZ) / object mask (public)
def disc(pil, shape, r_fixed):
    rgb = np.asarray(pil.convert('RGB')); H0, W0 = rgb.shape[:2]; s = 800 / max(H0, W0)
    small = cv2.resize(rgb, (round(W0 * s), round(H0 * s)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV); h, sat, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = cv2.morphologyEx(((h >= 3) & (h <= 25) & (sat >= 60) & (v >= 60)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    if n < 2: return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); gh, gw = shape; yy, xx = np.mgrid[:gh, :gw]
    cy, cx = cen[i, 1] / small.shape[0], cen[i, 0] / small.shape[1]
    return np.hypot((yy + .5) / gh - cy, ((xx + .5) / gw - cx) * small.shape[1] / small.shape[0]) <= r_fixed
def obj_mask(path, shape):
    a = np.asarray(Image.open(path).convert('L').resize((256, 256)), dtype=np.float32)
    border = np.r_[a[0], a[-1], a[:, 0], a[:, -1]]; diff = np.abs(a - np.median(border)).astype(np.uint8)
    _, m = cv2.threshold(diff, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)); n, lab_, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2: return None
    m = ndimage.binary_fill_holes(lab_ == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))
    return cv2.resize(m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
R_FIX = 0.144   # median colour-disc radius of good test photographs (script 17), as a fraction of image height

# ============================================================================= [1]-[3] BONTAZ
CFG = json.load(open(os.path.join(ROOT, 'metadata', 'sealed_model_v2.json')))['config']; SIZE = (CFG['img_h'], CFG['img_w'])
K = pd.read_csv(os.path.join(FIN, 'manifest_v3.csv'), dtype=str); K['path'] = [os.path.join(FIN, f) for f in K.file_v3]
B = torch.load(os.path.join(FIN, 'model_v3', 'bank_v3.pt'), map_location='cpu')
REC = Model('recipe', SIZE, B['mem'].float(), B['W'].float(), B['mu'].float())
TAU_REC = float(json.load(open(os.path.join(FIN, 'model_v3', 'model_v3.json')))['tau'])
matched = sorted(os.path.join(OLD, 'ts_ok', f) for f in os.listdir(os.path.join(OLD, 'ts_ok')) if f.lower().endswith(IMG) and num(f) in OK_PARTS)
tr, cal = K[K.split_v3 == 'train_ok'].path.tolist(), K[K.split_v3 == 'calib_ok'].path.tolist()
items = [(r.split_v3, r.path, int(r.label_v3 == 'NOK'), num(r.file_v3) if r.split_v3.startswith('msa') else -1)
         for r in K[K.split_v3.isin(['test_ok', 'test_nok', 'msa_ok', 'msa_nok'])].itertuples()] + [('matched_ok', p, 0, num(p)) for p in matched]
_bc, _tj = os.path.join(OUT, 'bontaz_round5.csv'), os.path.join(OUT, 'tau_standard.json')
_old = open(os.path.join(OUT, 'report_round5.txt')).read() if os.path.exists(os.path.join(OUT, 'report_round5.txt')) else ''
_m = re.search(r'standard ([0-9.]+) \(max score\)', _old)
if os.path.exists(_bc) and (os.path.exists(_tj) or _m):
    TAU_STD = json.load(open(_tj))['tau'] if os.path.exists(_tj) else float(_m.group(1))
    rows = pd.read_csv(_bc).to_dict('records'); log(f"[1] BONTAZ scores already computed - loaded {len(rows)} rows from bontaz_round5.csv")
else:
    t0 = time.time(); CACHE = os.path.join(OUT, 'cache_std_bontaz_memory.pt')
    if os.path.exists(CACHE): STD = Model('std', SIZE, torch.load(CACHE, map_location='cpu')); print('    loaded cached standard PatchCore memory')
    else: STD = fit_std(tr, SIZE, SEED); torch.save(STD.MEM.cpu(), CACHE)
    log(f"[1] standard PatchCore fitted on {len(tr)} training photographs: memory {STD.MEM.shape[0]} x {STD.MEM.shape[1]} ({time.time() - t0:.0f}s)")
    TAU_STD = float(np.quantile([STD.run(p)[0] for p in cal], Q))
    rows = []
    for name, M, gfun in (('recipe', REC, None), ('standard', STD, None)):
        G = np.stack([M.run(p)[2] for p in tr])
        for s, p, y, part in items:
            sc, d, g = M.run(p); pil = Image.open(p); rec = dict(model=name, set=s, y=y, part=part, score=sc, nn=float(np.linalg.norm(G - g, axis=1).min()))
            for k in (1.0, 1.5, 2.0):
                mk = disc(pil, d.shape, R_FIX * k)
                if mk is not None and mk.any() and not mk.all():
                    rec[f'in{k}'], rec[f'out{k}'], rec[f'peak{k}'] = float(d[mk].mean()), float(d[~mk].mean()), bool(mk.flat[int(d.argmax())])
            rows.append(rec)
    json.dump({'tau': TAU_STD}, open(_tj, 'w'))
Bz = pd.DataFrame(rows); Bz.to_csv(os.path.join(OUT, 'bontaz_round5.csv'), index=False)
chk = Bz[(Bz.model == 'recipe') & Bz.set.isin(['msa_ok', 'msa_nok'])]
log(f"    reproduction (recipe): gauge false alarms {(chk[chk.y == 0].score > TAU_REC).sum()}/25, misses {(chk[chk.y == 1].score <= TAU_REC).sum()}/25 (expected 1 and 2)")
log(f"    thresholds: recipe {TAU_REC:.4f} (mean score), standard {TAU_STD:.4f} (max score)")
SETS = (('held-out test', ['test_ok', 'test_nok']), ('gauge set', ['msa_ok', 'msa_nok']), ('matched set', ['matched_ok', 'msa_nok']))
for name, tau in (('recipe', TAU_REC), ('standard', TAU_STD)):
    log(f"\n  detector: {name}")
    for nm, ss in SETS:
        D = Bz[(Bz.model == name) & Bz.set.isin(ss)].reset_index(drop=True); idx = boot_idx(D.y)
        a = auc_ci(D.y, D.score, idx); fa = int((D[D.y == 0].score > tau).sum()); ms = int((D[D.y == 1].score <= tau).sum()); n0, n1 = int((D.y == 0).sum()), int(D.y.sum())
        la, lb = cp(fa, n0), cp(ms, n1)
        log(f"   {nm:13s} AUROC {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}] | alpha {fa}/{n0} [{la[0]:.1%}, {la[1]:.1%}] | beta {ms}/{n1} [{lb[0]:.1%}, {lb[1]:.1%}] | E {(n0 - fa + n1 - ms) / (n0 + n1):.0%}")
        log(f"   {'':13s} distance alone {roc_auc_score(D.y, D.nn):.3f} | model - distance {f3(paired(D.y, D.score, D.nn, idx))}")
        for k in (1.0, 1.5, 2.0):
            E_ = D.dropna(subset=[f'in{k}']).reset_index(drop=True)
            if E_.y.nunique() < 2: continue
            log(f"   {'':13s} disc {k:.1f}x: part {roc_auc_score(E_.y, E_[f'in{k}']):.3f} | background {roc_auc_score(E_.y, E_[f'out{k}']):.3f} | part - background {f3(paired(E_.y, E_[f'in{k}'], E_[f'out{k}']))} | "
                f"strongest patch on part: defective {E_[E_.y == 1][f'peak{k}'].mean():.0%}, good {E_[E_.y == 0][f'peak{k}'].mean():.0%}")
    g2 = Bz[(Bz.model == name) & (Bz.set == 'msa_ok')].set_index('part').score; g1 = Bz[(Bz.model == name) & (Bz.set == 'matched_ok')].set_index('part').score
    j = pd.concat([g2, g1], axis=1, keys=['later', 'first']).dropna()
    log(f"   same decision on both photographs of a good part: {int(((j['later'] > tau) == (j['first'] > tau)).sum())}/{len(j)}; rejected on first-occasion photo {int((j['first'] > tau).sum())}/{len(j)}")
    # [3] order adjustment on the first occasion, drift refitted in every replicate
    O = Bz[(Bz.model == name) & Bz.set.isin(['matched_ok', 'msa_nok'])].reset_index(drop=True)
    X = np.c_[np.ones(len(O)), O.part, O.y]; yv = O.score.values
    def fit_(ix):
        b = np.linalg.lstsq(X[ix], yv[ix], rcond=None)[0]; adj = yv[ix] - b[1] * O.part.values[ix]; return b, roc_auc_score(O.y.values[ix], adj)
    b0, a0 = fit_(np.arange(len(O))); bs = [fit_(ix) for ix in boot_idx(O.y)]
    log(f"   first occasion: drift {b0[1]:+.4f}/part, class effect {b0[2]:+.4f} [{np.percentile([b[0][2] for b in bs], 2.5):+.4f}, {np.percentile([b[0][2] for b in bs], 97.5):+.4f}], "
        f"order-adjusted AUROC {a0:.3f} [{np.percentile([b[1] for b in bs], 2.5):.3f}, {np.percentile([b[1] for b in bs], 97.5):.3f}] (drift refitted per replicate)")

# ============================================================================= [4] PUBLIC CONTROLS
def shortcut(img, kind, s):
    if kind == 'brightness': return ImageEnhance.Brightness(img).enhance(s)
    if kind == 'gradient':
        a = np.asarray(img, dtype=np.float32); ramp = np.linspace(1.0, 1.0 + s, a.shape[1])[None, :, None]
        return Image.fromarray(np.clip(a * ramp, 0, 255).astype(np.uint8))
    if kind == 'zoom':
        w, h = img.size; cw, ch = int(w / s), int(h / s); l, t = (w - cw) // 2, (h - ch) // 2
        return img.crop((l, t, l + cw, t + ch)).resize((w, h), Image.BILINEAR)
    if kind == 'blur': return img.filter(ImageFilter.GaussianBlur(s))
pub = []
_pc = os.path.join(OUT, 'public_round5.csv')
if os.path.exists(_pc):
    pub = pd.read_csv(_pc).to_dict('records'); print(f'  resuming: {len(pub)} public conditions already done')
DONE = {(r['category'], int(r['seed']), r['model'], r['shortcut'], float(r['strength'])) for r in pub}
def cached_model(name, fit):
    fp = os.path.join(OUT, name)
    if os.path.exists(fp):
        c = torch.load(fp, map_location='cpu'); return Model(c['kind'], tuple(c['size']), c['mem'], c['W'], c['mu'])
    M = fit(); torch.save(dict(kind=M.kind, size=list(M.size), mem=M.MEM.cpu(), W=None if M.W is None else M.W.cpu(), mu=None if M.mu is None else M.mu.cpu()), fp); return M
if PUB is None: log("\n[4] MVTec folder not found - public controls skipped")
for seed, cat in [(sd, ct) for sd in PUB_SEEDS for ct in ('metal_nut', 'screw')]:
    base = os.path.join(PUB, cat) if PUB else None
    if not base or not os.path.isdir(base): continue
    trall = sorted(os.path.join(base, 'train', 'good', f) for f in os.listdir(os.path.join(base, 'train', 'good')) if f.lower().endswith(IMG))
    te = [(os.path.join(base, 'test', dt, f), int(dt != 'good')) for dt in sorted(os.listdir(os.path.join(base, 'test')))
          for f in sorted(os.listdir(os.path.join(base, 'test', dt))) if f.lower().endswith(IMG)]
    yt = np.array([y for _, y in te]); masks = {}
    if True:
        tr_ = trall[:]; random.Random(seed).shuffle(tr_); cut = int(.8 * len(tr_)); bank, cal_ = tr_[:cut], tr_[cut:]
        for mname in (['recipe', 'standard'] if seed == SEED else ['recipe']):
            todo = [c for c in [('clean', 0)] + (SHORTCUTS if mname == 'recipe' else []) if (cat, seed, mname, c[0], float(c[1])) not in DONE]
            if not todo: print(f'  {cat} seed {seed} {mname}: already done, skipped'); continue
            M = cached_model(f'cache_{cat}_{seed}_{mname}.pt', lambda: fit_recipe(bank, (360, 360), seed) if mname == 'recipe' else fit_std(bank, (360, 360), seed))
            G = np.stack([M.run(p)[2] for p in bank]); tau = float(np.quantile([M.run(p)[0] for p in cal_], Q))
            for cond in todo:
                sc, nn, ins, outs = [], [], [], []
                for p, y in te:
                    img = Image.open(p).convert('RGB'); img = shortcut(img, *cond) if (y == 1 and cond[0] != 'clean') else img
                    s_, d, g = M.run(img); mk = masks.get(p)
                    if mk is None: mk = masks[p] = obj_mask(p, d.shape)
                    sc.append(s_); nn.append(float(np.linalg.norm(G - g, axis=1).min()))
                    ins.append(float(d[mk].mean()) if mk is not None and mk.any() else np.nan); outs.append(float(d[~mk].mean()) if mk is not None and mk.any() else np.nan)
                sc, nn, ins, outs = map(np.array, (sc, nn, ins, outs)); ok = ~np.isnan(ins); idx = boot_idx(yt)
                d3 = paired(yt, sc, nn, idx); d4 = paired(yt[ok], ins[ok], outs[ok])
                rec = dict(category=cat, seed=seed, model=mname, shortcut=cond[0], strength=cond[1], auroc=roc_auc_score(yt, sc), distance=roc_auc_score(yt, nn),
                           obj=roc_auc_score(yt[ok], ins[ok]), bg=roc_auc_score(yt[ok], outs[ok]), d3=d3[0], d3_lo=d3[1], d3_hi=d3[2], d4=d4[0], d4_lo=d4[1], d4_hi=d4[2],
                           alpha=float((sc[yt == 0] > tau).mean()), beta=float((sc[yt == 1] <= tau).mean()), flag_d3=bool(d3[1] <= 0), flag_d4=bool(d4[1] <= 0))
                pub.append(rec)
                log(f"  {cat:9s} seed {seed:6d} {mname:8s} {cond[0]:10s} {cond[1]:4.2f}: AUROC {rec['auroc']:.3f} | distance {rec['distance']:.3f} | model-distance {f3(d3)} {'FLAG' if rec['flag_d3'] else 'ok  '} | "
                    f"object {rec['obj']:.3f} bg {rec['bg']:.3f} | object-bg {f3(d4)} {'FLAG' if rec['flag_d4'] else 'ok'}")
            pd.DataFrame(pub).to_csv(os.path.join(OUT, 'public_round5.csv'), index=False)
P = pd.DataFrame(pub)
if len(P):
    R_ = P[P.model == 'recipe']; clean = R_[R_.shortcut == 'clean']; sh = R_[R_.shortcut != 'clean'].copy()
    base_auc = clean.set_index(['category', 'seed']).auroc
    sh['gain'] = [r.auroc - base_auc[(r.category, r.seed)] for r in sh.itertuples()]
    log(f"\n[4] summary (recipe): clean conditions flagged by D3 {clean.flag_d3.sum()}/{len(clean)}, by D4 {clean.flag_d4.sum()}/{len(clean)}")
    for lab, S in (('all shortcut conditions', sh), ('shortcuts that raised AUROC by >= 0.02', sh[sh.gain >= .02])):
        log(f"    {lab}: n={len(S)} | flagged by D3 {S.flag_d3.sum()} | by D4 {S.flag_d4.sum()} | by either {(S.flag_d3 | S.flag_d4).sum()}")
    log("    by shortcut type (flag rate D3 / D4, mean AUROC gain):\n" + sh.groupby(['shortcut', 'strength']).agg(D3=('flag_d3', 'mean'), D4=('flag_d4', 'mean'), gain=('gain', 'mean')).round(2).to_string())
    S_ = P[(P.model == 'standard') & (P.shortcut == 'clean')]
    if len(S_): log("    standard PatchCore, clean public test: " + ", ".join(f"{r.category} AUROC {r.auroc:.3f}" for r in S_.itertuples()))
print(f"\nsaved -> {OUT}\nSend report_round5.txt to Claude.")
