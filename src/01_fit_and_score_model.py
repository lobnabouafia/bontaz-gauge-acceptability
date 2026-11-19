# =============================================================================
# 11  FINAL MODEL ON DATASET v3 (Colab, GPU, ~15-20 min)
# Same recipe as the sealed model (read from sealed_model_v2.json + its bank dimensions):
#   ResNeXt-50 layer2+3, 3x3 averaging, input 360x480, PCA, greedy k-centre coreset bank, mean patch distance,
#   tau = Q0.96 of calib_ok.  Fitted on v3 train_ok only. Nothing is tuned on test or MSA.
# Scores calib / val / test / MSA and writes everything the paper tables and figures need.
# Outputs: MyDrive/dataset_150360_v3/model_v3/
# =============================================================================
import os, io, json, time, hashlib, shutil
import numpy as np, pandas as pd
import torch, torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from scipy.stats import beta as _beta
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

ROOT = os.environ.get('BONTAZ_ROOT', '/content/drive/MyDrive/dataset_150360_clean')
_FINAL = '/content/drive/MyDrive/dataset_150360_final'                 # built by script 13; used automatically when it exists
V3 = os.environ.get('BONTAZ_V3', _FINAL if os.path.isdir(_FINAL) else '/content/drive/MyDrive/dataset_150360_v3')
print('dataset folder:', V3)
OUT = os.path.join(V3, 'model_v3'); os.makedirs(OUT, exist_ok=True)
SEED, POOL_PER_IMG, Q = 150360, int(os.environ.get('BONTAZ_POOL', 500)), 0.96
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
LOG = io.StringIO()
def log(*a): print(*a); print(*a, file=LOG)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
if not os.path.isfile(os.path.join(V3, 'manifest_v3.csv')): raise SystemExit("Run 10_reorganize_dataset.py (MODE='build') first.")
if DEV == 'cpu': print("WARNING: no GPU - Runtime > Change runtime type > GPU.")

ART = json.load(open(os.path.join(ROOT, 'metadata', 'sealed_model_v2.json')))
CFG = ART['config']; SCORE_METHOD = ART.get('score_method', 'mean_all')
b0 = torch.load(os.path.join(ROOT, 'metadata', ART['memory_bank_file']), map_location='cpu')
PCA_D, NBANK = int(b0['W'].shape[1]), int(b0['mem'].shape[0]); del b0
net = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2 if os.environ.get('BONTAZ_PRETRAINED', '1') == '1' else None).eval().to(DEV)
for p_ in net.parameters(): p_.requires_grad_(False)
TFM = transforms.Compose([transforms.Resize((CFG['img_h'], CFG['img_w'])), transforms.ToTensor(),
                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
@torch.no_grad()
def feats(path):
    fs = {}
    h1 = net.layer2.register_forward_hook(lambda a, b, o: fs.__setitem__('a', o))
    h2 = net.layer3.register_forward_hook(lambda a, b, o: fs.__setitem__('b', o))
    net(TFM(Image.open(path).convert('RGB')).unsqueeze(0).to(DEV)); h1.remove(); h2.remove()
    fa = F.avg_pool2d(fs['a'], 3, 1, 1); fb = F.avg_pool2d(fs['b'], 3, 1, 1)
    fb = F.interpolate(fb, size=fa.shape[-2:], mode='bilinear', align_corners=False)
    f = torch.cat([fa, fb], 1); return f[0].permute(1, 2, 0).reshape(-1, f.shape[1]).float()

K = pd.read_csv(os.path.join(V3, 'manifest_v3.csv'), dtype=str)
K['path'] = [os.path.join(V3, f) for f in K.file_v3]; K['y'] = (K.label_v3 == 'NOK').astype(int)
log(f"device {DEV} | recipe: input {CFG['img_h']}x{CFG['img_w']}, PCA {PCA_D}, bank {NBANK}, tau = Q{Q} of calib_ok | {time.strftime('%Y-%m-%d %H:%M')}")
log("dataset v3: " + str(K.split_v3.value_counts().to_dict()))
json.dump(dict(written=time.strftime('%Y-%m-%d %H:%M'), manifest_sha256=sha(os.path.join(V3, 'manifest_v3.csv')),
               recipe=dict(pca_dim=PCA_D, bank=NBANK, pool_per_image=POOL_PER_IMG, tau=f'Q{Q} calib_ok', seed=SEED, score=SCORE_METHOD)),
          open(os.path.join(OUT, 'protocol_model_v3.json'), 'w'), indent=1)

# ----------------------------------------------------------------------------- fit on train_ok only
g = torch.Generator().manual_seed(SEED); pool = []; tr = K[K.split_v3 == 'train_ok'].path.tolist(); t0 = time.time()
for k, p in enumerate(tr, 1):
    f = feats(p); pool.append(f[torch.randperm(f.shape[0], generator=g)[:POOL_PER_IMG].to(DEV)].half())
    if k % 50 == 0: print(f"  train {k}/{len(tr)} ({time.time() - t0:.0f}s)", end='\r')
P = torch.cat(pool).float(); del pool; print()
mu = P.mean(0); C = (P - mu).T @ (P - mu) / (P.shape[0] - 1)
ev, V = torch.linalg.eigh(C); W = V[:, torch.argsort(ev, descending=True)[:PCA_D]]
Z = (P - mu) @ W; del P; n = min(NBANK, Z.shape[0]); gg = torch.Generator().manual_seed(SEED)
sel = [int(torch.randint(0, Z.shape[0], (1,), generator=gg))]; dmin = torch.cdist(Z, Z[sel[0]:sel[0] + 1]).squeeze(1)
for _ in range(n - 1):
    j = int(torch.argmax(dmin)); sel.append(j); dmin = torch.minimum(dmin, torch.cdist(Z, Z[j:j + 1]).squeeze(1))
MEM = Z[torch.tensor(sel, device=DEV)]; del Z
torch.save(dict(mem=MEM.cpu(), W=W.cpu(), mu=mu.cpu()), os.path.join(OUT, 'bank_v3.pt'))
log(f"bank: {n} patches from {len(tr)} train_ok images | sha256 {sha(os.path.join(OUT, 'bank_v3.pt'))[:16]}")

@torch.no_grad()
def score(p):
    d = torch.cdist((feats(p) - mu) @ W, MEM).min(1).values
    return float(d.mean()) if SCORE_METHOD == 'mean_all' else float(d.max())
E = K[K.split_v3 != 'train_ok'].copy(); E['score'] = [score(p) for p in E.path]
TAU = float(np.quantile(E[E.split_v3 == 'calib_ok'].score, Q))
E['pred_nok'] = (E.score > TAU).astype(int)
E.drop(columns=['path']).to_csv(os.path.join(OUT, 'scores_v3.csv'), index=False)
json.dump(dict(tau=TAU, q=Q, n_calib=int((E.split_v3 == 'calib_ok').sum()), bank_sha256=sha(os.path.join(OUT, 'bank_v3.pt'))),
          open(os.path.join(OUT, 'model_v3.json'), 'w'), indent=1)

def cp(k, n, a=0.05):
    return (0.0 if k == 0 else _beta.ppf(a / 2, k, n - k + 1), 1.0 if k == n else _beta.ppf(1 - a / 2, k + 1, n - k))
def rates(D, name):
    ok, nok = D[D.y == 0].score.values, D[D.y == 1].score.values
    fa, ms = int((ok > TAU).sum()), int((nok <= TAU).sum()); a, b = cp(fa, len(ok)), cp(ms, len(nok))
    auc = roc_auc_score(D.y, D.score) if D.y.nunique() == 2 else float('nan')
    log(f"  {name:30s} AUROC {auc:.3f} | false alarms {fa}/{len(ok)} = {fa / len(ok):.1%} [{a[0]:.1%}, {a[1]:.1%}] | "
        f"misses {ms}/{len(nok)} = {ms / len(nok):.1%} [{b[0]:.1%}, {b[1]:.1%}] | effectiveness {(len(ok) - fa + len(nok) - ms) / (len(ok) + len(nok)):.1%}")
log(f"\ntau = {TAU:.4f}")
ver = E['verified'].astype(str).str.lower() != 'false' if 'verified' in E else pd.Series(True, index=E.index)
SETS = [('val', E.split_v3.isin(['val_ok', 'val_nok'])), ('test', E.split_v3.isin(['test_ok', 'test_nok'])),
        ('MSA (main, 50 parts)', E.split_v3.isin(['msa_ok', 'msa_nok'])),
        ('MSA (main), verified photos', E.split_v3.isin(['msa_ok', 'msa_nok']) & ver),
        ('MSA-A 16 May, all', E.split_v3.isin(['msaA_ok', 'msaA_nok'])),
        ('MSA-A 16 May, verified', E.split_v3.isin(['msaA_ok', 'msaA_nok']) & ver),
        ('MSA-B 17 May OK + 16 May NOK', E.split_v3.isin(['msaB_ok', 'msaA_nok'])),
        ('MSA check (same-session OK)', E.split_v3.isin(['msa_check_ok', 'msa_nok'])),
        ('MSA check, verified photos', E.split_v3.isin(['msa_check_ok', 'msa_nok']) & ver)]
for name, mask in SETS:
    D = E[mask]
    if len(D) and D.y.nunique() == 2: rates(D, name)
m = E[E.split_v3.str.startswith('msa')].sort_values('score')
log("\nMSA parts that the model gets wrong:\n" + (m[m.y != m.pred_nok][['image_id', 'label_v3', 'score']].to_string(index=False) or 'none'))
open(os.path.join(OUT, 'model_v3_report.txt'), 'w').write(LOG.getvalue())
print(f"\nsaved -> {OUT}\nSend model_v3_report.txt and scores_v3.csv to Claude.")
