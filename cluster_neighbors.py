import numpy as np, pandas as pd, random, pathlib, os, sys

# --- SETTINGS ---
CLUSTER_ID = 7      # <--- change this (use -1 for noise)
N_QUERIES  = 10
K_NEIGHBORS = 10

random.seed(42)
np.random.seed(42)

CSV = "cluster_results.csv"
FEATS = "features.npy"

df = pd.read_csv(CSV)
X = np.load(FEATS)

if len(df) != X.shape[0]:
    raise SystemExit(f"Row mismatch: CSV {len(df)} vs feats {X.shape[0]}")

# Select rows in the target cluster
idxs = df.index[df["cluster_id"] == CLUSTER_ID].tolist()
if len(idxs) == 0:
    raise SystemExit(f"No rows found for cluster_id={CLUSTER_ID}")
if len(idxs) < N_QUERIES:
    print(f"Warning: cluster has only {len(idxs)} rows; sampling all.")
    sample = idxs
else:
    sample = random.sample(idxs, N_QUERIES)

# Cosine similarity (normalize)
Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

# Output
out = pathlib.Path(f"knn_report_cluster_{CLUSTER_ID}.html")
html = []
html.append("<html><head><meta charset='utf-8'>")
html.append(f"<title>kNN report cluster {CLUSTER_ID}</title>")
html.append("""
<style>
body{font-family:system-ui;margin:20px}
.row{display:flex;gap:12px;align-items:flex-start;margin-bottom:28px}
img{width:120px;height:120px;object-fit:contain;background:#fff;border:1px solid #ddd;border-radius:8px}
.small img{width:90px;height:90px}
.card{display:flex;flex-direction:column;gap:6px}
.meta{font-size:12px;color:#333;max-width:200px;word-break:break-all}
</style>
""")
html.append("</head><body>")
html.append(f"<h1>kNN report — cluster_id={CLUSTER_ID}</h1>")
html.append(f"<p>{len(sample)} random queries from this cluster, top-{K_NEIGHBORS} neighbors by cosine similarity.</p>")

def url_from_path(p: str) -> str:
    """
    Your server serves /trivalaya_data/... at http://localhost:8080/trivalaya_data/...
    So we generate root-relative URLs.
    """
    p = str(p)
    if "/trivalaya_data/" in p:
        p = p.split("/trivalaya_data/", 1)[1]
        return "/trivalaya_data/" + p
    if p.startswith("trivalaya_data/"):
        return "/" + p
    return "/" + p.lstrip("/")

for i in sample:
    sims = Xn @ Xn[i]
    nn = np.argsort(-sims)[:(K_NEIGHBORS + 1)]
    nn = [j for j in nn if j != i][:K_NEIGHBORS]

    qpath = df.loc[i, "image_path"]
    qlabel = df.loc[i, "period"] if "period" in df.columns else ""
    qcluster = df.loc[i, "cluster_id"]

    html.append("<div class='row'>")
    html.append("<div class='card'>")
    html.append(f"<div><b>QUERY</b> idx={i} cluster={qcluster} label={qlabel}</div>")
    html.append(f"<img src='{url_from_path(qpath)}' />")
    html.append(f"<div class='meta'>{qpath}</div>")
    html.append("</div>")

    html.append("<div class='card small'>")
    html.append(f"<div><b>Top-{K_NEIGHBORS} neighbors</b></div>")
    html.append("<div style='display:flex;flex-wrap:wrap;gap:10px'>")
    for j in nn:
        p = df.loc[j, "image_path"]
        lbl = df.loc[j, "period"] if "period" in df.columns else ""
        cl = df.loc[j, "cluster_id"]
        s = float(sims[j])
        html.append("<div class='card'>")
        html.append(f"<div class='meta'>sim={s:.3f}<br>cluster={cl}<br>{lbl}</div>")
        html.append(f"<img src='{url_from_path(p)}' />")
        html.append("</div>")
    html.append("</div></div></div><hr>")

html.append("</body></html>")
out.write_text("\n".join(html), encoding="utf-8")
print("Wrote:", out.resolve())
print("Open (adjust path if needed):", f"http://localhost:8080/cluster_output/{out.name}")
