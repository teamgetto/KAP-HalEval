# =============================================================================
# HÜCRE 1: Kurulum
# =============================================================================
import os, json, re, random, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
import anthropic
warnings.filterwarnings('ignore')
random.seed(42); np.random.seed(42)
 
import torch
print(f"GPU: {torch.cuda.is_available()}")
 
try:
    from google.colab import userdata
    api_key = userdata.get('sci')
    print("API key alindi")
except Exception:
    api_key = os.environ.get('ANTHROPIC_API_KEY', '')
 
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY bulunamadi! Colab Secrets'a ekle.")
 
client = anthropic.Anthropic(api_key=api_key)
 
print("API baglantisi test ediliyor...")
try:
    test = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        messages=[{"role": "user", "content": "test"}]
    )
    MODEL_NAME = "claude-haiku-4-5-20251001"
    print(f"API calisiyor (model: {MODEL_NAME})")
except Exception as e:
    print(f"Haiku denenecek alternatif: {e}")
    try:
        test = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=10,
            messages=[{"role": "user", "content": "test"}]
        )
        MODEL_NAME = "claude-haiku-4-5"
        print(f"API calisiyor (model: {MODEL_NAME})")
    except Exception as e2:
        test = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": "test"}]
        )
        MODEL_NAME = "claude-3-haiku-20240307"
        print(f"API calisiyor (model: {MODEL_NAME})")
 
# =============================================================================
# HÜCRE 2: Dataset Yükle
# =============================================================================
print("\nDataset yukleniyor...")
ds = load_dataset("finansai/kap-turkish-financial-sentiment")
df_raw = ds['train'].to_pandas()
 
def parse_messages(row):
    u, a = "", ""
    for m in row['messages']:
        if m['role'] == 'user':        u = m['content']
        elif m['role'] == 'assistant': a = m['content']
    return pd.Series({'source_document': u, 'llm_response': a})
 
df = df_raw.apply(parse_messages, axis=1)
df = df[df['source_document'].str.len() > 300].reset_index(drop=True)
print(f"{len(df)} belge yuklendi")
 
# =============================================================================
# HÜCRE 3: Claude API ile Özet Üretimi
# =============================================================================
N_SAMPLES = 200
 
df_to_sum = df.sample(min(N_SAMPLES, len(df)), random_state=42).reset_index(drop=True)
 
PROMPT = """Aşağıdaki KAP bildirimini 3-5 cümleyle özetle. Önemli sayısal değerleri ve finansal sonuçları mutlaka belirt. Sadece özeti yaz.
 
Bildirim:
{doc}
 
Özet:"""
 
def summarize(doc, retries=3):
    for attempt in range(retries):
        try:
            r = client.messages.create(
                model=MODEL_NAME,
                max_tokens=250,
                messages=[{"role": "user",
                           "content": PROMPT.format(doc=doc[:1500])}]
            )
            text = r.content[0].text.strip()
            if len(text) > 20:
                return text
            return None
        except anthropic.RateLimitError:
            time.sleep(10)
        except anthropic.APIStatusError as e:
            print(f"  API hatasi (deneme {attempt+1}): {e.status_code} - {e.message}")
            time.sleep(3)
        except Exception as e:
            print(f"  Beklenmedik hata (deneme {attempt+1}): {type(e).__name__}: {e}")
            time.sleep(2)
    return None
 
print(f"\n{N_SAMPLES} belge ozetleniyor (model: {MODEL_NAME})...")
 
rows = []
failed = 0
for idx, row in tqdm(df_to_sum.iterrows(), total=len(df_to_sum)):
    summary = summarize(row['source_document'])
    if summary:
        rows.append({
            'source_document': row['source_document'],
            'original_summary': summary,
        })
    else:
        failed += 1
 
print(f"\nBasarili: {len(rows)} | Basarisiz: {failed}")
 
if len(rows) == 0:
    raise RuntimeError("Hic ozet uretilemedi! API key veya model adini kontrol et.")
 
df_summaries = pd.DataFrame(rows)
df_summaries.to_csv('kap_summaries.csv', index=False)
print(f"kap_summaries.csv kaydedildi")
 
# =============================================================================
# HÜCRE 4: Hallucination Enjeksiyonu (ağırlıklı atama, doc_id etiketli)
# =============================================================================
def extract_numbers(text):
    return re.findall(r'\d+(?:[.,]\d+)?\s*(?:milyon|milyar|bin|TL|USD|EUR|%)?', text, re.IGNORECASE)
 
def distort_number(num_str):
    try:
        n = float(re.search(r'\d+(?:[.,]\d+)?', num_str).group().replace(',', '.'))
        mult = random.choice([2, 3, 0.5, 5])
        new_n = n * mult
        new_str = str(int(new_n)) if new_n == int(new_n) else f"{new_n:.1f}"
        return num_str.replace(re.search(r'\d+(?:[.,]\d+)?', num_str).group(), new_str)
    except Exception:
        return num_str
 
DIR_PAIRS = [
    ("artış", "düşüş"), ("büyüme", "gerileme"), ("kar", "zarar"),
    ("kazanç", "kayıp"), ("yükseldi", "düştü"), ("arttı", "azaldı"),
    ("olumlu", "olumsuz"), ("güçlü", "zayıf"), ("iyileşme", "kötüleşme"),
]
 
FAKE_SENTENCES = [
    " Şirket yönetimi olağanüstü genel kurul toplantısı yapılacağını duyurdu.",
    " Bu gelişme sonucunda şirketin borsa değeri %15 geriledi.",
    " Düzenleyici kurumlar şirket hakkında inceleme başlattı.",
    " Yabancı yatırımcılar bu haberin ardından satış yaptı.",
]
 
def hall_type1(text):  # Sayısal distorsiyon
    nums = extract_numbers(text)
    if not nums:
        return text, False
    t = random.choice(nums)
    d = distort_number(t)
    return (text.replace(t, d, 1), True) if t != d else (text, False)
 
def hall_type2(text):  # Yön değişimi
    for orig, rep in DIR_PAIRS:
        if orig in text.lower():
            return re.sub(orig, rep, text, count=1, flags=re.IGNORECASE), True
    return text, False
 
def hall_type3(text):  # Sahte cümle ekleme
    sentences = text.split('.')
    pos = random.randint(0, max(0, len(sentences) - 1))
    sentences.insert(pos, random.choice(FAKE_SENTENCES).strip())
    return '.'.join(sentences), True
 
def hall_type4(text):  # Composite
    t, _ = hall_type1(text)
    t, ok = hall_type2(t)
    return t, True
 
HALL_FUNCS = {
    1: ("Numerical Distortion", hall_type1),
    2: ("Direction Reversal",   hall_type2),
    3: ("Fabricated Context",   hall_type3),
    4: ("Composite Error",      hall_type4),
}
 
# Sabit, uniform-olmayan ağırlıklar (Bölüm 4.3): Type 3'ün her özete
# uygulanabilir olması (100% availability) nedeniyle bilinçli olarak
# düşük ağırlıklandırılmıştır.
HALL_WEIGHTS = {1: 0.325, 2: 0.30, 3: 0.05, 4: 0.325}
 
print("\nHallucination enjeksiyonu...")
ground_rows, hall_rows = [], []
 
for doc_idx, row in df_summaries.iterrows():
    ground_rows.append({
        'source_document':   row['source_document'],
        'generated_summary': row['original_summary'],
        'label': 0, 'hall_type_name': 'Grounded',
        'doc_id': doc_idx
    })
 
    ht = random.choices(
        population=list(HALL_WEIGHTS.keys()),
        weights=list(HALL_WEIGHTS.values()),
        k=1
    )[0]
    hname, hfunc = HALL_FUNCS[ht]
    htext, ok = hfunc(row['original_summary'])
    if not ok:
        htext, _ = hall_type3(row['original_summary'])
        hname = "Fabricated Context"
 
    hall_rows.append({
        'source_document':   row['source_document'],
        'generated_summary': htext,
        'label': 1, 'hall_type_name': hname,
        'doc_id': doc_idx
    })
 
df_all = pd.DataFrame(ground_rows + hall_rows).reset_index(drop=True)
print(f"Toplam: {len(df_all)} | G:{(df_all['label']==0).sum()} | H:{(df_all['label']==1).sum()}")
print(df_all[df_all['label'] == 1]['hall_type_name'].value_counts())
 
# =============================================================================
# HÜCRE 4.5: Document-Level Train/Validation/Test Split (leakage önleme)
# =============================================================================
n_docs = len(df_summaries)
doc_ids = np.arange(n_docs)
 
# Table 2: validation = 80 instances (20%), test = 320 instances (80%).
# Bu oran belge sayisinda da korunur: 40 belge -> 80 ornek val,
# 160 belge -> 320 ornek test. Her belge (grounded+hallucinated)
# ayni partition'a duser.
val_doc_ids, test_doc_ids = train_test_split(
    doc_ids, test_size=0.80, random_state=42
)
 
assert set(val_doc_ids).isdisjoint(set(test_doc_ids)), "Leakage tespit edildi!"
 
val_mask  = df_all['doc_id'].isin(val_doc_ids)
test_mask = df_all['doc_id'].isin(test_doc_ids)
 
df_val  = df_all[val_mask].sample(frac=1, random_state=42).reset_index(drop=True)
df_test = df_all[test_mask].sample(frac=1, random_state=42).reset_index(drop=True)
 
print(f"\nValidation: {len(df_val)} ornek / {len(val_doc_ids)} belge")
print(f"Test:       {len(df_test)} ornek / {len(test_doc_ids)} belge")
 
labels_val  = df_val['label'].tolist()
labels_test = df_test['label'].tolist()
 
# =============================================================================
# HÜCRE 5: Embedding + Similarity (val ve test ayrı hesaplanır)
# =============================================================================
print("\nModeller yukleniyor...")
model_mpnet  = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
model_minilm = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
try:
    model_tr = SentenceTransformer('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')
    use_tr = True
    print("Turkish BERT yuklendi")
except Exception:
    use_tr = False
 
def get_sims(model, df, desc=""):
    sims = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        e1 = model.encode([str(row['source_document'])[:512]],    convert_to_numpy=True)
        e2 = model.encode([str(row['generated_summary'])[:512]],  convert_to_numpy=True)
        sims.append(float(cosine_similarity(e1, e2)[0][0]))
    return sims
 
print("\nSimilarity hesaplaniyor (validation)...")
sims_mpnet_val  = get_sims(model_mpnet,  df_val, "MPNet-Val")
sims_minilm_val = get_sims(model_minilm, df_val, "MiniLM-Val")
if use_tr:
    sims_tr_val = get_sims(model_tr, df_val, "TR-Val")
 
print("\nSimilarity hesaplaniyor (test)...")
sims_mpnet_test  = get_sims(model_mpnet,  df_test, "MPNet-Test")
sims_minilm_test = get_sims(model_minilm, df_test, "MiniLM-Test")
if use_tr:
    sims_tr_test = get_sims(model_tr, df_test, "TR-Test")
 
df_test['sim_mpnet']  = sims_mpnet_test
df_test['sim_minilm'] = sims_minilm_test
if use_tr:
    df_test['sim_tr'] = sims_tr_test
 
for name, sims in [('MPNet', sims_mpnet_test), ('MiniLM', sims_minilm_test)]:
    g = [s for s, l in zip(sims, labels_test) if l == 0]
    h = [s for s, l in zip(sims, labels_test) if l == 1]
    print(f"\n{name} (test): Grounded={np.mean(g):.4f} | Hallucinated={np.mean(h):.4f} | "
          f"Delta={np.mean(g)-np.mean(h):.4f}")
 
# =============================================================================
# HÜCRE 6: Threshold (validation-only) + Evaluation (test-only)
# =============================================================================
def optimize_tau(scores, labels):
    best_f1, best_tau = 0, 0.5
    rows = []
    for tau in np.arange(0.05, 0.95, 0.025):
        preds = [1 if s < tau else 0 for s in scores]
        f1 = f1_score(labels, preds, zero_division=0)
        rows.append({'tau': round(tau, 3),
                      'f1': f1,
                      'accuracy':  accuracy_score(labels, preds),
                      'precision': precision_score(labels, preds, zero_division=0),
                      'recall':    recall_score(labels, preds, zero_division=0)})
        if f1 > best_f1:
            best_f1, best_tau = f1, tau
    return best_tau, pd.DataFrame(rows)
 
def eval_model(scores, labels, tau):
    preds = [1 if s < tau else 0 for s in scores]
    return {'Precision': round(precision_score(labels, preds, zero_division=0), 4),
            'Recall':    round(recall_score(labels, preds, zero_division=0), 4),
            'F1-Score':  round(f1_score(labels, preds, zero_division=0), 4),
            'Accuracy':  round(accuracy_score(labels, preds), 4)}
 
# Esik SADECE validation setinde optimize edilir
tau_mpnet,  tdf_mpnet  = optimize_tau(sims_mpnet_val,  labels_val)
tau_minilm, tdf_minilm = optimize_tau(sims_minilm_val, labels_val)
if use_tr:
    tau_tr, tdf_tr = optimize_tau(sims_tr_val, labels_val)
 
# Tum metrikler SADECE test setinde raporlanir
model_results = {
    'MPNet':  eval_model(sims_mpnet_test,  labels_test, tau_mpnet),
    'MiniLM': eval_model(sims_minilm_test, labels_test, tau_minilm),
}
if use_tr:
    model_results['Turkish-BERT'] = eval_model(sims_tr_test, labels_test, tau_tr)
 
results_df = pd.DataFrame(model_results).T
print("\n" + "=" * 60)
print("TABLO 4: Model Karsilastirma (test set)")
print("=" * 60)
print(results_df.to_string())
print(f"\nKalibre edilen esikler (validation): "
      f"MPNet={tau_mpnet:.3f} MiniLM={tau_minilm:.3f}"
      + (f" TurkishBERT={tau_tr:.3f}" if use_tr else ""))
 
# Korelasyon (test set)
print("\nKorelasyon Analizi (test):")
for name, sims in [('MPNet', sims_mpnet_test), ('MiniLM', sims_minilm_test)] + \
                   ([('Turkish-BERT', sims_tr_test)] if use_tr else []):
    pr, pp = pearsonr(sims, labels_test)
    sig = "OK" if pp < 0.05 else "!!"
    print(f"  {name:15s}: r={pr:+.4f} p={pp:.2e} [{sig}]")
 
# Hallucination tipi analizi (test set)
print("\nHallucination Tipi Bazli Detection (test):")
type_rows = []
for tname in ["Numerical Distortion", "Direction Reversal", "Fabricated Context", "Composite Error"]:
    mask = (df_test['hall_type_name'] == tname) & (df_test['label'] == 1)
    if mask.sum() < 1:
        continue
    scs = df_test.loc[mask, 'sim_mpnet'].values
    preds = [1 if s < tau_mpnet else 0 for s in scs]
    det = np.mean(preds)
    type_rows.append({'Type': tname, 'N': int(mask.sum()),
                       'Mean Sim': round(float(np.mean(scs)), 4),
                       'Std': round(float(np.std(scs)), 4),
                       'Detection Rate': round(float(det), 4)})
    print(f"  {tname:25s}: N={mask.sum():3d} | Sim={np.mean(scs):.4f} | Det={det*100:.1f}%")
 
type_df = pd.DataFrame(type_rows)
 
# Reduction analizi (test set)
print("\nHallucination Reduction (test):")
baseline = np.mean(labels_test) * 100
red_rows = {}
for name, sims, tau in [('MPNet', sims_mpnet_test, tau_mpnet),
                         ('MiniLM', sims_minilm_test, tau_minilm)] + \
                        ([('Turkish-BERT', sims_tr_test, tau_tr)] if use_tr else []):
    acc_i = [i for i, s in enumerate(sims) if s >= tau]
    filt = np.mean([labels_test[i] for i in acc_i]) * 100 if acc_i else 0
    cov = len(acc_i) / len(sims) * 100
    red_rows[name] = {'Baseline(%)': round(baseline, 1), 'Filtered(%)': round(filt, 1),
                       'Reduction(%)': round(baseline - filt, 1),
                       'RelReduction(%)': round((baseline - filt) / baseline * 100, 1) if baseline else 0,
                       'Coverage(%)': round(cov, 1)}
    print(f"  {name}: {baseline:.1f}% -> {filt:.1f}% (rel. red.={red_rows[name]['RelReduction(%)']:.1f}%, "
          f"cov={cov:.1f}%)")
 
reduction_df = pd.DataFrame(red_rows).T
 
# =============================================================================
# HÜCRE 6.5: Threshold Stability Across 5 Random Splits (Bölüm 3.5)
# =============================================================================
print("\n" + "=" * 60)
print("Esik stabilitesi: 5 bagimsiz rastgele validation split")
print("=" * 60)
 
taus_across_splits = {'MPNet': [], 'MiniLM': []}
if use_tr:
    taus_across_splits['Turkish-BERT'] = []
 
for seed in [42, 43, 44, 45, 46]:
    v_ids, _ = train_test_split(doc_ids, test_size=0.80, random_state=seed)
    v_mask = df_all['doc_id'].isin(v_ids)
    dv = df_all[v_mask].reset_index(drop=True)
    lv = dv['label'].tolist()
 
    sv_mpnet  = get_sims(model_mpnet,  dv, f"seed{seed}-MPNet")
    sv_minilm = get_sims(model_minilm, dv, f"seed{seed}-MiniLM")
 
    t_mpnet,  _ = optimize_tau(sv_mpnet,  lv)
    t_minilm, _ = optimize_tau(sv_minilm, lv)
    taus_across_splits['MPNet'].append(t_mpnet)
    taus_across_splits['MiniLM'].append(t_minilm)
 
    if use_tr:
        sv_tr = get_sims(model_tr, dv, f"seed{seed}-TR")
        t_tr, _ = optimize_tau(sv_tr, lv)
        taus_across_splits['Turkish-BERT'].append(t_tr)
 
print()
stability_rows = []
for name, taus in taus_across_splits.items():
    if taus:
        std = float(np.std(taus))
        print(f"{name}: tau* values = {[round(t,3) for t in taus]}, std = {std:.4f}")
        stability_rows.append({'Model': name, 'Taus': taus, 'Std': std})
 
stability_df = pd.DataFrame(stability_rows)
stability_df.to_csv('threshold_stability_v6.csv', index=False)
 
# =============================================================================
# HÜCRE 7: Görselleştirmeler (test set üzerinden)
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Turkish Financial Hallucination Detection - KAP Dataset (Test Set)',
             fontsize=13, fontweight='bold')
 
ax = axes[0, 0]
g_sc = [s for s, l in zip(sims_mpnet_test, labels_test) if l == 0]
h_sc = [s for s, l in zip(sims_mpnet_test, labels_test) if l == 1]
ax.hist(g_sc, bins=25, alpha=0.65, color='#2196F3', label=f'Grounded (n={len(g_sc)})')
ax.hist(h_sc, bins=25, alpha=0.65, color='#F44336', label=f'Hallucinated (n={len(h_sc)})')
ax.axvline(x=tau_mpnet, color='black', ls='--', lw=2, label=f'tau*={tau_mpnet:.3f}')
ax.set_xlabel('Semantic Similarity (MPNet)'); ax.set_ylabel('Frequency')
ax.set_title('Fig 1: Similarity Distribution (Test)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
 
ax = axes[0, 1]
ax.plot(tdf_mpnet['tau'], tdf_mpnet['f1'], 'b-o', lw=2, ms=3, label='F1')
ax.plot(tdf_mpnet['tau'], tdf_mpnet['accuracy'], 'g-s', lw=2, ms=3, label='Accuracy')
ax.plot(tdf_mpnet['tau'], tdf_mpnet['precision'], 'r-^', lw=2, ms=3, label='Precision')
ax.plot(tdf_mpnet['tau'], tdf_mpnet['recall'], 'm-v', lw=2, ms=3, label='Recall')
ax.axvline(x=tau_mpnet, color='black', ls='--', lw=2, label=f'tau*={tau_mpnet:.3f}')
ax.set_xlabel('Threshold (tau)'); ax.set_ylabel('Score')
ax.set_title('Fig 2: Threshold Sensitivity (Validation)'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
 
ax = axes[0, 2]
metrics = ['F1-Score', 'Accuracy', 'Precision', 'Recall']
x = np.arange(len(metrics)); w = 0.8 / len(results_df)
cm = plt.cm.get_cmap('tab10')
for i, (mname, row) in enumerate(results_df.iterrows()):
    ax.bar(x + i * w, [row[m] for m in metrics], w, label=mname, color=cm(i), alpha=0.85)
ax.set_xticks(x + w * (len(results_df) - 1) / 2); ax.set_xticklabels(metrics, fontsize=8)
ax.set_ylim(0, 1.15); ax.set_ylabel('Score'); ax.set_title('Fig 3: Model Comparison (Test)')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis='y')
 
ax = axes[1, 0]
if len(type_df) > 0:
    colors_t = ['#F44336', '#FF9800', '#4CAF50', '#9C27B0']
    bars = ax.barh(type_df['Type'], type_df['Detection Rate'],
                    color=colors_t[:len(type_df)], alpha=0.85, edgecolor='white')
    ax.set_xlim(0, 1.15); ax.set_xlabel('Detection Rate')
    ax.set_title('Fig 4: Detection by Hallucination Type (Test)')
    for bar, val in zip(bars, type_df['Detection Rate']):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{val:.2f}', va='center', fontsize=9)
ax.grid(True, alpha=0.3, axis='x')
 
ax = axes[1, 1]
box_data, box_labels = [], []
for tname in ["Numerical Distortion", "Direction Reversal", "Fabricated Context", "Composite Error"]:
    mask = (df_test['hall_type_name'] == tname) & (df_test['label'] == 1)
    if mask.sum() > 1:
        box_data.append(df_test.loc[mask, 'sim_mpnet'].values)
        box_labels.append(tname.replace(' ', '\n'))
box_data.append(df_test.loc[df_test['label'] == 0, 'sim_mpnet'].values)
box_labels.append('Grounded')
bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
colors_b = ['#F44336', '#FF9800', '#4CAF50', '#9C27B0', '#2196F3']
for patch, color in zip(bp['boxes'], colors_b):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax.axhline(y=tau_mpnet, color='black', ls='--', lw=1.5, label=f'tau*={tau_mpnet:.3f}')
ax.set_ylabel('Similarity Score'); ax.set_title('Fig 5: Similarity by Hall. Type (Test)')
ax.legend(fontsize=7); ax.tick_params(axis='x', labelsize=6); ax.grid(True, alpha=0.3, axis='y')
 
ax = axes[1, 2]
taus_p = np.arange(0.05, 0.95, 0.025)
covs, h_rates = [], []
for tau in taus_p:
    acc_i = [i for i, s in enumerate(sims_mpnet_test) if s >= tau]
    if acc_i:
        h_rates.append(np.mean([labels_test[i] for i in acc_i]) * 100)
        covs.append(len(acc_i) / len(sims_mpnet_test) * 100)
    else:
        h_rates.append(0); covs.append(0)
sc = ax.scatter(covs, h_rates, c=taus_p[:len(covs)], cmap='RdYlGn_r', s=50, alpha=0.85)
plt.colorbar(sc, ax=ax, label='Threshold tau')
ax.axhline(y=baseline, color='red', ls='--', lw=1.5, label=f'Baseline {baseline:.0f}%')
ax.set_xlabel('Coverage (%)'); ax.set_ylabel('Hallucination Rate (%)')
ax.set_title('Fig 6: Coverage-Hallucination Trade-off (Test)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
 
plt.tight_layout()
plt.savefig('hallucination_results_v6.png', dpi=150, bbox_inches='tight')
plt.show()
print("hallucination_results_v6.png kaydedildi")
 
# Kaydet
df_test.to_csv('kap_test_results_v6.csv', index=False)
df_val.to_csv('kap_val_results_v6.csv', index=False)
results_df.to_csv('model_comparison_v6.csv')
type_df.to_csv('hallucination_type_v6.csv', index=False)
reduction_df.to_csv('reduction_table_v6.csv')
 
print("\n" + "=" * 60)
print("TAMAMLANDI (v6 - document-level split, val/test ayrik)")
print("=" * 60)
best_m = results_df['F1-Score'].idxmax()
print(f"  En iyi model : {best_m}")
print(f"  F1-Score     : {results_df.loc[best_m,'F1-Score']:.4f}")
print(f"  Accuracy     : {results_df.loc[best_m,'Accuracy']:.4f}")
print(f"  Precision    : {results_df.loc[best_m,'Precision']:.4f}")
print(f"  Recall       : {results_df.loc[best_m,'Recall']:.4f}")
