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
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import pearsonr, spearmanr, chi2
import anthropic
warnings.filterwarnings('ignore')
random.seed(42); np.random.seed(42)

import torch
print(f"GPU: {torch.cuda.is_available()}")

def _get_key(name, colab_secret_name=None):
    try:
        from google.colab import userdata
        if colab_secret_name:
            v = userdata.get(colab_secret_name)
            if v:
                return v
    except Exception:
        pass
    return os.environ.get(name, '')

api_key = _get_key('ANTHROPIC_API_KEY', 'sci')
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY bulunamadi! Colab Secrets'a ekle.")
client = anthropic.Anthropic(api_key=api_key)

openai_key = _get_key('OPENAI_API_KEY')
together_key = _get_key('TOGETHER_API_KEY')

USE_GPT4O = bool(openai_key)
USE_LLAMA = bool(together_key)
print(f"GPT-4o baseline aktif mi: {USE_GPT4O}")
print(f"Llama-3-8B baseline aktif mi: {USE_LLAMA}")

if USE_GPT4O:
    from openai import OpenAI
    openai_client = OpenAI(api_key=openai_key)

if USE_LLAMA:
    from together import Together
    together_client = Together(api_key=together_key)

print("Anthropic API baglantisi test ediliyor...")
try:
    test = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=10,
        messages=[{"role": "user", "content": "test"}]
    )
    MODEL_NAME = "claude-haiku-4-5-20251001"
except Exception:
    try:
        test = client.messages.create(
            model="claude-haiku-4-5", max_tokens=10,
            messages=[{"role": "user", "content": "test"}]
        )
        MODEL_NAME = "claude-haiku-4-5"
    except Exception:
        test = client.messages.create(
            model="claude-3-haiku-20240307", max_tokens=10,
            messages=[{"role": "user", "content": "test"}]
        )
        MODEL_NAME = "claude-3-haiku-20240307"
print(f"Anthropic API calisiyor (model: {MODEL_NAME})")

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
                model=MODEL_NAME, max_tokens=250,
                messages=[{"role": "user", "content": PROMPT.format(doc=doc[:1500])}]
            )
            text = r.content[0].text.strip()
            return text if len(text) > 20 else None
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
rows, failed = [], 0
for idx, row in tqdm(df_to_sum.iterrows(), total=len(df_to_sum)):
    summary = summarize(row['source_document'])
    if summary:
        rows.append({'source_document': row['source_document'], 'original_summary': summary})
    else:
        failed += 1
print(f"\nBasarili: {len(rows)} | Basarisiz: {failed}")
if len(rows) == 0:
    raise RuntimeError("Hic ozet uretilemedi!")

df_summaries = pd.DataFrame(rows)
df_summaries.to_csv('kap_summaries.csv', index=False)

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

def hall_type1(text):
    nums = extract_numbers(text)
    if not nums:
        return text, False
    t = random.choice(nums)
    d = distort_number(t)
    return (text.replace(t, d, 1), True) if t != d else (text, False)

def hall_type2(text):
    for orig, rep in DIR_PAIRS:
        if orig in text.lower():
            return re.sub(orig, rep, text, count=1, flags=re.IGNORECASE), True
    return text, False

def hall_type3(text):
    sentences = text.split('.')
    pos = random.randint(0, max(0, len(sentences) - 1))
    sentences.insert(pos, random.choice(FAKE_SENTENCES).strip())
    return '.'.join(sentences), True

def hall_type4(text):
    t, _ = hall_type1(text)
    t, ok = hall_type2(t)
    return t, True

HALL_FUNCS = {
    1: ("Numerical Distortion", hall_type1),
    2: ("Direction Reversal",   hall_type2),
    3: ("Fabricated Context",   hall_type3),
    4: ("Composite Error",      hall_type4),
}
HALL_WEIGHTS = {1: 0.325, 2: 0.30, 3: 0.05, 4: 0.325}

print("\nHallucination enjeksiyonu...")
ground_rows, hall_rows = [], []
for doc_idx, row in df_summaries.iterrows():
    ground_rows.append({
        'source_document': row['source_document'], 'generated_summary': row['original_summary'],
        'label': 0, 'hall_type_name': 'Grounded', 'doc_id': doc_idx
    })
    ht = random.choices(list(HALL_WEIGHTS.keys()), weights=list(HALL_WEIGHTS.values()), k=1)[0]
    hname, hfunc = HALL_FUNCS[ht]
    htext, ok = hfunc(row['original_summary'])
    if not ok:
        htext, _ = hall_type3(row['original_summary'])
        hname = "Fabricated Context"
    hall_rows.append({
        'source_document': row['source_document'], 'generated_summary': htext,
        'label': 1, 'hall_type_name': hname, 'doc_id': doc_idx
    })

df_all = pd.DataFrame(ground_rows + hall_rows).reset_index(drop=True)
print(f"Toplam: {len(df_all)} | G:{(df_all['label']==0).sum()} | H:{(df_all['label']==1).sum()}")
print(df_all[df_all['label'] == 1]['hall_type_name'].value_counts())

# =============================================================================
# HÜCRE 4.5: Document-Level Train/Validation/Test Split
# =============================================================================
n_docs = len(df_summaries)
doc_ids = np.arange(n_docs)
val_doc_ids, test_doc_ids = train_test_split(doc_ids, test_size=0.80, random_state=42)
assert set(val_doc_ids).isdisjoint(set(test_doc_ids)), "Leakage tespit edildi!"

val_mask  = df_all['doc_id'].isin(val_doc_ids)
test_mask = df_all['doc_id'].isin(test_doc_ids)
df_val  = df_all[val_mask].sample(frac=1, random_state=42).reset_index(drop=True)
df_test = df_all[test_mask].sample(frac=1, random_state=42).reset_index(drop=True)
labels_val  = df_val['label'].tolist()
labels_test = df_test['label'].tolist()
print(f"\nValidation: {len(df_val)} ornek / {len(val_doc_ids)} belge")
print(f"Test:       {len(df_test)} ornek / {len(test_doc_ids)} belge")

# =============================================================================
# HÜCRE 5: Embedding + Similarity (val ve test ayrı hesaplanır)
# =============================================================================
print("\nModeller yukleniyor...")
model_mpnet  = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
model_minilm = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
try:
    model_tr = SentenceTransformer('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')
    use_tr = True
except Exception:
    use_tr = False

def get_sims(model, df, desc=""):
    sims = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        e1 = model.encode([str(row['source_document'])[:512]], convert_to_numpy=True)
        e2 = model.encode([str(row['generated_summary'])[:512]], convert_to_numpy=True)
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

# =============================================================================
# HÜCRE 6: Threshold (validation-only) + Evaluation (test-only) — Exp 1 & 2
# =============================================================================
def optimize_tau(scores, labels):
    best_f1, best_tau = 0, 0.5
    rows = []
    for tau in np.arange(0.05, 0.95, 0.025):
        preds = [1 if s < tau else 0 for s in scores]
        f1 = f1_score(labels, preds, zero_division=0)
        rows.append({'tau': round(tau, 3), 'f1': f1,
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

def preds_from_sims(scores, tau):
    return [1 if s < tau else 0 for s in scores]

tau_mpnet,  tdf_mpnet  = optimize_tau(sims_mpnet_val,  labels_val)
tau_minilm, tdf_minilm = optimize_tau(sims_minilm_val, labels_val)
if use_tr:
    tau_tr, tdf_tr = optimize_tau(sims_tr_val, labels_val)

model_results = {
    'MPNet':  eval_model(sims_mpnet_test,  labels_test, tau_mpnet),
    'MiniLM': eval_model(sims_minilm_test, labels_test, tau_minilm),
}
if use_tr:
    model_results['Turkish-BERT'] = eval_model(sims_tr_test, labels_test, tau_tr)

results_df = pd.DataFrame(model_results).T
print("\n" + "=" * 60)
print("TABLO 4: Sentence-Encoder Karsilastirma (test set)")
print("=" * 60)
print(results_df.to_string())
print(f"\nKalibre edilen esikler (validation): MPNet={tau_mpnet:.3f} MiniLM={tau_minilm:.3f}"
      + (f" TurkishBERT={tau_tr:.3f}" if use_tr else ""))

# --- Coverage hesapla (Tablo 4'teki Cov. sütunu) ---
for name, sims, tau in [('MPNet', sims_mpnet_test, tau_mpnet), ('MiniLM', sims_minilm_test, tau_minilm)] + \
                        ([('Turkish-BERT', sims_tr_test, tau_tr)] if use_tr else []):
    cov = np.mean([1 if s >= tau else 0 for s in sims]) * 100
    print(f"  {name} coverage: {cov:.1f}%")

# =============================================================================
# HÜCRE 7: Lexical & Token-Level Baselines (TF-IDF, BERTScore-P)
# =============================================================================
print("\n" + "=" * 60)
print("Lexical / token-level baseline hesaplaniyor")
print("=" * 60)

# TF-IDF cosine: korpus = validation+test tum source+summary metinleri
corpus = (df_val['source_document'].tolist() + df_val['generated_summary'].tolist() +
          df_test['source_document'].tolist() + df_test['generated_summary'].tolist())
tfidf = TfidfVectorizer(max_features=20000)
tfidf.fit(corpus)

def tfidf_sims(df):
    D = tfidf.transform(df['source_document'].tolist())
    S = tfidf.transform(df['generated_summary'].tolist())
    return [float(cosine_similarity(D[i], S[i])[0][0]) for i in range(D.shape[0])]

sims_tfidf_val  = tfidf_sims(df_val)
sims_tfidf_test = tfidf_sims(df_test)
tau_tfidf, _ = optimize_tau(sims_tfidf_val, labels_val)
tfidf_result = eval_model(sims_tfidf_test, labels_test, tau_tfidf)
tfidf_cov = np.mean([1 if s >= tau_tfidf else 0 for s in sims_tfidf_test]) * 100
print(f"TF-IDF Cosine: {tfidf_result} | Coverage={tfidf_cov:.1f}%")

# BERTScore-P (precision component), bert-base-multilingual-cased
try:
    from bert_score import score as bertscore_fn
    def bertscore_p_sims(df):
        P, R, F1 = bertscore_fn(
            cands=df['generated_summary'].tolist(),
            refs=df['source_document'].tolist(),
            model_type='bert-base-multilingual-cased', lang='tr', verbose=False
        )
        return P.tolist()
    sims_bsp_val  = bertscore_p_sims(df_val)
    sims_bsp_test = bertscore_p_sims(df_test)
    tau_bsp, _ = optimize_tau(sims_bsp_val, labels_val)
    bsp_result = eval_model(sims_bsp_test, labels_test, tau_bsp)
    bsp_cov = np.mean([1 if s >= tau_bsp else 0 for s in sims_bsp_test]) * 100
    print(f"BERTScore-P: {bsp_result} | Coverage={bsp_cov:.1f}%")
    HAVE_BSP = True
except Exception as e:
    print(f"BERTScore-P atlandi (kutuphane/model hatasi): {e}")
    HAVE_BSP = False

# =============================================================================
# HÜCRE 8: LLM Zero-Shot Detector Baselines (100% coverage)
# =============================================================================
print("\n" + "=" * 60)
print("LLM zero-shot detector baseline'lari hesaplaniyor")
print("=" * 60)

VERDICT_PROMPT = """Aşağıda bir kaynak belge ve onun özeti verilmiştir. Özetin
kaynak belgeyle tam olarak tutarlı (grounded) mı yoksa kaynakta olmayan/kaynakla
çelişen bilgi (hallucination) içerip içermediğini değerlendir.

Sadece tek kelime cevap ver: "GROUNDED" veya "HALLUCINATED".

Kaynak belge:
{doc}

Özet:
{summary}

Cevap:"""

def parse_verdict(text):
    t = text.strip().upper()
    if "HALLUCINAT" in t:
        return 1
    if "GROUND" in t:
        return 0
    return None  # belirsiz

def claude_haiku_verdict(doc, summary, retries=3):
    for attempt in range(retries):
        try:
            r = client.messages.create(
                model=MODEL_NAME, max_tokens=10,
                messages=[{"role": "user",
                           "content": VERDICT_PROMPT.format(doc=doc[:1500], summary=summary)}]
            )
            v = parse_verdict(r.content[0].text)
            if v is not None:
                return v
        except Exception:
            time.sleep(2)
    return random.choice([0, 1])  # cozulemeyen durumda rastgele (nadir olmali)

def gpt4o_verdict(doc, summary, retries=3):
    for attempt in range(retries):
        try:
            r = openai_client.chat.completions.create(
                model="gpt-4o", max_tokens=10,
                messages=[{"role": "user",
                           "content": VERDICT_PROMPT.format(doc=doc[:1500], summary=summary)}]
            )
            v = parse_verdict(r.choices[0].message.content)
            if v is not None:
                return v
        except Exception:
            time.sleep(2)
    return random.choice([0, 1])

def llama3_verdict(doc, summary, retries=3):
    for attempt in range(retries):
        try:
            r = together_client.chat.completions.create(
                model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo", max_tokens=10,
                messages=[{"role": "user",
                           "content": VERDICT_PROMPT.format(doc=doc[:1500], summary=summary)}]
            )
            v = parse_verdict(r.choices[0].message.content)
            if v is not None:
                return v
        except Exception:
            time.sleep(2)
    return random.choice([0, 1])

def run_llm_detector(verdict_fn, df, desc):
    preds = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        preds.append(verdict_fn(row['source_document'], row['generated_summary']))
    return preds

# Claude Haiku (her zaman calisir, ayni model ozet uretiminde de kullanildi)
preds_haiku_test = run_llm_detector(claude_haiku_verdict, df_test, "Claude-Haiku-verdict")
haiku_result = {'Precision': round(precision_score(labels_test, preds_haiku_test, zero_division=0), 4),
                 'Recall':    round(recall_score(labels_test, preds_haiku_test, zero_division=0), 4),
                 'F1-Score':  round(f1_score(labels_test, preds_haiku_test, zero_division=0), 4),
                 'Accuracy':  round(accuracy_score(labels_test, preds_haiku_test), 4)}
print(f"Claude Haiku zero-shot: {haiku_result}")

if USE_GPT4O:
    preds_gpt4o_test = run_llm_detector(gpt4o_verdict, df_test, "GPT-4o-verdict")
    gpt4o_result = {'Precision': round(precision_score(labels_test, preds_gpt4o_test, zero_division=0), 4),
                     'Recall':    round(recall_score(labels_test, preds_gpt4o_test, zero_division=0), 4),
                     'F1-Score':  round(f1_score(labels_test, preds_gpt4o_test, zero_division=0), 4),
                     'Accuracy':  round(accuracy_score(labels_test, preds_gpt4o_test), 4)}
    print(f"GPT-4o zero-shot: {gpt4o_result}")
else:
    preds_gpt4o_test = None
    print("GPT-4o atlandi (OPENAI_API_KEY yok)")

if USE_LLAMA:
    preds_llama_test = run_llm_detector(llama3_verdict, df_test, "Llama-3-verdict")
    llama_result = {'Precision': round(precision_score(labels_test, preds_llama_test, zero_division=0), 4),
                     'Recall':    round(recall_score(labels_test, preds_llama_test, zero_division=0), 4),
                     'F1-Score':  round(f1_score(labels_test, preds_llama_test, zero_division=0), 4),
                     'Accuracy':  round(accuracy_score(labels_test, preds_llama_test), 4)}
    print(f"Llama-3-8B zero-shot: {llama_result}")
else:
    preds_llama_test = None
    print("Llama-3-8B atlandi (TOGETHER_API_KEY yok)")

# =============================================================================
# HÜCRE 9: Tablo 5 (kapsamlı karşılaştırma) + McNemar + Bootstrap CI — Exp 1
# =============================================================================
print("\n" + "=" * 60)
print("TABLO 5: Kapsamli Model Karsilastirma (test set)")
print("=" * 60)

table5_rows = {'TF-IDF Cosine': tfidf_result}
if HAVE_BSP:
    table5_rows['BERTScore-P'] = bsp_result
if USE_LLAMA:
    table5_rows['Llama-3-8B'] = llama_result
table5_rows['Claude Haiku'] = haiku_result
if USE_GPT4O:
    table5_rows['GPT-4o'] = gpt4o_result
table5_rows['Turkish BERT'] = model_results.get('Turkish-BERT', {})
table5_rows['MiniLM'] = model_results['MiniLM']
table5_rows['MPNet'] = model_results['MPNet']

table5_df = pd.DataFrame(table5_rows).T
print(table5_df.to_string())
table5_df.to_csv('table5_comprehensive_comparison_v7.csv')

# --- McNemar testleri ---
def mcnemar_test(preds_a, preds_b, labels):
    correct_a = [int(p == l) for p, l in zip(preds_a, labels)]
    correct_b = [int(p == l) for p, l in zip(preds_b, labels)]
    b = sum(1 for ca, cb in zip(correct_a, correct_b) if ca == 1 and cb == 0)
    c = sum(1 for ca, cb in zip(correct_a, correct_b) if ca == 0 and cb == 1)
    if b + c == 0:
        return 0.0, 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)  # continuity correction
    p = float(chi2.sf(stat, df=1))
    return float(stat), p

preds_mpnet_test  = preds_from_sims(sims_mpnet_test,  tau_mpnet)
preds_minilm_test = preds_from_sims(sims_minilm_test, tau_minilm)
mcnemar_rows = []
stat, p = mcnemar_test(preds_mpnet_test, preds_minilm_test, labels_test)
mcnemar_rows.append({'Model A': 'MPNet', 'Model B': 'MiniLM', 'chi2': stat, 'p': p})

if use_tr:
    preds_tr_test = preds_from_sims(sims_tr_test, tau_tr)
    stat, p = mcnemar_test(preds_mpnet_test, preds_tr_test, labels_test)
    mcnemar_rows.append({'Model A': 'MPNet', 'Model B': 'Turkish BERT', 'chi2': stat, 'p': p})
    stat, p = mcnemar_test(preds_minilm_test, preds_tr_test, labels_test)
    mcnemar_rows.append({'Model A': 'MiniLM', 'Model B': 'Turkish BERT', 'chi2': stat, 'p': p})

if USE_GPT4O:
    stat, p = mcnemar_test(preds_mpnet_test, preds_gpt4o_test, labels_test)
    mcnemar_rows.append({'Model A': 'MPNet', 'Model B': 'GPT-4o', 'chi2': stat, 'p': p})

mcnemar_df = pd.DataFrame(mcnemar_rows)
print("\nTABLO 6: McNemar test sonuclari")
print(mcnemar_df.to_string(index=False))
mcnemar_df.to_csv('table6_mcnemar_v7.csv', index=False)

# --- Bootstrap CI (F1, 10000 resample) ---
def bootstrap_f1_ci(sims, labels, tau, n_boot=10000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(labels)
    f1s = []
    sims_arr = np.array(sims); labels_arr = np.array(labels)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        preds = [1 if s < tau else 0 for s in sims_arr[idx]]
        f1s.append(f1_score(labels_arr[idx], preds, zero_division=0))
    return float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))

ci_rows = []
for name, sims, tau in [('MPNet', sims_mpnet_test, tau_mpnet), ('MiniLM', sims_minilm_test, tau_minilm)] + \
                        ([('Turkish-BERT', sims_tr_test, tau_tr)] if use_tr else []):
    lo, hi = bootstrap_f1_ci(sims, labels_test, tau)
    ci_rows.append({'Model': name, 'F1_CI_low': round(lo, 4), 'F1_CI_high': round(hi, 4)})
ci_df = pd.DataFrame(ci_rows)
print("\nBootstrap %95 F1 guven araliklari:")
print(ci_df.to_string(index=False))
ci_df.to_csv('bootstrap_ci_v7.csv', index=False)

# =============================================================================
# HÜCRE 10: Korelasyon Analizi — Exp 5
# =============================================================================
print("\n" + "=" * 60)
print("TABLO 9/10: Korelasyon Analizi (test set)")
print("=" * 60)
corr_rows, groupstat_rows = [], []
for name, sims in [('MPNet', sims_mpnet_test), ('MiniLM', sims_minilm_test)] + \
                   ([('Turkish-BERT', sims_tr_test)] if use_tr else []):
    pr, pp = pearsonr(sims, labels_test)
    sr, sp = spearmanr(sims, labels_test)
    corr_rows.append({'Model': name, 'Pearson_r': round(pr, 4), 'Pearson_p': pp,
                       'Spearman_rho': round(sr, 4), 'Spearman_p': sp})
    g = [s for s, l in zip(sims, labels_test) if l == 0]
    h = [s for s, l in zip(sims, labels_test) if l == 1]
    groupstat_rows.append({'Model': name, 'Grounded_mu': round(np.mean(g), 4),
                            'Grounded_sigma': round(np.std(g), 4),
                            'Hallucinated_mu': round(np.mean(h), 4),
                            'Hallucinated_sigma': round(np.std(h), 4),
                            'Delta_mu': round(np.mean(g) - np.mean(h), 4)})
corr_df = pd.DataFrame(corr_rows)
groupstat_df = pd.DataFrame(groupstat_rows)
print(corr_df.to_string(index=False))
print(groupstat_df.to_string(index=False))
corr_df.to_csv('table9_correlation_v7.csv', index=False)
groupstat_df.to_csv('table10_groupstats_v7.csv', index=False)

# =============================================================================
# HÜCRE 11: Per-Type Detection — Exp 3 (Table 7)
# =============================================================================
print("\n" + "=" * 60)
print("TABLO 7: Per-Type Hallucination Detection (MPNet, test set)")
print("=" * 60)
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
g_mask = df_test['label'] == 0
type_rows.append({'Type': 'Grounded (reference)', 'N': int(g_mask.sum()),
                   'Mean Sim': round(float(df_test.loc[g_mask, 'sim_mpnet'].mean()), 4),
                   'Std': round(float(df_test.loc[g_mask, 'sim_mpnet'].std()), 4),
                   'Detection Rate': None})
type_df = pd.DataFrame(type_rows)
type_df.to_csv('table7_per_type_v7.csv', index=False)

# =============================================================================
# HÜCRE 12: Hallucination Reduction & Coverage — Exp 4 (Table 8)
# =============================================================================
print("\n" + "=" * 60)
print("TABLO 8: Hallucination Reduction (test set)")
print("=" * 60)
baseline = np.mean(labels_test) * 100
red_rows = {}
for name, sims, tau in [('MPNet', sims_mpnet_test, tau_mpnet), ('MiniLM', sims_minilm_test, tau_minilm)] + \
                        ([('Turkish-BERT', sims_tr_test, tau_tr)] if use_tr else []):
    acc_i = [i for i, s in enumerate(sims) if s >= tau]
    filt = np.mean([labels_test[i] for i in acc_i]) * 100 if acc_i else 0
    cov = len(acc_i) / len(sims) * 100
    red_rows[name] = {'Baseline(%)': round(baseline, 1), 'Filtered(%)': round(filt, 1),
                       'Abs_Red(pp)': round(baseline - filt, 1),
                       'Rel_Red(%)': round((baseline - filt) / baseline * 100, 1) if baseline else 0,
                       'Coverage(%)': round(cov, 1)}
    print(f"  {name}: {baseline:.1f}% -> {filt:.1f}% "
          f"(rel.red={red_rows[name]['Rel_Red(%)']:.1f}%, cov={cov:.1f}%)")
reduction_df = pd.DataFrame(red_rows).T
reduction_df.to_csv('table8_reduction_v7.csv')

# =============================================================================
# HÜCRE 13: Threshold Stability Across 5 Random Splits
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

stability_rows = []
for name, taus in taus_across_splits.items():
    if taus:
        std = float(np.std(taus))
        print(f"{name}: tau* values = {[round(t,3) for t in taus]}, std = {std:.4f}")
        stability_rows.append({'Model': name, 'Taus': taus, 'Std': std})
pd.DataFrame(stability_rows).to_csv('threshold_stability_v7.csv', index=False)

# =============================================================================
# HÜCRE 14: Sentence-Level Aggregation Ablation (dilution effect testi)
# =============================================================================
print("\n" + "=" * 60)
print("Sentence-level aggregation ablasyonu (MPNet, test set)")
print("=" * 60)

def get_sentence_level_sims(model, df, agg='min'):
    sims = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"SentAgg-{agg}"):
        doc_emb = model.encode([str(row['source_document'])[:512]], convert_to_numpy=True)
        sentences = [s.strip() for s in str(row['generated_summary']).split('.') if s.strip()]
        if not sentences:
            sims.append(0.0); continue
        sent_embs = model.encode(sentences, convert_to_numpy=True)
        sent_sims = cosine_similarity(doc_emb, sent_embs)[0]
        sims.append(float(sent_sims.min()) if agg == 'min' else float(sent_sims.max()))
    return sims

sims_min_val  = get_sentence_level_sims(model_mpnet, df_val,  agg='min')
sims_min_test = get_sentence_level_sims(model_mpnet, df_test, agg='min')
sims_max_val  = get_sentence_level_sims(model_mpnet, df_val,  agg='max')
sims_max_test = get_sentence_level_sims(model_mpnet, df_test, agg='max')

tau_min, _ = optimize_tau(sims_min_val, labels_val)
tau_max, _ = optimize_tau(sims_max_val, labels_val)

agg_results = {
    'Whole-document (baseline)': eval_model(sims_mpnet_test, labels_test, tau_mpnet),
    'Sentence-level min':        eval_model(sims_min_test, labels_test, tau_min),
    'Sentence-level max':        eval_model(sims_max_test, labels_test, tau_max),
}
agg_df = pd.DataFrame(agg_results).T
print(agg_df.to_string())

mask_t3 = (df_test['hall_type_name'] == 'Fabricated Context') & (df_test['label'] == 1)
if mask_t3.sum() > 0:
    idx_t3 = df_test.index[mask_t3]
    det_whole_t3 = np.mean([1 if sims_mpnet_test[i] < tau_mpnet else 0 for i in idx_t3])
    det_min_t3   = np.mean([1 if sims_min_test[i]   < tau_min   else 0 for i in idx_t3])
    print(f"\nType 3 detection - whole-document: {det_whole_t3*100:.1f}%")
    print(f"Type 3 detection - sentence-level min: {det_min_t3*100:.1f}%")

agg_df.to_csv('sentence_aggregation_ablation_v7.csv')

# =============================================================================
# HÜCRE 15: Source Document Truncation Ablation — Exp 6 (Table 11)
# =============================================================================
print("\n" + "=" * 60)
print("Exp 6: Source Document Truncation Ablation (MPNet, test set)")
print("=" * 60)

def split_into_token_chunks(text, tokenizer, chunk_size=256):
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = [ids[i:i+chunk_size] for i in range(0, len(ids), chunk_size)]
    return [tokenizer.decode(c) for c in chunks] if chunks else [text]

def get_sims_truncated(model, df, max_tokens, desc=""):
    """Truncation gercek token sayisina gore yapilir (model.max_seq_length)."""
    orig_len = model.max_seq_length
    model.max_seq_length = max_tokens
    sims = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        e1 = model.encode([str(row['source_document'])], convert_to_numpy=True)
        e2 = model.encode([str(row['generated_summary'])], convert_to_numpy=True)
        sims.append(float(cosine_similarity(e1, e2)[0][0]))
    model.max_seq_length = orig_len
    return sims

def get_sims_hierarchical(model, df, chunk_size=256, desc=""):
    tokenizer = model.tokenizer
    sims = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        chunks = split_into_token_chunks(str(row['source_document']), tokenizer, chunk_size)
        chunk_embs = model.encode(chunks, convert_to_numpy=True)
        doc_emb = chunk_embs.mean(axis=0, keepdims=True)
        summ_emb = model.encode([str(row['generated_summary'])], convert_to_numpy=True)
        sims.append(float(cosine_similarity(doc_emb, summ_emb)[0][0]))
    return sims

# Trunc-256
sims_t256_val  = get_sims_truncated(model_mpnet, df_val,  256, "Trunc256-Val")
sims_t256_test = get_sims_truncated(model_mpnet, df_test, 256, "Trunc256-Test")
tau_t256, _ = optimize_tau(sims_t256_val, labels_val)
res_t256 = eval_model(sims_t256_test, labels_test, tau_t256)

# Trunc-512 (default, karakter-tabanli sims_mpnet_test ile TUTARLI OLMASI icin
# ayni token-tabanli yontemle yeniden hesaplaniyor)
sims_t512_val  = get_sims_truncated(model_mpnet, df_val,  512, "Trunc512-Val")
sims_t512_test = get_sims_truncated(model_mpnet, df_test, 512, "Trunc512-Test")
tau_t512, _ = optimize_tau(sims_t512_val, labels_val)
res_t512 = eval_model(sims_t512_test, labels_test, tau_t512)

# Hierarchical (256-token segmentler, mean-pooled)
sims_hier_val  = get_sims_hierarchical(model_mpnet, df_val,  256, "Hier-Val")
sims_hier_test = get_sims_hierarchical(model_mpnet, df_test, 256, "Hier-Test")
tau_hier, _ = optimize_tau(sims_hier_val, labels_val)
res_hier = eval_model(sims_hier_test, labels_test, tau_hier)

trunc_results = {
    'Trunc-256': res_t256,
    'Trunc-512': res_t512,
    'Hierarchical': res_hier,
}
trunc_df = pd.DataFrame(trunc_results).T
print(trunc_df.to_string())
trunc_df.to_csv('table11_truncation_ablation_v7.csv')

# =============================================================================
# HÜCRE 16: Görselleştirmeler (test set)
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
plot_type_df = type_df[type_df['Detection Rate'].notna()]
if len(plot_type_df) > 0:
    colors_t = ['#F44336', '#FF9800', '#4CAF50', '#9C27B0']
    bars = ax.barh(plot_type_df['Type'], plot_type_df['Detection Rate'],
                    color=colors_t[:len(plot_type_df)], alpha=0.85, edgecolor='white')
    ax.set_xlim(0, 1.15); ax.set_xlabel('Detection Rate')
    ax.set_title('Fig 4: Detection by Hallucination Type (Test)')
    for bar, val in zip(bars, plot_type_df['Detection Rate']):
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
plt.savefig('hallucination_results_v7.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# HÜCRE 17: Kaydet
# =============================================================================
df_test.to_csv('kap_test_results_v7.csv', index=False)
df_val.to_csv('kap_val_results_v7.csv', index=False)
results_df.to_csv('table4_sentence_encoders_v7.csv')

print("\n" + "=" * 60)
print("TAMAMLANDI (v7 - full coverage)")
print("=" * 60)
best_m = results_df['F1-Score'].idxmax()
print(f"  En iyi model : {best_m}")
print(f"  F1-Score     : {results_df.loc[best_m,'F1-Score']:.4f}")
print(f"  Accuracy     : {results_df.loc[best_m,'Accuracy']:.4f}")
print(f"  Precision    : {results_df.loc[best_m,'Precision']:.4f}")
print(f"  Recall       : {results_df.loc[best_m,'Recall']:.4f}")
print("\nUretilen CSV dosyalari:")
for f in ['table4_sentence_encoders_v7.csv', 'table5_comprehensive_comparison_v7.csv',
          'table6_mcnemar_v7.csv', 'bootstrap_ci_v7.csv', 'table7_per_type_v7.csv',
          'table8_reduction_v7.csv', 'table9_correlation_v7.csv', 'table10_groupstats_v7.csv',
          'table11_truncation_ablation_v7.csv', 'threshold_stability_v7.csv',
          'sentence_aggregation_ablation_v7.csv']:
    print(f"  - {f}")
