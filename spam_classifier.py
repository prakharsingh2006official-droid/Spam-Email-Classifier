"""Spam classifier: TF-IDF features + Naive Bayes / Linear SVM."""
import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

MODEL_PATH = "spam_model.joblib"


def load_data(path="data/sms.tsv"):
    """Load a labelled dataset. Supports:
       - .tsv with no header: label<TAB>text   (bundled SMS Spam Collection)
       - .csv with columns label,text  or  v1,v2 (Kaggle spam.csv)  or  Category,Message
    Returns a DataFrame with columns: label (1=spam, 0=ham), text."""
    if path.endswith(".tsv"):
        df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"])
    else:
        df = pd.read_csv(path, encoding="latin-1")
        cols = {c.lower(): c for c in df.columns}
        lab = next(cols[c] for c in ("label", "v1", "category", "class") if c in cols)
        txt = next(cols[c] for c in ("text", "v2", "message", "email", "body") if c in cols)
        df = df[[lab, txt]].rename(columns={lab: "label", txt: "text"})
    df = df.dropna().drop_duplicates()
    df["label"] = df["label"].astype(str).str.lower().map({"spam": 1, "ham": 0, "1": 1, "0": 0})
    return df.dropna(subset=["label"]).astype({"label": int}).reset_index(drop=True)


def clean(text):
    """Light normalisation: lowercase, mask URLs/numbers so the model sees 'there is a link/number'."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"\d+", " numtoken ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_model(kind="nb"):
    vec = TfidfVectorizer(preprocessor=clean, stop_words="english", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    clf = MultinomialNB(alpha=0.1) if kind == "nb" else LinearSVC(C=1.0, class_weight="balanced")
    return Pipeline([("tfidf", vec), ("clf", clf)])


def compare_models(df, test_size=0.2, seed=42):
    """Train both models, return fitted pipelines + metrics + the held-out test split."""
    X_tr, X_te, y_tr, y_te = train_test_split(df["text"], df["label"], test_size=test_size,
                                              stratify=df["label"], random_state=seed)
    results, models = {}, {}
    for kind, name in (("nb", "Naive Bayes"), ("svm", "Linear SVM")):
        m = build_model(kind).fit(X_tr, y_tr)
        pred = m.predict(X_te)
        results[name] = {
            "accuracy": accuracy_score(y_te, pred),
            "f1_spam": f1_score(y_te, pred),
            "confusion": confusion_matrix(y_te, pred),
            "report": classification_report(y_te, pred, target_names=["ham", "spam"], digits=3),
        }
        models[name] = m
    return models, results, (X_te, y_te)


def top_spam_words(model, n=15):
    """Words most associated with spam (works for NB and SVM)."""
    vec, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    names = vec.get_feature_names_out()
    score = clf.feature_log_prob_[1] - clf.feature_log_prob_[0] if hasattr(clf, "feature_log_prob_") else clf.coef_[0]
    return [names[i] for i in score.argsort()[::-1][:n]]


def save(model, path=MODEL_PATH):
    joblib.dump(model, path)


def load(path=MODEL_PATH):
    return joblib.load(path)


def predict(model, text):
    """Returns (label, confidence) where confidence is the spam probability/score squashed to 0-1."""
    clf = model.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        p = float(model.predict_proba([text])[0][1])
    else:
        import math
        p = 1 / (1 + math.exp(-2 * float(model.decision_function([text])[0])))
    return ("SPAM" if p >= 0.5 else "HAM"), p
