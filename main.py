"""CLI for the spam classifier.

  python main.py train [--data data/sms.tsv] [--model nb|svm]   train, compare both models, save the chosen one
  python main.py predict "Congratulations! You won a free prize, click here"
  python main.py file messages.txt            classify every line of a text file
  python main.py interactive
"""
import argparse
import os
import sys
import spam_classifier as sc


def cmd_train(a):
    df = sc.load_data(a.data)
    print(f"Loaded {len(df)} messages  ({df.label.sum()} spam / {(df.label == 0).sum()} ham)\n")
    models, results, _ = sc.compare_models(df)
    for name, r in results.items():
        tn, fp, fn, tp = r["confusion"].ravel()
        print(f"=== {name} ===")
        print(f"Accuracy: {r['accuracy']:.4f}   Spam F1: {r['f1_spam']:.4f}")
        print(f"Confusion matrix -> ham correct: {tn}, ham flagged as spam: {fp}, spam missed: {fn}, spam caught: {tp}")
        print(r["report"])
    chosen = "Naive Bayes" if a.model == "nb" else "Linear SVM"
    # refit the chosen model on ALL data before saving
    final = sc.build_model(a.model).fit(df["text"], df["label"])
    sc.save(final)
    print(f"Saved {chosen} (trained on all data) to {sc.MODEL_PATH}")
    print("Top spam indicators:", ", ".join(sc.top_spam_words(models[chosen])))


def get_model():
    if not os.path.exists(sc.MODEL_PATH):
        print("No trained model found - run: python main.py train"); sys.exit(1)
    return sc.load()


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description="Spam Email Classifier")
    sub = p.add_subparsers(dest="cmd")
    t = sub.add_parser("train"); t.add_argument("--data", default="data/sms.tsv"); t.add_argument("--model", choices=["nb", "svm"], default="nb")
    pr = sub.add_parser("predict"); pr.add_argument("text")
    f = sub.add_parser("file"); f.add_argument("path")
    sub.add_parser("interactive")
    a = p.parse_args()
    if a.cmd == "train":
        cmd_train(a)
    elif a.cmd == "predict":
        label, p_spam = sc.predict(get_model(), a.text)
        print(f"{label}  (spam score: {p_spam:.2f})")
    elif a.cmd == "file":
        m = get_model()
        for line in open(a.path, encoding="utf-8"):
            if line.strip():
                label, s = sc.predict(m, line.strip())
                print(f"[{label:4}] {s:.2f}  {line.strip()[:80]}")
    elif a.cmd == "interactive":
        m = get_model()
        print("Type a message to classify (blank line to quit)")
        while (text := input("> ").strip()):
            label, s = sc.predict(m, text)
            print(f"  -> {label} (spam score {s:.2f})")
    else:
        p.print_help()


if __name__ == "__main__":
    main()
