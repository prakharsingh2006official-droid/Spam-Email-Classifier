# Spam Email Classifier

Detects spam from message content using **TF-IDF features** and two classifiers that are compared head to head: **Multinomial Naive Bayes** and **Linear SVM**.

## How it works
1. **Cleaning**: lowercase; URLs and numbers are replaced with `urltoken` / `numtoken` (spam is full of links and phone numbers).
2. **Features**: TF-IDF on unigrams + bigrams, English stop words removed.
3. **Models**: Naive Bayes and Linear SVM, trained on an 80/20 stratified split.
4. **Evaluation**: accuracy, precision, recall, F1 for spam, and a confusion matrix. The dataset is imbalanced (~13% spam), so F1 and recall matter more than accuracy alone.

## Results (held-out 20%)
| Model | Accuracy | Spam F1 | Ham wrongly flagged | Spam missed |
|---|---|---|---|---|
| Naive Bayes | 98.65% | 0.945 | 3 | 11 |
| Linear SVM | 98.45% | 0.938 | 5 | 11 |

## Run
```bash
pip install -r requirements.txt
python main.py train                  # compares both models, saves the chosen one (default: nb; use --model svm)
python main.py predict "WINNER! Claim your free prize now"
python main.py file messages.txt      # classify each line of a text file
python main.py interactive
```

## Dataset
Bundled: SMS Spam Collection (UCI), 5,572 labelled messages (5,169 after removing duplicates) in `data/sms.tsv`.
To train on real emails instead, pass any CSV with label/text columns (e.g. Kaggle "Spam Email" `spam.csv` with `v1,v2`, or `Category,Message`):
```bash
python main.py train --data data/emails.csv
```

## Files
```
spam_classifier.py   data loading, cleaning, models, evaluation, prediction
main.py              command-line interface
data/sms.tsv         dataset
requirements.txt
```

## Limitations and ideas
Short SMS text differs from long emails, so retrain on an email dataset for real email use. Ideas: add email headers/subject as features, tune hyperparameters with GridSearchCV, add a Streamlit web UI.
