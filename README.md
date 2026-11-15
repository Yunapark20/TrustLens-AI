# TrustLens AI

**Track:** AI + Finance  
**Pitch:** TrustLens AI helps people pause before sending money by screening suspicious Vietnamese messages, explaining warning signals and suggesting safe verification steps.

## What it does
- Classifies pasted SMS/chat/payment-request text using a TF-IDF word + character n-gram model and Logistic Regression.
- Displays a model score and a review-oriented risk band.
- Adds transparent pattern checks for OTP requests, urgent transfers, threats, impersonation cues and links.
- Gives concrete next steps before a user pays or shares credentials.
- Does not store submitted messages in an application database.

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
The first run downloads the public training CSV and trains the model. It caches `trustlens_model.joblib` locally. Internet is needed for the first run.

## Reproduce the evaluation
```bash
python train_evaluate.py
```
This uses the dataset's published train.csv and independent test.csv. It reports a majority-class baseline and the model's precision, recall, F1 and confusion matrix. Do not copy benchmark numbers from other papers as your own results. Paste the actual output from your run into the write-up before submission.

## Model design
- Input: one user-pasted Vietnamese message.
- Features: word TF-IDF (1–2 grams) + character TF-IDF (3–5 grams).
- Classifier: class-weighted Logistic Regression.
- Baseline: `DummyClassifier(strategy="most_frequent")`.
- Evaluation: fixed official train/test split; report spam/scam precision and recall, not accuracy alone.
- User-facing safety rules are a separate transparent layer; they do not override the model's learned prediction.

## Important limitations
- The source dataset is labelled for SMS spam/scam detection, not all forms of financial fraud.
- The score is a model score, not a calibrated probability of fraud.
- A safe-looking message may still be fraudulent; a legitimate promotional or bank message may be flagged.
- Links are not automatically proof of fraud.
- The system does not verify sender identity, bank accounts, domains, or live transactions.
- Do not enter real OTPs, passwords, card numbers or other sensitive data.
- This is a prototype and is not financial advice.

## Sources and credits
- Dataset: Tran et al., Vietnamese Quality-Assured SMS Scam Dataset, CC BY 4.0. Official repository: https://github.com/trannguyenthaituan251209/vietnamese_sms_dataset
- Dataset paper citation is provided in the source repository: *Vietnamese SMS Dataset with Quality Assurance*, IEEE Access (2026).
- Machine-learning libraries: scikit-learn (https://scikit-learn.org/), pandas (https://pandas.pydata.org/), joblib (https://joblib.readthedocs.io/), Streamlit (https://streamlit.io/).
- AI assistance: ChatGPT (OpenAI), used to help debug project code.
