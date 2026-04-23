# 🏦 SecureTrust AI – Credit Risk Assessment Engine

SecureTrust AI is an industry-grade machine learning system designed to automate and optimize loan approval decisions using applicant financial data.

---

## 📌 Project Overview

SecureTrust Bank processes hundreds of loan applications daily. Manual verification is slow, biased, and error-prone.

This system predicts whether a loan should be **APPROVED** or **REJECTED**, enabling faster, fairer, and more consistent decision-making.

---

## ⚠️ Business Risks

- **Type I Error (False Positive)**  
  Approving a risky loan → Financial loss  

- **Type II Error (False Negative)**  
  Rejecting a good loan → Loss of business  

📌 Since financial loss is more critical, the system is optimized for **high precision**.

---

## 🚀 Key Features

- End-to-end ML pipeline (EDA → preprocessing → modeling → evaluation)
- Feature engineering for improved accuracy
- Multiple model comparison
- Threshold tuning for precision optimization
- Explainable predictions with risk flags
- Clean Streamlit UI for real-time predictions

---

## 🤖 Models Used

1. Logistic Regression (Baseline model)  
2. Gaussian Naive Bayes (Probabilistic model)  
3. K-Nearest Neighbors (Distance-based model)  

---

## 📊 Evaluation Metrics

- Accuracy  
- Precision (**Primary Focus**)  
- Recall  
- F1 Score  
- ROC-AUC  

---

## 🧠 How It Works

1. Data is preprocessed (missing values, encoding, scaling)  
2. Features are engineered to capture non-linear patterns  
3. Models are trained and compared  
4. Best model is selected based on **precision**  
5. Predictions are generated with explanations  

---

## 🖥️ Streamlit App

- Users input applicant details manually  
- No dataset upload required (ensures consistency & security)  
- System outputs:
  - Approval / Rejection  
  - Probability score  
  - Risk explanation  

---
## 📁 Project Structure
├── streamlit_app.py
├── main_pipeline.py
├── loan_approval_data.csv
├── requirements.txt
└── README.md



---

## 🎯 Business Impact

- Reduces risky loan approvals  
- Improves decision speed  
- Ensures consistency and fairness  
- Supports scalable banking operations  

---

## 🔮 Future Improvements

- Deploy with Flask / FastAPI  
- Add XGBoost / Random Forest  
- Connect to real banking APIs  
- Add authentication system  

---

## 👨‍💻 Author

**Ammar Husain Gheewala**  
Aspiring AI Engineer | FinTech Enthusiast

## 📁 Project Structure
