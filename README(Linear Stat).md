# 🎓 Predicting Graduate School Admission using Logistic Regression

**Author:** Kayode Okunola  
**Course:** Linear Statistics  
**Project:** Final Project Report  

---

## 📘 Introduction

The pursuit of postgraduate degrees often involves uncertainty regarding admission standards and competitiveness.  
This project aims to **develop a logistic regression model** that predicts a student's **chance of admission into graduate school** based on three key factors:

- **GRE Score (Graduate Record Examination)**  
- **CGPA (Cumulative Grade Point Average)**  
- **University Rank (Institution Rating)**  

The goal is to help **students** estimate their admission probability and assist **institutions** in understanding how these predictors influence admissions.

---

## 🧮 Methodology

Since the dependent variable (admission) is **binary** (1 = admitted, 0 = not admitted), a **logistic regression** model is used:

\[
\ln\left(\frac{p}{1 - p}\right) = \beta_0 + \beta_1 \text{GRE} + \beta_2 \text{GPA} + \beta_3 \text{Rank}
\]

where \( p \) is the probability of admission.

The model parameters are estimated using **Maximum Likelihood Estimation (MLE)** through R’s `glm()` function.

---

## 🧩 Model Summary

| Predictor | Coefficient | Std. Error | p-value | Odds Ratio | Interpretation |
|------------|--------------|-------------|-----------|--------------|----------------|
| GRE        | 0.0683       | 0.0323      | 0.0345    | 1.071        | Higher GRE increases admission odds |
| GPA        | 4.2786       | 0.8219      | 0.0000    | 72.145       | Higher GPA greatly increases admission odds |
| Rank       | -0.4656      | 0.3271      | 0.1546    | 0.628        | Lower-ranked institutions slightly reduce odds |

- **AIC:** 137.8  
- **McFadden R²:** 0.45 (Good fit)  
- **Accuracy:** 95%  
- **AUC:** 0.9332 (Excellent model performance)

---

## ⚙️ Model Validation

- **Hosmer-Lemeshow Test:** p = 0.8873 → *Model fits well*  
- **No multicollinearity:** VIF < 2 for all predictors  
- **Linearity (Box-Tidwell Test):** Not violated  
- **Cross-validation Accuracy:** 90.95%  
- **Confusion Matrix:**  

|                | Predicted No | Predicted Yes |
|----------------|--------------|----------------|
| Actual No      | 7            | 1              |
| Actual Yes     | 3            | 69             |

---

## 📈 ROC Curve & AUC

The model’s **ROC curve** shows strong classification ability with  
an **AUC of 0.9332**, indicating high discrimination power.

---

## 🔍 Key Insights

- **GPA** has the most significant impact on admission probability.  
- **GRE** scores also contribute positively.  
- **University Rank** has minimal negative influence and is not statistically significant.

---

## 🧠 Conclusion

The **logistic regression model** accurately predicts graduate admission chances.  
Results suggest that **academic performance (GPA)** and **standardized test scores (GRE)** are the most important factors influencing admissions.  

Students aiming for graduate programs should prioritize **strong academic reco**
