# 🎓 Personalized Learning Recommendation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2+-orange?style=for-the-badge&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-1.5+-green?style=for-the-badge&logo=pandas)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)
![Batch](https://img.shields.io/badge/Batch-2026-purple?style=for-the-badge)

**ML Internship Track | Project 1 | Batch 2026 | INNOVEXA**

</div>

---

## 📌 Project Overview

With thousands of online courses available across platforms like Coursera, Udemy, edX — students struggle to find the right course. This project builds a **Hybrid ML Recommendation System** that analyzes learner skills, interests, and experience level to recommend the most relevant courses.

---

## 📁 Project Structure

```
personalized-learning-recommendation-system/
├── COMPLETE_PROJECT_COLAB.py     ← Main project script
├── online_courses_dataset.csv    ← Dataset (298 courses, 10 categories)
├── eda_plots.png                 ← EDA charts
├── evaluation_plots.png          ← Evaluation charts
├── requirements.txt              ← Dependencies
└── README.md                     ← This file
```

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| Total Courses | 298 |
| Platforms | Coursera, Udemy, edX, Udacity, LinkedIn Learning |
| Categories | 10 (Data Science, ML, Deep Learning, Web Dev, etc.) |
| Features | 13 columns |

---

## 🔧 Tech Stack

Python | Pandas | NumPy | Scikit-learn | Matplotlib | Seaborn | Pickle

---

## 🧠 How It Works

```
Learner Profile (Skills + Category + Level)
        ↓
Content-Based Filtering (60%)  +  Collaborative Filtering (40%)
  TF-IDF + Cosine Similarity       User-User Similarity Matrix
        ↓
    Hybrid Score = CB×0.6 + CF×0.4
        ↓
  Top-N Personalized Course Recommendations
```

---

## 📏 Evaluation Results @ K=10

| Metric | Score |
|--------|-------|
| Precision@10 | 0.872 |
| Recall@10 | 0.814 |
| F1-Score@10 | 0.842 |
| MAP@10 | 0.886 |
| NDCG@10 | 0.903 |

---

## ▶️ How to Run

### Google Colab (Easiest)
1. Upload `online_courses_dataset.csv` and `COMPLETE_PROJECT_COLAB.py` to Colab
2. Run: `exec(open('COMPLETE_PROJECT_COLAB.py').read())`

### Local
```bash
git clone https://github.com/YOUR_USERNAME/personalized-learning-recommendation-system.git
cd personalized-learning-recommendation-system
pip install -r requirements.txt
python COMPLETE_PROJECT_COLAB.py
```

---

## 🚀 Future Enhancements
1. Real-time user feedback integration
2. Deep Learning (Neural Collaborative Filtering)
3. Multi-modal recommendations
4. Cross-platform deployment

---

<div align="center">
🚀 <b>INNOVEXA | Batch 2026 | Project 1 | ML Internship Track</b><br>
⭐ Star this repo if you found it useful!
</div>
