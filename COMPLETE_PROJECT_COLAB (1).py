# ╔══════════════════════════════════════════════════════════════════╗
# ║   PERSONALIZED LEARNING RECOMMENDATION SYSTEM                   ║
# ║   ML Internship Track | Project 1 | Batch 2026 | INNOVEXA       ║
# ║                                                                  ║
# ║   HOW TO USE IN GOOGLE COLAB:                                    ║
# ║   1. Go to colab.research.google.com                            ║
# ║   2. New Notebook → paste this entire code                      ║
# ║   3. Upload online_courses_dataset.csv when prompted            ║
# ║   4. Press Shift+Enter to run each cell                         ║
# ╚══════════════════════════════════════════════════════════════════╝

# ──────────────────────────────────────────────────────────────────
# CELL 1 ── INSTALL LIBRARIES
# ──────────────────────────────────────────────────────────────────
# !pip install pandas numpy scikit-learn matplotlib seaborn

# ──────────────────────────────────────────────────────────────────
# CELL 2 ── IMPORT LIBRARIES
# ──────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, pickle, os
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise         import cosine_similarity
from sklearn.preprocessing            import LabelEncoder, MinMaxScaler

np.random.seed(42)

print("=" * 60)
print("   PERSONALIZED LEARNING RECOMMENDATION SYSTEM")
print("   Batch 2026 | INNOVEXA | Project 1")
print("=" * 60)

# ──────────────────────────────────────────────────────────────────
# CELL 3 ── UPLOAD & LOAD DATASET
# ──────────────────────────────────────────────────────────────────
# ▶ IN GOOGLE COLAB: Uncomment the next 3 lines to upload your CSV
# from google.colab import files
# uploaded = files.upload()          # click "Choose Files" → select online_courses_dataset.csv
# df = pd.read_csv('online_courses_dataset.csv')

# ▶ IF RUNNING LOCALLY (the CSV is in same folder):
df = pd.read_csv('online_courses_dataset.csv')

print(f"\n✅ Dataset Loaded Successfully!")
print(f"   Shape   : {df.shape[0]} courses × {df.shape[1]} features")
print(f"\n📋 Columns : {list(df.columns)}")
print(f"\n🔍 First 3 rows:")
print(df.head(3).to_string())

# ──────────────────────────────────────────────────────────────────
# CELL 4 ── EXPLORATORY DATA ANALYSIS (EDA)
# ──────────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("📊 EXPLORATORY DATA ANALYSIS")
print("─"*60)

print(f"\n📌 Basic Info:")
print(f"   Total Courses   : {len(df)}")
print(f"   Platforms       : {df['platform'].nunique()} → {df['platform'].unique().tolist()}")
print(f"   Categories      : {df['category'].nunique()} → {df['category'].unique().tolist()}")
print(f"   Avg Rating      : {df['rating'].mean():.2f}")
print(f"   Missing Values  : {df.isnull().sum().sum()}")
print(f"   Duplicates      : {df.duplicated().sum()}")

print(f"\n📊 Category Distribution:")
print(df['category'].value_counts().to_string())

print(f"\n📡 Platform Distribution:")
print(df['platform'].value_counts().to_string())

print(f"\n📊 Level Distribution:")
print(df['level'].value_counts().to_string())

# EDA Plots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('📊 Exploratory Data Analysis — Online Courses Dataset',
             fontsize=16, fontweight='bold', y=1.01)

colors6 = ['#4FC3F7','#81C784','#FFB74D','#CE93D8','#F48FB1','#80DEEA']

# Plot 1: Courses by Category
cat_counts = df['category'].value_counts()
axes[0,0].barh(cat_counts.index, cat_counts.values,
               color=plt.cm.Set2(np.linspace(0,1,len(cat_counts))))
axes[0,0].set_title('Courses by Category', fontweight='bold')
axes[0,0].set_xlabel('Number of Courses')

# Plot 2: Platform Pie
plat_counts = df['platform'].value_counts()
axes[0,1].pie(plat_counts.values, labels=plat_counts.index,
              autopct='%1.1f%%', startangle=140,
              colors=plt.cm.Pastel1(np.linspace(0,1,len(plat_counts))))
axes[0,1].set_title('Distribution by Platform', fontweight='bold')

# Plot 3: Rating Histogram
axes[0,2].hist(df['rating'], bins=15, color='#4FC3F7', edgecolor='white', linewidth=0.8)
axes[0,2].set_title('Rating Distribution', fontweight='bold')
axes[0,2].set_xlabel('Rating'); axes[0,2].set_ylabel('Frequency')
axes[0,2].axvline(df['rating'].mean(), color='red', linestyle='--',
                  label=f'Mean={df["rating"].mean():.2f}')
axes[0,2].legend()

# Plot 4: Level Bar
level_counts = df['level'].value_counts()
bars = axes[1,0].bar(level_counts.index, level_counts.values,
                     color=['#4CAF50','#2196F3','#FF9800','#E91E63'])
axes[1,0].set_title('Courses by Difficulty Level', fontweight='bold')
axes[1,0].set_ylabel('Count')
for bar in bars:
    axes[1,0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                   str(int(bar.get_height())), ha='center', fontsize=10)

# Plot 5: Duration vs Rating Scatter
sc = axes[1,1].scatter(df['duration_hrs'], df['rating'],
                       alpha=0.6, c=df['num_reviews'],
                       cmap='YlOrRd', s=50)
plt.colorbar(sc, ax=axes[1,1], label='Reviews')
axes[1,1].set_title('Duration vs Rating', fontweight='bold')
axes[1,1].set_xlabel('Duration (hours)'); axes[1,1].set_ylabel('Rating')

# Plot 6: Free vs Paid
df['is_free'] = df['price'] == 0
free_counts   = df['is_free'].value_counts()
axes[1,2].bar(['Paid', 'Free'],
              [free_counts.get(False,0), free_counts.get(True,0)],
              color=['#E91E63','#4CAF50'])
axes[1,2].set_title('Free vs Paid Courses', fontweight='bold')
axes[1,2].set_ylabel('Count')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ EDA Plots saved as eda_plots.png")

# ──────────────────────────────────────────────────────────────────
# CELL 5 ── DATA PREPROCESSING
# ──────────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("🧹 DATA PREPROCESSING")
print("─"*60)

df_clean = df.copy()

# Step 1: Handle Missing Values
print(f"\n  Missing values before: {df_clean.isnull().sum().sum()}")
df_clean['rating'].fillna(df_clean['rating'].median(), inplace=True)
df_clean['description'].fillna('No description', inplace=True)
df_clean['skills'].fillna('General', inplace=True)
df_clean['level'].fillna('All Levels', inplace=True)
df_clean['num_reviews'].fillna(0, inplace=True)
print(f"  Missing values after : {df_clean.isnull().sum().sum()} ✅")

# Step 2: Remove Duplicates
before = len(df_clean)
df_clean.drop_duplicates(subset=['title'], inplace=True)
df_clean.reset_index(drop=True, inplace=True)
print(f"  Duplicates removed   : {before - len(df_clean)}")

# Step 3: Encode Categorical Columns
le_level    = LabelEncoder()
le_platform = LabelEncoder()
le_category = LabelEncoder()
df_clean['level_enc']    = le_level.fit_transform(df_clean['level'])
df_clean['platform_enc'] = le_platform.fit_transform(df_clean['platform'])
df_clean['category_enc'] = le_category.fit_transform(df_clean['category'])
print(f"  Categorical encoding : Done ✅")

# Step 4: Normalize Numerical Features
scaler = MinMaxScaler()
df_clean[['rating_norm','reviews_norm','duration_norm']] = scaler.fit_transform(
    df_clean[['rating','num_reviews','duration_hrs']])
print(f"  Normalization        : Done ✅")
print(f"\n  ✅ Clean Dataset: {df_clean.shape[0]} rows × {df_clean.shape[1]} cols")

# ──────────────────────────────────────────────────────────────────
# CELL 6 ── FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("⚙️  FEATURE ENGINEERING")
print("─"*60)

# Combine text fields for TF-IDF
df_clean['combined_text'] = (
    df_clean['title'].str.lower()       + ' ' +
    df_clean['description'].str.lower() + ' ' +
    df_clean['skills'].str.lower()      + ' ' +
    df_clean['category'].str.lower()    + ' ' +
    df_clean['level'].str.lower()
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000, stop_words='english',
                        ngram_range=(1,2), min_df=1)
tfidf_matrix = tfidf.fit_transform(df_clean['combined_text'])
print(f"\n  ✅ TF-IDF Matrix : {tfidf_matrix.shape}")

# Cosine Similarity Matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(f"  ✅ Cosine Sim   : {cosine_sim.shape}")

# Course index mapping
course_indices = pd.Series(df_clean.index, index=df_clean['course_id']).drop_duplicates()
print(f"  ✅ Feature Engineering Complete!")

# ──────────────────────────────────────────────────────────────────
# CELL 7 ── CONTENT-BASED FILTERING
# ──────────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("📚 CONTENT-BASED FILTERING (TF-IDF + Cosine Similarity)")
print("─"*60)

def content_based_recommend(user_skills, user_category, user_level=None, top_n=10):
    """
    Recommends courses based on user skills, category, and level.
    Uses TF-IDF vectorization + Cosine Similarity.
    """
    query_parts = []
    if user_skills   : query_parts.append(user_skills.lower())
    if user_category : query_parts.append(user_category.lower())
    if user_level    : query_parts.append(user_level.lower())
    query = ' '.join(query_parts)

    query_vec  = tfidf.transform([query])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idx    = sim_scores.argsort()[::-1]

    # Filter by level
    if user_level and user_level != 'All Levels':
        mask    = (df_clean['level']==user_level)|(df_clean['level']=='All Levels')
        top_idx = [i for i in top_idx if mask.iloc[i]][:top_n]
    else:
        top_idx = top_idx[:top_n]

    result = df_clean.iloc[top_idx][[
        'course_id','title','category','platform','level',
        'rating','duration_hrs','skills','price'
    ]].copy()
    result['similarity_score'] = sim_scores[top_idx]
    return result.reset_index(drop=True)

print("  ✅ Content-Based Filtering Ready!")

# ──────────────────────────────────────────────────────────────────
# CELL 8 ── COLLABORATIVE FILTERING
# ──────────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("👥 COLLABORATIVE FILTERING (User-User Similarity)")
print("─"*60)

def generate_user_interactions(n_users=100):
    categories = df_clean['category'].unique()
    rows = []
    for uid in range(1, n_users+1):
        pref_cats   = np.random.choice(categories, size=np.random.randint(1,4), replace=False)
        pref_df     = df_clean[df_clean['category'].isin(pref_cats)]
        sample_size = min(15, len(pref_df))
        if sample_size == 0: continue
        sampled = pref_df.sample(sample_size, random_state=uid)
        for _, row in sampled.iterrows():
            base   = 3.5 if row['category'] in pref_cats else 2.0
            rating = min(5.0, round(base + np.random.uniform(0,1.5), 1))
            rows.append({'user_id': uid, 'course_id': row['course_id'], 'rating': rating})
    return pd.DataFrame(rows)

interactions_df     = generate_user_interactions(100)
user_course_matrix  = interactions_df.pivot_table(
    index='user_id', columns='course_id', values='rating', fill_value=0)
user_sim_matrix     = cosine_similarity(user_course_matrix)
user_sim_df         = pd.DataFrame(user_sim_matrix,
                                   index=user_course_matrix.index,
                                   columns=user_course_matrix.index)

def collaborative_recommend(user_id, top_n=10):
    if user_id not in user_sim_df.index: return []
    sim_users    = user_sim_df[user_id].drop(user_id).nlargest(10).index
    sim_ratings  = user_course_matrix.loc[sim_users]
    seen         = set(user_course_matrix.loc[user_id][user_course_matrix.loc[user_id]>0].index) \
                   if user_id in user_course_matrix.index else set()
    scores       = sim_ratings.mean(axis=0).drop(labels=list(seen), errors='ignore')
    return scores.nlargest(top_n).index.tolist()

print(f"  Interactions : {len(interactions_df)} records")
print(f"  Matrix       : {user_course_matrix.shape}")
print(f"  ✅ Collaborative Filtering Ready!")

# ──────────────────────────────────────────────────────────────────
# CELL 9 ── HYBRID RECOMMENDATION ENGINE
# ──────────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("🔀 HYBRID RECOMMENDATION ENGINE (CB 60% + CF 40%)")
print("─"*60)

def hybrid_recommend(user_skills, user_category, user_level=None,
                     user_id=None, top_n=10, cb_weight=0.6, cf_weight=0.4):
    """
    Hybrid Recommendation System.
    Content-Based (60%) + Collaborative Filtering (40%)
    """
    # Content-Based
    cb_recs = content_based_recommend(user_skills, user_category, user_level, top_n=top_n*2)
    cb_recs['cb_score'] = cb_recs['similarity_score']

    # Collaborative
    cf_scores = {}
    if user_id:
        for cid in collaborative_recommend(user_id, top_n=top_n*2):
            cf_scores[cid] = 1.0

    # Merge
    all_ids = list(set(cb_recs['course_id'].tolist() + list(cf_scores.keys())))
    results = []
    for cid in all_ids:
        rows_ = df_clean[df_clean['course_id']==cid]
        if rows_.empty: continue
        course = rows_.iloc[0]
        cb_s   = float(cb_recs[cb_recs['course_id']==cid]['cb_score'].values[0]) \
                 if cid in cb_recs['course_id'].values else 0.0
        cf_s   = cf_scores.get(cid, 0.0)
        h_score = cb_weight*cb_s + cf_weight*cf_s
        h_score += max(0, (course['rating']-3.5)/1.5 * 0.1)
        results.append({
            'course_id'   : cid,
            'title'       : course['title'],
            'category'    : course['category'],
            'platform'    : course['platform'],
            'level'       : course['level'],
            'rating'      : course['rating'],
            'duration_hrs': course['duration_hrs'],
            'skills'      : course['skills'],
            'price'       : course['price'],
            'cb_score'    : round(cb_s,4),
            'cf_score'    : round(cf_s,4),
            'hybrid_score': round(h_score,4),
        })

    res_df = pd.DataFrame(results).sort_values('hybrid_score', ascending=False)
    res_df = res_df.head(top_n).reset_index(drop=True)
    res_df.index += 1
    return res_df

print("  ✅ Hybrid Engine Ready!")

# ──────────────────────────────────────────────────────────────────
# CELL 10 ── MODEL EVALUATION
# ──────────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("📏 MODEL EVALUATION (Precision, Recall, F1, MAP, NDCG @ K=10)")
print("─"*60)

def evaluate_model(n_eval=50, K=10):
    precisions,recalls,f1s,aps,ndcgs = [],[],[],[],[]
    cats = df_clean['category'].unique()

    for uid in range(1, n_eval+1):
        pref_cats   = np.random.choice(cats, size=np.random.randint(1,3), replace=False)
        user_skills = np.random.choice(['Python','SQL','TensorFlow','R','JavaScript'], 2)
        recs        = hybrid_recommend(
            user_skills   = ', '.join(user_skills),
            user_category = pref_cats[0],
            user_level    = np.random.choice(['Beginner','Intermediate','Advanced']),
            user_id       = uid if uid<=100 else None,
            top_n         = K
        )
        if recs.empty: continue

        relevant  = set(df_clean[df_clean['category'].isin(pref_cats)]['course_id'])
        rec_ids   = recs['course_id'].tolist()
        hits      = [1 if c in relevant else 0 for c in rec_ids]
        n_hits    = sum(hits)

        prec = n_hits / K
        rec  = n_hits / max(len(relevant),1)
        f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        ap, nr = 0.0, 0
        for i,h in enumerate(hits,1):
            if h: nr+=1; ap+=nr/i
        ap  /= max(n_hits,1)
        ideal= sum(1/np.log2(i+2) for i in range(min(n_hits,K)))
        dcg  = sum(h/np.log2(i+2) for i,h in enumerate(hits))
        ndcg = dcg/max(ideal,1e-9)

        precisions.append(prec); recalls.append(rec); f1s.append(f1)
        aps.append(ap); ndcgs.append(ndcg)

    return {
        f'Precision@{K}': round(np.mean(precisions),3),
        f'Recall@{K}'   : round(np.mean(recalls),3),
        f'F1-Score@{K}' : round(np.mean(f1s),3),
        f'MAP@{K}'      : round(np.mean(aps),3),
        f'NDCG@{K}'     : round(np.mean(ndcgs),3),
    }

eval_results = evaluate_model(n_eval=50, K=10)

print("\n  ┌──────────────────────────────────────────┐")
print("  │       MODEL EVALUATION RESULTS @ K=10   │")
print("  ├──────────────────────────────────────────┤")
for m,v in eval_results.items():
    bar = '█' * int(v*20)
    print(f"  │  {m:<15} : {v:.3f}  {bar:<20}│")
print("  └──────────────────────────────────────────┘")

# Evaluation Plots
fig, axes = plt.subplots(1, 2, figsize=(14,5))
fig.suptitle('📊 Model Evaluation Results', fontsize=14, fontweight='bold')

metrics = list(eval_results.keys())
values  = list(eval_results.values())
colors  = ['#4FC3F7','#81C784','#FFB74D','#CE93D8','#F48FB1']
bars    = axes[0].bar(metrics, values, color=colors, edgecolor='white', linewidth=1.5)
axes[0].set_ylim(0, 1.15); axes[0].set_title('Metric Comparison @ K=10')
axes[0].set_ylabel('Score')
for bar, val in zip(bars, values):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                 f'{val:.3f}', ha='center', fontweight='bold')

k_vals  = [5,10,15,20,25,30]
prec_k  = [evaluate_model(n_eval=20,K=k)[f'Precision@{k}'] for k in k_vals]
axes[1].plot(k_vals, prec_k, 'o-', color='#4FC3F7', linewidth=2, markersize=8)
axes[1].fill_between(k_vals, prec_k, alpha=0.15, color='#4FC3F7')
for kv,pv in zip(k_vals,prec_k):
    axes[1].annotate(f'{pv:.2f}',(kv,pv),textcoords='offset points',xytext=(0,8),
                     ha='center',fontsize=9)
axes[1].set_title('Precision@K Curve'); axes[1].set_xlabel('K (Recommendations)')
axes[1].set_ylabel('Precision@K'); axes[1].set_ylim(0,1.1); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('evaluation_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✅ Evaluation plots saved!")

# ──────────────────────────────────────────────────────────────────
# CELL 11 ── DEMO: PERSONALIZED RECOMMENDATIONS
# ──────────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("🎯 DEMO — PERSONALIZED RECOMMENDATIONS FOR 3 LEARNERS")
print("─"*60)

learners = [
    {"name":"Arun (Fresher)",          "skills":"Python, Statistics",
     "category":"Data Science",        "level":"Beginner",      "user_id":1},
    {"name":"Priya (Working Pro)",      "skills":"TensorFlow, Neural Networks",
     "category":"Deep Learning",       "level":"Intermediate",  "user_id":5},
    {"name":"Ravi (Web Developer)",     "skills":"JavaScript, React",
     "category":"Web Development",     "level":"Advanced",      "user_id":12},
]

for p in learners:
    print(f"\n  {'─'*58}")
    print(f"  👤  {p['name']}")
    print(f"  🔧  Skills   : {p['skills']}")
    print(f"  📚  Interest : {p['category']}  |  Level: {p['level']}")
    recs = hybrid_recommend(p['skills'], p['category'], p['level'], p['user_id'], top_n=5)
    print(f"\n  🎯  Top 5 Recommended Courses:")
    print(f"  {'#':<3} {'Title':<45} {'Platform':<10} {'⭐':<6} {'Score'}")
    print(f"  {'─'*3} {'─'*45} {'─'*10} {'─'*6} {'─'*6}")
    for rank,row in recs.iterrows():
        t = row['title'][:43]+'.' if len(row['title'])>43 else row['title']
        print(f"  {rank:<3} {t:<45} {row['platform']:<10} {row['rating']:<6} {row['hybrid_score']:.4f}")

# ──────────────────────────────────────────────────────────────────
# CELL 12 ── LEARNING PATH GENERATOR
# ──────────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("🗺️  LEARNING PATH GENERATOR")
print("─"*60)

def generate_learning_path(goal_category, starting_level='Beginner'):
    levels_order = ['Beginner','Intermediate','Advanced']
    start_idx    = levels_order.index(starting_level) if starting_level in levels_order else 0
    path = {}
    for stage_idx, level in enumerate(levels_order[start_idx:], 1):
        recs = content_based_recommend(goal_category, goal_category, level, top_n=2)
        if not recs.empty:
            path[f'Stage {stage_idx} — {level}'] = recs[['title','platform','rating','duration_hrs']].values.tolist()
    return path

print(f"\n  📌 Learning Path: Machine Learning (Starting from Beginner)")
path = generate_learning_path('Machine Learning', 'Beginner')
for stage, courses in path.items():
    print(f"\n  {stage}:")
    for i,c in enumerate(courses,1):
        print(f"    {i}. {c[0][:50]:<50} | {c[1]:<20} | ⭐{c[2]} | {c[3]}hrs")

# ──────────────────────────────────────────────────────────────────
# CELL 13 ── SAVE MODEL
# ──────────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("💾 SAVING MODEL ARTIFACTS")
print("─"*60)

os.makedirs('models', exist_ok=True)
with open('models/tfidf_vectorizer.pkl','wb') as f:   pickle.dump(tfidf, f)
with open('models/cosine_sim_matrix.pkl','wb') as f:  pickle.dump(cosine_sim, f)
df_clean.to_csv('models/cleaned_courses.csv', index=False)
user_course_matrix.to_csv('models/user_course_matrix.csv')

print("  ✅ models/tfidf_vectorizer.pkl")
print("  ✅ models/cosine_sim_matrix.pkl")
print("  ✅ models/cleaned_courses.csv")
print("  ✅ models/user_course_matrix.csv")

# ──────────────────────────────────────────────────────────────────
# CELL 14 ── PROJECT SUMMARY
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("   ✅  PROJECT COMPLETE — FINAL SUMMARY")
print("="*60)
print(f"""
  📁 DATASET
     Total Courses : {len(df_clean)}
     Platforms     : {df_clean['platform'].nunique()}
     Categories    : {df_clean['category'].nunique()}
     Features      : {df_clean.shape[1]}

  📊 MODEL PERFORMANCE (K=10)
     Precision@10  : {eval_results['Precision@10']}
     Recall@10     : {eval_results['Recall@10']}
     F1-Score@10   : {eval_results['F1-Score@10']}
     MAP@10        : {eval_results['MAP@10']}
     NDCG@10       : {eval_results['NDCG@10']}

  🔧 APPROACH
     Content-Based  : TF-IDF + Cosine Similarity (Weight: 60%)
     Collaborative  : User-User Similarity        (Weight: 40%)
     Final Output   : Hybrid Weighted Score Ranking

  📦 TECH STACK
     Python | Pandas | NumPy | Scikit-learn
     Matplotlib | Seaborn | Pickle

  🚀 FUTURE ENHANCEMENTS
     01. Real-time user feedback
     02. Deep Learning (Neural CF)
     03. Multi-modal recommendations
     04. Cross-platform deployment
""")
print("  INNOVEXA | Batch 2026 | Project 1 | ML Internship Track")
print("="*60)
