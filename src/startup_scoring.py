"""
Startup Evaluation Engine
------------------------
This script simulates a credit-score-like evaluation for startups using a composite scoring methodology.

Sections:
1. Data Loading & Exploration
2. Data Preprocessing & Normalization
3. Scoring Formula & Feature Weighting
4. Ranking & Interpretation
5. Visualization
6. Documentation & Insights
7. Bonus: ML Extension (KMeans Clustering)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Data Loading & Exploration
# -----------------------------
# Load the dataset into a pandas DataFrame for analysis.
DATA_PATH = '../Startup_Scoring_Dataset.csv' if os.path.exists('../Startup_Scoring_Dataset.csv') else 'Startup_Scoring_Dataset.csv'
df = pd.read_csv(DATA_PATH)
print('First 5 rows:')
print(df.head())

# 2. Data Preprocessing & Normalization
# -------------------------------------
# Normalize all numeric columns to a 0-1 range using Min-Max normalization.
# For negatively correlated metrics (like burn rate), invert the normalization so lower is better.
good_cols = ['team_experience', 'market_size_million_usd', 'monthly_active_users', 'funds_raised_inr', 'valuation_inr']  # Higher is better
bad_cols = ['monthly_burn_rate_inr']  # Higher is worse

# Normalize 'good' columns (higher is better)
for col in good_cols:
    df[col + '_norm'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
# Normalize and invert 'bad' columns (higher is worse)
for col in bad_cols:
    df[col + '_norm'] = 1 - (df[col] - df[col].min()) / (df[col].max() - df[col].min())

print('\nNormalized columns:')
print(df[[c+'_norm' for c in good_cols + bad_cols]].head())

# 3. Scoring Formula & Feature Weighting
# --------------------------------------
# Assign weights to each feature based on business logic and perceived impact.
# The sum of weights should be 1.0 (or 100% when scaled).
weights = {
    'team_experience_norm': 0.15,           # Team quality
    'market_size_million_usd_norm': 0.20,   # Market opportunity
    'monthly_active_users_norm': 0.25,      # Traction
    'monthly_burn_rate_inr_norm': 0.10,     # Efficiency (inverted)
    'funds_raised_inr_norm': 0.10,          # Fundraising
    'valuation_inr_norm': 0.20              # Perceived value
}

# Compute the composite score as a weighted sum, scaled to 100.
df['composite_score'] = sum(df[col] * w for col, w in weights.items()) * 100
print('\nComposite scores:')
print(df[['startup_id', 'composite_score']].head())

# 4. Ranking & Interpretation
# ---------------------------
# Rank startups by their composite score. Identify top and bottom performers.
df_sorted = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
top10 = df_sorted.head(10)
bottom10 = df_sorted.tail(10)

print('\nTop 10 Startups:')
print(top10[['startup_id', 'composite_score']])
print('\nBottom 10 Startups:')
print(bottom10[['startup_id', 'composite_score']])

# Example: Print details for the top and bottom scorer for interpretation.
top_row = top10.iloc[0]
bottom_row = bottom10.iloc[0]
print(f"\nTop Scorer: {top_row['startup_id']}\n", top_row)
print(f"\nBottom Scorer: {bottom_row['startup_id']}\n", bottom_row)

# 5. Visualization
# ----------------
# Create output directory if it doesn't exist.
os.makedirs('../outputs', exist_ok=True) if os.path.exists('../outputs') else os.makedirs('outputs', exist_ok=True)
OUTPUT_DIR = '../outputs' if os.path.exists('../outputs') else 'outputs'

# Bar chart: Composite scores for all startups, sorted.
plt.figure(figsize=(14,4))
plt.bar(df_sorted['startup_id'], df_sorted['composite_score'])
plt.title('Startup Composite Scores (Sorted)')
plt.xlabel('Startup ID')
plt.ylabel('Score')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'bar_chart_scores.png'))
plt.close()

# Correlation heatmap: Shows relationships between normalized features.
plt.figure(figsize=(8,6))
sns.heatmap(df[[c+'_norm' for c in good_cols + bad_cols]].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap (Normalized Features)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'))
plt.close()

# Score distribution: Histogram of composite scores.
plt.figure(figsize=(8,4))
sns.histplot(df['composite_score'], bins=20, kde=True)
plt.title('Score Distribution')
plt.xlabel('Composite Score')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'score_distribution.png'))
plt.close()

# 6. Documentation & Insights
# ---------------------------
# Append key insights and methodology to the README for transparency.
with open('README.md', 'a') as f:
    f.write('\n\n## Insights from Analysis (auto-appended)\n')
    f.write('- Weights chosen based on perceived business impact (traction, market, team, etc.).\n')
    f.write('- Burn Rate is inverted so lower burn is better.\n')
    f.write('- See script output and plots in outputs/ for more details.\n')

print(f"\nPlots saved to {OUTPUT_DIR}/.\nDocumentation appended to README.md.")

# 7. Bonus: ML Extension - Clustering Startups with KMeans
# --------------------------------------------------------
# Cluster startups into archetypes using KMeans on normalized features.
# Visualize clusters using PCA for dimensionality reduction.
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Prepare data for clustering (normalized features only)
feature_cols = [c+'_norm' for c in good_cols + bad_cols]
X = df[feature_cols]

# Choose number of clusters (e.g., 3 archetypes)
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X)

# Reduce to 2D for visualization using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
df['pca1'] = X_pca[:,0]
df['pca2'] = X_pca[:,1]

# Plot clusters in PCA-reduced space
plt.figure(figsize=(8,6))
for i in range(n_clusters):
    plt.scatter(df[df['cluster']==i]['pca1'], df[df['cluster']==i]['pca2'], label=f'Cluster {i}')
plt.title('Startup Clusters (KMeans, PCA-reduced)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'kmeans_clusters.png'))
plt.close()

# Show mean feature values for each cluster (archetype profile)
grouped = df.groupby('cluster')[feature_cols + ['composite_score']].mean()
print('\nKMeans Cluster Centers (mean normalized features):')
print(grouped)

# Append clustering explanation and summary to README
with open('README.md', 'a') as f:
    f.write('\n\n## Bonus: ML Extension - Startup Clustering\n')
    f.write('We applied KMeans clustering (k=3) to the normalized features to identify archetypes among startups.\n')
    f.write('Clusters were visualized using PCA for dimensionality reduction.\n')
    f.write('Cluster centers (mean feature values) reveal typical profiles, e.g., high-growth/low-burn, high-burn/low-traction, etc.\n')
    f.write('See the plot in outputs/kmeans_clusters.png and the summary table below.\n')
    f.write('\n\nCluster Centers (mean normalized features):\n')
    f.write(grouped.to_string())
    f.write('\n')

print(f"\nKMeans clustering complete. Cluster plot saved to {OUTPUT_DIR}/kmeans_clusters.png. Explanation appended to README.md.") 