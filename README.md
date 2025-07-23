# Startup Evaluation Engine

A comprehensive, data-driven engine to evaluate and rank startups based on key business indicators—similar to a credit score, but for startups. This project demonstrates a clear, structured approach to scoring, ranking, and interpreting startup potential using Python and machine learning best practices.

---

## Note on Submission Format

I personally prefer working with well-commented Python scripts. However, to align with the assignment's preference for Jupyter/Colab notebooks—which keep code, output, and reasoning in one place—I have also provided a Google Colab-compatible notebook version of the analysis. This ensures the project is accessible and reviewable in the format most convenient for evaluators.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [How to Run](#how-to-run)
- [Methodology](#methodology)
- [Feature Weighting & Scoring](#feature-weighting--scoring)
- [Visualization Outputs](#visualization-outputs)
- [Bonus: ML Extension (KMeans Clustering)](#bonus-ml-extension-kmeans-clustering)
- [Insights & Future Improvements](#insights--future-improvements)

---

## Project Overview
This project simulates a startup evaluation engine that generates a composite score (out of 100) for each startup based on:
- Team experience
- Market size
- User traction
- Burn rate (inverted)
- Funds raised
- Valuation

The higher the score, the stronger the startup’s potential. The project includes data preprocessing, custom scoring, ranking, visualization, and a bonus ML clustering extension.

---

## Folder Structure
```
Startup-Health_ScoringModel-Task1/
  src/
    startup_scoring.py         # Main analysis script
  outputs/
    bar_chart_scores.png       # Bar chart of sorted scores
    correlation_heatmap.png    # Correlation heatmap
    score_distribution.png     # Score distribution histogram
    kmeans_clusters.png        # KMeans cluster visualization
  requirements.txt             # Python dependencies
  README.md                    # Project documentation
  Startup_Scoring_Dataset.csv  # Input dataset
```

---

## How to Run
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the analysis script:**
   ```bash
   python src/startup_scoring.py
   ```
3. **View results:**
   - Plots and visualizations are saved in the `outputs/` directory.
   - Key insights and cluster summaries are appended to this `README.md`.

---

## Methodology
- **Data Preprocessing:**
  - All numeric features are normalized to a 0-1 range using Min-Max normalization.
  - Burn rate is inverted so that lower values are better (since high burn is a negative indicator).
- **Feature Weighting:**
  - Each feature is assigned a weight based on business logic and impact (see below).
- **Composite Score:**
  - The final score is a weighted sum of normalized features, scaled to 100.
- **Ranking:**
  - Startups are ranked by their composite score. Top 10 and Bottom 10 are highlighted.
- **Visualization:**
  - Bar chart, correlation heatmap, and score distribution histogram are generated.
- **Interpretation:**
  - The script prints and appends explanations for the top and bottom performers.

---

## Feature Weighting & Scoring
| Feature                   | Weight (%) | Rationale                        |
|---------------------------|------------|-----------------------------------|
| Team Experience           | 15         | Quality of founding team          |
| Market Size               | 20         | Opportunity size                  |
| Monthly Active Users      | 25         | Traction and growth               |
| Monthly Burn Rate (inv.)  | 10         | Efficiency/sustainability         |
| Funds Raised              | 10         | Fundraising ability               |
| Valuation                 | 20         | Perceived market value            |

- **Burn Rate** is inverted so that lower burn is better.
- **Composite Score** = Weighted sum of normalized features × 100.

---

## Visualization Outputs
- **bar_chart_scores.png:** Bar chart of all startup scores (sorted)
- **correlation_heatmap.png:** Correlation heatmap of normalized features
- **score_distribution.png:** Histogram of score distribution
- **kmeans_clusters.png:** KMeans cluster visualization (see below)

All plots are saved in the `outputs/` directory after running the script.

---

## Bonus: ML Extension (KMeans Clustering)
To further analyze the startup landscape, we applied KMeans clustering (k=3) to the normalized features. This groups startups into archetypes based on their business indicators:
- **Cluster 0:** May represent high-growth, low-burn startups
- **Cluster 1:** May represent high-burn, low-traction startups
- **Cluster 2:** May represent balanced or average performers

Clusters are visualized using PCA (Principal Component Analysis) to reduce feature space to 2D for easy plotting. The cluster centers (mean feature values) help interpret the typical profile of each group.

See the plot in `outputs/kmeans_clusters.png` and the summary table below for details.

---

## Insights & Future Improvements

### Key Insights
- **Top Performers:** Startups with high scores typically exhibit a strong combination of experienced teams, large market opportunities, high user traction, and efficient (low) burn rates. These factors collectively drive their high composite scores.
- **Low Performers:** Startups with lower scores often struggle with high burn rates, limited user traction, or smaller market sizes, even if they have raised significant funds. This highlights the importance of balanced growth and sustainability.
- **Feature Correlations:** The correlation heatmap reveals relationships between features. For example, valuation often correlates with funds raised and user growth, while burn rate may be inversely related to efficiency and sustainability.

### Recommendations for Future Work
- **Feature Engineering:**
  - Incorporate additional features such as growth rates, churn, customer acquisition cost, or qualitative assessments (e.g., founder background, product differentiation).
  - Analyze time-series data if available to capture trends and momentum.
- **Advanced Modeling:**
  - Experiment with other machine learning models (e.g., regression, decision trees, or ensemble methods) to predict outcomes like future valuation or user growth.
  - Use feature importance techniques to refine the weighting scheme based on data-driven insights.
- **Clustering & Segmentation:**
  - Explore different numbers of clusters (k) or alternative clustering algorithms (e.g., hierarchical clustering, DBSCAN) to uncover more nuanced startup archetypes.
  - Profile each cluster in detail to provide actionable recommendations for each group.
- **Validation & Robustness:**
  - Apply cross-validation or bootstrapping to test the stability of the scoring and ranking methodology.
  - Compare results across different time periods or datasets to ensure generalizability.
- **Visualization & Reporting:**
  - Develop interactive dashboards (e.g., using Plotly Dash or Streamlit) for dynamic exploration of results.
  - Automate the generation of PDF or HTML reports for stakeholders.
- **Collaboration & Feedback:**
  - Engage with domain experts (e.g., investors, founders) to validate the scoring logic and feature selection.
  - Incorporate feedback to continuously improve the evaluation engine.

---

*For more details, see the code in `src/startup_scoring.py` and the generated plots in the `outputs/` directory.*
