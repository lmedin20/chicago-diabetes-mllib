# Diabetes Risk & Chicago Neighborhood Clustering (PySpark / AWS EMR)

Two independent analyses built on Apache Spark (PySpark MLlib), run on an AWS EMR cluster.

## 1. Diabetes Risk — Logistic Regression
Predicts diabetes likelihood from clinical measurements using a logistic regression
classifier.

- **Data**: Clinical dataset (403 individuals) with age, gender, family history,
  height, weight, cholesterol, and diabetes status.
- **Process**:
  - Cleaned data and removed missing records
  - Engineered BMI from height/weight (converted to metric units)
  - Encoded categorical fields (gender, diabetic status) as binary
  - Split into 80/20 train/test sets (fixed seed for reproducibility)
  - Trained a `LogisticRegression` model (PySpark ML) using age, gender, family
    history, BMI, and cholesterol as features
- **Result**: Achieved 83% accuracy on held-out test data.

## 2. Chicago Neighborhoods — KMeans Clustering
Segments Chicago neighborhoods into 3 clusters based on socioeconomic and
demographic characteristics.

- **Data**: Racial composition, income, and unemployment data for 70+ Chicago
  neighborhoods.
- **Process**:
  - Assembled features (Black %, Hispanic %, unemployment rate, income)
  - Fit a `KMeans` model (k=3) using PySpark ML
  - Visualized clusters (unemployment vs. income, colored by cluster)
  - Computed average demographic/economic profile per cluster
- **Result**: Identified distinct neighborhood groupings and analyzed how income
  and unemployment patterns relate to racial composition across clusters.

## Tech Stack
- Apache Spark / PySpark (MLlib: `LogisticRegression`, `KMeans`, `VectorAssembler`)
- AWS EMR (Spark cluster, manually scaled)
- Python (pandas, matplotlib for visualization)

## Files
- `LogisticRegression.ipynb` — diabetes classification model
- `kMeansClustering.ipynb` — neighborhood clustering analysis
