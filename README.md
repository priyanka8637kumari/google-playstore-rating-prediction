# Google Play Store - App Rating Prediction

## Report

### Problem and Dataset

I chose to work on predicting whether a Google Play Store app is
highly rated (4.0 or above). The dataset had 10,841 rows with 13
columns including category, reviews, size, installs, price, and rating.

What I liked about this dataset was that it wasn't clean. it had
real missing values, mixed formats, and dirty rows. That made it
more challenging but also more realistic.

---

### Data Preparation

This was actually the biggest part of the project. Here's what I did:

**Handling missing values**
- Rating had 1,474 missing values. since this is the column I'm
  predicting from, I couldn't guess or fill it, so I dropped those rows
- Type, Content Rating, Current Ver, and Android Ver had very few
  missing values (1–8 rows each), so I filled them with the most
  common value (mode)

**Cleaning messy columns**
- Size had values like "19M" and "8.7k" — I converted these to plain numbers
- Installs had values like "10,000+" — I removed the "+" and ","
- Price had values like "$4.99" — I removed the "$"
- During visualization I spotted dirty rows where Category was "1.9"
  and Type was "0" — these were clearly data entry errors, so I removed them

**Converting text to numbers**
- Columns like Category, Type, and Content Rating were text.
  machine learning models can't read text, so I used Label Encoding
  to convert each unique value to a number

**Final dataset:** 9,366 rows with 7 features

---

### Data Leakage Fix

After finishing the project, I reviewed my work with an AI tool which pointed out that I had included
Rating as a feature but Rating is exactly what I used to create
the target variable (HighRating = Rating >= 4.0). That means the
model was basically predicting Rating using Rating itself, which
is cheating.

I fixed this by removing Rating from the features completely.
The final features used are:
Category, Reviews, Size, Installs, Type, Price, Content Rating

After the fix, the 80% accuracy is genuine, the model learned
real patterns from app metadata, not from the target column itself.

---

### Model Selection and Results

I tested three models:

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Logistic Regression (original) | 0.79 | 0.88 |
| Logistic Regression (balanced) | 0.48 | 0.53 |
| Random Forest (balanced) | 0.80 | 0.88 |

The first model looked good at 79% accuracy but it was actually
just predicting everything as highly rated but it completely failed
on low rated apps (recall 0.00 for class 0).

I added class_weight='balanced' to fix this, but that overcorrected
and the model started missing too many highly rated apps.

**Random Forest** with balanced weights gave the best overall result —
80% accuracy with a good balance between catching both classes.

---

### Limitations and Reflection

The biggest challenge was the imbalanced dataset — 79% of apps are
highly rated, which kept confusing the models. I researched and found that this is a common
problem in real-world classification tasks.

In practice, this kind of model could help app developers understand
what factors matter before launching an app. But there are clear
limitations:

- The model still struggles to correctly identify low rated apps
- Ratings can be influenced by things not in the dataset, like
  marketing campaigns or app store featuring
- The data is from 2018 and the Play Store has changed a lot since then

---

### Possible Improvements

- Use SMOTE to create synthetic samples of the minority class
  and balance the dataset properly
- Add more features like sentiment analysis of app descriptions
- Try more advanced models like XGBoost or a neural network
  and compare results
