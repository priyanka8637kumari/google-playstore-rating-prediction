## Final Report

### Problem and Dataset
The goal of this project was to predict whether a Google Play Store 
app is highly rated (4.0 or above) based on features like category, 
reviews, size, installs, type, price, and content rating. The dataset 
contained 10,841 apps with real missing values and messy data formats, 
making it a realistic and practical dataset to work with.

### Data Preparation
- Removed 1,474 rows where Rating was missing (target column)
- Filled minor missing values in Type, Content Rating, Current Ver 
  and Android Ver with the most common value (mode)
- Cleaned messy columns — Size (converted from "19M" to numbers), 
  Installs (removed "+" and ","), Price (removed "$")
- Removed dirty rows where Category was "1.9" and Type was "0"
- Converted categorical columns (Category, Type, Content Rating) 
  to numeric using Label Encoding
- Final dataset: 9,366 rows with 7 features

### Model Selection and Results
Three models were tested:

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Logistic Regression (original) | 0.79 | 0.88 |
| Logistic Regression (balanced) | 0.48 | 0.53 |
| Random Forest (balanced) | 0.80 | 0.88 |

Random Forest was the best performing model — achieving 80% accuracy 
with a good balance between precision and recall.

### Limitations and Reflection
The biggest challenge was the imbalanced dataset — 79% of apps are 
highly rated, which caused the original model to simply predict 
everything as highly rated. This is a common real-world problem.

In practice, this model could help app developers understand what 
factors contribute to high ratings before launching an app. However 
it has limitations:

- It still struggles to identify low rated apps correctly
- Ratings can be influenced by external factors like marketing and 
  timing that are not in the dataset
- The dataset is from 2018 and may not reflect current Play Store trends

Possible improvements:
- Use SMOTE to artificially balance the dataset
- Add more features like app description sentiment
- Try more advanced models like XGBoost or neural networks
