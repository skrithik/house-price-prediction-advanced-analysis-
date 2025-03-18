# house-price-prediction-advanced-analysis
Below is a simplified `README.md` file that highlights the main steps and insights from your project without being overly formal:

```markdown
# House Price Prediction System

This project is built to show how detailed exploratory data analysis (EDA) and careful data handling can really improve model performance—even with a simple model. It walks through how we handled data preprocessing, outliers, selective feature engineering, and model optimization using Lasso regression and GridSearchCV.

## What We Did

### 1. Data Collection & Preprocessing
- **Data Collection:**  
  Collected the dataset from an online platform.
  
- **Handling Missing Values & Duplicates:**  
  Cleaned the data by addressing missing values and removing duplicate entries to ensure quality.

### 2. Exploratory Data Analysis (EDA)
- **Initial EDA:**  
  Performed univariate, bivariate, and multivariate analyses to understand the data and extract insights.
  
- **Post-Outlier Handling EDA:**  
  Ran another round of EDA after handling outliers to see the improvements in data distribution and insights.

### 3. Outlier Handling
- **Method:**  
  Used the IQR (Interquartile Range) method to cap outliers.
  
- **Code Example:**
  ```python
  # Calculate the 25th and 75th percentiles and IQR
  Q1 = train_data_capped.quantile(0.25)
  Q3 = train_data_capped.quantile(0.75)
  IQR = Q3 - Q1
  
  # Cap outliers within the range [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
  train_data_capped = train_data_capped.clip(lower=(Q1 - 1.5 * IQR), upper=(Q3 + 1.5 * IQR), axis=1)
  ```

### 4. Feature Engineering
- **Creating New Features:**  
  Added new ratio-based features:
  - `rooms_per_household` = total_rooms / households
  - `population_per_household` = population / households

- **Handling Categorical Data:**  
  Modified latitude and longitude categorical features by aggregating some categories and dropping redundant ones.

- **Code Example:**
  ```python
  # Ratio-based features
  train_data_feature_engineered['rooms_per_household'] = train_data_feature_engineered['total_rooms'] / train_data_feature_engineered['households']
  train_data_feature_engineered['population_per_household'] = train_data_feature_engineered['population'] / train_data_feature_engineered['households']
  
  # Backup and combine categorical latitude and longitude features
  train_data_feature_engineered["lat_category_high"] = train_data_feature_engineered["lat_category_0"]
  train_data_feature_engineered["lon_category_high"] = train_data_feature_engineered["lon_category_0"]
  
  train_data_feature_engineered["lat_category_low"] = train_data_feature_engineered["lat_category_3"] + train_data_feature_engineered["lat_category_4"]
  train_data_feature_engineered["lon_category_low"] = train_data_feature_engineered["lon_category_3"] + train_data_feature_engineered["lon_category_4"]
  
  # Drop old categorical columns
  lat_lon_cols = [col for col in train_data_feature_engineered.columns if "lat_category_" in col or "lon_category_" in col]
  lat_lon_cols.remove("lat_category_high")
  lat_lon_cols.remove("lon_category_high")
  lat_lon_cols.remove("lat_category_low")
  lat_lon_cols.remove("lon_category_low")
  train_data_feature_engineered.drop(columns=lat_lon_cols, inplace=True)
  ```

### 5. Model Evaluation & Optimization
- **Model Choice:**  
  Used Lasso regression as a simple yet effective model.
  
- **Training & Evaluation:**  
  Trained the model on the processed data and evaluated it using both training and cross-validation scores.
  
- **Hyperparameter Tuning:**  
  Applied GridSearchCV to find the best Lasso parameter (`alpha`) by plotting training vs. CV errors to choose the optimum value.

- **Pipeline Approach:**  
  Used a pipeline to streamline model training and hyperparameter tuning.
  
- **Code Example:**
  ```python
  from sklearn.pipeline import Pipeline
  from sklearn.linear_model import Lasso
  from sklearn.model_selection import GridSearchCV
  
  # Create a pipeline with Lasso regression
  pipeline = Pipeline([
      ('model', Lasso())
  ])
  
  # Grid of alpha values to search over
  param_grid = {'model__alpha': [0.01, 0.1, 1, 10, 100]}
  
  # Grid Search with Cross-Validation
  grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
  grid_search.fit(X_train, y_train)
  
  # Get the best alpha
  optimal_alpha = grid_search.best_params_['model__alpha']
  ```

## Summary
This project highlights:
- Detailed and thorough EDA at multiple stages
- Rigorous data preprocessing (handling missing values, duplicates, and outliers)
- Selective feature engineering to improve model inputs
- Effective model optimization using GridSearchCV and visualization of Lasso parameter vs. error
- Optimization of hypermparameter by plotting graph between train and cross validation errors with respect to lasso parameter 

Even a simple model like Lasso regression can be made robust and accurate with careful data handling and tuning!
```

This version focuses on clearly describing each step of the process and highlighting the key areas you mentioned without being overly formal. Feel free to modify any sections to better suit your needs.
