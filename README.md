
# Predicting-Cancer-Diagnosis-using-Classification-Models
## Project Objective:
In honor of Breast Cancer Awareness Month, this project aims to develop and evaluate machine learning models to classify breast cancer tumors as either malignant or benign. The project will involve:
1. Understanding the dataset and performing necessary data cleanup.
2. Building classification models and fine-tuning hyperparameters.
3. Comparing evaluation metrics to determine the most accurate and efficient model.
This project is dedicated to raising awareness and advancing early diagnosis efforts in the fight against breast cancer.
## Tools Used
1. Python
2. Power BI
## Data 
The data used in this was downloaded from kaggle.Download [here](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)
- Breast cancer is the most common cancer amongst women in the world. It accounts for 25% of all cancer cases, and affected over 2.1 Million people in 2015 alone. It starts
  when cells in the breast begin to grow out of control. These cells usually form tumors that can be seen via X-ray or felt as lumps in the breast area.
- The key challenges against it’s detection is how to classify tumors into malignant (cancerous) or benign(non cancerous).
  ### Data description:
  The data has the following columns:
  1. id - This is the unique identifier for each patient or observation.
  2. diagnosis - The diagnosis of breast cancer, indicating whether the cancer is Malignant (M) or Benign (B).
  3. radius_mean - The average distance from the center to points on the perimeter of the nucleus. This provides a measureof     the size of the cell nuclei.
  4. texture_mean - The standard deviation of gray-scale values, capturing the variation in the texture or smoothness of the     surface of the cell nuclei.
  5. perimeter_mean - The average perimeter of the cell nucleus. A larger perimeter generally correlates with a larger,           possibly malignant cell.
  6. area_mean - The average area of the cell nucleus. Larger cell areas are more likely to be malignant.
  7. smoothness_mean - The mean of local variations in the radius lengths. It measures how smooth the edges of the cell           nucleus are. Less smoothness might indicate malignancy.
  8. compactness_mean -This captures how compact the cell nucleus is. Less compactness may be a sign of malignancy.
  9. concavity_mean - The mean extent of concave (inward-curved) portions of the cell nucleus boundary. Larger concavities        are associated with malignant cells.
  10. concave points_mean - The mean number of concave points (points where the perimeter curves inward) on the boundary of        the cell nucleus. More concave points suggest malignancy.
  11. symmetry_mean - The mean measure of how symmetric the cell nucleus is. Malignant cells often exhibit more asymmetry.
  12. fractal_dimension_mean - The mean "coastline" fractal dimension, which quantifies the complexity of the boundary.            Malignant cells tend to have more complex, irregular boundaries.
  13. radius_se - The standard error of the radius measurements. Higher values indicate variability in cell sizes, which may       indicate malignancy.
  14. texture_se - The standard error of texture measurements, indicating variability in surface texture.
  15. perimeter_se - The standard error of the perimeter measurements. Higher values reflect variability in the cell               boundaries.
  16. area_se - The standard error of the area measurements. Large variability in cell areas is often associated with              malignancy.
  17. smoothness_se - The standard error of smoothness measurements, reflecting variability in edge smoothness.
  18. compactness_se - The standard error of compactness measurements, indicating variability in how tightly packed the cell       is.
  19. concavity_se - The standard error of concavity measurements, showing the variability in the inward curvature of the          cell boundaries.
  20. concave points_se - The standard error of the number of concave points, reflecting the variability in boundary               concavities.
  21. symmetry_se - The standard error of symmetry measurements, indicating how variable the symmetry is between different         cells.
  22. fractal_dimension_se - The standard error of fractal dimension measurements, capturing the variability in boundary           complexity.
  23. radius_worst - The largest mean radius observed across all cells.
  24. texture_worst - The worst texture measurement, representing the most extreme variation in surface smoothness among the       cells.
  25. perimeter_worst - The largest perimeter measurement observed, indicating the most extreme cell size.
  26. area_worst - The largest area measurement observed across cells.
  27. smoothness_worst - The worst smoothness value, representing the least smooth or roughest cell boundary.
  28. compactness_worst - The most extreme compactness value observed.
  29. concavity_worst - The largest concavity observed, representing the cell with the most inward curvature.
  30. concave points_worst - The most concave points observed across cells.
  31. symmetry_worst - The least symmetric cell observed.
  32. fractal_dimension_worst - The most complex or irregular cell boundary observed across all cells.
  ### Data cleaning
1. Checked for missing values - there were no missing values.
2. Checked for duplicates - there were no duplicates
3. Checked for errors, and data types to ensure consistency and acuracy of the data.
   - Since the diagnosis column is of string data type, I 
   
 ## Exploratory Data Analysis
 1. ### Univariate EDA
    a) What is the distribution of the target variable diagnosis.
    
    ![image](https://github.com/user-attachments/assets/daee4189-7c9f-49e6-8257-142a4ebda551)

    b) What are the distributions of key features like radius_mean, perimeter_mean, area_mean, and                                  concavity_mean?

    ![image](https://github.com/user-attachments/assets/53fb99ea-8c87-4a53-8eac-77f6db766272)

  2. ### Bivariate EDA
     c) How are radius_mean and perimeter_mean related?

     ![image](https://github.com/user-attachments/assets/549861d6-c71c-4090-bf10-d7fabe8d11a6)

     d) What is the correlation between numerical features?

     ![image](https://github.com/user-attachments/assets/b6e51be0-ab14-4bdb-ae52-9e83d789cb6f)

     e) How does compactness_mean vary between malignant and benign diagnoses?

     ![image](https://github.com/user-attachments/assets/f95e1795-35b1-4589-bb7c-dc555340a6dc)
     
  3. ### Multivariate EDA
     f) How do radius_mean, texture_mean, and area_mean interact across different diagnoses?

     ![image](https://github.com/user-attachments/assets/197a5c75-430b-4640-a7e3-f86ad73a4214)

     g) What are the principal components that explain the variance in the dataset?

     ![image](https://github.com/user-attachments/assets/85c6aeca-edf4-4330-bf20-2df174655239)

## Data preprocessing 
In order to use the data to train the models, we need to pre- process it. 
1. We first encode the diagnosis column, since it is an object column, we encode it to transform it to a numerical column.
2. We then select the columns that are highly correlated with the target variable.
   - We get the correlation values.
   - We ten select the values using a thershold of 0.2.
   - 25 rows are selected i.e (radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean,     
     concavity_mean, concave points_mean, symmetry_mean, radius_se, perimeter_se, area_se, compactness_se, concavity_se, 
     concave points_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, 
     concavity_worst, concave points_worst, symmetry_worst, fractal_dimension_worst)
## Model Building
In this project, we will use the following models : 
1. ### Logistic Regression:
- Logistic Regression is a widely used model for binary classification problems, such as predicting whether breast cancer 
  is benign or malignant. It works by modeling the probability that an instance belongs to a particular class (e.g., 
  malignant) using a sigmoid function, which maps input features to a probability between 0 and 1. The model calculates a 
  decision boundary that separates the two classes and is particularly useful when the relationship between the features 
  and the outcome is linear. For breast cancer prediction, you can train the model on labeled data (where diagnosis is 
  known), and it will learn the weight of each feature to predict the probability of malignancy.

2. ### Decision Tree:
- A Decision Tree classifier makes predictions by recursively splitting the dataset based on feature values, creating a 
  tree structure where each node represents a decision based on a feature, and the leaf nodes represent the final 
  classification (e.g., benign or malignant). It is interpretable and visually intuitive, allowing for straightforward 
  explanations of why a particular diagnosis was made. For breast cancer prediction, the Decision Tree will learn the most 
  important features (e.g., tumor size, texture) that best split the data to classify the type of cancer accurately.

3. ### K-Nearest Neighbors (KNN):
- KNN is a simple, non-parametric model that classifies a data point based on the majority class of its nearest neighbors. 
  It works by calculating the distance (often Euclidean) between the new data point and all other points in the training 
  set. For predicting breast cancer, KNN can determine whether a tumor is malignant or benign based on the labels of the 
  most similar cases in the dataset. However, KNN’s performance depends heavily on the number of neighbors (K) and the 
  distance metric used, making it sensitive to noisy data and the curse of dimensionality.

4. ### Random Forest Classifier:
- Random Forest is an ensemble learning method that combines multiple Decision Trees to create a more robust and accurate 
  model. Each tree is trained on a random subset of the data, and the final prediction is made by averaging the predictions   of all trees (in classification, through a majority vote). For breast cancer prediction, Random Forest reduces   
  overfitting compared to a single Decision Tree and improves accuracy by leveraging the wisdom of many trees. It is 
  particularly useful for datasets with a large number of features, as it can handle high-dimensional data well.

5. ### XGBoost Classifier:
- XGBoost (Extreme Gradient Boosting) is an advanced, highly efficient boosting algorithm that builds models iteratively, 
  where each new model corrects the errors of the previous one. It uses a combination of decision trees that focus on the 
  most challenging cases to classify. For breast cancer prediction, XGBoost can achieve high accuracy by optimizing model 
  performance through regularization and handling imbalanced datasets effectively. Its ability to capture complex patterns 
  and interactions in the data makes it suitable for this task, though it requires careful tuning of hyperparameters.

- I defined the independend(x) and response(y) variables. 
- The data was split into train and test sets. The ratio of 80:20 was used.
- I then standardized the independent variable using StandardScaler.
- StandardScaler is used to standardize features by removing the mean and scaling them to unit variance, making it useful   
  for preparing data before training machine learning models.
- Since I will be fitting 5 models, I defined a function that fits the model on the training data, makes predictions on the 
  test data, computes evaluation metrics, and returns a DataFrame with the results.
  ## Results
  1. Logistic regression
     ### Model Evaluation Metrics

| Accuracy | F1 Score | Precision | Recall | Balanced Accuracy |
|----------|----------|-----------|--------|-------------------|
| 0.964912 | 0.952381 | 0.975610  | 0.930233 | 0.958074         |

  2. 





    

