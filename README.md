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
  3. radius_mean - The average distance from the center to points on the perimeter of the nucleus. This provides a measure of the size of the cell nuclei.
  4. texture_mean - The standard deviation of gray-scale values, capturing the variation in the texture or smoothness of the surface of the cell nuclei.
  5. perimeter_mean - The average perimeter of the cell nucleus. A larger perimeter generally correlates with a larger, possibly malignant cell.
  6. area_mean - The average area of the cell nucleus. Larger cell areas are more likely to be malignant.
  7. smoothness_mean - The mean of local variations in the radius lengths. It measures how smooth the edges of the cell nucleus are. Less smoothness might indicate
     malignancy.
  8. compactness_mean -This captures how compact the cell nucleus is. Less compactness may be a sign of malignancy.
  9. concavity_mean - The mean extent of concave (inward-curved) portions of the cell nucleus boundary. Larger concavities are associated with malignant cells.
  10. concave points_mean - The mean number of concave points (points where the perimeter curves inward) on the boundary of the cell nucleus. More concave points suggest
     malignancy.
  13. 'symmetry_mean',
  14. 'fractal_dimension_mean',
  15. 'radius_se',
  16. 'texture_se',
  17. 'perimeter_se',
  18. 'area_se',
  19. 'smoothness_se',
  20. 'compactness_se',
  21. 'concavity_se',
  22. 'concave points_se',
  23. 'symmetry_se',
  24. 'fractal_dimension_se',
  25. 'radius_worst',
  26. 'texture_worst',
  27. 'perimeter_worst',
  28. 'area_worst',
  29. 'smoothness_worst',
  30. 'compactness_worst',
  31. 'concavity_worst',
  32.  'concave points_worst',
  33.   symmetry_worst',
  34.   'fractal_dimension_worst
