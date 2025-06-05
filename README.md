# TASK--7
Support Vector Machines (SVM)

# Objective :
 To use Support Vector Machines (SVM) for solving a binary classification problem on the Breast Cancer dataset.

# Tools Used :
 - Python
 - Scikit-learn
 - Numpy
 - Matplotlib
 - Pandas

# Steps Performed :
 1. Loaded and cleaned the dataset
    - Removed any non-numeric or missing values
    - Separated features and target label
 2. Date Preprocessing
    - Scaled features using standerdscaler
    - Split the date into training and testing sets (80/20 split)
 3. Model Training
    - Trained two SVM classifiers
    - Linear kernel- suitable  for linearaly  separable date
    - RBF Kernel- handles non-linear separation using kernel trick
 4. Hyperparameter Tuning
    - Used GridSearchCV to find the best combination of c and gamma for the RBF kernel
 5. Model Evaluation
    - Evaluation models using:
    - confusion Matrix
    - Classification Report(Precision,Recall,F1-score)
 6. Visualization
    - Applied PCA to reduce feature dimentions to 2D
    - Plotted the decision boundary using the best SVM model trained on PCA-reduced data.

# Output :
First 5 rows of the dataset:
         id diagnosis  radius_mean  texture_mean  perimeter_mean  area_mean  \
0    842302         M        17.99         10.38          122.80     1001.0   
1    842517         M        20.57         17.77          132.90     1326.0   
2  84300903         M        19.69         21.25          130.00     1203.0   
3  84348301         M        11.42         20.38           77.58      386.1   
4  84358402         M        20.29         14.34          135.10     1297.0   

   smoothness_mean  compactness_mean  concavity_mean  concave points_mean  \
0          0.11840           0.27760          0.3001              0.14710   
1          0.08474           0.07864          0.0869              0.07017   
2          0.10960           0.15990          0.1974              0.12790   
3          0.14250           0.28390          0.2414              0.10520   
4          0.10030           0.13280          0.1980              0.10430   

   ...  radius_worst  texture_worst  perimeter_worst  area_worst  \
0  ...         25.38          17.33           184.60      2019.0   
1  ...         24.99          23.41           158.80      1956.0   
2  ...         23.57          25.53           152.50      1709.0   
3  ...         14.91          26.50            98.87       567.7   
4  ...         22.54          16.67           152.20      1575.0   

   smoothness_worst  compactness_worst  concavity_worst  concave points_worst  \
0            0.1622             0.6656           0.7119                0.2654   
1            0.1238             0.1866           0.2416                0.1860   
2            0.1444             0.4245           0.4504                0.2430   
3            0.2098             0.8663           0.6869                0.2575   
4            0.1374             0.2050           0.4000                0.1625   

   symmetry_worst  fractal_dimension_worst  
0          0.4601                  0.11890  
1          0.2750                  0.08902  
2          0.3613                  0.08758  
3          0.6638                  0.17300  
4          0.2364                  0.07678  

[5 rows x 32 columns]

Column names before filtering:
Index(['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst'],
      dtype='object')

Unique values in the target column 'diagnosis': ['M' 'B']
Converting target column 'diagnosis' to numeric...
Conversion complete. Unique numeric values in target: [1 0]

--- Linear Kernel SVM ---
[[68  3]
 [ 2 41]]
              precision    recall  f1-score   support

           0       0.97      0.96      0.96        71
           1       0.93      0.95      0.94        43

    accuracy                           0.96       114
   macro avg       0.95      0.96      0.95       114
weighted avg       0.96      0.96      0.96       114


--- RBF Kernel SVM (Tuned) ---
Best Parameters: {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}
[[71  0]
 [ 2 41]]
              precision    recall  f1-score   support

           0       0.97      1.00      0.99        71
           1       1.00      0.95      0.98        43

    accuracy                           0.98       114
   macro avg       0.99      0.98      0.98       114
weighted avg       0.98      0.98      0.98       114

![piccc](https://github.com/user-attachments/assets/3b4c9d6b-e774-4b9f-9071-0133e0436677)

    
  
