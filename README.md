<h1>Breast Cancer Prediction using Machine Learning</h1>
<h2>Team Details</h2>
<br/>
<h3>Jay Kapadiya - 9471899901</h3>
<br/>

<h3>Jaydev Gupta - 8657307877<h3>
  
<br/>

<h2>Introduction</h2>

Breast cancer is one of the most common types of cancer affecting women worldwide. Early detection can significantly improve survival rates by allowing timely medical intervention. In this project, we build a machine learning model to predict whether a tumor is benign or malignant based on various clinical features. This model can serve as a decision support tool for healthcare professionals in identifying cancerous tumors early.

<h2>Dataset Description</h2>

The dataset contains features related to breast cancer diagnostics, such as the size, shape, and texture of cell nuclei. Each record represents a patient, and the dataset includes the following columns:

	•	radius_mean: Mean of distances from the center to points on the perimeter of the tumor.
	•	texture_mean: Standard deviation of gray-scale values.
	•	perimeter_mean: Mean size of the tumor perimeter.
	•	area_mean: Mean size of the tumor area.
	•	smoothness_mean: Local variation in radius lengths.
	•	compactness_mean: Perimeter² / area - 1.0.
	•	concavity_mean: Severity of concave portions of the contour.
	•	concave points_mean: Number of concave portions of the contour.
	•	symmetry_mean: Symmetry of the tumor.
	•	fractal_dimension_mean: “Coastline approximation” - 1.
	•	diagnosis: The target variable, indicating whether the tumor is benign (B) or malignant (M).

<h2>Dataset Source:</h2>

The dataset is available on the UCI Machine Learning Repository and is widely used for binary classification tasks in the medical domain.

Dataset Split Info

The dataset is split into training and testing sets to evaluate the performance of the machine learning model:

	•	Training Set: 80% of the data is used for training the model.
	•	Test Set: 20% of the data is used for testing the model’s performance on unseen data.

Example of Split:

	•	Total Dataset Size: 569 rows
	•	Training Set Size: 455 rows
	•	Test Set Size: 114 rows

<h2>Exploratory Data Analysis (EDA)</h2>

Exploratory Data Analysis (EDA) helps in understanding the underlying patterns and relationships in the data before building the model. Key steps taken in EDA include:

1. Data Visualization

	•	Target Distribution: Checked the distribution of benign (B) and malignant (M) tumors.
	•	Feature Correlation: Created a heatmap to visualize the correlation between different features and the target variable.
	•	Histograms: Plotted histograms for continuous features like radius_mean, area_mean, compactness_mean to observe their distributions.
	•	Pairplots: Used pairplots to visualize the relationships between multiple features and how they differ for benign and malignant tumors.

2. Outlier Detection

	•	Box Plots: Plotted box plots to identify outliers in features like perimeter_mean and concavity_mean, as extreme values can affect model performance.

3. Feature Correlation

	•	Correlation Matrix: Examined the correlation matrix to identify features that are highly correlated with the target variable and with each other. This helps to detect multicollinearity, which can influence model performance.

4. Class Imbalance

	•	Class Balance Check: Verified whether the dataset is imbalanced in terms of benign and malignant cases. If the data was imbalanced, techniques like SMOTE or class weighting were considered to balance the dataset.

Key Findings from EDA:

	•	The dataset is relatively balanced, with a slightly higher number of benign cases.
	•	Certain features like radius_mean, area_mean, and concavity_mean showed significant differences between benign and malignant tumors, suggesting their predictive power.
	•	Strong correlations were found between radius_mean and area_mean, indicating potential redundancy.
<h2> Images of EDA<h2>
<img width="973" alt="Screenshot 2024-09-13 at 11 23 56 AM" src="https://github.com/user-attachments/assets/28b5cba6-8778-4e14-974b-1bf7ce014ff7">
<img width="1222" alt="Screenshot 2024-09-13 at 11 23 32 AM" src="https://github.com/user-attachments/assets/4ac593b3-b477-4c1a-9dc8-083d861384ca">
<img width="1208" alt="Screenshot 2024-09-13 at 11 23 16 AM" src="https://github.com/user-attachments/assets/5e041724-b2dc-46ec-80f8-bac2a107302f">
<img width="1208" alt="Screenshot 2024-09-13 at 11 23 10 AM" src="https://github.com/user-attachments/assets/68c3aa6e-61c5-4d92-a49e-b80bb75db173">
<img width="616" alt="Screenshot 2024-09-13 at 11 24 50 AM" src="https://github.com/user-attachments/assets/35f4432d-ca25-4697-a673-bd8ded41ae02">
<img width="1208" alt="Screenshot 2024-09-13 at 11 23 00 AM" src="https://github.com/user-attachments/assets/72656d22-2e05-4658-89f8-39181c0ed85a">

<h2>Approach</h2>

The approach to solving the breast cancer prediction problem is broken down into several steps:

1. Data Preprocessing

	•	Handling Missing Values: Checked and handled any missing or incorrect values.
	•	Feature Encoding: Encoded the target variable diagnosis using label encoding (0 for benign, 1 for malignant).
	•	Feature Scaling: Normalized continuous variables using standard scaling to ensure all features are on the same scale.

2. Model Selection

Several machine learning classifiers were explored for model building, including:

	•	Logistic Regression
	•	Random Forest Classifier
	•	Support Vector Machines (SVM)
	•	k-Nearest Neighbors (k-NN)
	•	Naive Bayes

3. Hyperparameter Tuning

For optimizing model performance, GridSearchCV was used to fine-tune the hyperparameters for the chosen classifiers:

	•	Number of trees for Random Forest.
	•	Regularization strength for Logistic Regression.
	•	Kernel type for SVM.
 <img width="1375" alt="Screenshot 2024-09-13 at 11 30 38 AM" src="https://github.com/user-attachments/assets/89917b29-90d9-4afc-ad38-88fdf13df75c">

4. Model Training and Evaluation

The models were trained using the training data and evaluated on the test set using the following metrics:


	•	Accuracy: The proportion of correct predictions.
	•	Precision: The proportion of true positive cases out of all predicted positive cases.
	•	Recall: The proportion of actual positive cases that were correctly predicted.
	•	F1-Score: The harmonic mean of precision and recall.
<img width="1336" alt="Screenshot 2024-09-13 at 11 32 13 AM" src="https://github.com/user-attachments/assets/f0fdf470-1ad6-4507-93a0-f1c6f2cec5bc">

5. Cross-Validation

To ensure robust performance, we used 5-fold cross-validation to evaluate the model across different subsets of the data, reducing overfitting and improving generalization.

<h2>Novelty Factor</h2>

This project showcases the importance of using feature engineering, hyperparameter tuning, and cross-validation to build robust models in the context of medical data. Furthermore, it demonstrates the application of interpretability techniques like feature importance analysis and explores the deployment potential of such a model in real-world medical decision support systems. The novelty also lies in combining multiple machine learning techniques and comparing their performance on the breast cancer dataset.

<h2>Results</h2>

	•	The best model achieved an F1_score of 0.9459459459459458 on the test set.
	•	Random Forest was the top-performing model, with the highest accuracy after hyperparameter tuning.
 <img width="1336" alt="Screenshot 2024-09-13 at 11 39 48 AM" src="https://github.com/user-attachments/assets/2b6db5c0-5913-450c-aac4-0770af4de485">
 <img width="593" alt="Screenshot 2024-09-13 at 11 50 22 AM" src="https://github.com/user-attachments/assets/2a8619c3-6e6a-4435-8d9a-0db58561573f">


Conclusion

The Random Forest Classifier provided the best performance in predicting whether a breast tumor is benign or malignant. With further optimization, this model can serve as an effective tool for early detection of breast cancer.
