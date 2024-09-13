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

4. Model Training and Evaluation

The models were trained using the training data and evaluated on the test set using the following metrics:

	•	Accuracy: The proportion of correct predictions.
	•	Precision: The proportion of true positive cases out of all predicted positive cases.
	•	Recall: The proportion of actual positive cases that were correctly predicted.
	•	F1-Score: The harmonic mean of precision and recall.

5. Cross-Validation

To ensure robust performance, we used 5-fold cross-validation to evaluate the model across different subsets of the data, reducing overfitting and improving generalization.

<h2>Results</h2>

	•	The best model achieved an accuracy of X% on the test set.
	•	Random Forest was the top-performing model, with the highest accuracy after hyperparameter tuning.
	•	Visualizations such as confusion matrices and ROC curves were used to evaluate model performance and trade-offs between precision and recall.

Conclusion

The Random Forest Classifier provided the best performance in predicting whether a breast tumor is benign or malignant. With further optimization, this model can serve as an effective tool for early detection of breast cancer.
