# **Udacity Machine Learning Engineer Nanodegree Program**
# **Capstone Project**

## Introduction
This is the final project of the Udacity Machine Learning Engineer Nanodegree Program.

The project is derived from cancer classification and prediction by gene expression at the molecular level.

The original dataset of this project comes from a research study by Professor Golub. It described a generic method for automatically determining the type of cancer between acute lymphocytic leukemia (ALL) and acute myeloid leukemia (AML). The published paper showed the possibility of cancer classification based only on the gene expressions without relevant biological knowledge.

The goal of the project is to find a good classification method to classify leukemia patients into acute lymphocytic leukemia (ALL) and acute myeloid leukemia (AML).

## Machine Learning Workflow

* Data Loading and Exploration
  * Data Loading
  * Exploratory Data Analysis
  * Data Cleaning and Pre-processing
* Feature Engineering and Data Transformation
  * Comparison of StandardScaler and MinMaxScaler
  * Normalization
  * Dimensionality reduction
* Training Model
  * Define and Create Estimator:
    * Benchmark Model
    * K-Means Clustering Model
    * Naive Bayes Model
    * K-Nearest Neighbors Model
    * Logistic Regression Model
    * Support Vector Machine Model
    * Gradient Boosting Model
  * Model tuning
  * Deploy the trained model
  * Evaluate the Performance
* Clean up Resources


## Documents in this project
* data folder - original dataset downloaded from [Kaggle](https://www.kaggle.com/crawford/gene-expression/download)
* gene_data folder - cleaned and pre-processed data frames for model training
* Cancer_Type_Classifier.ipynb - analysis, code and evaluation result
* Capstone Project.pdf - Project report
* Proposal.pdf - Project proposal


## Environment and Libraries
This project is developed in Python 3.6
The libraries used in this project are shown below.

* pandas
* numpy
* matplotlib
* seaborn
* mpl_toolkits.mplot3d
* scikit-learn
  * StandardScaler from sklearn.preprocessing
  * MinMaxScaler from sklearn.preprocessing
  * PCA from sklearn.decomposition
  * roc_auc_score, accuracy_score, confusion_matrix from sklearn.metrics
  * GridSearchCV from sklearn.model_selection
  * KMeans from sklearn.cluster
  * GaussianNB from sklearn.naive_bayes
  * KNeighborsClassifier from sklearn.neighbors
  * LogisticRegression from sklearn.linear_model
  * SVC from sklearn.svm
  * GradientBoostingClassifier from sklearn.ensemble
