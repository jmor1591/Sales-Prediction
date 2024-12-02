# Import necessary libraries
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the supermarket_sales.csv file into a DataFrame called df
df = pd.read_csv('supermarket_sales.csv')

# Get a list of all column names in the DataFrame
list_1 = list(df.columns)

# Initialize an empty list to store the names of categorical features
list_cate = []

# Iterate through the column names to identify categorical features
for i in list_1:
    # Check if the column's data type is 'object' (indicating categorical data)
    if df[i].dtype == 'object':
        # Add the categorical column name to the list
        list_cate.append(i)

# Create an instance of LabelEncoder
le = LabelEncoder()

# Iterate through the list of categorical features to encode them
for i in list_cate:
    # Transform the categorical column into numerical values
    df[i] = le.fit_transform(df[i])

# After encoding, print the mapping for 'Gender' specifically
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Label Mapping (Gender):", label_mapping)

# Define the target variable 'y' as the 'Gender' column from the DataFrame
y = df['Gender']

# Define the feature set 'x' by dropping the 'Gender' column from the DataFrame
x = df.drop(columns=['Invoice ID', 'Gender', 'Branch'])
# print(df.head())

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    random_state=123,
    test_size=0.1
)

# Function to plot confusion matrix


def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Male', 'Predicted Female'],
                yticklabels=['Actual Male', 'Actual Female'])
    plt.title(title)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()


# XGBClassifier
xgb = XGBClassifier(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1,
                    max_depth=5, alpha=10, n_estimators=10)

xgb.fit(x_train, y_train)

y_pred = xgb.predict(x_test)

print("XGB Classification Report is:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
plot_confusion_matrix(cm, title='XGBoost Confusion Matrix')

# Calculate accuracy using TP and TN
TP = cm[1, 1]  # True Positives
TN = cm[0, 0]  # True Negatives
FP = cm[0, 1]  # False Positives
FN = cm[1, 0]  # False Negatives

accuracy = (TP + TN) / (TP + TN + FP + FN) * 100
print("Custom Accuracy Score (using TP and TN):\n", accuracy)

# K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=7)

# Fit the KNN model on the training data
knn.fit(x_train, y_train)

# Use the trained model to make predictions on the test data
y_pred = knn.predict(x_test)

# Print the classification report to evaluate model performance
print("K-Nearest Neighbor (7) Classification Report is:\n",
      classification_report(y_test, y_pred))

# Print the confusion matrix to visualize the performance of the model
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
plot_confusion_matrix(cm, title='KNN Confusion Matrix')

# Calculate accuracy using TP and TN
TP = cm[1, 1]  # True Positives
TN = cm[0, 0]  # True Negatives
FP = cm[0, 1]  # False Positives
FN = cm[1, 0]  # False Negatives

accuracy = (TP + TN) / (TP + TN + FP + FN) * 100
print("Custom Accuracy Score (using TP and TN): \n", accuracy)

# Gaussian Naive Bayes Classifiers
nb = GaussianNB()
nb.fit(x_train, y_train)

y_pred = nb.predict(x_test)
print("Naive Bayes Classification Report is:\n",
      classification_report(y_test, y_pred))

# Print the confusion matrix for Naive Bayes
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
plot_confusion_matrix(cm, title='Naive Bayes Confusion Matrix')

# Calculate accuracy using TP and TN
TP = cm[1, 1]  # True Positives
TN = cm[0, 0]  # True Negatives
FP = cm[0, 1]  # False Positives
FN = cm[1, 0]  # False Negatives

accuracy = (TP + TN) / (TP + TN + FP + FN) * 100
print("Custom Accuracy Score (using TP and TN):\n", accuracy)
