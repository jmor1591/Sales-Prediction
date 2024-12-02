# Import necessary libraries
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
import pandas as pd

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
# x_train and y_train are the training data, while x_test and y_test are the testing data
# random_state=0 ensures reproducibility of the split
# test_size=0.2 indicates that 20% of the data will be used for testing
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    random_state=123,
    test_size=0.2
)

# Start the first model: K-Nearest Neighbors (KNN)
# Create an instance of KNeighborsClassifier with 7 neighbors
knn = KNeighborsClassifier(n_neighbors=7)

# Fit the KNN model on the training data
knn.fit(x_train, y_train)

# Use the trained model to make predictions on the test data
y_pred = knn.predict(x_test)

# Print the classification report to evaluate model performance
print("K-Nearest Neighbor (7) Classification Report is:\n",
      classification_report(y_test, y_pred))

# Print the confusion matrix to visualize the performance of the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Print the training accuracy score to see how well the model performed on the training data
print("Training Accuracy Score:\n", knn.score(x_train, y_train)*100)

# Calculate permutation importance
result = permutation_importance(
    knn, x_test, y_test, n_repeats=30, random_state=0)

# Create a DataFrame for visualization
perm_importance_df = pd.DataFrame({
    'Feature': x.columns,
    'Importance': result.importances_mean
})

# Sort the DataFrame by importance
perm_importance_df = perm_importance_df.sort_values(
    by='Importance', ascending=False)

# Plotting permutation importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=perm_importance_df)
plt.title('Permutation Feature Importance for KNN')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

# Gaussian Naive Bayes Classifiers
nb = GaussianNB()
nb.fit(x_train, y_train)

y_pred = nb.predict(x_test)
print("Naive Bayes Classification Report is:\n",
      classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Training Accuracy Score:\n", nb.score(x_train, y_train)*100)

# Decision Tree Classifier

# Define the parameter grid for max_depth
param_grid = {'max_depth': range(1, 21)}

# Create an instance of DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(criterion='entropy')

# Set up GridSearchCV with cross-validation
# Set return_train_score=True to get training scores
grid_search = GridSearchCV(estimator=decision_tree, param_grid=param_grid,
                           scoring='accuracy', cv=5, n_jobs=-1, return_train_score=True)

# Fit the model to the training data
grid_search.fit(x_train, y_train)

# Get the best parameters and the best score
best_depth = grid_search.best_params_['max_depth']
best_score = grid_search.best_score_

print(f"The optimal depth found using GridSearchCV is: {best_depth}")
print(f"The best cross-validated accuracy score is: {best_score * 100:.2f}%")

# Now you train the final model with the optimal depth
final_decision_tree = DecisionTreeClassifier(
    max_depth=best_depth, criterion='entropy')
final_decision_tree.fit(x_train, y_train)

# Evaluate the final model
y_pred = final_decision_tree.predict(x_test)

# Print the classification report
print("Decision Tree Classification Report is:\n",
      classification_report(y_test, y_pred))

# Get confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Extract TP, TN, FP, FN from confusion matrix
TP = cm[1, 1]  # True Positives
TN = cm[0, 0]  # True Negatives
FP = cm[0, 1]  # False Positives
FN = cm[1, 0]  # False Negatives

# Calculate accuracy using TP and TN
accuracy = (TP + TN) / (TP + TN + FP + FN) * 100
print("Custom Accuracy Score (using TP and TN):\n", accuracy)

# Print the decision tree structure in the terminal
tree_rules = export_text(final_decision_tree, feature_names=list(x.columns))
print("Decision Tree Structure:\n", tree_rules)

""" # Plotting the decision tree
plt.figure(figsize=(12, 8))
plot_tree(final_decision_tree, filled=True, feature_names=x.columns,
          class_names=['Class 0', 'Class 1'])  # Class 0 is female
plt.title('Decision Tree Visualization')
plt.show()

# Plotting the results to visualize the performance across different depths
plt.figure(figsize=(10, 6))
plt.plot(param_grid['max_depth'], grid_search.cv_results_[
         'mean_train_score'], label='Training Accuracy', marker='o')
plt.plot(param_grid['max_depth'], grid_search.cv_results_[
         'mean_test_score'], label='Testing Accuracy', marker='o')
plt.axvline(x=best_depth, color='r', linestyle='--',
            # Highlight the optimal depth
            label=f'Optimal Depth = {best_depth}')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree Depth vs Accuracy')
plt.legend()
plt.grid()
plt.show() """

# Support Vector Machine (SVM)

svc = SVC()
svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)
print("Support Vector Machine Classification Report is:\n",
      classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# This score never changes even if the random state of the train test split does
print("Training Accuracy Score:\n", svc.score(x_train, y_train)*100)


# Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)

y_pred = rfc.predict(x_test)

# Print classification report
print("Random Forest Classification Report is:\n",
      classification_report(y_test, y_pred))

# Get confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Extract TP, TN, FP, FN from confusion matrix
TP = cm[1, 1]  # True Positives
TN = cm[0, 0]  # True Negatives
FP = cm[0, 1]  # False Positives
FN = cm[1, 0]  # False Negatives

# Calculate accuracy using TP and TN
accuracy = (TP + TN) / (TP + TN + FP + FN)
print("Custom Accuracy Score:\n", accuracy * 100)

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[
            'Female', 'Male'], yticklabels=['Female', 'Male'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')
plt.show()

# Feature Importance
# Get feature importances from the Random Forest Classifier
importances = rfc.feature_importances_
feature_names = x_train.columns

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame(
    {'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(
    by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance from Random Forest Classifier')
plt.show()

# End of Random Forest Classifier


# AdaBoost Classifier with a specific base estimator
adb = AdaBoostClassifier()
adb.fit(x_train, y_train)  # Train the AdaBoost model on the training data

y_pred = adb.predict(x_test)  # Make predictions on the test data

# Print classification report for AdaBoost
print("AdaBoost Classification Report is:\n",
      classification_report(y_test, y_pred))

# Get confusion matrix for AdaBoost
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Extract True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN) from confusion matrix
TP = cm[1, 1]  # True Positives
TN = cm[0, 0]  # True Negatives
FP = cm[0, 1]  # False Positives
FN = cm[1, 0]  # False Negatives

# Calculate accuracy using TP and TN
accuracy = (TP + TN) / (TP + TN + FP + FN) * 100
print("Custom Accuracy Score (using TP and TN):\n", accuracy)

# Gradient Boosting Classifier
gbc = GradientBoostingClassifier()
# Train the Gradient Boosting model on the training data
gbc.fit(x_train, y_train)

y_pred = gbc.predict(x_test)  # Make predictions on the test data

# Print classification report for Gradient Boosting
print("Gradient Boost Classification Report is:\n",
      classification_report(y_test, y_pred))

# Get confusion matrix for Gradient Boosting
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Extract True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN) from confusion matrix
TP = cm[1, 1]  # True Positives
TN = cm[0, 0]  # True Negatives
FP = cm[0, 1]  # False Positives
FN = cm[1, 0]  # False Negatives

# Calculate accuracy using TP and TN
accuracy = (TP + TN) / (TP + TN + FP + FN) * 100
print("Custom Accuracy Score (using TP and TN):\n", accuracy)

# XGBClassifier

xgb = XGBClassifier(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1,
                    max_depth=5, alpha=10, n_estimators=10)

xgb.fit(x_train, y_train)

y_pred = xgb.predict(x_test)
print("XGB Classification Report is:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Training Accuracy Score:\n", xgb.score(x_train, y_train)*100)


# Extra Trees Classifier
etc = ExtraTreesClassifier(n_estimators=100, random_state=0)
etc.fit(x_train, y_train)

# Make predictions on the test data
y_pred = etc.predict(x_test)

# Print the classification report
print("Extra Trees Classification Report is:\n",
      classification_report(y_test, y_pred))

# Get confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Extract TP, TN, FP, FN from confusion matrix
TP = cm[1, 1]  # True Positives
TN = cm[0, 0]  # True Negatives
FP = cm[0, 1]  # False Positives
FN = cm[1, 0]  # False Negatives

# Calculate accuracy using TP and TN
accuracy = (TP + TN) / (TP + TN + FP + FN) * 100
print("Custom Accuracy Score (using TP and TN):\n", accuracy)

# Bagging Classifier
model = BaggingClassifier()
model.fit(x_train, y_train)

# Make predictions on the test data
y_pred = model.predict(x_test)

# Print the classification report
print("Bagging Classifier Classification Report is:\n",
      classification_report(y_test, y_pred))

# Get confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Calculate and print the accuracy score
accuracy = model.score(x_test, y_test) * 100
print("Training Accuracy Score:\n", accuracy)
