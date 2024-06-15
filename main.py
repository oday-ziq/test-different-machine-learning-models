import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


TEST_SIZE = 0.3
K=3

def evaluate(labels, predictions):
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    cm = confusion_matrix(labels, predictions)
    print("Confusion Matrix:")
    print(cm)


    return accuracy, precision, recall, f1



# Load the dataset
file_path = r'C:\Users\user\Downloads\heart.csv'
heart_data = pd.read_csv(file_path)

# Replacing the noisy data from the RestingBP feature with the median
min_valid_bp = 60
max_valid_bp = 250
median_resting_bp = heart_data['RestingBP'].median()
heart_data['RestingBP'] = heart_data['RestingBP'].apply(lambda x: median_resting_bp if x < min_valid_bp or x > max_valid_bp else x)

# Replacing the noisy data from the Cholesterol feature with the median
min_validCol = 50
max_validCol = 500
medianCol = heart_data['Cholesterol'].median()
heart_data['Cholesterol'] = heart_data['Cholesterol'].apply(lambda x: medianCol if x < min_validCol or x > max_validCol else x)


# Checking for missing values in the dataset
missing_values = heart_data.isnull().sum()

print("*************************************************************************************")
# Display the count of missing values for each feature
print("The number of missing values for each feature is : ")
print(missing_values)
print("*************************************************************************************")
print(f"Number of examples : \n{heart_data.shape[0]}")
print(f"Number of features : \n {heart_data.shape[1]}")
print("*************************************************************************************")

print("*************************************************************************************")
print("Possible values for the feature ChestPainType")
unique_chest_pain_types = heart_data['ChestPainType'].unique()
print(unique_chest_pain_types)
print("*************************************************************************************")

print("*************************************************************************************")
print("Possible values for the feature Sex")
uniqueSex = heart_data['Sex'].unique()
print(uniqueSex)
print("*************************************************************************************")

print("*************************************************************************************")
print("Possible values for the feature RestingECG")
uniqueRestingECG = heart_data['RestingECG'].unique()
print(uniqueRestingECG)
print("*************************************************************************************")

print("*************************************************************************************")
minRBP = heart_data['RestingBP'].min()
maxRBP = heart_data['RestingBP'].max()
print(f"Range of RestingBP: {minRBP} to {maxRBP}")
print("*************************************************************************************")

print("*************************************************************************************")
minCol = heart_data['Cholesterol'].min()
maxCol = heart_data['Cholesterol'].max()
print(f"Range of Cholesterol: {minCol} to {maxCol}")
print("*************************************************************************************")

print("*************************************************************************************")
min_age = heart_data['Age'].min()
max_age = heart_data['Age'].max()
print(f"Range of Age: {min_age} to {max_age}")
print("*************************************************************************************")

print("*************************************************************************************")
print("Possible values for the feature FastingBS")
uniqueFastingBS = heart_data['FastingBS'].unique()
print(uniqueFastingBS)
print("*************************************************************************************")

print("*************************************************************************************")
min_HR = heart_data['MaxHR'].min()
max_HR = heart_data['MaxHR'].max()
print(f"Range of MaxHR: {min_HR} to {max_HR}")
print("*************************************************************************************")

print("*************************************************************************************")
print("Possible values for the feature ExerciseAngina")
uniqueEA = heart_data['ExerciseAngina'].unique()
print(uniqueEA)
print("*************************************************************************************")

print("*************************************************************************************")
minOP = heart_data['Oldpeak'].min()
maxOP = heart_data['Oldpeak'].max()
print(f"Range of Oldpeak: {minOP} to {maxOP}")
print("*************************************************************************************")

print("*************************************************************************************")
print("Possible values for the feature ST_Slope")
uniqueST = heart_data['ST_Slope'].unique()
print(uniqueST)
print("*************************************************************************************")

print()
# Descriptive statistics of the dataset
print("Here are descriptive statistics of the dataset\n")
selected_columns = ['Age', 'RestingBP', 'Cholesterol','FastingBS','MaxHR','Oldpeak']
subset_heart_data = heart_data[selected_columns]
descriptive_stats = subset_heart_data.describe()
print(descriptive_stats)
print()


# Histograms for numerical features
heart_data.hist(bins=15, figsize=(15, 10), layout=(3, 4))


# Box plots for numerical features
plt.figure(figsize=(15, 10))
sns.boxplot(data=heart_data.drop(['Sex', 'RestingECG', 'ExerciseAngina', 'FastingBS', 'HeartDisease','Oldpeak'], axis=1))
plt.title('Box Plots for the numerical Features')
plt.show()

# Pie chart for 'Sex'
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
heart_data['Sex'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Sex Distribution')
plt.ylabel('')  # Hide the y-label

# Pie chart for 'RestingECG'
plt.subplot(1, 3, 2)
heart_data['RestingECG'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('RestingECG Distribution')
plt.ylabel('')

# Pie chart for 'ExerciseAngina'
plt.subplot(1, 3, 3)
heart_data['ExerciseAngina'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('ExerciseAngina Distribution')
plt.ylabel('')

plt.tight_layout()
plt.show()




# Preprocessing the data: Encoding categorical variables and scaling features
label_encoders = {}
for column in ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']:
    le = LabelEncoder()
    heart_data[column] = le.fit_transform(heart_data[column])

X = heart_data.drop('HeartDisease', axis=1)
y = heart_data['HeartDisease']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


############################################### kNN Classifier #####################################

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Baseline Model using Nearest Neighbor (kNN)

print("*************************************************************************************")
print("                                 kNN                                      ")
# For k=1
print("Evaluation Metrics For k=1")
knn_1 = KNeighborsClassifier(n_neighbors=1)
knn_1.fit(X_train, y_train)
accuracy1, precision, recall, f1 = evaluate(y_test, knn_1.predict(X_test))

# Evaluate and print training accuracy for k=1
train_accuracy1 = accuracy_score(y_train, knn_1.predict(X_train))
print("Training Accuracy (k=1): ", train_accuracy1)

# Evaluate and print testing accuracy for k=1
test_accuracy1 = accuracy_score(y_test, knn_1.predict(X_test))
print("Testing Accuracy (k=1): ", test_accuracy1)
print("Precision (k=1): ", precision)
print("Recall (k=1): ", recall)
print("F1 Score (k=1): ", f1)
print("*************************************************************************************")


print("*************************************************************************************")
# For k=3
print("                                 kNN                                      ")
print("Evaluation Metrics For k=3")
knn_3 = KNeighborsClassifier(n_neighbors=3)
knn_3.fit(X_train, y_train)
accuracy2, precision2, recall2, f1_2 = evaluate(y_test, knn_3.predict(X_test))

# Evaluate and print training accuracy for k=3
train_accuracy3 = accuracy_score(y_train, knn_3.predict(X_train))
print("Training Accuracy (k=3): ", train_accuracy3)

# Evaluate and print testing accuracy for k=3
test_accuracy3 = accuracy_score(y_test, knn_3.predict(X_test))
print("Testing Accuracy (k=3): ", test_accuracy3)
print("Precision (k=3): ", precision2)
print("Recall (k=3): ", recall2)
print("F1 Score (k=3): ", f1_2)
print("*************************************************************************************")


##################################################### SVM #######################################################


# Preprocessing the data
label_encoders = {}
for column in ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']:
    le = LabelEncoder()
    heart_data[column] = le.fit_transform(heart_data[column])

X = heart_data.drop('HeartDisease', axis=1)
y = heart_data['HeartDisease']

# Scaling the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Here, data is split into 60% training and 20% validation and 20% testing
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.4, random_state=42)
X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define a range of C values to explore
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

# Create an SVM model with a linear kernel
svm_model = SVC(kernel='linear')

# Create a GridSearchCV object with the SVM model and parameter grid
grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy')


print("Choosing the hyper-parameter C using the validation set")
# Fit the GridSearchCV to validation data
grid_search.fit(X_validation, y_validation)

# Get the best C value from the grid search
best_C = grid_search.best_params_['C']
print("Best C value:", best_C)

# Create and train a new SVM model with the best C value on the combined training and validation data
best_svm_model = SVC(kernel='linear', C=best_C)
best_svm_model.fit(X_train, y_train)

# Predict on the test set using the best model
y_pred_best_svm = best_svm_model.predict(X_test)

# Calculate testing accuracy, precision, recall, and F1-score for the best model
accuracy_best_svm = accuracy_score(y_test, y_pred_best_svm)
precision_best_svm = precision_score(y_test, y_pred_best_svm)
recall_best_svm = recall_score(y_test, y_pred_best_svm)
f1_best_svm = f1_score(y_test, y_pred_best_svm)

# Predict on the validation set using the best model
y_validation_pred_best_svm = best_svm_model.predict(X_validation)

# Calculate validation accuracy, precision, recall, and F1-score for the best model
accuracy_validation_best_svm = accuracy_score(y_validation, y_validation_pred_best_svm)


# Predict on the training set using the best model
y_train_pred_best_svm = best_svm_model.predict(X_train)
# Calculate training accuracy for the best model
accuracy_train_best_svm = accuracy_score(y_train, y_train_pred_best_svm)
# Print the results for both validation and testing sets
print("*************************************************************************************")
print("                                 Soft SVM                                       ")
print("Validation Accuracy with Best C :", accuracy_validation_best_svm)
print("Training Accuracy :", accuracy_train_best_svm)
print("Testing Accuracy with Best C :", accuracy_best_svm)
print("Precision with Best C :", precision_best_svm)
print("Recall with Best C :", recall_best_svm)
print("Testing F1-Score with Best C :", f1_best_svm)
print("The Confusion Matrix For the SVM : ")
evaluate(y_test, y_pred_best_svm)
print("*************************************************************************************")


################################################### Random Forest ############################################

# Preprocessing the data
label_encoders = {}
for column in ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']:
    le = LabelEncoder()
    heart_data[column] = le.fit_transform(heart_data[column])

X = heart_data.drop('HeartDisease', axis=1)
y = heart_data['HeartDisease']

# Scaling the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Here, data is split into 60% training and 20% validation and 20% testing
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.4, random_state=42)
X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define hyperparameter values to search
param_grid = {
    'n_estimators': [10, 50, 100, 200],  # Number of trees in the forest
}

# Create a RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)

# Create a GridSearchCV object with the RandomForestClassifier and parameter grid
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy')

# Fit the GridSearchCV to your validation data
grid_search.fit(X_validation, y_validation)

# Get the best hyperparameters from the grid search
best_params = grid_search.best_params_

# Train a new RandomForestClassifier using the best hyperparameters on the combined training and validation data
best_rf_model = RandomForestClassifier(random_state=42, **best_params)
print("*************************************************************************************")
print("                                 Random Forest                                       ")
best_rf_model.fit(X_train, y_train)

# Predict on the test set using the best model
y_pred_best_rf = best_rf_model.predict(X_test)

# Calculate testing accuracy, precision, recall, and F1-score for the best model
accuracy_best_rf = accuracy_score(y_test, y_pred_best_rf)
precision_best_rf = precision_score(y_test, y_pred_best_rf)
recall_best_rf = recall_score(y_test, y_pred_best_rf)
f1_best_rf = f1_score(y_test, y_pred_best_rf)

# Predict on the training set using the best model
y_train_pred_best_rf = best_rf_model.predict(X_train)

# Calculate training accuracy for the best model
accuracy_train_best_rf = accuracy_score(y_train, y_train_pred_best_rf)

# Print the results for both validation and testing sets
print("Best Hyperparameters:", best_params)
print("Training Accuracy with Best Model:", accuracy_train_best_rf)
print("Testing Accuracy with Best Model:", accuracy_best_rf)
print("Precision with Best Model:", precision_best_rf)
print("Recall with Best Model:", recall_best_rf)
print("Testing F1-Score with Best Model:", f1_best_rf)
print("The Confusion Matrix For the Random Forest : ")
evaluate(y_test, y_pred_best_rf)
print("*************************************************************************************")



print("Indices of the Misclassified examples by the random forest classifier : ")
missclassified = np.where(y_test != y_pred_best_rf)[0]
print(missclassified)
