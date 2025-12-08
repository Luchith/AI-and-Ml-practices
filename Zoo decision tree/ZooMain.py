# =========================
# Zoo Decision Tree Project
# =========================

# Import libraries
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Load the Dataset
# =========================

zoo = pd.read_csv("zoo.csv")

print("First 5 rows of data:")
print(zoo.head().to_string())

print("\nData types:")
print(zoo.dtypes)

print("\nMissing values:")
print(zoo.isnull().sum())

# Drop non-useful column
zoo = zoo.drop(columns=['animal_name'])

# =========================
# Visualise the Data
# =========================

# Bar chart of class distribution
fig = zoo.class_type.value_counts().sort_index().plot(kind='bar')
fig.set_xlabel('Animal Category')
fig.set_ylabel('Animal Count')
plt.title("Class Distribution")
plt.show()

# Correlation heatmap
corr_matrix = zoo.corr()
plt.figure(figsize=(9, 8))
sns.heatmap(data=corr_matrix, cmap='BrBG', annot=True, linewidths=0.2)
plt.title("Correlation Heatmap")
plt.show()

# =========================
# Define X and y
# =========================

features = [
    'hair','feathers','eggs','milk','airborne','aquatic','predator',
    'toothed','backbone','breathes','venomous','fins','legs','tail','domestic'
]

X = zoo[features].values
y = zoo['class_type']

# =========================
# Train / Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# =========================
# Train the Model
# =========================

dtc = DecisionTreeClassifier(criterion='entropy')
dtree = dtc.fit(X_train, y_train)

# =========================
# Make Predictions
# =========================

y_pred = dtree.predict(X_test)

# =========================
# Evaluate the Model
# =========================

print("\nAccuracy:", metrics.accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=dtc.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dtc.classes_)
disp.plot()
plt.show()

# Kappa Statistic
print("Cohen Kappa:", metrics.cohen_kappa_score(y_test, y_pred))

# K-Fold Cross Validation
print("Validation Mean Accuracy:",
      cross_val_score(dtree, X, y, cv=3, scoring='accuracy').mean())

# =========================
# Display the Decision Tree
# =========================

plt.figure(figsize=(20, 10))
tree.plot_tree(dtree, feature_names=features)
plt.show()

