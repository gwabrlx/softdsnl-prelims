import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('dataset.csv')

# Split features and target
X = df[['feature1', 'feature2', 'feature3']]  # Your feature columns
y = df['target']  # Your target column

# Label Encoding if target is categorical
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)  # RandomForest with 100 trees
model.fit(X_train, y_train)

# Save the trained model and label encoder
joblib.dump(model, 'model.pkl')
joblib.dump(encoder, 'label_encoder.pkl')

# Optional: Save visualizations

# 1. Correlation Heatmap (between features)
plt.figure(figsize=(8, 6))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.savefig('correlation_heatmap.png')
plt.close()

# 2. Pairplot (to visualize relationships between features and target)
sns.pairplot(df, hue="target", palette="Set2")
plt.savefig('pairplot.png')
plt.close()

# 3. Feature Importance (from the RandomForest model)
feature_importance = model.feature_importances_
features = X.columns

# Plot the feature importance
plt.figure(figsize=(8, 6))
sns.barplot(x=features, y=feature_importance, palette='Blues_d')
plt.title("Feature Importance")
plt.ylabel('Importance')
plt.savefig('feature_importance.png')
plt.close()

print("Model training complete and visualizations saved!")
