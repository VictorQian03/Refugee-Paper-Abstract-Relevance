import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Read the dataset into a DataFrame
df = pd.read_excel('RefugeeTraining.xlsx')

# Replace missing values with empty strings
df['Abstract'].fillna('', inplace=True)

# Combine the sets
X = df['Abstract']
y = df['Relevance']

# Oversample the minority class using RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X.values.reshape(-1, 1), y)

# Apply text vectorization using CountVectorizer
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X_resampled.ravel())

# Create a RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform ten-fold cross-validation
k = 10  
scores = cross_val_score(rf_model, X_vectorized, y_resampled, cv=k, scoring='accuracy')

print(f'Cross-validation scores: {scores}')
print(f'Mean accuracy: {scores.mean()}')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y_resampled, test_size=0.2, random_state=42)

# Fit the model on the training data
rf_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_model.predict(X_test)

# Evaluate performance using classification report
print(classification_report(y_test, y_pred))
