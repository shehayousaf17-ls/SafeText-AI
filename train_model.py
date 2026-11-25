import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# 1. Find the dataset automatically
dataset_files = list(
    Path("data").rglob("cyberbullying_tweets.csv")
)

if not dataset_files:
    raise FileNotFoundError(
        "Dataset not found! Check that "
        "cyberbullying_tweets.csv is inside the data folder."
    )

DATA_PATH = dataset_files[0]

# 2. Load the Kaggle dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset location:", DATA_PATH)
print("Original dataset shape:", df.shape)
print("Available columns:", df.columns.tolist())

# 3. Use the actual Kaggle column names
text_column = "tweet_text"
label_column = "cyberbullying_type"

if text_column not in df.columns:
    raise ValueError(f"Missing column: {text_column}")

if label_column not in df.columns:
    raise ValueError(f"Missing column: {label_column}")

# 4. Remove missing and duplicate text entries
df = df.dropna(subset=[text_column, label_column])
df[text_column] = df[text_column].astype(str).str.strip()
df[label_column] = df[label_column].astype(str).str.strip()

df = df[
    (df[text_column] != "") &
    (df[label_column] != "")
]

df = df.drop_duplicates(subset=[text_column])

print("\nCleaned dataset shape:", df.shape)

# 5. Display the number of samples in each category
print("\nCyberbullying category counts:")
print(df[label_column].value_counts())

# 6. Separate text and target labels
X = df[text_column]
y = df[label_column]

# 7. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 8. Create the TF-IDF and Logistic Regression pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            stop_words="english"
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    )
])

# 9. Train the model
print("\nTraining the model. Please wait...")
model.fit(X_train, y_train)

print("Training completed successfully!")

# 10. Evaluate the model on unseen test data
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy: {:.2f}%".format(accuracy * 100))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

# 11. Save the trained model
models_folder = Path("models")
models_folder.mkdir(exist_ok=True)

model_path = models_folder / "cyberbullying_model.joblib"
joblib.dump(model, model_path)

print("\nModel saved successfully!")
print("Saved location:", model_path.resolve())
print("\nYou can now use this model in your Streamlit website.")
