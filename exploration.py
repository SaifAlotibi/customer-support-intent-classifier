import pandas as pd
import matplotlib.pyplot as plt
import joblib 

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, ConfusionMatrixDisplay, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline


# Load the dataset
df = pd.read_csv("customer_support.csv")

# Remove duplicate instructions
df = df.drop_duplicates(
    subset="instruction"
).reset_index(drop=True)

print("\nShape after removing duplicates:")
print(df.shape)

print("\nRemaining duplicated messages:")
print(df["instruction"].duplicated().sum())


# Features and target
X = df["instruction"]
y = df["intent"]


# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# TF-IDF
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF testing shape:", X_test_tfidf.shape)


# Logistic Regression
lg_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lg_model.fit(
    X_train_tfidf,
    y_train
)

lg_predictions = lg_model.predict(
    X_test_tfidf
)


# Linear SVM
svc_model = LinearSVC(
    random_state=42
)

svc_model.fit(
    X_train_tfidf,
    y_train
)

svc_predictions = svc_model.predict(
    X_test_tfidf
)


# Compare models
lg_accuracy = accuracy_score(
    y_test,
    lg_predictions
)

svc_accuracy = accuracy_score(
    y_test,
    svc_predictions
)

lg_f1 = f1_score(
    y_test,
    lg_predictions,
    average="macro"
)

svc_f1 = f1_score(
    y_test,
    svc_predictions,
    average="macro"
)


print("\nLogistic Regression")
print("Accuracy:", lg_accuracy)
print("Macro F1:", lg_f1)


print("\nLinear SVM")
print("Accuracy:", svc_accuracy)
print("Macro F1:", svc_f1)


# Confusion Matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    svc_predictions,
    xticks_rotation="vertical"
)

plt.title("Linear SVM Confusion Matrix")
plt.tight_layout()
plt.show()


# Final Pipeline
final_pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2)
        )
    ),
    (
        "model",
        LinearSVC(
            random_state=42
        )
    )
])


# Train the complete pipeline
final_pipeline.fit(
    X_train,
    y_train
)


# Predictions using raw text
final_predictions = final_pipeline.predict(
    X_test
)


# Final evaluation
final_f1 = f1_score(
    y_test,
    final_predictions,
    average="macro"
)

final_accuracy = accuracy_score(
    y_test,
    final_predictions
)


print("\nFinal Pipeline")
print("Macro F1:", final_f1)
print("Accuracy:", final_accuracy)

# Test on completely new messages

my_messages = [
    "The delivery address needs to be changed",
    "I can't remember how to access my account",
    "The money was taken but my payment isn't showing",
    "Can you tell me when my order is coming?",
    "I don't want this purchase anymore"
]

my_predictions = final_pipeline.predict(my_messages)

for message, prediction in zip(my_messages, my_predictions):
    print(f"\n{message}")
    print(f"→ {prediction}")

# find misclassified massage

result = pd.DataFrame({
    "massage": X_test.values, 
    "actual": y_test.values, 
    "predicted": final_predictions
})

error = result[result["actual"] != result["predicted"]]

print("\n Number of misclasssified massages", len(error))

print("\nMisclassified examples")
print(error.to_string(index=False))

error_count = error.groupby(
    ["actual", "predicted"]
    ).size().sort_values(ascending=False)

print("\n Most common errors: ")
print(error_count.head(15))

#Save:
joblib.dump(
    final_pipeline, 
    "intent_calssifier.pkl")

print("\n Model Saved Succsessfully!")
