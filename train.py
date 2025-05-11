from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# small sample dataset
texts = [
    "I loved this movie, it was amazing!",
    "Worst film I have ever seen.",
    "Absolutely fantastic acting and direction.",
    "Terrible plot and bad acting.",
    "It was okay, not great but not bad."
]
labels = [1, 0, 1, 0, 1]  # 1: Positive, 0: Negative

# Training vectorizer along with model
vec = CountVectorizer()
X = vec.fit_transform(texts)
model = LogisticRegression()
model.fit(X, labels)

os.makedirs("model", exist_ok=True)
joblib.dump(vec, "model/vectorizer.joblib")
joblib.dump(model, "model/model.joblib")
print("Model and vectorizer saved.")
