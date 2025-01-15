import re
import pandas as pandas
import numpy as numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# 1. Data pre-processing

# Dataset from
data_path = "dataset/JobLevelData.xlsx"
data_frame = pandas.read_excel(data_path, sheet_name="in")

# Label merge
def preprocessing_labels(line):
    labels = [line["Column 1"], line["Column 2"], line["Column 3"], line["Column 4"]]
    result = []
    for label in labels:
      if pandas.notnull(label):
        result.append(label)
    return result

data_frame["Labels"] = data_frame.apply(preprocessing_labels, axis=1)

# Text cleanup
def text_cleaner(text):
    text = text.lower() # Register
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text) # Wildcards
    words = text.split() # Tokenization

    stop_words = {"at", "and", "to", "of", "in", "the", "a", "an", "for", "on", "with", "by"} # Noise
    filtered_words = []
    for word in words:
        if word not in stop_words:
            filtered_words.append(word)
    cleaned_text = " ".join(filtered_words)

    return cleaned_text

data_frame["Title"] = data_frame["Title"].apply(text_cleaner)


# 2. TF-IDF vectorization
vector = TfidfVectorizer(max_features=5000)
X_titles = vector.fit_transform(data_frame["Title"]) # Title matrix

# Labels in multi-label
all_labels = []
for labels in data_frame["Labels"]:
    for label in labels:
        all_labels.append(label)

unique_labels = list(set(all_labels))  # Remove duplicates

# Converting text labels to numbers
label_binarizer = {}
for i, label in enumerate(unique_labels):
    label_binarizer[label] = i

def encode_labels(labels):
    encoded = numpy.zeros(len(unique_labels))

    for label in labels:
        encoded[label_binarizer[label]] = 1

    return encoded

Y_labels = numpy.array(data_frame["Labels"].apply(encode_labels).tolist()) # Labels matrix


# 3. 80/20 data split
X_train, X_test, Y_train, Y_test = train_test_split(X_titles, Y_labels, test_size=0.20, random_state=1)

# 4. Training
model = MultiOutputClassifier(LogisticRegression(max_iter=100, random_state=1))
model.fit(X_train, Y_train)

# 5. Result
Y_predct = model.predict(X_test)

results = classification_report(Y_test, Y_predct, target_names=unique_labels, zero_division=0)
accuracy = accuracy_score(Y_test, Y_predct)

# Print metrics
print(f"Classification Report:\n{results}")
print(f"Total model accuracy:\n{accuracy}")