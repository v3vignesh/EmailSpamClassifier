import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix


# Page settings
st.set_page_config(
    page_title="Spam Message Classifier",
    page_icon="📩",
    layout="centered"
)


# Train the Machine Learning model
@st.cache_resource
def train_model():
    data = pd.read_csv("spam.csv", encoding="latin-1")

    data = data[["v1", "v2"]]
    data.columns = ["label", "message"]
    data = data.drop_duplicates()

    X = data["message"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer()

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LinearSVC()
    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(
        y_test, predictions
    ) * 100

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=["ham", "spam"]
    )

    return model, vectorizer, accuracy, cm, len(data)


model, vectorizer, accuracy, cm, total_messages = train_model()


# Website title
st.title("📩 Spam Message Classifier")

st.write(
    "Enter a message below to check whether it is Spam or Ham."
)

st.metric("Model Accuracy", f"{accuracy:.2f}%")

st.caption(
    f"Dataset messages after cleaning: {total_messages}"
)


# Message input box
message = st.text_area(
    "Enter your message",
    placeholder="Example: Congratulations! You won a free prize...",
    height=130
)


# Check Message button
if st.button("🔍 Check Message", use_container_width=True):

    if not message.strip():
        st.warning("Please enter a message first.")

    else:
        message_tfidf = vectorizer.transform([message])

        prediction = model.predict(message_tfidf)[0]

        if prediction == "spam":
            st.error("🚨 SPAM MESSAGE")
            st.write(
                "The model classified this as a spam message."
            )

        else:
            st.success("✅ HAM — NORMAL MESSAGE")
            st.write(
                "The model classified this as a normal message."
            )


# Model evaluation
st.divider()

st.subheader("📊 Model Evaluation")
st.write("Confusion Matrix")

fig, ax = plt.subplots(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["HAM", "SPAM"],
    yticklabels=["HAM", "SPAM"],
    ax=ax
)

ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")
ax.set_title("Spam Classifier Confusion Matrix")

st.pyplot(fig)
plt.close(fig)


st.caption(
    "Note: This project uses an SMS spam dataset. "
    "Predictions may not always be correct."
)