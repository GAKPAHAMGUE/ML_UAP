import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from wordcloud import WordCloud
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
import streamlit as st

# Streamlit UI for uploading and processing
st.title("Sentiment Analysis and Text Classification")
st.write("Upload your dataset and perform text processing, visualization, and modeling.")

uploaded_file = st.file_uploader("Upload CSV file", type="csv")
if uploaded_file:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(df.head())
    
    # Data Info
    st.write("### Dataset Information")
    st.write(df.info())
    
    # Check missing values
    st.write("### Missing Values")
    st.write(df.isnull().sum())
    
    # Distribution of Scores
    if 'score' in df.columns:
        st.write("### Distribution of Scores")
        fig, ax = plt.subplots()
        sns.countplot(x='score', data=df, palette="viridis", ax=ax)
        st.pyplot(fig)
    
    # Add text length column
    if 'content' in df.columns:
        df['text_length'] = df['content'].apply(lambda x: len(str(x)))
        st.write("### Text Length Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['text_length'], kde=True, color='purple', ax=ax)
        st.pyplot(fig)
    
    # Wordcloud
    st.write("### WordCloud")
    all_text = ' '.join(df['content'].dropna().astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
    
    # Preprocessing function
    st.write("### Preprocessing")
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in stop_words])
        stemmer = PorterStemmer()
        text = ' '.join([stemmer.stem(word) for word in text.split()])
        return text

    df['clean_content'] = df['content'].astype(str).apply(preprocess_text)
    st.write(df[['content', 'clean_content']].head())

    # Feature extraction with TF-IDF
    st.write("### Feature Extraction")
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_features = tfidf_vectorizer.fit_transform(df['clean_content'])
    st.write(f"TF-IDF Features Shape: {tfidf_features.shape}")

    # Splitting dataset
    st.write("### Model Training")
    if 'score' in df.columns:
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['score'] > 3)
        X_train, X_test, y_train, y_test = train_test_split(tfidf_features, df['label'], test_size=0.2, random_state=42)
        
        # Logistic Regression
        logreg = LogisticRegression(max_iter=200)
        logreg.fit(X_train, y_train)
        y_pred_logreg = logreg.predict(X_test)
        st.write("#### Logistic Regression")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred_logreg):.2f}")
        st.text(classification_report(y_test, y_pred_logreg))
        
        # Naive Bayes
        nb = MultinomialNB()
        nb.fit(X_train, y_train)
        y_pred_nb = nb.predict(X_test)
        st.write("#### Naive Bayes")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred_nb):.2f}")
        st.text(classification_report(y_test, y_pred_nb))
        
        # Support Vector Machine (SVM)
        svm = SVC(kernel='linear')
        svm.fit(X_train, y_train)
        y_pred_svm = svm.predict(X_test)
        st.write("#### Support Vector Machine (SVM)")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.2f}")
        st.text(classification_report(y_test, y_pred_svm))

        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        st.write("#### Random Forest")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")
        st.text(classification_report(y_test, y_pred_rf))

    # LSTM Model
    st.write("### LSTM Neural Network")
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['clean_content'])
    sequences = tokenizer.texts_to_sequences(df['clean_content'])
    padded = pad_sequences(sequences, maxlen=100)

    X_train, X_test, y_train, y_test = train_test_split(padded, df['label'], test_size=0.2, random_state=42)

    model = Sequential([
        Embedding(input_dim=5000, output_dim=16),
        Bidirectional(LSTM(32)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), verbose=1)

    st.write("#### LSTM Model Results")
    st.write(f"Accuracy: {model.evaluate(X_test, y_test, verbose=0)[1]:.2f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title("Model Accuracy")
    ax1.legend()

    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title("Model Loss")
    ax2.legend()

    st.pyplot(fig)
