import os
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="SafeText AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# FILE PATHS
# =====================================================

MODEL_PATH = "models/cyberbullying_model.joblib"

DATASET_PATHS = [
    "data/data/cyberbullying_tweets.csv",
    "data/cyberbullying_tweets.csv",
    "data_data/cyberbullying_tweets.csv",
    "cyberbullying_tweets.csv"
]


# =====================================================
# CUSTOM THEME
# =====================================================

st.markdown("""
<style>


/* ===== SAFETEXT AI DESIGN ===== */

.stApp {
    background-color: #F7F2FF;
    color: #29213A;
}

/* Main headings */
h1 {
    color: #6425B5 !important;
    font-size: 40px !important;
    font-weight: 800 !important;
}

h2 {
    color: #7134C5 !important;
    font-size: 32px !important;
    font-weight: 750 !important;
}

h3 {
    color: #873CC7 !important;
    font-size: 25px !important;
    font-weight: 700 !important;
}

/* Normal text */
p, li, label, .stMarkdown {
    font-size: 18px !important;
    line-height: 1.7 !important;
    color: #30263D;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #E9D9FF 0%,
        #F5EFFF 100%
    ) !important;
}

section[data-testid="stSidebar"] h1 {
    font-size: 29px !important;
    color: #6425B5 !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
    font-size: 19px !important;
    font-weight: 600 !important;
    color: #43206D !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(
        90deg,
        #7138D0,
        #C344B5
    ) !important;
    color: white !important;
    font-size: 19px !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 22px !important;
    min-height: 48px;
}

.stButton > button:hover {
    background: #5723A5 !important;
    color: white !important;
    border: 2px solid #E2B9FF !important;
}

/* Text input and text area */
.stTextInput input,
.stTextArea textarea {
    font-size: 18px !important;
    border: 2px solid #C9A5F5 !important;
    border-radius: 10px !important;
    color: #29213A !important;
}

/* Select boxes */
.stSelectbox div,
.stMultiSelect div {
    font-size: 17px !important;
}

/* Cards and information boxes */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
    font-size: 18px !important;
}

div[data-testid="stMetric"] {
    background: #EFE3FF;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #D5B9F5;
}

/* Horizontal separators */
hr {
    border-color: #D9C5F0 !important;
}

/* Responsive text */
@media (max-width: 768px) {
    h1 {
        font-size: 32px !important;
    }

    h2 {
        font-size: 27px !important;
    }

    p, li {
        font-size: 16px !important;
    }
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    try:
        return joblib.load(MODEL_PATH)

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


model = load_model()


# =====================================================
# LOAD DATASET
# =====================================================

@st.cache_data
def load_dataset():

    for path in DATASET_PATHS:

        if os.path.exists(path):

            try:

                df = pd.read_csv(path)

            except UnicodeDecodeError:

                df = pd.read_csv(
                    path,
                    encoding="latin-1"
                )

            return df, path

    return None, None


dataset, dataset_path = load_dataset()


# =====================================================
# SESSION STATE
# =====================================================

if "history" not in st.session_state:

    st.session_state.history = []


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def is_bullying_label(label):

    label = str(label).strip().lower()

    non_bullying_labels = [
        "not_cyberbullying",
        "not cyberbullying",
        "non-cyberbullying",
        "non cyberbullying",
        "nonbullying",
        "not bullying",
        "0",
        "false"
    ]

    if label in non_bullying_labels:

        return False

    if label in [
        "1",
        "true",
        "bullying",
        "cyberbullying"
    ]:

        return True

    return "cyberbullying" in label


def predict_text(text):

    prediction = model.predict([text])[0]

    bullying_score = 0.0

    try:

        probabilities = model.predict_proba([text])[0]

        classes = model.classes_

        for label, probability in zip(
            classes,
            probabilities
        ):

            if is_bullying_label(label):

                bullying_score += float(probability)

    except Exception:

        bullying_score = 0.0

    return prediction, bullying_score


def display_prediction(text):

    prediction, bullying_score = predict_text(text)

    is_bullying = is_bullying_label(prediction)

    st.divider()

    if is_bullying:

        st.error(
            "⚠️ Potential Cyberbullying Detected"
        )

        st.write(
            "The model identified this text as "
            "potentially harmful or bullying-related."
        )

    else:

        st.success(
            "✅ Non-Cyberbullying"
        )

        st.write(
            "The model classified this text "
            "as non-cyberbullying."
        )

    st.subheader("📊 Prediction Result")

    st.metric(
        "Estimated Bullying Probability",
        f"{bullying_score * 100:.2f}%"
    )

    st.progress(
        min(
            max(bullying_score, 0.0),
            1.0
        )
    )

    st.caption(
        "This score is an estimated model probability, "
        "not a guarantee of correctness."
    )

    st.session_state.history.append({

        "Time": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "Text": text,

        "Prediction": (
            "Cyberbullying"
            if is_bullying
            else "Non-Cyberbullying"
        ),

        "Bullying Probability": round(
            bullying_score * 100,
            2
        )

    })


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🛡️ SafeText AI")

st.sidebar.subheader(
    "AI-Based Cyberbullying Detection"
)

st.sidebar.divider()

page = st.sidebar.radio(

    "NAVIGATION",

    [
        "Home",
        "Detect Cyberbullying",
        "Dataset Explorer",
        "Sample Testing",
        "Detection History"
    ]

)

st.sidebar.divider()


st.sidebar.markdown("""
<div style="
    background-color: #F0E6FF;
    color: #3B176B !important;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #D8BFFF;
    font-weight: 600;
    font-size: 14px;
    line-height: 1.6;
    text-align: center;
">
    💜 Promoting safer and kinder digital
    communication through Artificial Intelligence.
</div>
""", unsafe_allow_html=True)


# =====================================================
# HOME PAGE
# =====================================================

if page == "Home":

    # Main heading

    st.title("🛡️ SafeText AI")

    st.header(
        "Making Digital Communication Safer"
    )

    st.write(
        "An intelligent machine learning application "
        "designed to identify potentially harmful language "
        "and cyberbullying in text messages."
    )

    st.divider()

    # About project

    st.header("✨ About Our Project")

    st.write(
        "SafeText AI uses Natural Language Processing "
        "and Machine Learning to analyse text and identify "
        "potentially harmful communication."
    )

    # Information cards

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("### 🤖 AI Powered")

        st.write(
            "Logistic Regression classification."
        )

    with col2:

        st.markdown("### 📝 Text Analysis")

        st.write(
            "TF-IDF based text representation."
        )

    with col3:

        st.markdown("### 🛡️ Safer Internet")

        st.write(
            "Helps identify potentially harmful text."
        )

    st.divider()

    # Objectives

    st.header("🎯 Project Objectives")

    st.markdown("""
    - Apply Natural Language Processing techniques.
    - Use machine learning for text classification.
    - Detect potentially harmful text.
    - Display understandable prediction results.
    - Promote awareness about cyberbullying.
    """)

    st.divider()

    # Project information

    st.header("💡 Project Information")

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            "**Algorithm:** Logistic Regression"
        )

    with col2:

        st.info(
            "**Feature Extraction:** TF-IDF"
        )

    st.warning(
        "This is an educational screening tool. "
        "Predictions may be incorrect and require context."
    )

    if model is None:

        st.error(
            "Model not found. Please run "
            "python train_model.py."
        )


# =====================================================
# DETECT CYBERBULLYING
# =====================================================

elif page == "Detect Cyberbullying":

    st.title("🔍 Cyberbullying Detection")

    st.write(
        "Enter a message below to analyse it "
        "using the trained machine learning model."
    )

    st.divider()

    text_input = st.text_area(

        "💬 Enter your message",

        placeholder="Type or paste a message here...",

        height=180,

        max_chars=5000

    )

    if st.button(
        "✨ Analyse Text",
        use_container_width=True
    ):

        if not text_input.strip():

            st.warning(
                "Please enter a message first."
            )

        elif model is None:

            st.error(
                "Model not found. "
                "Please train your model first."
            )

        else:

            display_prediction(text_input)


# =====================================================
# DATASET EXPLORER
# =====================================================

elif page == "Dataset Explorer":

    st.title("📊 Dataset Explorer")

    st.write(
        "Explore the dataset used for training "
        "the cyberbullying detection model."
    )

    st.divider()

    if dataset is None:

        st.error(
            "Dataset not found. "
            "Please check your CSV file location."
        )

    else:

        st.success(
            f"Dataset loaded successfully: {dataset_path}"
        )

        required_columns = [
            "tweet_text",
            "cyberbullying_type"
        ]

        if not all(
            column in dataset.columns
            for column in required_columns
        ):

            st.error(
                "The dataset does not contain the expected "
                "columns tweet_text and cyberbullying_type."
            )

            st.write(
                "Available columns:",
                list(dataset.columns)
            )

        else:

            # Dataset statistics

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "📄 Total Records",
                    f"{len(dataset):,}"
                )

            with col2:

                st.metric(
                    "🏷️ Categories",
                    dataset[
                        "cyberbullying_type"
                    ].nunique()
                )

            with col3:

                st.metric(
                    "📌 Columns",
                    len(dataset.columns)
                )

            st.divider()

            st.header("📋 Dataset Preview")

            st.write(
                "Each row contains a tweet and "
                "its corresponding category label."
            )

            # Search

            search_text = st.text_input(
                "🔎 Search Tweet Text",
                placeholder="Enter a word to search..."
            )

            # Category filter

            categories = sorted(
                dataset[
                    "cyberbullying_type"
                ]
                .dropna()
                .astype(str)
                .unique()
            )

            selected_category = st.selectbox(
                "🏷️ Filter by Category",
                ["All Categories"] + categories
            )

            filtered_dataset = dataset.copy()

            if selected_category != "All Categories":

                filtered_dataset = filtered_dataset[
                    filtered_dataset[
                        "cyberbullying_type"
                    ].astype(str)
                    == selected_category
                ]

            if search_text:

                filtered_dataset = filtered_dataset[
                    filtered_dataset[
                        "tweet_text"
                    ].astype(str)
                    .str.contains(
                        search_text,
                        case=False,
                        na=False,
                        regex=False
                    )
                ]

            # Prepare table

            display_dataset = (
                filtered_dataset
                .reset_index(drop=True)
                .copy()
            )

            display_dataset.insert(
                0,
                "S.No.",
                range(
                    1,
                    len(display_dataset) + 1
                )
            )

            # Display Excel-style table

            st.dataframe(

                display_dataset,

                use_container_width=True,

                hide_index=True,

                height=500,

                column_config={

                    "S.No.": st.column_config.NumberColumn(
                        "S.No.",
                        width="small"
                    ),

                    "tweet_text": st.column_config.TextColumn(
                        "Tweet Text",
                        width="large"
                    ),

                    "cyberbullying_type": st.column_config.TextColumn(
                        "Cyberbullying Type",
                        width="medium"
                    )

                }

            )

            st.caption(
                f"Displaying {len(filtered_dataset):,} "
                f"records out of {len(dataset):,} total records."
            )

            st.divider()

            # Category distribution

            st.header("📈 Dataset Category Distribution")

            category_counts = (

                dataset[
                    "cyberbullying_type"
                ]

                .value_counts()

                .rename_axis("Category")

                .reset_index(
                    name="Number of Records"
                )

            )

            st.dataframe(

                category_counts,

                use_container_width=True,

                hide_index=True

            )

            st.bar_chart(

                category_counts.set_index(
                    "Category"
                )

            )

            st.divider()

            # Download dataset

            st.download_button(

                label="⬇️ Download Dataset",

                data=dataset.to_csv(
                    index=False
                ).encode("utf-8"),

                file_name="cyberbullying_tweets.csv",

                mime="text/csv",

                use_container_width=True

            )


# =====================================================
# SAMPLE TESTING
# =====================================================

elif page == "Sample Testing":

    st.title("🧪 Sample Testing")

    st.write(
        "Test the model using sample messages."
    )

    st.divider()

    samples = {

        "Friendly Greeting":
            "I hope you have a wonderful day!",

        "Supportive Message":
            "You can do this. I believe in you.",

        "Insult":
            "You are useless and nobody likes you.",

        "Threat":
            "I will hurt you.",

        "Neutral Statement":
            "The class starts at ten in the morning."

    }

    selected_sample = st.selectbox(

        "Choose a sample message",

        list(samples.keys())

    )

    st.text_area(

        "Selected Message",

        value=samples[selected_sample],

        height=120,

        disabled=True

    )

    if st.button(
        "✨ Test Sample",
        use_container_width=True
    ):

        if model is None:

            st.error(
                "Model not found. "
                "Please train the model first."
            )

        else:

            display_prediction(
                samples[selected_sample]
            )


# =====================================================
# DETECTION HISTORY
# =====================================================

elif page == "Detection History":

    st.title("📜 Detection History")

    st.write(
        "View the messages analysed during "
        "your current website session."
    )

    st.caption(
        "History is stored only during the current session."
    )

    st.divider()

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(

            history_df,

            use_container_width=True,

            hide_index=True

        )

        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(

            label="⬇️ Download History",

            data=csv_data,

            file_name="safetext_detection_history.csv",

            mime="text/csv"

        )

        if st.button("🗑️ Clear History"):

            st.session_state.history = []

            st.rerun()

    else:

        st.info(
            "No detections yet. "
            "Analyse a message to create history."
        )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.subheader("🛡️ SafeText AI")

st.write(
    "AI-Based Cyberbullying Text Detection"
)

st.caption(
    "BCA AI/ML Mini Project | Project No. 78"
)

st.markdown(
    "💜 Promoting a Safer Digital World"
)
