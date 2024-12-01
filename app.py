import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

# Load data and model
full_df = joblib.load('full_df_half.joblib')  # Load DataFrame from saved file
model = joblib.load('churn_analyzer_model.joblib')

st.set_page_config(
    page_title="Churnanalyzer App",
    page_icon="📉",
    layout="centered",
)

# Create session state for storing random data
if "random_data" not in st.session_state:
    st.session_state.random_data = None

# Function to generate random dataset
def random_dataset(count):
    st.session_state.random_data = full_df.sample(n=count)

# Function for prediction
def predict(threshold):
    if st.session_state.random_data is not None:
        test_data = st.session_state.random_data
        x_df = test_data.drop(columns=['Churn'])
        y_df = test_data['Churn']
        
        # Get model predictions probabilities
        y_probs = model.predict_proba(x_df)[:, 1]  # Probability of class 1
        
        # Set threshold
        threshold = threshold / 100
        y_pred_threshold = (y_probs >= threshold).astype(int)

        # Calculate accuracy
        accuracy = accuracy_score(y_df, y_pred_threshold)

        
        # Display results in Streamlit
        st.subheader("Model Analysis Results")
        st.success(f"Accuracy with threshold ({threshold * 100} %) : {accuracy *100:.2f} %")
        
        # Create Scatter Plot and Confusion Matrix
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Scatter Plot
        correct = (y_pred_threshold == y_df).to_numpy()  # True if prediction is correct
        colors = np.where(correct, 'blue', 'red')  # Blue for correct, red for incorrect
        axes[0].axhspan(threshold, 1, color='lightblue', alpha=0.3, label='Above Threshold (Light Blue)')
        axes[0].axhspan(0, threshold, color='lightcoral', alpha=0.3, label='Below Threshold (Light Red)')
        axes[0].scatter(range(len(y_probs)), y_probs, c=colors, label='Predicted Probabilities', edgecolor='k', alpha=0.7)
        axes[0].axhline(y=threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold = {threshold}')
        axes[0].set_xlabel('Index')
        axes[0].set_ylabel('Predicted Probability')
        axes[0].set_title('Scatter Plot of Predicted Probabilities with Background Colors')
        axes[0].legend(loc='upper right')
        axes[0].grid(alpha=0.3)

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_df, y_pred_threshold)
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[1])
        axes[1].set_xlabel('Predicted Labels')
        axes[1].set_ylabel('True Labels')
        axes[1].set_title('Confusion Matrix')

        # Display graphs in Streamlit
        st.pyplot(fig)

# Toggle Button
page = st.toggle(f"Switch Mode")

if page:
    # Model analysis page
    st.title("Churn Model Analysis")
    st.subheader("Control the process with the buttons below")

    # Random dataset input and button
    count = st.number_input('Number of Samples', value=10, step=1, min_value=1, max_value=int(full_df.shape[0]))
    if st.button("Generate Random Dataset"):
        random_dataset(count)
        st.success(f"Random dataset of {count} samples generated successfully!")

    # Display random dataset
    if st.session_state.random_data is not None:
        st.subheader("Randomly Sampled Data")
        st.dataframe(st.session_state.random_data)

    # Line separator between data and prediction
    if st.session_state.random_data is not None:
        st.markdown("---")

    # Predict button and prediction
    if st.session_state.random_data is not None:
        threshold = st.slider('Select Threshold', min_value=0, max_value=100, value=50)
        if st.button("Predict"):
            predict(threshold)

else:
    # Hello World Page
    st.title("Upload file to predict")
    st.write("Welcome! This is a Churnanalyzer app.")

    # File upload section
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        # Read uploaded CSV file
        df = pd.read_csv(uploaded_file)

        # Display DataFrame
        st.write("Uploaded Data :")
        st.dataframe(df)

        st.markdown("---")

        if st.button('Predict'):
            # Predict probability for class 1
            y_probs = model.predict_proba(df)[:, 1]

            # Set threshold
            threshold = 0.5
            y_pred_threshold = (y_probs >= threshold).astype(int)

            # Add prediction result to customer id
            df['Prediction'] = y_pred_threshold

            # Display customer id and prediction
            result_df = df[['CustomerID', 'Prediction']]
            st.write("Prediction Results each customerID :")
            st.dataframe(result_df)
            st.markdown("---")

          # สร้างฟังก์ชันสำหรับการคำนวณเปอร์เซ็นต์ Churn
            def calculate_churn_percentage(df, groupby_col):
                  group = df.groupby(groupby_col)['Prediction'].mean() * 100
                  return group.reset_index(name='Churn Percentage')
    
            # แสดงผลใน Streamlit
            st.subheader("Churn Analysis by Gender, Subscription Type, and Contract Length")
    
            col1, col2, col3 = st.columns(3)
    
            # Gender Analysis
            with col1:
                st.write("Churn Percentage by Gender \n    ")
                gender_churn = calculate_churn_percentage(df, 'Gender')
                st.bar_chart(gender_churn.set_index('Gender')['Churn Percentage'])
    
            # Subscription Type Analysis
            with col2:
                st.write("Churn Percentage by \n Subscription Type")
                subscription_churn = calculate_churn_percentage(df, 'Subscription Type')
                st.bar_chart(subscription_churn.set_index('Subscription Type')['Churn Percentage'])
    
            # Contract Length Analysis
            with col3:
                st.write("Churn Percentage \n by Contract Length")
                contract_churn = calculate_churn_percentage(df, 'Contract Length')
                st.bar_chart(contract_churn.set_index('Contract Length')['Churn Percentage'])
    else:
        st.warning("Please upload a CSV file to start.")
