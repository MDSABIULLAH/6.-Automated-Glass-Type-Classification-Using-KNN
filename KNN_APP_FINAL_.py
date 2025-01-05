import streamlit as st
import pandas as pd
import joblib
import pickle
from sqlalchemy import create_engine

# Load the necessary models and transformers
column_transformer = joblib.load('columntransformer_minmaxscaler.joblib')
knn_model = pickle.load(open('knn.pkl', 'rb'))

# Streamlit UI setup
st.title('Batch Glass Type Classification')
st.write('Upload a CSV file with glass properties for batch predictions.')

# Database credentials input
st.write('Provide your database connection details:')
db_root_name = st.text_input('Root Username')
db_password = st.text_input('Password', type='password')
db_name = st.text_input('Database Name')

# File uploader for CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

def process_and_predict(file, db_root_name, db_password, db_name):
    try:
        # Read the CSV file
        data = pd.read_csv(file)

        # Check if all necessary columns are present
        required_columns = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
        if all(col in data.columns for col in required_columns):
            # Preprocess the data using column transformer
            data_preprocessed = pd.DataFrame(column_transformer.transform(data[required_columns]), columns=required_columns)
            
            # Predict the glass type
            predictions = knn_model.predict(data_preprocessed)

            # Map predictions to glass type labels
            glass_types = {1: 'Type 1', 2: 'Type 2', 3: 'Type 3', 5: 'Type 5', 6: 'Type 6', 7: 'Type 7'}
            data['Predicted Type'] = [glass_types.get(p, f'Unknown ({p})') for p in predictions]

            # Connect to the database and save the results
            engine = create_engine(f'mysql+pymysql://{db_root_name}:{db_password}@localhost/{db_name}')
            data.to_sql('glass_predictions', con=engine, if_exists='append', index=False)

            # Display results and success message
            st.write(data)
            st.success('Data successfully saved to the database!')

        else:
            st.error("CSV doesn't contain required columns. Please check the file.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Button to trigger prediction
if uploaded_file is not None:
    if db_root_name and db_password and db_name:
        st.write(f'You have uploaded {uploaded_file.name}. Processing...')
        process_and_predict(uploaded_file, db_root_name, db_password, db_name)
    else:
        st.error('Please provide valid database credentials.')

