import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import pickle

# Add custom CSS
st.markdown("""
    <style>
        .header {
            font-size: 36px;
            font-weight: bold;
            color: #2C3E50;
            text-align: center;
        }
        .subheader {
            font-size: 24px;
            font-weight: bold;
            color: #34495E;
        }
        .input-label {
            font-size: 16px;
        }
        .stButton>button {
            background-color: #3498db;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #2980b9;
        }
    </style>
""", unsafe_allow_html=True)



# Feature Engineering Function (same as before)
def feature_engineering(data):
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Calculate transaction_age in months
    current_date = datetime.now()
    data['transaction_age'] = data['timestamp'].apply(
        lambda x: (current_date.year - x.year) * 12 + (current_date.month - x.month)
    )

    # Temperature Range
    data['temp_range_10H'] = data['temperature_10H_max (\u00b0C)'] - data['temperature_10H_min (\u00b0C)']

    # Average Temperature
    data['avg_temperature'] = data[['temperature-1', 'temperature-2', 'temperature-3']].mean(axis=1)

    # Voltage Range
    data['voltage_range'] = data[['Voltage-L', 'Voltage-R', 'Voltage-M']].max(axis=1) - data[['Voltage-L', 'Voltage-R', 'Voltage-M']].min(axis=1)

    # Total Current
    data['total_current'] = data[['Current-M', 'Current-R', 'Current-T']].sum(axis=1)

    # RPM Standard Deviation
    data['rpm_std'] = data[['RPM-1', 'RPM-2', 'RPM-3']].std(axis=1)

    # Average Vibration
    data['avg_vibration'] = data[['Vibration-1', 'Vibration-2']].mean(axis=1)

    # Drop columns that are already represented by new features
    columns_to_drop = [
        'temperature_10H_max (\u00b0C)', 'temperature_10H_min (\u00b0C)',
        'temperature-1', 'temperature-2', 'temperature-3',
        'Voltage-L', 'Voltage-R', 'Voltage-M',
        'Current-M', 'Current-R', 'Current-T',
        'RPM-1', 'RPM-2', 'RPM-3',
        'Vibration-1', 'Vibration-2'
    ]
    data.drop(columns=columns_to_drop, inplace=True)

    # Fill missing values in Power_Backup
    data['Power_Backup'].fillna('No', inplace=True)

    # Ordinal Encoding for Power_Backup
    ordinal_encoder = OrdinalEncoder(categories=[['No', 'Yes']])
    data['Power_Backup'] = ordinal_encoder.fit_transform(data[['Power_Backup']])

    # Convert Last Maintenance to maintenance_age_months
    data['Last Maintenance'] = pd.to_datetime(data['Last Maintenance'])
    data['maintenance_age_months'] = data['transaction_age'] - (
        data['Last Maintenance'].dt.year * 12 + data['Last Maintenance'].dt.month
    )

    # Drop Last Maintenance column
    data.drop(['Last Maintenance'], axis=1, inplace=True)

    # One-Hot Encoding for Status Sparepart and Priority
    status_sparepart = ['Broken', 'Empty', 'In Use', 'On Check', 'Ready', 'Repair']
    for status in status_sparepart:
        data[f'Status Sparepart_{status}'] = (data['Status Sparepart'] == status).astype(int)

    priority = ['High', 'Medium', 'Low']
    for pr in priority:
        data[f'Priority_{pr}'] = (data['Priority'] == pr).astype(int)

    # Drop original columns after encoding
    data.drop(columns=['Status Sparepart', 'Priority'], inplace=True)

    # Add Age column for manual input

    # Reorder columns
    column_order = [
        'apparent_temperature_max', 'apparent_temperature_min', 'humidity', 'RPM', 'Power', 'Power_Backup', 
        'transaction_age', 'temp_range_10H', 'avg_temperature', 'voltage_range', 'total_current', 'rpm_std', 
        'avg_vibration', 'Age', 
        'Status Sparepart_Broken', 'Status Sparepart_Empty', 'Status Sparepart_In Use', 'Status Sparepart_On Check', 
        'Status Sparepart_Ready', 'Status Sparepart_Repair', 
        'Priority_High', 'Priority_Low', 'Priority_Medium', 
        'maintenance_age_months'
    ]
    data = data[column_order]

    return data

# Standardize data function
def standardize_data(data, scaler=None):
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    if scaler is None:
        scaler = StandardScaler()
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
    else:
        data[numeric_columns] = scaler.transform(data[numeric_columns])

    return data, scaler

# Main function for the app
def main():
    st.title("🔧 Machine Fault Classification App")

    st.write("""
        Input machine data such as temperature, vibration, and power to get quick fault predictions:
        - ✅ **Normal**: No immediate action needed.
        - ⚠️ **Warning**: Keep an eye on the system.
        - ❌ **Breakdown**: Immediate attention required.

        This app helps streamline maintenance decisions, reducing downtime and improving efficiency.
    """)

    default_values = {
        'temperature_10H_max': 23.71,
        'temperature_10H_min': -17.91,
        'temperature_1': 24.5,
        'temperature_2': 23.2,
        'temperature_3': 22.3,
        'humidity': 45.17,
        'Voltage_L': 228.31,
        'Voltage_R': 229.40,
        'Voltage_M': 230.50,
        'Current_M': 193.31,
        'Current_R': 180.50,
        'Current_T': 160.40,
        'RPM': 2991.06,
        'RPM_1': 2989.45,
        'RPM_2': 2990.78,
        'RPM_3': 2992.10,
        'Vibration_1': 30.94,
        'Vibration_2': 28.72,
        'apparent_temperature_max': 23.71,
        'apparent_temperature_min': -17.91,
        'Power': 1877.73,
        'Age': 10,
        'Power_Backup': "No",
        'Last_Maintenance': datetime(2020, 1, 16),
        'Status_Sparepart': "Ready",
        'Priority': "Medium"
    }

    # Sidebar for input section
    with st.sidebar:
        st.header("📝 Input Data")
        
        # 🌡️ Temperature & Humidity Inputs
       # Ensure default_values contains floats for numerical inputs
        def ensure_float(value):
            return float(value) if isinstance(value, (int, float)) else 0.0

        # Temperature & Humidity Inputs
        st.subheader("🌡️ Temperature & Humidity Inputs")
        temperature_10H_max = st.number_input("Temperature 10H Max (°C)", format="%.2f", value=ensure_float(default_values['temperature_10H_max']))
        temperature_10H_min = st.number_input("Temperature 10H Min (°C)", format="%.2f", value=ensure_float(default_values['temperature_10H_min']))
        temperature_1 = st.number_input("Temperature-1", format="%.2f", value=ensure_float(default_values['temperature_1']))
        temperature_2 = st.number_input("Temperature-2", format="%.2f", value=ensure_float(default_values['temperature_2']))
        temperature_3 = st.number_input("Temperature-3", format="%.2f", value=ensure_float(default_values['temperature_3']))
        humidity = st.number_input("Humidity", format="%.2f", value=ensure_float(default_values['humidity']))

        # ⚡ Electrical Inputs
        st.subheader("⚡ Electrical Inputs")
        Voltage_L = st.number_input("Voltage-L", format="%.2f", value=ensure_float(default_values['Voltage_L']))
        Voltage_R = st.number_input("Voltage-R", format="%.2f", value=ensure_float(default_values['Voltage_R']))
        Voltage_M = st.number_input("Voltage-M", format="%.2f", value=ensure_float(default_values['Voltage_M']))
        Current_M = st.number_input("Current-M", format="%.2f", value=ensure_float(default_values['Current_M']))
        Current_R = st.number_input("Current-R", format="%.2f", value=ensure_float(default_values['Current_R']))
        Current_T = st.number_input("Current-T", format="%.2f", value=ensure_float(default_values['Current_T']))

        # 🔄 RPM & Vibration Inputs
        st.subheader("🔄 RPM & Vibration Inputs")
        RPM = st.number_input("RPM", format="%.2f", value=ensure_float(default_values['RPM']))
        RPM_1 = st.number_input("RPM-1", format="%.2f", value=ensure_float(default_values['RPM_1']))
        RPM_2 = st.number_input("RPM-2", format="%.2f", value=ensure_float(default_values['RPM_2']))
        RPM_3 = st.number_input("RPM-3", format="%.2f", value=ensure_float(default_values['RPM_3']))
        Vibration_1 = st.number_input("Vibration-1", format="%.2f", value=ensure_float(default_values['Vibration_1']))
        Vibration_2 = st.number_input("Vibration-2", format="%.2f", value=ensure_float(default_values['Vibration_2']))

        # 🛠️ Other Inputs
        st.subheader("🛠️ Other Inputs")
        apparent_temperature_max = st.number_input("Apparent Temperature Max", format="%.2f", value=ensure_float(default_values['apparent_temperature_max']))
        apparent_temperature_min = st.number_input("Apparent Temperature Min", format="%.2f", value=ensure_float(default_values['apparent_temperature_min']))
        Power = st.number_input("Power", format="%.2f", value=ensure_float(default_values['Power']))
        Age = st.number_input("Age", format="%.2f", value=ensure_float(default_values['Age']))
        Power_Backup = st.selectbox("Power Backup", ["No", "Yes"], index=["No", "Yes"].index(default_values['Power_Backup']))
        Last_Maintenance = st.date_input("Last Maintenance", min_value=datetime(2000, 1, 1), max_value=datetime.today(), value=default_values['Last_Maintenance'])
        Status_Sparepart = st.selectbox("Status Sparepart", ["Broken", "Empty", "In Use", "On Check", "Ready", "Repair"], index=["Broken", "Empty", "In Use", "On Check", "Ready", "Repair"].index(default_values['Status_Sparepart']))
        Priority = st.selectbox("Priority", ["High", "Medium", "Low"], index=["High", "Medium", "Low"].index(default_values['Priority']))


        # Button to process manual input
        if st.sidebar.button("The Data is Complete"):
            manual_data = {
                'temperature_10H_max (\u00b0C)': [temperature_10H_max],
                'temperature_10H_min (\u00b0C)': [temperature_10H_min],
                'temperature-1': [temperature_1],
                'temperature-2': [temperature_2],
                'temperature-3': [temperature_3],
                'apparent_temperature_max': [apparent_temperature_max],
                'apparent_temperature_min': [apparent_temperature_min],
                'timestamp': [datetime.today()],
                'humidity': [humidity],
                'Age':[Age],
                'RPM': [RPM],
                'Voltage-L': [Voltage_L],
                'Voltage-R': [Voltage_R],
                'Voltage-M': [Voltage_M],
                'Current-M': [Current_M],
                'Current-R': [Current_R],
                'Current-T': [Current_T],
                'RPM-1': [RPM_1],
                'RPM-2': [RPM_2],
                'RPM-3': [RPM_3],
                'Vibration-1': [Vibration_1],
                'Vibration-2': [Vibration_2],
                'Power': [Power],
                'Power_Backup': [Power_Backup],
                'Last Maintenance': [Last_Maintenance],
                'Status Sparepart': [Status_Sparepart],
                'Priority': [Priority]
            }

            manual_df = pd.DataFrame(manual_data)

            # Process the manual input
            processed_manual_data = feature_engineering(manual_df)

            # Standardize the data
            scaler = None
            try:
                with open("E:\\DOWNLOAD\\scaler.pkl", 'rb') as f:
                    scaler = pickle.load(f)
            except FileNotFoundError:
                pass

            processed_manual_data, scaler = standardize_data(processed_manual_data, scaler)

            # Store processed data in session state for prediction
            st.session_state.processed_data = processed_manual_data

    # Prediction Section
    st.subheader("🔮 Make a Prediction")

    if st.button("Predict Fault"):
        # Check if processed data is available
        if 'processed_data' in st.session_state:
            processed_data = st.session_state.processed_data

            # Load the model and make the prediction
            model = pickle.load(open("E:\\DOWNLOAD\\voting_classifier_model.pkl", "rb"))
            prediction = model.predict(processed_data)

            # Display the prediction
            if prediction[0] == 0:
                prediction_label = "Normal ✅"
            elif prediction[0] == 1:
                prediction_label = "Warning ⚠️"
            else:
                prediction_label = "Breakdown ❌"
                
            st.subheader(f"Prediction: {prediction_label}")
        else:
            st.warning("Please process the input data first.")

if __name__ == "__main__":
    main()
