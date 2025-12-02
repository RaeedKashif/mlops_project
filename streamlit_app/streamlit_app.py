import sqlite3
import pandas as pd
import streamlit as st
import json
from streamlit_autorefresh import st_autorefresh
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add to your existing streamlit_app.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import cv2

class MultiModalPredictor(nn.Module):
    def __init__(self, time_series_size=3, text_embedding_size=128, 
                 image_embedding_size=512, hidden_size=256, output_size=2):
        super(MultiModalPredictor, self).__init__()
        
        # Time series branch (existing health metrics)
        self.time_series_fc1 = nn.Linear(time_series_size, 64)
        self.time_series_fc2 = nn.Linear(64, 128)
        
        # Text branch (for medical notes, descriptions)
        self.text_fc1 = nn.Linear(text_embedding_size, 128)
        self.text_fc2 = nn.Linear(128, 128)
        
        # Image branch (for medical imaging)
        self.image_fc1 = nn.Linear(image_embedding_size, 256)
        self.image_fc2 = nn.Linear(256, 128)
        
        # Fusion layer
        total_features = 128 + 128 + 128  # time_series + text + image
        self.fusion_fc1 = nn.Linear(total_features, hidden_size)
        self.fusion_fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.output_layer = nn.Linear(hidden_size // 2, output_size)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, time_series, text_embeddings, image_embeddings):
        # Time series processing
        ts_out = F.relu(self.time_series_fc1(time_series))
        ts_out = self.dropout(ts_out)
        ts_out = F.relu(self.time_series_fc2(ts_out))
        
        # Text processing
        text_out = F.relu(self.text_fc1(text_embeddings))
        text_out = self.dropout(text_out)
        text_out = F.relu(self.text_fc2(text_out))
        
        # Image processing
        img_out = F.relu(self.image_fc1(image_embeddings))
        img_out = self.dropout(img_out)
        img_out = F.relu(self.image_fc2(img_out))
        
        # Feature fusion
        fused = torch.cat([ts_out, text_out, img_out], dim=1)
        fused = F.relu(self.fusion_fc1(fused))
        fused = self.dropout(fused)
        fused = F.relu(self.fusion_fc2(fused))
        
        output = self.output_layer(fused)
        return output
    

# Add this class to your streamlit_app.py

class DataDriftDetector:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.reference_data = None
        self.drift_history = []
        
    def set_reference(self, data):
        """Set reference distribution for drift detection"""
        self.reference_data = data
        
    def detect_drift(self, current_data, feature_names=None, alpha=0.05):
        """Detect data drift using multiple statistical tests"""
        if self.reference_data is None:
            self.set_reference(current_data)
            return {"drift_detected": False, "message": "Reference set"}
            
        drift_results = {}
        
        # Kolmogorov-Smirnov test for each feature
        for i in range(current_data.shape[1]):
            feature_name = feature_names[i] if feature_names else f"feature_{i}"
            ks_stat, p_value = stats.ks_2samp(
                self.reference_data[:, i], 
                current_data[:, i]
            )
            
            drift_results[feature_name] = {
                "ks_statistic": ks_stat,
                "p_value": p_value,
                "drift_detected": p_value < alpha
            }
        
        # Overall drift assessment
        drift_detected = any([result["drift_detected"] for result in drift_results.values()])
        
        # Calculate drift magnitude
        drift_magnitude = self._calculate_drift_magnitude(current_data)
        
        result = {
            "drift_detected": drift_detected,
            "drift_magnitude": drift_magnitude,
            "feature_drifts": drift_results,
            "timestamp": datetime.now().isoformat()
        }
        
        self.drift_history.append(result)
        return result
    
    def _calculate_drift_magnitude(self, current_data):
        """Calculate overall drift magnitude"""
        magnitude = 0
        for i in range(current_data.shape[1]):
            ref_mean = np.mean(self.reference_data[:, i])
            curr_mean = np.mean(current_data[:, i])
            ref_std = np.std(self.reference_data[:, i])
            
            if ref_std > 0:  # Avoid division by zero
                magnitude += abs((curr_mean - ref_mean) / ref_std)
        
        return magnitude / current_data.shape[1]

# ---- Federated Learning Model Definition ----
class HealthRiskPredictor(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, output_size=2):
        super(HealthRiskPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# ---- Config ----
DB_PATH = os.environ.get("DB_PATH", "/app/db/events.db")
st.set_page_config(page_title="IoT Monitoring Dashboard", layout="wide")
st.title("🔴 Real-Time IoT Monitoring Dashboard")

# Refresh interval
refresh_rate = st.slider("Refresh interval (seconds)", 1, 10, 2)
st_autorefresh(interval=refresh_rate * 1000, key="autorefresh")

# ---- Load events from SQLite ----
@st.cache_data(ttl=5)
def load_events():
    if not os.path.exists(DB_PATH):
        st.warning(f"Database not found at {DB_PATH}. Waiting for data...")
        return pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)
    try:
        # First, let's check what columns actually exist
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(events)")
        columns = [col[1] for col in cursor.fetchall()]
        print(f"Available columns: {columns}")
        
        # Build query based on available columns
        select_columns = []
        if 'event_id' in columns:
            select_columns.append('event_id')
        if 'timestamp' in columns:
            select_columns.append('timestamp')
        if 'node_id' in columns:
            select_columns.append('node_id')
        if 'type' in columns:
            select_columns.append('type')
        if 'seq' in columns:
            select_columns.append('seq')
        if 'payload' in columns:
            select_columns.append('payload')
        if 'target_node' in columns:
            select_columns.append('target_node')
        
        if not select_columns:
            st.error("No valid columns found in events table")
            return pd.DataFrame()
            
        query = f"SELECT {', '.join(select_columns)} FROM events ORDER BY timestamp DESC LIMIT 500"
        df = pd.read_sql_query(query, conn)
        
    except Exception as e:
        st.error(f"Failed to read from DB: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

    if not df.empty:
        print(f"Loaded {len(df)} events")
        # Parse payload JSON if payload column exists
        if 'payload' in df.columns:
            df["payload"] = df["payload"].apply(lambda x: json.loads(x) if x else {})
            
            # Extract metrics from payload
            df["temperature"] = df["payload"].apply(lambda x: x.get("temperature") if isinstance(x, dict) else None)
            df["humidity"] = df["payload"].apply(lambda x: x.get("humidity") if isinstance(x, dict) else None)
            df["heart_rate"] = df["payload"].apply(lambda x: x.get("heart_rate") if isinstance(x, dict) else None)
            df["cholesterol"] = df["payload"].apply(lambda x: x.get("cholesterol") if isinstance(x, dict) else None)
            df["blood_pressure"] = df["payload"].apply(lambda x: x.get("blood_pressure") if isinstance(x, dict) else None)
        
        # Convert timestamp to datetime for proper time series
        if 'timestamp' in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"])
        
        # Ensure required columns exist
        if 'type' not in df.columns:
            df['type'] = 'unknown'
        if 'target_node' not in df.columns:
            df['target_node'] = None

    return df

df = load_events()

def show_enhanced_ai_tab():
    st.subheader("🤖 Multi-Modal AI with Federated Learning")
    
    # Model Configuration
    st.write("### Model Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        data_types = st.multiselect(
            "Data Types to Use",
            ["Time Series", "Text", "Images"],
            default=["Time Series"]
        )
        
    with col2:
        fl_rounds = st.slider("Federated Learning Rounds", 1, 100, 10)
        drift_threshold = st.slider("Drift Detection Sensitivity", 0.01, 0.1, 0.05)
    
    # Data Type Visualization
    st.write("### Multi-Modal Data Overview")
    
    if not df.empty:
        # Show data distribution across types
        col1, col2, col3 = st.columns(3)
        
        with col1:
            time_series_count = len(df[df['type'].isin(['heart', 'weather'])])
            st.metric("Time Series Data", time_series_count)
            
        with col2:
            # Simulate text data availability
            text_data_count = len(df[df['node_id'].str.contains('hospital')])
            st.metric("Text Data (Simulated)", text_data_count)
            
        with col3:
            # Simulate image data availability
            image_data_count = len(df[df['seq'] % 10 == 0])  # Every 10th record
            st.metric("Image Data (Simulated)", image_data_count)
    
    # Data Drift Monitoring
    st.write("### Data Drift Detection")
    
    if not df.empty and 'heart_rate' in df.columns:
        # Create sample data for drift detection
        heart_data = df[df['type'] == 'heart'][['heart_rate', 'cholesterol', 'blood_pressure']].dropna()
        
        if len(heart_data) > 10:
            # Split into reference and current data
            split_idx = len(heart_data) // 2
            reference_data = heart_data.iloc[:split_idx].values
            current_data = heart_data.iloc[split_idx:].values
            
            # Initialize drift detector
            drift_detector = DataDriftDetector()
            drift_detector.set_reference(reference_data)
            
            # Detect drift
            drift_result = drift_detector.detect_drift(
                current_data, 
                ['heart_rate', 'cholesterol', 'blood_pressure'],
                alpha=drift_threshold
            )
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                status = "🚨 DRIFT" if drift_result['drift_detected'] else "✅ STABLE"
                st.metric("Data Distribution", status)
                
            with col2:
                st.metric("Drift Magnitude", f"{drift_result['drift_magnitude']:.3f}")
                
            with col3:
                drifting_features = sum([
                    1 for feature in drift_result['feature_drifts'].values() 
                    if feature['drift_detected']
                ])
                st.metric("Drifting Features", drifting_features)
            
            # Show feature-level drift details
            st.write("#### Feature-level Drift Analysis")
            for feature_name, result in drift_result['feature_drifts'].items():
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.write(f"**{feature_name}**")
                    
                with col2:
                    st.write(f"KS: {result['ks_statistic']:.3f}")
                    
                with col3:
                    st.write(f"p-value: {result['p_value']:.3f}")
                    
                with col4:
                    if result['drift_detected']:
                        st.error("🚨 Drift")
                    else:
                        st.success("✅ Stable")
    
    # Federated Learning Progress
    st.write("### Federated Learning Progress")
    
    # Simulate FL progress (replace with actual FL tracking)
    fl_progress = st.progress(0)
    for i in range(fl_rounds + 1):
        # Simulate round completion
        if st.button(f"Simulate FL Round {i}") or i == 0:
            progress = i / fl_rounds
            fl_progress.progress(progress)
            
            # Show model performance metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Simulated accuracy
                accuracy = 0.7 + (progress * 0.25) + random.uniform(-0.05, 0.05)
                st.metric("Global Model Accuracy", f"{accuracy:.2%}")
                
            with col2:
                # Simulated client participation
                active_clients = min(5, int(progress * 10))
                st.metric("Active Clients", active_clients)
                
            with col3:
                # Data variety
                data_variety = len(data_types)
                st.metric("Data Types Used", data_variety)

# Add this new tab to your existing tab structure
# Modify your tabs line to include the new AI tab:
# tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🌦 Weather", "❤️ Heart", "🔗 Network", "🤖 FL Monitoring", "🧠 AI Analytics", "📊 Raw Data"])

# Then add:
# with tab5:
#     show_enhanced_ai_tab()

# ---- Federated Learning Tab Function ----
def show_federated_learning_tab():
    st.subheader("Federated Learning Monitoring")
    
    # Model performance
    st.write("### Model Performance")
    
    # Check if model files exist
    models_dir = "/app/models"
    model_files = []
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    
    if model_files:
        for model_file in sorted(model_files)[-5:]:  # Show last 5 models
            st.success(f"✅ Model found: {model_file}")
    else:
        st.warning("⏳ No trained models found yet. Federated learning is in progress...")
    
    # Client status
    st.write("### Client Status")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Check if hospital_A has data
        hospital_a_data = len(df[(df['node_id'] == 'hospital_A') & (df['type'] == 'heart')])
        status = "Active" if hospital_a_data > 0 else "Waiting for data"
        st.metric("Hospital A Client", status, delta=f"{hospital_a_data} records")
    
    with col2:
        # Check if hospital_B has data
        hospital_b_data = len(df[(df['node_id'] == 'hospital_B') & (df['type'] == 'heart')])
        status = "Active" if hospital_b_data > 0 else "Waiting for data"
        st.metric("Hospital B Client", status, delta=f"{hospital_b_data} records")
    
    with col3:
        # Check if continent_asia has data
        asia_data = len(df[(df['node_id'] == 'continent_asia') & (df['type'] == 'weather')])
        status = "Active" if asia_data > 0 else "Waiting for data"
        st.metric("Asia Client", status, delta=f"{asia_data} records")
    
    with col4:
        total_clients = len(df['node_id'].unique()) if not df.empty else 0
        st.metric("Total Clients", total_clients)
    
    # Real-time prediction demo
    st.write("### Health Risk Prediction Demo")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        heart_rate = st.slider("Heart Rate (BPM)", 40, 200, 72)
    with col2:
        cholesterol = st.slider("Cholesterol (mg/dL)", 100, 400, 200)
    with col3:
        blood_pressure = st.slider("Blood Pressure (mmHg)", 80, 200, 120)
    
    if st.button("Predict Health Risk"):
        # Load the latest model
        latest_model = None
        if model_files:
            latest_model_file = sorted(model_files)[-1]  # Get most recent model
            latest_model = os.path.join(models_dir, latest_model_file)
        
        if latest_model and os.path.exists(latest_model):
            try:
                model = HealthRiskPredictor()
                model.load_state_dict(torch.load(latest_model, map_location='cpu'))
                model.eval()
                
                # Make prediction
                with torch.no_grad():
                    features = torch.FloatTensor([[heart_rate, cholesterol, blood_pressure]])
                    output = model(features)
                    prediction = torch.softmax(output, dim=1)
                    risk_level = prediction[0][1].item()  # Probability of high risk
                
                st.write(f"### Prediction Result")
                st.metric("Health Risk Probability", f"{risk_level:.2%}")
                
                if risk_level > 0.7:
                    st.error("🚨 High health risk detected! Please consult a doctor.")
                elif risk_level > 0.3:
                    st.warning("⚠️ Moderate health risk. Monitor your health.")
                else:
                    st.success("✅ Low health risk. Keep maintaining healthy habits!")
                    
                # Show model info
                st.info(f"Using model: {os.path.basename(latest_model)}")
                
            except Exception as e:
                st.error(f"Error loading model: {e}")
                st.info("The model format might be incompatible. Waiting for new training round.")
        else:
            st.info("🤖 Model is still training. Please wait for federated learning to complete.")
    
    # Training Progress
    st.write("### Training Progress")
    
    if not df.empty:
        # Show data distribution for FL
        heart_data_by_client = df[df['type'] == 'heart'].groupby('node_id').size()
        weather_data_by_client = df[df['type'] == 'weather'].groupby('node_id').size()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not heart_data_by_client.empty:
                st.write("**Heart Data Distribution**")
                fig_heart = px.pie(values=heart_data_by_client.values, 
                                 names=heart_data_by_client.index,
                                 title="Heart Data by Client")
                st.plotly_chart(fig_heart, use_container_width=True)
            else:
                st.info("No heart data available for FL")
        
        with col2:
            if not weather_data_by_client.empty:
                st.write("**Weather Data Distribution**")
                fig_weather = px.pie(values=weather_data_by_client.values, 
                                   names=weather_data_by_client.index,
                                   title="Weather Data by Client")
                st.plotly_chart(fig_weather, use_container_width=True)
            else:
                st.info("No weather data available for FL")
    
    # FL Statistics
    st.write("### Federated Learning Statistics")
    
    if not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_heart = len(df[df['type'] == 'heart'])
            st.metric("Total Heart Records", total_heart)
        
        with col2:
            total_weather = len(df[df['type'] == 'weather'])
            st.metric("Total Weather Records", total_weather)
        
        with col3:
            active_heart_clients = len(df[df['type'] == 'heart']['node_id'].unique())
            st.metric("Active Heart Clients", active_heart_clients)
        
        with col4:
            active_weather_clients = len(df[df['type'] == 'weather']['node_id'].unique())
            st.metric("Active Weather Clients", active_weather_clients)

# ---- Display Stats ----
if not df.empty:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_events = len(df)
        st.metric("Total Events", total_events)
    
    with col2:
        if 'type' in df.columns:
            weather_events = len(df[df["type"] == "weather"])
            st.metric("Weather Events", weather_events)
        else:
            st.metric("Weather Events", 0)
    
    with col3:
        if 'type' in df.columns:
            heart_events = len(df[df["type"] == "heart"])
            st.metric("Heart Events", heart_events)
        else:
            st.metric("Heart Events", 0)
    
    with col4:
        if 'node_id' in df.columns:
            unique_nodes = df["node_id"].nunique()
            st.metric("Unique Nodes", unique_nodes)
        else:
            st.metric("Unique Nodes", 0)
else:
    st.warning("No data available yet. Waiting for events...")

# ---- Tabs ----
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🌦 Weather", "❤️ Heart", "🔗 Network", "🤖 FL Monitoring", "📊 Raw Data"])

# ---- Weather Chart ----
with tab1:
    st.subheader("Weather Sensor Data")
    
    if not df.empty and 'type' in df.columns:
        weather_df = df[df["type"] == "weather"].copy()
        
        if not weather_df.empty and 'datetime' in weather_df.columns:
            weather_df = weather_df.sort_values("datetime")
            
            # Temperature over time
            if 'temperature' in weather_df.columns and weather_df["temperature"].notna().any():
                fig_temp = px.line(weather_df, x="datetime", y="temperature", 
                                 color="node_id", title="Temperature Over Time",
                                 labels={"temperature": "Temperature (°F)", "datetime": "Time"})
                st.plotly_chart(fig_temp, use_container_width=True)
            else:
                st.info("No temperature data available in weather events")
            
            # Humidity over time
            if 'humidity' in weather_df.columns and weather_df["humidity"].notna().any():
                fig_humidity = px.line(weather_df, x="datetime", y="humidity",
                                     color="node_id", title="Humidity Over Time",
                                     labels={"humidity": "Humidity (%)", "datetime": "Time"})
                st.plotly_chart(fig_humidity, use_container_width=True)
            else:
                st.info("No humidity data available in weather events")
            
            # Current readings by continent
            st.subheader("Current Weather Readings by Continent")
            if 'node_id' in weather_df.columns:
                latest_weather = weather_df.drop_duplicates("node_id", keep="first")
                if not latest_weather.empty:
                    cols = st.columns(min(3, len(latest_weather)))
                    for idx, (_, row) in enumerate(latest_weather.iterrows()):
                        with cols[idx % 3]:
                            temp = row.get('temperature', 'N/A')
                            humidity = row.get('humidity', 'N/A')
                            node_name = row.get('node_id', 'Unknown').replace("continent_", "").title()
                            
                            if temp != 'N/A' and humidity != 'N/A':
                                st.metric(
                                    label=node_name,
                                    value=f"{temp:.1f}°F",
                                    delta=f"{humidity:.1f}% humidity"
                                )
                            else:
                                st.metric(label=node_name, value="No data")
        else:
            st.info("No weather data available yet. Waiting for sensor data...")
    else:
        st.info("No weather data available yet. Waiting for sensor data...")

# ---- Heart Data Chart ----
with tab2:
    st.subheader("Heart Health Data")
    
    if not df.empty and 'type' in df.columns:
        heart_df = df[df["type"] == "heart"].copy()
        
        if not heart_df.empty and 'datetime' in heart_df.columns:
            heart_df = heart_df.sort_values("datetime")
            
            # Heart rate over time
            if 'heart_rate' in heart_df.columns and heart_df["heart_rate"].notna().any():
                fig_heart = px.line(heart_df, x="datetime", y="heart_rate",
                                  color="node_id", title="Heart Rate Over Time",
                                  labels={"heart_rate": "Heart Rate (BPM)", "datetime": "Time"})
                st.plotly_chart(fig_heart, use_container_width=True)
            else:
                st.info("No heart rate data available")
            
            # Cholesterol levels
            if 'cholesterol' in heart_df.columns and heart_df["cholesterol"].notna().any():
                fig_chol = px.line(heart_df, x="datetime", y="cholesterol",
                                color="node_id", title="Cholesterol Levels Over Time",
                                labels={"cholesterol": "Cholesterol (mg/dL)", "datetime": "Time"})
                st.plotly_chart(fig_chol, use_container_width=True)
            else:
                st.info("No cholesterol data available")
            
            # Blood pressure
            if 'blood_pressure' in heart_df.columns and heart_df["blood_pressure"].notna().any():
                fig_bp = px.line(heart_df, x="datetime", y="blood_pressure",
                               color="node_id", title="Blood Pressure Over Time",
                               labels={"blood_pressure": "Blood Pressure (mmHg)", "datetime": "Time"})
                st.plotly_chart(fig_bp, use_container_width=True)
            else:
                st.info("No blood pressure data available")
            
            # Current health metrics by hospital
            st.subheader("Current Health Metrics by Hospital")
            if 'node_id' in heart_df.columns:
                latest_heart = heart_df.drop_duplicates("node_id", keep="first")
                if not latest_heart.empty:
                    for _, row in latest_heart.iterrows():
                        col1, col2, col3 = st.columns(3)
                        node_name = row.get('node_id', 'Unknown')
                        
                        with col1:
                            hr = row.get('heart_rate')
                            st.metric(
                                label=f"{node_name} Heart Rate",
                                value=f"{hr:.0f} BPM" if pd.notna(hr) else "N/A"
                            )
                        with col2:
                            chol = row.get('cholesterol')
                            st.metric(
                                label=f"{node_name} Cholesterol",
                                value=f"{chol:.0f} mg/dL" if pd.notna(chol) else "N/A"
                            )
                        with col3:
                            bp = row.get('blood_pressure')
                            st.metric(
                                label=f"{node_name} Blood Pressure",
                                value=f"{bp:.0f} mmHg" if pd.notna(bp) else "N/A"
                            )
        else:
            st.info("No heart data available yet. Waiting for medical device data...")
    else:
        st.info("No heart data available yet. Waiting for medical device data...")

# ---- Network Visualization ----
with tab3:
    st.subheader("IoT Network Data Flow")
    
    if not df.empty and 'node_id' in df.columns:
        G = nx.DiGraph()
        
        # Add nodes with their types
        for node in df["node_id"].unique():
            if 'continent' in str(node):
                node_type = "continent"
            else:
                node_type = "hospital"
            G.add_node(node, type=node_type, title=node)
        
        # Add edges with weights based on connection frequency
        if 'target_node' in df.columns:
            edge_weights = {}
            for _, row in df.iterrows():
                if pd.notna(row.get("target_node")):
                    edge = (row["node_id"], row["target_node"])
                    edge_weights[edge] = edge_weights.get(edge, 0) + 1
            
            for edge, weight in edge_weights.items():
                G.add_edge(edge[0], edge[1], weight=weight, title=f"Messages: {weight}")
        
        if G.number_of_nodes() > 0:
            # Create network visualization
            net = Network(height="600px", width="100%", directed=True)
            net.from_nx(G)
            
            # Customize node appearance
            for node in net.nodes:
                if node.get("type") == "continent":
                    node["color"] = "#1f77b4"  # Blue for continents
                    node["size"] = 25
                else:
                    node["color"] = "#ff7f0e"  # Orange for hospitals
                    node["size"] = 20
            
            # Save and display
            net.save_graph("/app/network.html")
            with open("/app/network.html", "r", encoding="utf-8") as f:
                components.html(f.read(), height=600)
            
            # Network statistics
            st.subheader("Network Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Nodes", G.number_of_nodes())
            with col2:
                st.metric("Total Connections", G.number_of_edges())
            with col3:
                continent_nodes = len([n for n in G.nodes() if 'continent' in str(n)])
                st.metric("Continents", continent_nodes)
        else:
            st.info("No nodes found for network visualization")
    else:
        st.info("No network data available yet. Waiting for sensor communications...")

# ---- Federated Learning Tab ----
with tab4:
    show_federated_learning_tab()

# ---- Raw Data Tab ----
with tab5:
    st.subheader("Raw Event Data")
    if not df.empty:
        # Show available columns
        display_columns = [col for col in ['event_id', 'timestamp', 'node_id', 'type', 'temperature', 'humidity', 'heart_rate'] 
                         if col in df.columns]
        display_df = df[display_columns].head(20)
        st.dataframe(display_df)
        
        # Data summary
        st.subheader("Data Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'type' in df.columns:
                st.write("**Data Types Distribution**")
                type_counts = df["type"].value_counts()
                st.bar_chart(type_counts)
            else:
                st.write("No type data available")
        
        with col2:
            if 'node_id' in df.columns:
                st.write("**Node Activity**")
                node_counts = df["node_id"].value_counts().head(10)
                st.bar_chart(node_counts)
            else:
                st.write("No node data available")
    else:
        st.info("No data available yet. Waiting for events...")

# ---- Footer ----
st.markdown("---")
st.markdown("IoT Monitoring Dashboard | Real-time Health & Weather Data | Federated Learning")