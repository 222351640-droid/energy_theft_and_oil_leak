# ---------------------------------------------------
# 1️⃣ Import Libraries
# ---------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

# ---------------------------------------------------
# 2️⃣ Paths
# ---------------------------------------------------
# Find project root by looking for .git or README.md
def find_project_root(start_path: Path = Path.cwd()) -> Path:
    current = start_path.absolute()
    while current != current.parent:
        if (current / '.git').exists() or (current / 'README.md').exists():
            return current
        current = current.parent
    raise RuntimeError('Could not find project root. Please run from within the project directory.')

# Set up paths relative to project root
PROJECT_ROOT = find_project_root(Path(__file__))
DATA_DIR = PROJECT_ROOT / 'src' / 'data'
energy_data_path = DATA_DIR / 'energy_features.csv'
pipeline_data_path = DATA_DIR / 'pipeline_features.csv'
energy_model_path = DATA_DIR / 'energy_model.joblib'
pipeline_model_path = DATA_DIR / 'pipeline_model.joblib'
energy_scaler_path = DATA_DIR / 'energy_scaler.joblib'
pipeline_scaler_path = DATA_DIR / 'pipeline_scaler.joblib'

# ---------------------------------------------------
# 3️⃣ Load Data and Models
# ---------------------------------------------------
@st.cache_data
def load_data():
    energy_df = pd.read_csv(energy_data_path)
    pipeline_df = pd.read_csv(pipeline_data_path)
    return energy_df, pipeline_df

@st.cache_resource
def load_models():
    energy_model = joblib.load(energy_model_path)
    pipeline_model = joblib.load(pipeline_model_path)
    energy_scaler = joblib.load(energy_scaler_path)
    pipeline_scaler = joblib.load(pipeline_scaler_path)
    return energy_model, pipeline_model, energy_scaler, pipeline_scaler

@st.cache_data
def calculate_metrics(energy_predictions, pipeline_predictions, X_energy_scaled, X_pipeline_scaled):
    # Calculate active alerts
    energy_alerts = sum(1 for x in energy_predictions if x == -1)
    pipeline_alerts = sum(1 for x in pipeline_predictions if x == -1)
    total_alerts = energy_alerts + pipeline_alerts
    
    # Calculate detection confidence using decision function
    energy_scores = np.abs(energy_model.decision_function(X_energy_scaled))
    pipeline_scores = np.abs(pipeline_model.decision_function(X_pipeline_scaled))
    
    # Convert scores to a confidence metric between 0 and 100
    energy_confidence = (np.mean(energy_scores) / np.max(energy_scores) * 100)
    pipeline_confidence = (np.mean(pipeline_scores) / np.max(pipeline_scores) * 100)
    avg_confidence = (energy_confidence + pipeline_confidence) / 2
    
    # Calculate other metrics
    response_time = round(energy_df['rolling_mean_3'].mean() / 60, 1) if 'rolling_mean_3' in energy_df.columns else 4.3
    monitored_zones = len(energy_df['id'].unique()) if 'id' in energy_df.columns else len(energy_df)
    
    result = {}
    result['active_alerts'] = total_alerts
    result['energy_alerts'] = energy_alerts
    result['pipeline_alerts'] = pipeline_alerts
    result['detection_accuracy'] = round(avg_confidence, 1)
    result['response_time'] = response_time
    result['monitored_zones'] = monitored_zones
    return result
    return metrics_dict

@st.cache_data
def get_predictions():
    # Energy theft predictions
    energy_features = ['consumption', 'rolling_mean_3', 'rolling_std_3', 'pct_change', 
                      'lag_1', 'lag_2', 'z_score']
    X_energy = energy_df[energy_features]
    X_energy = X_energy.replace([np.inf, -np.inf], np.nan)
    for col in X_energy.columns:
        if col in ['rolling_std_3']:
            X_energy[col] = X_energy[col].fillna(0)
        else:
            X_energy[col] = X_energy[col].fillna(method='ffill').fillna(method='bfill')
    X_energy_scaled = energy_scaler.transform(X_energy)
    energy_predictions = energy_model.predict(X_energy_scaled)
    
    # Pipeline leak predictions
    pipeline_features = ['flow_mean_5', 'flow_std_5', 'pressure_mean_5', 'pressure_std_5', 
                        'flow_change', 'pressure_change', 'flow_pressure_corr']
    X_pipeline = pipeline_df[pipeline_features]
    X_pipeline = X_pipeline.replace([np.inf, -np.inf], np.nan)
    X_pipeline = X_pipeline.fillna(method='ffill').fillna(method='bfill').fillna(0)
    X_pipeline_scaled = pipeline_scaler.transform(X_pipeline)
    pipeline_predictions = pipeline_model.predict(X_pipeline_scaled)
    
    return energy_predictions, pipeline_predictions, X_energy_scaled, X_pipeline_scaled

# Initialize everything at startup
try:
    # Load data and models
    energy_df, pipeline_df = load_data()
    energy_model, pipeline_model, energy_scaler, pipeline_scaler = load_models()
    
    # Get predictions and calculate metrics
    energy_predictions, pipeline_predictions, X_energy_scaled, X_pipeline_scaled = get_predictions()
    metrics = calculate_metrics(energy_predictions, pipeline_predictions, X_energy_scaled, X_pipeline_scaled)
    
    # Calculate system health metrics
    energy_efficiency = 100 - (metrics.get('energy_alerts', 0) / len(energy_predictions) * 100)
    pipeline_integrity = 100 - (metrics.get('pipeline_alerts', 0) / len(pipeline_predictions) * 100)
    
    # Calculate total theft cases
    theft_cases = sum(1 for x in energy_predictions if x == -1)
except Exception as e:
    st.error(f"Error initializing the application: {str(e)}")
    metrics = {}
    metrics['active_alerts'] = 0
    metrics['energy_alerts'] = 0
    metrics['pipeline_alerts'] = 0
    metrics['detection_accuracy'] = 0
    metrics['response_time'] = 0
    metrics['monitored_zones'] = 0
    energy_efficiency = 0
    pipeline_integrity = 0
    theft_cases = 0

@st.cache_resource
def load_models():
    energy_model = joblib.load(energy_model_path)
    pipeline_model = joblib.load(pipeline_model_path)
    energy_scaler = joblib.load(energy_scaler_path)
    pipeline_scaler = joblib.load(pipeline_scaler_path)
    return energy_model, pipeline_model, energy_scaler, pipeline_scaler

@st.cache_data
def get_predictions():
    # Energy theft predictions
    energy_features = ['consumption', 'rolling_mean_3', 'rolling_std_3', 'pct_change', 
                      'lag_1', 'lag_2', 'z_score']
    X_energy = energy_df[energy_features]
    X_energy = X_energy.replace([np.inf, -np.inf], np.nan)
    for col in X_energy.columns:
        if col in ['rolling_std_3']:
            X_energy[col] = X_energy[col].fillna(0)
        else:
            X_energy[col] = X_energy[col].fillna(method='ffill').fillna(method='bfill')
    X_energy_scaled = energy_scaler.transform(X_energy)
    energy_predictions = energy_model.predict(X_energy_scaled)
    
    # Pipeline leak predictions
    pipeline_features = ['flow_mean_5', 'flow_std_5', 'pressure_mean_5', 'pressure_std_5', 
                        'flow_change', 'pressure_change', 'flow_pressure_corr']
    X_pipeline = pipeline_df[pipeline_features]
    X_pipeline = X_pipeline.replace([np.inf, -np.inf], np.nan)
    X_pipeline = X_pipeline.fillna(method='ffill').fillna(method='bfill').fillna(0)
    X_pipeline_scaled = pipeline_scaler.transform(X_pipeline)
    pipeline_predictions = pipeline_model.predict(X_pipeline_scaled)
    
    return energy_predictions, pipeline_predictions, X_energy_scaled, X_pipeline_scaled

@st.cache_data
def calculate_metrics(energy_predictions, pipeline_predictions, X_energy_scaled, X_pipeline_scaled):
    # Calculate active alerts
    energy_alerts = sum(1 for x in energy_predictions if x == -1)
    pipeline_alerts = sum(1 for x in pipeline_predictions if x == -1)
    total_alerts = energy_alerts + pipeline_alerts
    
    # Calculate detection confidence using decision function
    energy_scores = np.abs(energy_model.decision_function(X_energy_scaled))
    pipeline_scores = np.abs(pipeline_model.decision_function(X_pipeline_scaled))
    
    # Convert scores to a confidence metric between 0 and 100
    energy_confidence = (np.mean(energy_scores) / np.max(energy_scores) * 100)
    pipeline_confidence = (np.mean(pipeline_scores) / np.max(pipeline_scores) * 100)
    avg_confidence = (energy_confidence + pipeline_confidence) / 2
    
    # Calculate other metrics
    response_time = round(energy_df['rolling_mean_3'].mean() / 60, 1) if 'rolling_mean_3' in energy_df.columns else 4.3
    monitored_zones = len(energy_df['id'].unique()) if 'id' in energy_df.columns else len(energy_df)
    
    # Create and return metrics dictionary
    metrics = {}
    metrics['active_alerts'] = total_alerts
    metrics['energy_alerts'] = energy_alerts
    metrics['pipeline_alerts'] = pipeline_alerts
    metrics['detection_accuracy'] = round(avg_confidence, 1)
    metrics['response_time'] = response_time
    metrics['monitored_zones'] = monitored_zones
    return metrics

@st.cache_data
def calculate_risk_areas(energy_predictions):
    # Create time-based zones if no zone IDs exist
    periods = pd.qcut(range(len(energy_df)), q=3, labels=['A', 'B', 'C'])
    df_zones = pd.DataFrame({
        'period': periods,
        'anomaly': [1 if x == -1 else 0 for x in energy_predictions]
    })
    
    zone_stats = df_zones.groupby('period').agg(
        anomalies=('anomaly', 'sum'),
        total=('anomaly', 'count')
    ).reset_index()
    
    zone_stats['risk_level'] = zone_stats['anomalies'].apply(
        lambda x: 'High' if x > 5 else 'Medium' if x > 2 else 'Low'
    )
    
    risk_data = {}
    for _, row in zone_stats.iterrows():
        zone_name = f'Zone {row["period"]}-{int(row["total"])%12+1}'
        risk_data[zone_name] = (
            row['risk_level'], 
            int(row['anomalies']),
            0  # placeholder for leaks
        )
    
    return risk_data

def calculate_performance_metrics(energy_predictions, pipeline_predictions):
    # Calculate theft cases detected
    theft_cases = sum(1 for x in energy_predictions if x == -1)
    
    # Calculate pipeline integrity (percentage of normal operations)
    pipeline_integrity = round((1 - sum(1 for x in pipeline_predictions if x == -1) / len(pipeline_predictions)) * 100, 1)
    
    # Calculate energy efficiency
    normal_consumption = energy_df[energy_predictions != -1]['consumption'].mean()
    anomaly_consumption = energy_df[energy_predictions == -1]['consumption'].mean()
    energy_efficiency = round(100 - (anomaly_consumption / normal_consumption * 100), 1) if normal_consumption else 90.0
    
    return theft_cases, pipeline_integrity, energy_efficiency

# Load initial data
energy_df, pipeline_df = load_data()
energy_model, pipeline_model, energy_scaler, pipeline_scaler = load_models()

# Get predictions
energy_predictions, pipeline_predictions, X_energy_scaled, X_pipeline_scaled = get_predictions()

# ---------------------------------------------------
# 4️⃣ Dashboard Configuration
# ---------------------------------------------------
st.set_page_config(page_title="AI-Powered Energy Theft & Leak Detection System", layout="wide")

# Custom CSS for cards
st.markdown('''
<style>
    .card {
        padding: 20px;
        border-radius: 10px;
        background-color: #2b2c36;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-card {
        text-align: center;
        padding: 15px;
        border-radius: 8px;
        background-color: #2b2c36;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
    }
    .metric-label {
        font-size: 14px;
        color: #666;
    }
    .right-sidebar {
        height: 100%;
            width: 70%;
        background-color: #1e1f24;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    }
    .right-sidebar h3 {
        color: #ffffff;
        margin-bottom: 15px;
    }
</style>
''', unsafe_allow_html=True)

# Custom styles for buttons and navigation
st.markdown('''
<style>
    .stButton > button {
        width: 100%;
        background-color: #2b2c36;
        border: none;
        padding: 15px 10px;
        border-radius: 10px;
        color: #ffffff;
        font-size: 16px;
        margin-bottom: 10px;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: flex-start;
    }
    .stButton > button:hover {
        background-color: #3b3c46;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stButton > button:active {
        transform: translateY(0);
    }
    .nav-icon {
        font-size: 20px;
        margin-right: 10px;
    }
    .view-indicator {
        background-color: #1e1f24;
        padding: 10px;
        border-radius: 5px;
        margin-top: 20px;
        border-left: 4px solid #00acee;
    }
</style>
''', unsafe_allow_html=True)

# Initialize session state for view if not exists
if 'current_view' not in st.session_state:
    st.session_state.current_view = "Dashboard"

# Left Sidebar Navigation
with st.sidebar:
    st.title("Spectra")
    st.markdown("---")
    
    # Navigation buttons with icons
    if st.button("📊 Dashboard", key="dash_btn", use_container_width=True):
        st.session_state.current_view = "Dashboard"
    
    if st.button("⚡ Theft Detection", key="theft_btn", use_container_width=True):
        st.session_state.current_view = "Theft Detection"
    
    if st.button("🔍 Leak Monitoring", key="leak_btn", use_container_width=True):
        st.session_state.current_view = "Leak Monitoring"
    
    if st.button("📈 Reports", key="reports_btn", use_container_width=True):
        st.session_state.current_view = "Reports"
    
    if st.button("💬 Community", key="community_btn", use_container_width=True):
        st.session_state.current_view = "Community Chat"
    
    # Show current view indicator with custom styling
    st.markdown("""
    <div class='view-indicator'>
        <small style='color: #666;'>CURRENT VIEW</small><br>
        <strong style='color: #fff;'>{}</strong>
    </div>
    """.format(st.session_state.current_view), unsafe_allow_html=True)
    
    st.markdown("---")

# Use session state instead of radio button value
view = st.session_state.current_view

# Create columns for main content and right sidebar
col1, col2 = st.columns([2, 1])

# Main Content
with col1:
    st.title("AI-Powered Energy Theft & Leak Detection System")
    st.markdown("Real-time insights into electricity theft and pipeline monitoring")

    # ---------------------------------------------------
    # 5️⃣ Dashboard Views
    # ---------------------------------------------------
    if view == "Dashboard":
        # Calculate metrics
        metrics = calculate_metrics(energy_predictions, pipeline_predictions, X_energy_scaled, X_pipeline_scaled)
        
        # Top metrics row
        mc1, mc2, mc3, mc4 = st.columns(4)
        
        with mc1:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{metrics['active_alerts']}</div>
                <div class="metric-label">Active Alerts</div>
                <div style="color: {'red' if metrics['active_alerts'] > 5 else 'green'}; font-size: 12px;">
                    {metrics['energy_alerts']} energy, {metrics['pipeline_alerts']} pipeline
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        with mc2:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{metrics['detection_accuracy']}%</div>
                <div class="metric-label">Detection Accuracy</div>
                <div style="color: {'green' if metrics['detection_accuracy'] >= 90 else 'orange'}; font-size: 12px;">
                    ML model confidence
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        with mc3:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{metrics['response_time']}m</div>
                <div class="metric-label">Response Time</div>
                <div style="font-size: 12px;">average detection time</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with mc4:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{metrics['monitored_zones']}</div>
                <div class="metric-label">Monitored Zones</div>
                <div style="color: green; font-size: 12px;">
                    Active monitoring points
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        # Main content area
        st.markdown("### Monitoring Overview")
        mc1, mc2 = st.columns(2)
        
        with mc1:
            st.markdown('''
            <div class="card">
                <h4>Electricity Consumption Trends</h4>
            ''', unsafe_allow_html=True)
            
            # Plot
            sample = energy_df.sample(min(3000, len(energy_df)))
            sample['anomaly'] = [1 if x == -1 else 0 for x in energy_predictions[:len(sample)]]
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.scatterplot(data=sample, x='date', y='consumption', 
                           hue='anomaly', palette='coolwarm', ax=ax)
            plt.title('Energy Consumption Patterns')
            plt.xlabel('Date')
            plt.ylabel('Consumption')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
        with mc2:
            st.markdown('''
            <div class="card">
                <h4>Risk Areas</h4>
            ''', unsafe_allow_html=True)
            
            # Calculate risk areas from real data
            risk_data = calculate_risk_areas(energy_predictions)
            
            for zone, (risk, thefts, leaks) in risk_data.items():
                st.markdown(f'''
                <div style="margin-bottom: 15px;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>{zone}</span>
                        <span style="color: {'red' if risk == 'High' else 'orange' if risk == 'Medium' else 'green'};">{risk}</span>
                    </div>
                    <div style="font-size: 12px; color: #666;">
                        {thefts} theft{'s' if thefts != 1 else ''} • {leaks} leak{'s' if leaks != 1 else ''}
                    </div>
                </div>
                ''', unsafe_allow_html=True)
        
        # Bottom statistics
        st.markdown("### System Performance")
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        
        # Calculate performance metrics
        theft_cases, pipeline_integrity, energy_efficiency = calculate_performance_metrics(energy_predictions, pipeline_predictions)
        
        with perf_col1:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{theft_cases}</div>
                <div class="metric-label">Theft Cases Detected</div>
                <div style="color: {'red' if theft_cases > 50 else 'green'}; font-size: 12px;">
                    Total cases identified
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        with perf_col2:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{pipeline_integrity}%</div>
                <div class="metric-label">Pipeline Integrity</div>
                <div style="color: {'green' if pipeline_integrity > 95 else 'orange'}; font-size: 12px;">
                    System health
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        with perf_col3:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{energy_efficiency}%</div>
                <div class="metric-label">Energy Efficiency</div>
                <div style="color: {'green' if energy_efficiency > 90 else 'orange'}; font-size: 12px;">
                    Operational efficiency
                </div>
            </div>
            ''', unsafe_allow_html=True)

    elif view == "Theft Detection":
        st.subheader("🔌 Energy Theft Detection")
        
        # Display anomaly details
        st.markdown("### Anomaly Detection Results")
        anomaly_df = pd.DataFrame({
            'Date': energy_df['date'] if 'date' in energy_df.columns else range(len(energy_df)),
            'Consumption': energy_df['consumption'],
            'Anomaly': ['Yes' if x == -1 else 'No' for x in energy_predictions]
        })
        st.dataframe(anomaly_df[anomaly_df['Anomaly'] == 'Yes'].head(10))
        
        # Show feature importance if available
        if hasattr(energy_model, 'feature_importances_'):
            st.markdown("### Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': ['consumption', 'rolling_mean_3', 'rolling_std_3', 'pct_change', 
                           'lag_1', 'lag_2', 'z_score'],
                'Importance': energy_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(data=feature_importance, x='Importance', y='Feature')
            plt.title('Feature Importance in Theft Detection')
            st.pyplot(fig)

    elif view == "Leak Monitoring":
        st.subheader("🛢️ Pipeline Leak Monitoring")
        
        # Display leak detection results
        st.markdown("### Recent Leak Alerts")
        leak_df = pd.DataFrame({
            'Time': range(len(pipeline_df)),
            'Pressure': pipeline_df['pressure_mean_5'],
            'Flow': pipeline_df['flow_mean_5'],
            'Leak Detected': ['Yes' if x == -1 else 'No' for x in pipeline_predictions]
        })
        st.dataframe(leak_df[leak_df['Leak Detected'] == 'Yes'].head(10))
        
        # Pressure vs Flow visualization
        st.markdown("### Pressure vs Flow Analysis")
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['red' if x == -1 else 'blue' for x in pipeline_predictions]
        plt.scatter(pipeline_df['flow_mean_5'], pipeline_df['pressure_mean_5'], c=colors, alpha=0.5)
        plt.xlabel('Flow Rate')
        plt.ylabel('Pressure')
        plt.title('Pipeline Behavior Analysis')
        st.pyplot(fig)

    elif view == "Reports":
        st.subheader("📊 Reports")
        st.markdown("### Summary Statistics")
        
        report_col1, report_col2 = st.columns(2)
        
        with report_col1:
            st.markdown("#### Energy Theft Statistics")
            st.write(f"- Total Cases Detected: {theft_cases}")
            st.write(f"- Detection Accuracy: {metrics['detection_accuracy']}%")
            st.write(f"- Average Response Time: {metrics['response_time']}m")
            st.write(f"- Energy Efficiency: {energy_efficiency}%")
        
        with report_col2:
            st.markdown("#### Pipeline Health")
            st.write(f"- System Integrity: {pipeline_integrity}%")
            st.write(f"- Active Alerts: {metrics['pipeline_alerts']}")
            st.write(f"- Monitored Zones: {metrics['monitored_zones']}")

    elif view == "Community Chat":
        st.subheader("💬 Community Chat")
        st.info("This feature is coming soon! It will allow community members to discuss and share insights about energy theft and pipeline leak detection.")

# Right Sidebar
with col2:
    st.markdown('<div class="right-sidebar">', unsafe_allow_html=True)
    st.markdown("### 🤖 AI-Detected Solutions")
    
    if metrics['active_alerts'] > 0:
        st.error(f"🚨 {metrics['active_alerts']} Active Issues Detected")
        
        # Energy Theft Solutions
        if metrics['energy_alerts'] > 0:
            with st.expander("⚡ Energy Theft Solutions"):
                st.markdown("""
                **Recommended Actions:**
                1. Deploy field inspection team
                2. Install tamper-proof meters
                3. Implement smart meter monitoring
                4. Schedule routine audits
                
                **Prevention Measures:**
                - Regular meter inspection
                - Consumer education programs
                - Advanced metering infrastructure
                """)
        
        # Pipeline Leak Solutions
        if metrics['pipeline_alerts'] > 0:
            with st.expander("🔧 Pipeline Leak Solutions"):
                st.markdown("""
                **Immediate Actions:**
                1. Isolate affected section
                2. Deploy maintenance team
                3. Pressure reduction
                4. Emergency repair planning
                
                **Prevention Strategies:**
                - Regular maintenance schedule
                - Pressure monitoring
                - Corrosion prevention
                - Pipeline integrity testing
                """)
    else:
        st.success("✅ All Systems Normal")
    
    # System Health
    st.markdown("### 📊 System Health")
    # Ensure both metrics are positive before calculating health score
    safe_pipeline_integrity = max(0, min(100, pipeline_integrity))
    safe_energy_efficiency = max(0, min(100, energy_efficiency))
    health_score = (safe_pipeline_integrity + safe_energy_efficiency) / 2
    
    health_color = (
        "🔴 Critical" if health_score < 70 else
        "🟡 Moderate" if health_score < 90 else
        "🟢 Optimal"
    )
    
    st.metric("Overall Health", f"{health_color}")
    # Ensure progress value is between 0 and 1
    st.progress(max(0, min(1, health_score/100)))
    
    # Quick Stats
    st.markdown("### 📈 Quick Stats")
    st.markdown(f"""
    - Detection Accuracy: {metrics['detection_accuracy']}%
    - Response Time: {metrics['response_time']}m
    - Monitored Zones: {metrics['monitored_zones']}
    """)
    
    # Tips and Best Practices
    st.markdown("### 💡 Tips & Best Practices")
    with st.expander("View Tips"):
        st.markdown("""
        1. **Regular Monitoring**
           - Check system alerts daily
           - Review performance metrics weekly
           
        2. **Maintenance Schedule**
           - Monthly meter inspections
           - Quarterly pipeline integrity tests
           
        3. **Emergency Response**
           - Keep emergency contacts updated
           - Follow incident response protocols
           
        4. **Data Analysis**
           - Review historical patterns
           - Track seasonal variations
        """)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# 7️⃣ Footer
# ---------------------------------------------------
st.markdown("---")
st.markdown("Developed by **Constallation Group** — Elton, Shannon, Amelia, Rirhandzu, Nathi, Vinny & Unity")
st.caption("© 2025 AI-Powered Energy & Leak Detection System | Powered by Streamlit")