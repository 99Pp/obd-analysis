import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Function to load and cache the first segment from CSV
@st.cache_data
def load_segment():
    data = pd.read_csv('first_continuous_segment.csv')

    data.index = pd.to_timedelta(data.index * 100, unit='ms')
    data['Acceleration'] = data['Vehicle Speed Sensor [km/h]'].diff().fillna(0)
    data['Throttle Variability'] = data['Absolute Throttle Position [%]'].rolling(window=5).std().fillna(0)
    data['Estimated Engine Load'] = (data['Intake Manifold Absolute Pressure [kPa]'] * data['Engine RPM [RPM]']) / 1000
    data = data.drop(['Time','Date'],axis =1)
    print(data.dtypes)
    return data

def main():
    st.title("Comprehensive Vehicle Analysis")

    data = load_segment()

    # Sidebar options for feature visualization
    st.sidebar.header("Visualization Options")
    selected_feature = st.sidebar.selectbox("Select feature to visualize", data.columns)

    # Time-series plot
    st.header(f"Time Series Plot of {selected_feature}")
    plt.figure(figsize=(10, 5))
    plt.plot(data.index.total_seconds(), data[selected_feature], label=selected_feature)
    plt.xlabel('Time (seconds)')
    plt.ylabel(selected_feature)
    plt.title(f'{selected_feature} Over Time')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    # st.header("Correlation Matrix")
    # corr_matrix = data.corr()
    # sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    # st.pyplot(plt)

    # 3D Cluster Visualization
    st.header("3D Cluster Map")
    features = ['Vehicle Speed Sensor [km/h]', 'Acceleration', 'Throttle Variability']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[features].dropna())

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(scaled_features[:, 0], scaled_features[:, 1], scaled_features[:, 2], c=clusters, cmap='viridis', alpha=0.6)
    ax.set_xlabel('Vehicle Speed [km/h]')
    ax.set_ylabel('Acceleration')
    ax.set_zlabel('Throttle Variability')
    ax.set_title('3D Cluster Map')
    st.pyplot(plt)

    # Sustainability and Engine Health Insights
    st.header("Sustainability and Engine Health")
    st.write("### Estimated Engine Load Over Time")
    plt.figure(figsize=(10, 5))
    plt.plot(data.index.total_seconds(), data['Estimated Engine Load'], label='Estimated Engine Load', color='green')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Engine Load')
    plt.title('Estimated Engine Load Over Time')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    st.header("Driver Behavior and Environmental Impact")
    st.write("""
    - **Driver Behavior**: Can be Categorized into 3 Buckets Aggressive Normal and Highway Speed Cruncher.
    - **Pollution and Sustainability**: MAx Pollution in Aggressive and Highway Speed Crunching Mode.
    - **Engine Health**: High Revving and more than normal acceleration is damaging Engine.
    """)

if __name__ == "__main__":
    main()