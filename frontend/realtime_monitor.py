import time
import logging
import psutil
from datetime import datetime
from realtime_api import (
    RealtimeAPI,
    format_timestamp,
    format_event_type,
    get_status_color
)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Get the API client from the realtime_api module
from realtime_api import get_api_client


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize API
# api = RealtimeAPI() # This line is removed as per the edit hint.


def init_session_state():
    """Initializes the session state for the real-time monitor page."""
    session_defaults = {
        "realtime_events": [],
        "realtime_correlations": [],
        "system_metrics": [],
        "websocket_connected": False,
        "performance_data": []
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def update_realtime_data():
    """Fetches and updates real-time data from the API."""
    api_client = get_api_client()
    try:
        events_response = api_client.get_recent_events(limit=100)
        if events_response is not None:
            st.session_state.realtime_events = events_response
            logger.info(f"Updated events: {len(st.session_state.realtime_events)}")
        else:
            st.session_state.realtime_events = []

        correlations_response = api_client.get_recent_correlations(limit=20)
        if correlations_response is not None:
            st.session_state.realtime_correlations = correlations_response
            logger.info(f"Updated correlations: {len(st.session_state.realtime_correlations)}")
        else:
            st.session_state.realtime_correlations = []

        metrics_data = api_client.get_metrics(limit=20)
        if metrics_data:
            if "metrics" in metrics_data:
                st.session_state.system_metrics = metrics_data["metrics"][-20:]
                logger.info(f"Updated metrics history: {len(st.session_state.system_metrics)}")

            if "latest" in metrics_data and metrics_data["latest"]:
                st.session_state.system_metrics.append(metrics_data["latest"])
                if len(st.session_state.system_metrics) > 100:
                    st.session_state.system_metrics = st.session_state.system_metrics[-100:]
                logger.info("Added latest metrics")

    except Exception as e:
        st.error(f"Error updating data: {e}")
        logger.error(f"Error updating real-time data: {e}")


def create_metrics_dashboard():
    """Creates a dashboard with system metrics."""
    st.subheader("System Metrics")
    
    fig = go.Figure()
    
    if st.session_state.system_metrics:
        latest_metrics = st.session_state.system_metrics[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("CPU Usage", f"{latest_metrics.get('cpu_percent', 0):.1f}%")
        col2.metric("Memory Usage", f"{latest_metrics.get('memory_percent', 0):.1f}%")
        col3.metric("Active Connections", latest_metrics.get('active_connections', 0))
        col4.metric("Events Processed", latest_metrics.get('kafka_messages_processed', 0))

        if len(st.session_state.system_metrics) > 1:
            metrics_df = pd.DataFrame(st.session_state.system_metrics)
            metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
            
            fig.add_trace(go.Scatter(
                x=metrics_df['timestamp'], y=metrics_df['cpu_percent'],
                mode='lines+markers', name='CPU %'
            ))
            fig.add_trace(go.Scatter(
                x=metrics_df['timestamp'], y=metrics_df['memory_percent'],
                mode='lines+markers', name='Memory %'
            ))
            if 'disk_usage_percent' in metrics_df.columns:
                fig.add_trace(go.Scatter(
                    x=metrics_df['timestamp'], y=metrics_df['disk_usage_percent'],
                    mode='lines+markers', name='Disk %'
                ))
    else:
        st.info("Metrics will be displayed automatically as they become available.")

    fig.update_layout(
        title="System Metrics Over Time",
        xaxis_title="Time",
        yaxis_title="Percentage (%)",
        height=500,
        hovermode='x unified',
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)


def create_events_timeline():
    """Creates a timeline chart for real-time events."""
    st.subheader("Event Timeline")
    
    events_df = pd.DataFrame(st.session_state.realtime_events)
    if not events_df.empty:
        events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])

    fig = go.Figure()

    event_config = {
        'ERROR': {'color': 'red', 'symbol': 'x', 'size': 12, 'y': 1},
        'WARNING': {'color': 'orange', 'symbol': 'triangle-up', 'size': 10, 'y': 0.5},
        'INFO': {'color': 'blue', 'symbol': 'circle', 'size': 8, 'y': 0}
    }

    if not events_df.empty and 'type' in events_df.columns:
        for event_type, group in events_df.groupby('type'):
            config = event_config.get(event_type, {'color': 'grey', 'symbol': 'circle', 'size': 8, 'y': -0.5})
            fig.add_trace(go.Scatter(
                x=group['timestamp'],
                y=[config['y']] * len(group),
                mode='markers', name=event_type,
                marker=dict(color=config['color'], size=config['size'], symbol=config['symbol']),
                text=group.get('message', ''),
                customdata=group[['type', 'component']],
                hovertemplate='<b>%{text}</b><br>Time: %{x}<br>Type: %{customdata[0]}<br>Component: %{customdata[1]}<extra></extra>'
            ))
    else:
        st.info("Events will be displayed automatically as they appear.")

    fig.update_layout(
        title="Events Over Time",
        xaxis_title="Time",
        yaxis_title="Event Type",
        height=500,
        template="plotly_dark",
        yaxis=dict(
            tickmode='array',
            tickvals=[0, 0.5, 1],
            ticktext=['INFO', 'WARNING', 'ERROR'],
            range=[-1, 1.5]
        ),
        hovermode='closest'
    )
    st.plotly_chart(fig, use_container_width=True)


def create_correlations_chart():
    """Creates a Sankey chart for event correlations."""
    st.subheader("Correlations")
    
    fig = go.Figure()
    
    if st.session_state.realtime_correlations:
        correlations_df = pd.DataFrame(st.session_state.realtime_correlations)
        if not correlations_df.empty:
            fig.add_trace(go.Sankey(
                node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=["Anomalies", "Problems"]),
                link=dict(
                    source=[0] * len(correlations_df),
                    target=[1] * len(correlations_df),
                    value=correlations_df.get('similarity', pd.Series([0])).fillna(0) * 100,
                )
            ))
            st.dataframe(correlations_df)
    else:
        st.info("Correlations will be displayed here as they are detected.")

    fig.update_layout(
        title_text="Correlations Between Anomalies and Problems",
        height=500,
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)


def create_system_health():
    """Displays the health status of system components."""
    st.subheader("System Health")
    api_client = get_api_client()
    health_data = api_client.get_system_health()
    if health_data:
        status = health_data.get('status', 'unknown')
        st.write(f"**Overall Status:** {get_status_color(status)} {status.upper()}")
        
        st.write("**Components:**")
        for component, comp_status in health_data.get('components', {}).items():
            st.write(f"- {component}: {get_status_color(comp_status)} {comp_status}")
    else:
        st.warning("Could not retrieve system health status.")


def main():
    """Main function to render the real-time monitor page."""
    st.title("Real-time Monitor")
    st.markdown("Live monitoring of system logs and metrics.")
    
    init_session_state()
    
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh interval (sec)", 5, 60, 10)
    
    # Fetch data once at the top of the script run
    update_realtime_data()
    
    if auto_refresh:
        st.info(f"Auto-refreshing every {refresh_interval} seconds")
    
    # Dashboard Overview
    st.subheader("Real-time Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Events", len(st.session_state.realtime_events))
    col2.metric("Correlations", len(st.session_state.realtime_correlations))
    col3.metric("Metrics", len(st.session_state.system_metrics))
    col4.metric("Live CPU", f"{psutil.cpu_percent():.1f}%")

    # Tabs for different views
    create_metrics_dashboard()

    # Sidebar Controls
    st.sidebar.header("Controls")
    if st.sidebar.button("Refresh All Data"):
        st.rerun()
    
    # Sidebar Stats
    st.sidebar.header("Statistics")
    st.sidebar.write(f"**Events:** {len(st.session_state.realtime_events)}")
    st.sidebar.write(f"**Correlations:** {len(st.session_state.realtime_correlations)}")
    st.sidebar.write(f"**Metrics:** {len(st.session_state.system_metrics)}")

    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
