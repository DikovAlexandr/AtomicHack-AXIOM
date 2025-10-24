import os
import re
import io
import requests
from io import BytesIO
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
from datetime import datetime

from plotter import (
    plot_timeline_plotly,
    plot_sankey_diagram,
    plot_timeline_bar_chart
)

st.set_page_config(
    page_title="Log Analyzer",
    page_icon="static/icon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
ANALYZE_URL = f"{API_BASE}/analyze-logs"

# Initialize session state
if "events_df" not in st.session_state:
    st.session_state["events_df"] = pd.DataFrame()
if "res_df" not in st.session_state:
    st.session_state["res_df"] = pd.DataFrame()
if "all_components" not in st.session_state:
    st.session_state["all_components"] = []
if "llm_interpretation_result" not in st.session_state:
    st.session_state["llm_interpretation_result"] = ""

LINE_RE = re.compile(
    r"^(?P<ts>[\d\-T:.]+) (?P<level>INFO|ERROR|WARNING) (?P<component>[^:]+): (?P<message>.+)$"
)


def parse_log_files(
    log_files: List[st.runtime.uploaded_file_manager.UploadedFile]
) -> pd.DataFrame:
    """
    Parse uploaded log files into a pandas DataFrame.

    Args:
        log_files: A list of uploaded log files from Streamlit.

    Returns:
        A pandas DataFrame containing parsed log events, sorted by timestamp.
    """
    events = []
    for file in log_files:
        content = file.getvalue().decode('utf-8')
        filename = file.name
        for i, line in enumerate(content.splitlines(), 1):
            match = LINE_RE.match(line)
            if match:
                data = match.groupdict()
                try:
                    events.append({
                        "timestamp": datetime.fromisoformat(data['ts']),
                        "level": data['level'],
                        "component": data['component'],
                        "message": data['message'],
                        "file": filename,
                        "line": i
                    })
                except ValueError:
                    continue
    if not events:
        return pd.DataFrame()
    return pd.DataFrame(events).sort_values(by="timestamp").reset_index(drop=True)


@st.cache_data
def load_from_api() -> (pd.DataFrame, pd.DataFrame):
    """
    Load events and results data from the backend API.

    Returns:
        A tuple containing two pandas DataFrames: events and results.
    """
    events_url = f"{API_BASE}/events"
    results_url = f"{API_BASE}/results"
    events_df = pd.read_csv(events_url, parse_dates=["timestamp"])
    res_df = pd.read_csv(results_url)
    return events_df, res_df


def send_logs_to_api(
    anomalies_file: st.runtime.uploaded_file_manager.UploadedFile,
    log_files: List[st.runtime.uploaded_file_manager.UploadedFile]
) -> pd.DataFrame:
    """
    Send log files to the backend API for analysis.

    Args:
        anomalies_file: The uploaded anomalies CSV file.
        log_files: A list of uploaded log files.

    Returns:
        A pandas DataFrame with the analysis results.
    """
    files_payload = [
        ("anomalies_file", (anomalies_file.name, anomalies_file.getvalue(), "text/csv"))
    ]
    for log_file in log_files:
        files_payload.append(
            ("log_files", (log_file.name, log_file.getvalue(), "text/plain"))
        )

    try:
        r = requests.post(ANALYZE_URL, files=files_payload, timeout=300)
        r.raise_for_status()
    except requests.exceptions.Timeout as e:
        st.error(f"Error: Backend request timed out (5 minutes). Please check backend logs. {e}")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        st.error(f"Error sending request to backend: {e}")
        if hasattr(e, 'response') and e.response is not None:
            st.error(f"Response status: {e.response.status_code}")
            st.error(f"Response body: {e.response.text}")
        return pd.DataFrame()

    results_csv_content = r.content.decode('utf-8-sig')
    if not results_csv_content:
        return pd.DataFrame()

    return pd.read_csv(io.StringIO(results_csv_content))


def call_llm_interpretation(
    events_df: pd.DataFrame,
    res_df: pd.DataFrame
) -> str:
    """
    Call a Large Language Model for interpreting the analysis results.
    (Placeholder for future implementation)

    Args:
        events_df: DataFrame with log events.
        res_df: DataFrame with analysis results.

    Returns:
        A string with the LLM interpretation.
    """
    return "LLM interpretation feature is not implemented yet."


def display_log_analysis_page():
    """
    Display the main page for log analysis, including file upload and results visualization.
    """
    st.title("Application Log Analyzer")
    st.markdown("Upload logs to analyze errors and anomalies.")

    st.header("Log Upload")
    uploaded_files = st.file_uploader(
        "Select `anomalies_problems.csv` and log files (.txt, .log)",
        type=["csv", "txt", "log"],
        accept_multiple_files=True,
        help="Upload one CSV file and one or more log files."
    )

    anomalies_file = None
    log_files = []
    if uploaded_files:
        csv_files = [f for f in uploaded_files if f.name.endswith('.csv')]
        log_files = [f for f in uploaded_files if not f.name.endswith('.csv')]

        if len(csv_files) == 1:
            anomalies_file = csv_files[0]
            st.success(f"Anomalies file found: `{anomalies_file.name}`")
        elif len(csv_files) > 1:
            st.warning("Please upload only one `anomalies_problems.csv` file.")

        if log_files:
            st.info(f"Found {len(log_files)} log files for analysis.")

    run_clicked = st.button(
        "Analyze",
        use_container_width=True,
        disabled=(not anomalies_file or not log_files)
    )

    if run_clicked:
        with st.spinner("1/2: Parsing logs for visualization..."):
            events_df = parse_log_files(log_files)

        with st.spinner("2/2: Running analysis on the server..."):
            try:
                res_df = send_logs_to_api(anomalies_file, log_files)

                st.session_state["events_df"] = events_df
                st.session_state["res_df"] = res_df
                st.session_state["all_components"] = sorted(
                    events_df["component"].astype(str).unique()) if not events_df.empty else []

                if 'file_name' in res_df.columns and not events_df.empty:
                    merged_df = pd.merge(
                        res_df,
                        events_df[['file', 'line', 'component']],
                        left_on=['file_name', 'line_number'],
                        right_on=['file', 'line'],
                        how='left'
                    ).drop(columns=['file', 'line'])
                    st.session_state["res_df"] = merged_df

                st.success("Analysis complete.")
                st.rerun()

            except requests.HTTPError as e:
                try:
                    detail = e.response.json().get('detail', e.response.text)
                except Exception:
                    detail = e.response.text
                st.error(f"API Error ({e.response.status_code}): {detail}")
            except Exception as e:
                st.error(f"Failed to perform analysis: {e}")

    events_df = st.session_state["events_df"]
    res_df = st.session_state["res_df"]
    all_components = st.session_state["all_components"]

    if not res_df.empty:
        st.header("Identified Causal Links")

        res_display = res_df.rename(columns={
            'scenario_id': 'Scenario ID',
            'anomaly_id': 'Anomaly ID',
            'anomaly_text': 'Anomaly Text',
            'problem_id': 'Problem ID',
            'problem_text': 'Problem Text',
            'file_name': 'File',
            'line_number': 'Line',
            'log': 'Log Entry',
        })

        st.dataframe(
            res_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Scenario ID": st.column_config.NumberColumn(format="%d"),
                "Anomaly ID": st.column_config.NumberColumn(format="%d"),
                "Anomaly Text": st.column_config.TextColumn(width="large"),
                "Problem ID": st.column_config.NumberColumn(format="%d"),
                "Problem Text": st.column_config.TextColumn(width="large"),
                "Line": st.column_config.NumberColumn(format="%d"),
                "Log Entry": st.column_config.TextColumn(width="large"),
            }
        )

        col1, col2 = st.columns(2)
        with col1:
            csv_export = res_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="Export to CSV",
                data=csv_export,
                file_name="correlation_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col2:
            excel_buf = BytesIO()
            with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
                res_df.to_excel(writer, index=False, sheet_name="results")
            excel_buf.seek(0)
            st.download_button(
                label="Export to Excel",
                data=excel_buf,
                file_name="correlation_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        st.subheader("Sankey Diagram (Anomaly, Problem, and Component Flows)")
        if 'component' in res_df.columns:
            sankey_fig = plot_sankey_diagram(res_df)
            st.plotly_chart(sankey_fig, use_container_width=True)

        with st.expander("Interpreting the Sankey Diagram"):
            st.markdown("""
                **How to read the Sankey diagram:**
                - The diagram shows flows from **Anomalies** to **Problems** and then to **Components**.
                - Each column is an entity: Anomalies, Problems, Components.
                - The **width of the flows** is proportional to the number of related events.
                - The **color of the flows** corresponds to the related Problem.
            """)

        st.header("Event Timeline")
        plotter_res_df = res_df.rename(columns={
            'file_name': 'problem_file',
            'line_number': 'problem_line'
        })

        fig = plot_timeline_plotly(
            events_df,
            plotter_res_df,
            all_components=all_components
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Interpreting the Timeline"):
            st.markdown("""
                **How to read the timeline:**
                - Horizontal black lines are tracks for services/components.
                - Squares **above** the line are **ERROR** events, circles **below** are **WARNING** events.
                - Curves connect related **ERROR → WARNING** events.
            """)

        if st.button("Get LLM Interpretation", use_container_width=True):
            with st.spinner("Generating interpretation..."):
                st.session_state["llm_interpretation_result"] = call_llm_interpretation(events_df, res_df)

        if st.session_state["llm_interpretation_result"]:
            with st.expander("LLM Interpretation of Results", expanded=True):
                st.markdown(st.session_state["llm_interpretation_result"])

        st.header("Event Frequency Chart")
        time_grain_option = st.selectbox(
            "Select time granularity for the frequency chart",
            options=["h", "D", "W", "M"],
            format_func=lambda x: {"h": "Hourly", "D": "Daily", "W": "Weekly", "M": "Monthly"}[x]
        )

        filtered_events_df = events_df[events_df['level'].isin(['ERROR', 'WARNING'])]
        bar_chart_fig = plot_timeline_bar_chart(filtered_events_df, time_grain=time_grain_option)
        st.plotly_chart(bar_chart_fig, use_container_width=True)

    elif not events_df.empty and res_df.empty:
        st.info("Logs loaded, but no correlations found. Consider changing filters or checking data.")

    with st.expander("About the App"):
        st.markdown("""
            ### How to Use:
            1. **Upload log files** - .txt and .log formats are supported.
            2. **Click the analyze button** - files will be sent for processing.
            3. **View the results** - a table with detected problems will be displayed.
        """)


def main():
    """
    Main function to run the Streamlit application.
    """
    st.sidebar.title("🔍 Log Analyzer")
    st.sidebar.markdown("---")

    page = st.sidebar.selectbox(
        "Select a page:",
        ["📊 Log Analysis", "🔍 Real-time Monitor"]
    )

    st.sidebar.markdown("---")
    st.sidebar.header("🔌 Connection")
    st.sidebar.write(f"**API:** {API_BASE}")
    st.sidebar.write(f"**Status:** {'🟢 Connected' if API_BASE else '🔴 Disconnected'}")

    if page == "📊 Log Analysis":
        display_log_analysis_page()
    elif page == "🔍 Real-time Monitor":
        try:
            from realtime_monitor import main as realtime_main
            realtime_main()
        except ImportError as e:
            st.error(f"Error importing real-time monitor: {e}")
            st.info("Please ensure `realtime_monitor.py` exists.")
        except Exception as e:
            st.error(f"Error running real-time monitor: {e}")


if __name__ == "__main__":
    main()