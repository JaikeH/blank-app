import streamlit as st
import pandas as pd
import plotly.express as px
from st_aggrid import AgGrid, GridOptionsBuilder
from streamlit_plotly_events import plotly_events
import urllib
import time
from datetime import datetime
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from PIL import Image

# --- Set Streamlit Page Configuration ---
st.set_page_config(page_title="📈 Enhanced Sales Opportunity Dashboard", layout="wide")

# --- Initialize Session State ---
if 'selection_type' not in st.session_state:
    st.session_state['selection_type'] = None
if 'selection_value' not in st.session_state:
    st.session_state['selection_value'] = None

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_and_preprocess_data(uploaded_files):
    data_frames = []
    for f in uploaded_files:
        try:
            if f.name.endswith('.xlsx'):
                xls = pd.ExcelFile(f)
                df = pd.read_excel(xls)
            elif f.name.endswith('.csv'):
                df = pd.read_csv(f, encoding='utf-8', encoding_errors='replace')
            else:
                st.error(f"Unsupported file format: {f.name}. Please upload an Excel (.xlsx) or CSV (.csv) file.")
                continue

            required_columns = [
                'Created Date', 'Close Date', 'Expected Revenue', 'Amount',
                'Probability (%)', 'Fiscal Period', 'Stage', 'Account Name',
                'Opportunity Owner', 'Opportunity Name', 'Age'
            ]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"Missing columns {missing_columns} in {f.name}. Skipping this file.")
                continue

            # Data processing
            df['Created Date'] = pd.to_datetime(df['Created Date'], errors='coerce')
            df['Close Date'] = pd.to_datetime(df['Close Date'], errors='coerce')
            df['Probability (%)'] = pd.to_numeric(df['Probability (%)'], errors='coerce') / 100
            df['Likely Revenue'] = df['Expected Revenue'] * df['Probability (%)']
            df['File Name'] = f.name
            data_frames.append(df)
        except Exception as e:
            st.error(f"Error loading {f.name}: {e}")
            continue

    if not data_frames:
        return None
    return pd.concat(data_frames, ignore_index=True)

# --- Charting Functions ---
def create_bar_chart(df, x_col, y_col, title, labels, color_col=None, orientation='v'):
    if color_col:
        color_sequence = px.colors.qualitative.Plotly
        fig = px.bar(
            df,
            x=x_col,
            y=y_col,
            orientation=orientation,
            title=title,
            labels=labels,
            color=color_col,
            color_discrete_sequence=color_sequence
        )
    else:
        fig = px.bar(
            df,
            x=x_col,
            y=y_col,
            orientation=orientation,
            title=title,
            labels=labels,
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
    fig.update_layout(showlegend=False, clickmode='event+select')
    fig.update_traces(marker_line_width=1.5, opacity=0.8)
    return fig

# --- High-Level Dashboard Summary ---
def create_summary_dashboard(df):
    st.header("🔍 High-Level Metrics Summary")

    # Calculate metrics
    total_amount = df['Amount'].sum()
    total_expected_revenue = df['Expected Revenue'].sum()
    total_likely_revenue = df['Likely Revenue'].sum()
    avg_probability = df['Probability (%)'].mean() * 100
    num_opportunities = len(df)

    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Amount", f"${total_amount:,.0f}")
    col2.metric("Expected Revenue", f"${total_expected_revenue:,.0f}")
    col3.metric("Likely Revenue", f"${total_likely_revenue:,.0f}")
    col4.metric("Average Probability", f"{avg_probability:.1f}%")
    st.markdown("---")

# --- Dashboard Rendering ---
def render_dashboard(df, metric):
    st.title("📈 Enhanced Sales Opportunity Dashboard")

    # Display High-Level Summary
    create_summary_dashboard(df)

    # --- Overview Dashboard by Counts ---
    st.header("🔍 Opportunities Overview")

    # Aggregate counts
    count_salesperson = df.groupby('Opportunity Owner').size().reset_index(name='Count')
    count_fiscal = df.groupby('Fiscal Period').size().reset_index(name='Count')
    count_client = df.groupby('Account Name').size().reset_index(name='Count').sort_values(by='Count', ascending=False)

    # Calculate opportunities closing in the next 30 days
    thirty_days_from_now = datetime.now() + pd.DateOffset(days=30)
    count_closing_soon = df[df['Close Date'] <= thirty_days_from_now].groupby('Opportunity Name').size().reset_index(name='Count')

    # Display top entries for each category
    overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
    
    with overview_col1:
           st.markdown("<h5>👤 Opportunities by Salesperson</h5>", unsafe_allow_html=True)
           st.dataframe(count_salesperson.rename(columns={'Opportunity Owner': 'Salesperson'}).set_index('Salesperson'), width=250, height=500)
   
    with overview_col2:
        st.markdown("<h5>📅 Opportunities by Fiscal Period</h5>", unsafe_allow_html=True)
        st.dataframe(count_fiscal.set_index('Fiscal Period'), width=250, height=500)

    with overview_col3:
        st.markdown("<h5>🏢 Opportunities by Client</h5>", unsafe_allow_html=True)
        styled_client_df = count_client.rename(columns={'Account Name': 'Client'}).set_index('Client')
        st.dataframe(styled_client_df, width=400, height=500)

    with overview_col4:
        st.markdown("<h5>⏳ Closing in 30 Days</h5>", unsafe_allow_html=True)
        st.dataframe(count_closing_soon.set_index('Opportunity Name'), width=250, height=500)
   
    st.markdown("---")

    # --- Revenue by Fiscal Period Chart ---
    st.subheader("💰 Revenue by Fiscal Period")
    revenue_fiscal = df.groupby('Fiscal Period')[metric].sum().reset_index()
    revenue_fiscal['Fiscal Period'] = pd.Categorical(
        revenue_fiscal['Fiscal Period'],
        categories=sorted(df['Fiscal Period'].dropna().unique(), key=lambda x: pd.to_datetime(x, errors='ignore')),
        ordered=True
    )
    revenue_fiscal = revenue_fiscal.sort_values('Fiscal Period')
    fig_revenue_fiscal = create_bar_chart(
        revenue_fiscal,
        x_col='Fiscal Period',
        y_col=metric,
        title=f"Revenue by Fiscal Period ({metric})",
        labels={metric: metric, 'Fiscal Period': 'Fiscal Period'},
        color_col='Fiscal Period'
    )
    st.plotly_chart(fig_revenue_fiscal, use_container_width=True)

    st.markdown("---")

    # --- Revenue by Salesperson Chart and Email Feature ---
    st.subheader("👤 Revenue by Salesperson")
    revenue_salesperson = df.groupby('Opportunity Owner')[metric].sum().reset_index()
    fig_revenue_salesperson = create_bar_chart(
        revenue_salesperson,
        x_col='Opportunity Owner',
        y_col=metric,
        title=f"Revenue by Salesperson ({metric})",
        labels={'Opportunity Owner': 'Salesperson', metric: metric},
        color_col='Opportunity Owner'
    )
    selected_salesperson = plotly_events(fig_revenue_salesperson, click_event=True)

    st.markdown("### 📝 Related Opportunities")
    if selected_salesperson:
        salesperson_name = selected_salesperson[0]['x']
        filtered_df = df[df['Opportunity Owner'] == salesperson_name]

        if not filtered_df.empty:
            # Show table with all columns for detailed view
            gb = GridOptionsBuilder.from_dataframe(filtered_df)
            gb.configure_default_column(editable=False, sortable=True, filter=True)
            gridOptions = gb.build()
            AgGrid(filtered_df, gridOptions=gridOptions, height=300, allow_unsafe_jscode=True)

        else:
            st.write("No opportunities found for the selected Salesperson.")
    else:
        st.write("Click on a bar in the chart to view related opportunities.")

    st.markdown("---")

    # --- Total Expected Revenue by Client Chart ---
    st.subheader("🏆 Total Expected Revenue by Client (Top 20)")
    top_clients = (
        df.groupby('Account Name')[metric]
        .sum()
        .reset_index()
        .sort_values(by=metric, ascending=False)
        .head(20)
    )
    fig_top_clients = create_bar_chart(
        top_clients,
        x_col='Account Name',
        y_col=metric,
        title=f"Total {metric} by Client",
        labels={'Account Name': 'Client', metric: metric},
        color_col='Account Name'
    )
    st.plotly_chart(fig_top_clients, use_container_width=True)

    st.markdown("---")

# --- Main App ---
def main():
    st.sidebar.title("📁 File Upload & Filters")
    uploaded_files = st.sidebar.file_uploader(
        "Upload one or more Excel (.xlsx) or CSV (.csv) files",
        type=["xlsx", "csv"],
        accept_multiple_files=True,
        key='file_uploader_1'
    )

    if uploaded_files:
        with st.spinner("⏳ Loading and processing data..."):
            start_time = time.time()
            df = load_and_preprocess_data(uploaded_files)
            if df is not None and not df.empty:
                st.success(f"✅ Data loaded successfully in {int(time.time() - start_time)} seconds!")

                # --- Sidebar Filters ---
                st.sidebar.header("🔧 Filters")
                st.sidebar.header("🔍 Select Metric")
                available_metrics = ['Expected Revenue', 'Amount', 'Likely Revenue']
                metric = st.sidebar.selectbox("Choose a metric for analysis:", available_metrics, index=0)

                fiscal_periods = sorted(df['Fiscal Period'].dropna().unique())
                selected_fiscal_period = st.sidebar.multiselect("Fiscal Period", fiscal_periods, default=fiscal_periods)

                stages = sorted(df['Stage'].dropna().unique())
                selected_stage = st.sidebar.multiselect("Stage", stages, default=stages)

                min_close_date = df['Close Date'].min().date()
                max_close_date = df['Close Date'].max().date()
                selected_date_range = st.sidebar.date_input(
                    "Select Close Date Range",
                    [min_close_date, max_close_date],
                    min_value=min_close_date,
                    max_value=max_close_date
                )

                min_metric = float(df[metric].min())
                max_metric = float(df[metric].max())
                selected_metric_range = st.sidebar.slider(
                    f"{metric} Range",
                    min_value=min_metric,
                    max_value=max_metric,
                    value=(min_metric, max_metric)
                )

                # --- Apply Filters ---
                filtered_df = df[
                    df['Fiscal Period'].isin(selected_fiscal_period) &
                    df['Stage'].isin(selected_stage) &
                    (df['Close Date'] >= pd.to_datetime(selected_date_range[0])) &
                    (df['Close Date'] <= pd.to_datetime(selected_date_range[1])) &
                    (df[metric] >= selected_metric_range[0]) &
                    (df[metric] <= selected_metric_range[1])
                ]
                filtered_df = filtered_df[filtered_df['Stage'] != 'Won']

                render_dashboard(filtered_df, metric)
            else:
                st.error("❌ No data available after processing. Please check your uploaded files.")
    else:
        st.info("📂 Please upload one or more Excel files to get started.")

if __name__ == "__main__":
    main()
