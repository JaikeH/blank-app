import streamlit as st
import pandas as pd
import plotly.express as px
from st_aggrid import AgGrid, GridOptionsBuilder
from streamlit_plotly_events import plotly_events
import urllib
import time
from datetime import datetime

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

# --- Email Body Formatting ---
def format_salesperson_opportunities_email(filtered_df):
    body = "Here are the current opportunities for the selected salesperson:\n\n"
    for index, row in filtered_df.iterrows():
        body += (
            f"Client: {row['Account Name']}\n"
            f"Opportunity: {row['Opportunity Name']}\n"
            f"Close Date: {row['Close Date']}\n"
            "------------------------------------------\n"
        )
    return body

# --- Generate Mailto Link for Outlook Email ---
def create_outlook_link_for_salesperson(filtered_df):
    subject = "Opportunities for Selected Salesperson"
    body = format_salesperson_opportunities_email(filtered_df)
    mailto_link = f"mailto:?subject={urllib.parse.quote(subject)}&body={urllib.parse.quote(body)}"
    return mailto_link

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

# --- Dashboard Rendering ---
def render_dashboard(df, metric):
    st.title("📈 Enhanced Sales Opportunity Dashboard")

    # --- Top Dashboard: Counts by Salesperson, Fiscal Period, Client ---
    st.header("🔍 Overview Dashboard")

    # Calculate counts
    count_salesperson = df.groupby('Opportunity Owner').size().reset_index(name='Count')
    count_fiscal = df.groupby('Fiscal Period').size().reset_index(name='Count')
    count_client = df.groupby('Account Name').size().reset_index(name='Count')

    # Top 10 for Salesperson and Client
    count_salesperson_top = count_salesperson.sort_values('Count', ascending=False).head(10)
    count_fiscal_sorted = count_fiscal.sort_values('Fiscal Period')
    count_client_top = count_client.sort_values('Count', ascending=False).head(10)

    # Create three columns for counts
    overview_col1, overview_col2, overview_col3 = st.columns(3)

    with overview_col1:
        st.subheader("👤 Opportunities by Salesperson")
        st.metric(
            label="Total Salespersons",
            value=int(count_salesperson['Opportunity Owner'].nunique()),
            delta=int(count_salesperson['Count'].sum())
        )
        st.table(count_salesperson_top.rename(columns={'Opportunity Owner': 'Salesperson'}))

    with overview_col2:
        st.subheader("📅 Opportunities by Fiscal Period")
        st.metric(
            label="Total Fiscal Periods",
            value=int(count_fiscal['Fiscal Period'].nunique()),
            delta=int(count_fiscal['Count'].sum())
        )
        st.table(count_fiscal_sorted)

    with overview_col3:
        st.subheader("🏢 Opportunities by Client")
        st.metric(
            label="Total Clients",
            value=int(count_client['Account Name'].nunique()),
            delta=int(count_client['Count'].sum())
        )
        st.table(count_client_top.rename(columns={'Account Name': 'Client'}))

    st.markdown("---")

    # --- Revenue by Salesperson Chart ---
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

    # --- Display Related Opportunities Table and Email Button ---
    st.markdown("### 📝 Related Opportunities")
    if selected_salesperson:
        salesperson_name = selected_salesperson[0]['x']
        filtered_df = df[df['Opportunity Owner'] == salesperson_name]

        if not filtered_df.empty:
            gb = GridOptionsBuilder.from_dataframe(filtered_df[['Opportunity Name', 'Account Name', 'Close Date']])
            gb.configure_default_column(editable=False, sortable=True, filter=True)
            gridOptions = gb.build()
            AgGrid(filtered_df[['Opportunity Name', 'Account Name', 'Close Date']], gridOptions=gridOptions, height=300, allow_unsafe_jscode=True)

            # Email button
            st.header("Send Opportunities via Email")
            if st.button("Email Salesperson's Opportunities"):
                outlook_link = create_outlook_link_for_salesperson(filtered_df)
                st.markdown(f"[Click here to open email in Outlook]({outlook_link})", unsafe_allow_html=True)
        else:
            st.write("No opportunities found for the selected Salesperson.")
    else:
        st.write("Click on a bar in the chart to view related opportunities.")

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
