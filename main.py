import streamlit as st
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
from datetime import datetime, date
import streamlit_shadcn_ui as st_shadcn_ui
import gspread
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from google.oauth2.service_account import Credentials
import numpy as np
from streamlit_option_menu import option_menu

# --- Configuration and Constants ---
st.set_page_config(layout="wide", page_title="Pro Net Worth Tracker", initial_sidebar_state="expanded")

# --- CSS to hide the top header space ---
st.markdown("""
<style>
    header { visibility: hidden; }
    .stApp { background-color: #F0F2F6; }
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1, h2, h3 { font-weight: 600; color: #1E293B; }
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        border: 1px solid #E2E8F0; border-radius: 0.5rem; padding: 1rem 1.5rem 1.5rem 1.5rem;
        background-color: white; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- App Configuration from Secrets ---
WORKSHEET_NAME = "Data"
SPECIAL_ROWS = ['TOTAL', 'Liabilities', 'GRAND TOTAL', 'Increase']
CURRENCY_SYMBOL = "₹"
TEMPLATE_URL = st.secrets.google_sheets.template_url
YOUR_SHEET_URL = st.secrets.google_sheets.sheet_url


# --- Helper function for Indian Numbering System ---
def format_inr(num):
    if not isinstance(num, (int, float)): return num
    num = int(num)
    s = str(num)
    if num < 1000: return s
    last_three = s[-3:]
    rest = s[:-3]
    formatted_rest = re.sub(r'(\d)(?=(\d{2})+(?!\d))', r'\1,', rest)
    return f"{formatted_rest},{last_three}"


# --- Core Data Handling Functions (for multi-user support) ---
@st.cache_data
def get_sheet_id_from_url():
    # Use st.experimental_get_query_params() for robustness on Streamlit Cloud
    params = st.experimental_get_query_params()
    if "sheet_id" in params and params["sheet_id"]:
        # The result is a list, so we take the first element
        return params["sheet_id"][0]
    return None

@st.cache_resource(ttl=600)
def connect_to_gsheet(sheet_id):
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
    client = gspread.authorize(creds)
    try:
        spreadsheet = client.open_by_key(sheet_id)
        return spreadsheet
    except gspread.exceptions.APIError:
        st.error(f"Error accessing Google Sheet. Please check the Sheet ID and make sure you have shared it with the service account email: `{st.secrets.gcp_service_account.client_email}`", icon="🚨")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}", icon="🚨")
        return None

@st.cache_data(ttl=60)
def load_data(sheet_id):
    spreadsheet = connect_to_gsheet(sheet_id)
    if spreadsheet is None: st.stop()
    try:
        worksheet = spreadsheet.worksheet(WORKSHEET_NAME)
        df = get_as_dataframe(worksheet, evaluate_formulas=True, header=0)
        df.dropna(how='all', inplace=True)
        df['ID'] = df['ID'].astype(str)
        date_cols = df.columns[4:]
        for col in date_cols: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        id_cols = ['ID', 'Name', 'Mode', 'Category']
        sorted_dates = pd.to_datetime(date_cols).sort_values().strftime('%Y-%m-%d').tolist()
        return df[id_cols + sorted_dates]
    except gspread.exceptions.WorksheetNotFound:
        st.error(f"Worksheet '{WORKSHEET_NAME}' not found. Please ensure your sheet has a tab with this exact name and that you've given Editor permissions to the app's service email.", icon="🚨")
        st.stop()

def save_data(df, sheet_id):
    spreadsheet = connect_to_gsheet(sheet_id)
    if spreadsheet is None: st.stop()
    try:
        worksheet = spreadsheet.worksheet(WORKSHEET_NAME)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title=WORKSHEET_NAME, rows=100, cols=50)
    df.fillna('', inplace=True)
    set_with_dataframe(worksheet, df)
    st.cache_data.clear()
    st.cache_resource.clear()


# --- Page Rendering Functions ---

def render_welcome_page():
    st.title("👋 Welcome to the Pro Net Worth Tracker!")
    st.markdown("This app helps you track your net worth using your own private Google Sheet.")
    
    with st.container(border=True):
        st.header("1️⃣ Create Your Data Sheet")
        st.markdown("Click the button below to make a private copy of the data template in your own Google Drive.")
        st.link_button("📋 Make a Copy of the Template Sheet", TEMPLATE_URL, type="primary")

    with st.container(border=True):
        st.header("2️⃣ Share Your New Sheet with the App")
        st.markdown(f"""
        In your new sheet, click the **Share** button (top right) and invite the email address below as an **Editor**. 
        This allows the app to read and write your data. It cannot access any of your other files.
        """)
        st.code(st.secrets.gcp_service_account.client_email, language=None)

    with st.container(border=True):
        st.header("3️⃣ Get Your Personalized App Link")
        st.markdown("""
        - Look at the URL of your new sheet. It will be like: `.../spreadsheets/d/`**`1aBcDeFgHiJkLmNoPqRsTuVwXyZ...`**`/edit`
        - Copy that long string of characters in the middle (your **Sheet ID**).
        - Paste it below to generate your permanent, private link to your dashboard.
        """)
        user_sheet_id = st.text_input("Paste your Google Sheet ID here:", key="user_sheet_id_input")
        if user_sheet_id:
            st.success("✅ Success! Here is your permanent link. **Bookmark it!**")
            app_url = f"/?sheet_id={user_sheet_id}"
            st.link_button("Go to My Dashboard", app_url, use_container_width=True)

def render_dashboard(df):
    df = df.copy()
    st.title("🏠 Dashboard")
    st.markdown("Your financial command center. Set goals and track your progress.")

    date_cols = df.columns[4:]
    if len(date_cols) < 1:
        st.info("No data available. Please add data via the 'Update Data' or 'Manage Data' pages.")
        return

    df_meta = df[df['Name'].isin(SPECIAL_ROWS)].drop_duplicates(subset=['Name']).set_index('Name')
    latest_date = date_cols[-1]
    latest_grand_total = df_meta.loc['GRAND TOTAL', latest_date].item()
    latest_total_assets = df_meta.loc['TOTAL', latest_date].item()
    latest_liabilities = df_meta.loc['Liabilities', latest_date].item()
    
    latest_increase, cagr, required_monthly_increase = 0, 0.0, 0
    debt_to_asset_ratio = (latest_liabilities / latest_total_assets) if latest_total_assets > 0 else 0
    
    if len(date_cols) >= 2:
        prev_date = date_cols[-2]
        latest_increase = latest_grand_total - df_meta.loc['GRAND TOTAL', prev_date].item()
        grand_total_series = df_meta.loc['GRAND TOTAL', date_cols]
        first_non_zero = grand_total_series[grand_total_series > 0]
        if len(first_non_zero) > 1:
            start_date_pd, end_date_pd = pd.to_datetime(first_non_zero.index[0]), pd.to_datetime(first_non_zero.index[-1])
            num_years = (end_date_pd - start_date_pd).days / 365.25
            if num_years > 0: cagr = ((first_non_zero.iloc[-1] / first_non_zero.iloc[0]) ** (1 / num_years)) - 1
            
    months_remaining = (st.session_state.target_date.year - date.today().year) * 12 + st.session_state.target_date.month - date.today().month
    if months_remaining > 0:
        required_monthly_increase = (st.session_state.target_amount - latest_grand_total) / months_remaining

    with st.container(border=True):
        st.subheader("Financial Snapshot")
        cols = st.columns(5)
        with cols[0]:
            st_shadcn_ui.metric_card(title="Net Worth", content=f"{CURRENCY_SYMBOL}{format_inr(latest_grand_total)}", description=f"{CURRENCY_SYMBOL}{format_inr(latest_increase)} from last month" if len(date_cols) >=2 else "N/A")
        with cols[1]:
            st_shadcn_ui.metric_card(title="Total Assets", content=f"{CURRENCY_SYMBOL}{format_inr(latest_total_assets)}", description="All your investments")
        with cols[2]:
            st_shadcn_ui.metric_card(title="Total Liabilities", content=f"{CURRENCY_SYMBOL}{format_inr(latest_liabilities)}", description="Total outstanding debt")
        with cols[3]:
            st_shadcn_ui.metric_card(title="Portfolio CAGR", content=f"{cagr:.2%}", description="Annualized Growth" if cagr > 0 else "N/A")
        with cols[4]:
            st_shadcn_ui.metric_card(title="Debt/Asset Ratio", content=f"{debt_to_asset_ratio:.2%}", description="Financial leverage")

    with st.container(border=True):
        st.subheader("🎯 Goal Progress")
        progress = (latest_grand_total / st.session_state.target_amount) * 100 if st.session_state.target_amount > 0 else 0
        display_progress = min(100, int(progress))
        st.progress(display_progress, text=f"Progress: {progress:.2f}%")
        
        cols = st.columns([2, 3])
        with cols[0]:
            st.info(f"""
            **Target:** {CURRENCY_SYMBOL}{format_inr(st.session_state.target_amount)} by {st.session_state.target_date.strftime('%b %Y')}
            **Remaining:** {CURRENCY_SYMBOL}{format_inr(st.session_state.target_amount - latest_grand_total)}
            """)
        with cols[1]:
            if progress < 100 and required_monthly_increase > 0:
                st.success(f"**Required Monthly Growth:** {CURRENCY_SYMBOL}{format_inr(required_monthly_increase)}")
            else:
                st.success("**Goal Reached! Time to set a new one!** 🎉")

    df_assets = df[~df['Name'].isin(SPECIAL_ROWS)]
    df_dates = pd.to_datetime(date_cols)

    with st.container(border=True):
        st.subheader("📈 Net Worth Trajectory")
        grand_total_series = df_meta.loc['GRAND TOTAL', date_cols]
        fig_line = px.line(grand_total_series.reset_index(), x='index', y='GRAND TOTAL', markers=True, labels={'index': 'Date', 'GRAND TOTAL': f'Net Worth ({CURRENCY_SYMBOL})'})
        fig_line.update_traces(name='Net Worth', showlegend=True)
        if len(date_cols) >= 2:
            x_numeric = np.arange(len(df_dates))
            y_values = pd.to_numeric(grand_total_series, errors='coerce').fillna(0)
            if len(y_values[y_values > 0]) > 1:
                coeffs = np.polyfit(x_numeric, y_values, 1)
                trendline = coeffs[0] * x_numeric + coeffs[1]
                fig_line.add_trace(go.Scatter(x=df_dates, y=trendline, mode='lines', name='Trendline', line=dict(dash='dot', color='orange')))
        st.plotly_chart(fig_line, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1, st.container(border=True):
        st.subheader("📊 Asset Allocation by Category")
        allocation_data = df_assets[['Category', latest_date]].groupby('Category')[latest_date].sum().reset_index()
        fig_treemap = px.treemap(allocation_data[allocation_data[latest_date] > 0], path=[px.Constant("All Assets"), 'Category'], values=latest_date, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_treemap.update_layout(margin = dict(t=25, l=25, r=25, b=25))
        st.plotly_chart(fig_treemap, use_container_width=True)
        
    with col2, st.container(border=True):
        st.subheader("🔬 Holdings Breakdown")
        sunburst_df = df_assets[['Category', 'Name', latest_date]].copy()
        sunburst_df = sunburst_df[sunburst_df[latest_date] > 0]
        fig_sunburst = px.sunburst(sunburst_df, path=['Category', 'Name'], values=latest_date, color='Category', color_discrete_sequence=px.colors.qualitative.Pastel, hover_data={'Category':False, 'Name':False})
        fig_sunburst.update_traces(textinfo='label+percent parent', hovertemplate='<b>%{label}</b><br>Value: ' + CURRENCY_SYMBOL + '%{value:,.0f}<extra></extra>')
        fig_sunburst.update_layout(margin=dict(t=25, l=25, r=25, b=25))
        st.plotly_chart(fig_sunburst, use_container_width=True)
    
    with st.container(border=True):
        st.subheader("⏳ Asset Growth by Category")
        if len(date_cols) >= 2:
            df_melted = df_assets.melt(id_vars=['Category'], value_vars=date_cols, var_name='Date', value_name='Value')
            composition_data = df_melted.groupby(['Date', 'Category'])['Value'].sum().reset_index()
            fig_area = px.area(composition_data, x='Date', y='Value', color='Category', labels={'Value': f'Asset Value ({CURRENCY_SYMBOL})'})
            st.plotly_chart(fig_area, use_container_width=True)
        else:
            st.info("Area chart requires at least two months of data.")

    with st.expander("View & Export Raw Data"):
        st.dataframe(df)
        
def render_update_page(df, sheet_id):
    df = df.copy()
    st.title("✍️ Update Monthly Data")
    st.markdown("Enter values for a new month. Fields are pre-filled with the last known values for convenience.")

    date_cols = df.columns[4:]
    if not date_cols.any():
        st.warning("No data found. Please add an asset on the 'Manage Data' page to start.")
        return

    latest_date_dt = pd.to_datetime(date_cols).max()
    default_new_date = (latest_date_dt.to_pydatetime().date() + pd.DateOffset(months=1)).date()

    with st.form("update_data_form"):
        with st.container(border=True):
            st.subheader("1. Select New Month")
            new_date = st.date_input("Date for this update", value=default_new_date, min_value=latest_date_dt.date())

        with st.container(border=True):
            st.subheader("2. Enter Asset & Liability Values")
            df_assets = df[~df['Name'].isin(SPECIAL_ROWS)]
            user_inputs = {}
            for category in sorted(df_assets['Category'].unique()):
                st.markdown(f"**{category}**")
                assets_in_cat = df_assets[df_assets['Category'] == category]
                for _, row in assets_in_cat.iterrows():
                    user_inputs[row['ID']] = st.number_input(
                        label=row['Name'],
                        value=float(row[latest_date_dt.strftime('%Y-%m-%d')]),
                        step=1000.0, format="%.2f", key=f"asset_{row['ID']}"
                    )
            st.markdown("**Liabilities**")
            last_liabilities = float(df.loc[df['Name'] == 'Liabilities', latest_date_dt.strftime('%Y-%m-%d')].iloc[0])
            liabilities_input = st.number_input("Total Liabilities", value=last_liabilities, step=100.0, format="%.2f")

        submitted = st.form_submit_button("💾 Save Monthly Data", use_container_width=True, type="primary")

    if submitted:
        new_date_str = new_date.strftime('%Y-%m-%d')
        if new_date_str in df.columns:
            st.error(f"Data for {new_date_str} already exists.")
        else:
            with st.spinner("Saving data to Google Sheets..."):
                df_updated = df.copy()
                df_updated[new_date_str] = 0.0
                for asset_id, value in user_inputs.items():
                    df_updated.loc[df_updated['ID'] == str(asset_id), new_date_str] = value
                total_assets_new = df_updated.loc[~df_updated['Name'].isin(SPECIAL_ROWS), new_date_str].sum()
                grand_total_previous = df.loc[df['Name'] == 'GRAND TOTAL', latest_date_dt.strftime('%Y-%m-%d')].iloc[0]
                df_updated.loc[df['Name'] == 'Liabilities', new_date_str] = liabilities_input
                df_updated.loc[df['Name'] == 'TOTAL', new_date_str] = total_assets_new
                df_updated.loc[df['Name'] == 'GRAND TOTAL', new_date_str] = total_assets_new - liabilities_input
                df_updated.loc[df['Name'] == 'Increase', new_date_str] = (total_assets_new - liabilities_input) - grand_total_previous
                id_cols = ['ID', 'Name', 'Mode', 'Category']
                all_date_cols = sorted(list(df_updated.columns[4:]))
                df_to_save = df_updated[id_cols + all_date_cols]
                save_data(df_to_save, sheet_id)
            st.success(f"Successfully saved data for {new_date_str}!")
            st.balloons()
            st.rerun()

def render_manage_page(df, sheet_id):
    df = df.copy()
    st.title("⚙️ Manage Data")
    st.markdown("Add or remove assets from your portfolio.")
    
    st.info("ℹ️ To edit historical values, please directly modify your Google Sheet. The app will refresh with your changes automatically (within ~60 seconds).", icon="💡")
    st.link_button("✏️ Open Your Google Sheet to Edit", f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit", use_container_width=True, type="secondary")
    st.markdown("---")

    df_assets = df[~df['Name'].isin(SPECIAL_ROWS)]
    tab1, tab2, tab3 = st.tabs(["➕ Add Asset", "🗑️ Delete Asset", "🗓️ Delete Month"])

    with tab1:
        with st.container(border=True):
             with st.form("new_investment_form"):
                st.header("Add New Investment / Asset")
                add_new_opt = "--- Add New Category ---"
                new_name = st.text_input("Investment Name", placeholder="e.g., 'ICICI Bluechip Fund'")
                category_selection = st.selectbox("Category", options=sorted(df_assets['Category'].unique().tolist()) + [add_new_opt])
                new_category_name = ""
                if category_selection == add_new_opt:
                    new_category_name = st.text_input("New Category Name", placeholder="e.g., 'Large Cap MF'")
                new_mode = st.text_input("Mode", placeholder="e.g., 'SIP', 'Bank Account'")
                new_initial_value = st.number_input("Current Value", min_value=0.0, step=1000.0, format="%.2f")
                add_submitted = st.form_submit_button("➕ Add Investment", use_container_width=True)
        if add_submitted:
            if not new_name or (category_selection == add_new_opt and not new_category_name):
                st.error("Please fill all required fields (Name and Category).")
            else:
                with st.spinner("Adding new investment..."):
                    final_category = new_category_name if category_selection == add_new_opt else category_selection
                    numeric_ids = pd.to_numeric(df_assets['ID'], errors='coerce').dropna()
                    new_id = int(numeric_ids.max() + 1) if not numeric_ids.empty else 1
                    date_cols = df.columns[4:]
                    if not date_cols.any():
                        latest_date = datetime.now().strftime('%Y-%m-%d')
                        df[latest_date] = 0.0
                    else:
                        latest_date = pd.to_datetime(date_cols).max().strftime('%Y-%m-%d')
                    
                    new_row_data = {'ID': str(new_id), 'Name': new_name, 'Mode': new_mode, 'Category': final_category}
                    for date_col in df.columns[4:]: new_row_data[date_col] = 0.0
                    new_row_data[latest_date] = new_initial_value
                    new_row_df = pd.DataFrame([new_row_data])
                    df_special_rows = df[df['Name'].isin(SPECIAL_ROWS)]
                    df_assets_current = df[~df['Name'].isin(SPECIAL_ROWS)]
                    df_combined = pd.concat([df_assets_current, new_row_df, df_special_rows], ignore_index=True)
                    
                    total_assets_updated = df_combined.loc[~df_combined['Name'].isin(SPECIAL_ROWS), latest_date].sum()
                    liabilities_latest = df_combined.loc[df_combined['Name'] == 'Liabilities', latest_date].iloc[0]
                    df_combined.loc[df_combined['Name'] == 'TOTAL', latest_date] = total_assets_updated
                    df_combined.loc[df_combined['Name'] == 'GRAND TOTAL', latest_date] = total_assets_updated - liabilities_latest
                    save_data(df_combined, sheet_id)
                st.success(f"Successfully added '{new_name}'!"); st.rerun()
    with tab2:
        with st.container(border=True):
            st.header("Delete an Investment")
            st.warning("This action is permanent and will remove the investment and its entire history.", icon="⚠️")
            investment_to_delete = st.selectbox("Select Investment to Delete", options=df_assets['Name'].tolist())
            if st.button("🗑️ Delete Selected Investment", type="primary"):
                with st.spinner(f"Deleting '{investment_to_delete}'..."):
                    df_updated = df[df['Name'] != investment_to_delete].copy()
                    date_cols_recalc = df_updated.columns[4:]
                    for col in date_cols_recalc:
                        total_assets = df_updated.loc[~df_updated['Name'].isin(SPECIAL_ROWS), col].sum()
                        liabilities = df_updated.loc[df_updated['Name'] == 'Liabilities', col].iloc[0]
                        df_updated.loc[df_updated['Name'] == 'TOTAL', col] = total_assets
                        df_updated.loc[df_updated['Name'] == 'GRAND TOTAL', col] = total_assets - liabilities
                    save_data(df_updated, sheet_id)
                st.success(f"Successfully deleted '{investment_to_delete}'."); st.rerun()
    with tab3:
        with st.container(border=True):
            st.header("Delete a Month's Data")
            st.warning("This action is permanent and will remove the entire data column for the selected month.", icon="⚠️")
            date_cols = df.columns[4:]
            if date_cols.any():
                month_to_delete = st.selectbox("Select Month to Delete", options=date_cols)
                if st.button("🗓️ Delete Selected Month", type="primary"):
                    with st.spinner(f"Deleting data for {month_to_delete}..."):
                        df_updated = df.drop(columns=[month_to_delete])
                        save_data(df_updated, sheet_id)
                    st.success(f"Successfully deleted data for {month_to_delete}."); st.rerun()
            else:
                st.info("No monthly data available to delete.")

def render_projections_page(df):
    df = df.copy()
    st.title("🔮 Financial Projections")
    st.markdown("Forecast your future net worth based on different growth scenarios.")

    date_cols = df.columns[4:]
    if len(date_cols) < 2:
        st.warning("Projections require at least two months of data to calculate historical growth rates.")
        return
        
    df_meta = df[df['Name'].isin(SPECIAL_ROWS)].drop_duplicates(subset=['Name']).set_index('Name')
    latest_date = date_cols[-1]
    current_net_worth = df_meta.loc['GRAND TOTAL', latest_date].item()

    grand_total_series = df_meta.loc['GRAND TOTAL', date_cols]
    first_non_zero = grand_total_series[grand_total_series > 0]
    cagr = 0.0
    if len(first_non_zero) > 1:
        start_date, end_date = pd.to_datetime(first_non_zero.index[0]), pd.to_datetime(first_non_zero.index[-1])
        num_years = (end_date - start_date).days / 365.25
        if num_years > 0:
            cagr = ((first_non_zero.iloc[-1] / first_non_zero.iloc[0]) ** (1 / num_years)) - 1

    with st.container(border=True):
        st.subheader("Projection Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            years_to_project = st.slider("Years to Project", 1, 30, 10)
        with col2:
            annual_return = st.slider("Expected Annual Return (%)", 1.0, 30.0, cagr * 100, 0.5) / 100
        with col3:
            monthly_investment = st.number_input("Additional Monthly Investment", 0, 1000000, 25000, 1000)

    projection_data = []
    future_value = current_net_worth
    today = datetime.now()

    for year in range(1, years_to_project + 1):
        future_date = today.replace(year=today.year + year)
        fv_principal = future_value * ((1 + annual_return) ** 1)
        monthly_rate = (1 + annual_return)**(1/12) - 1
        fv_annuity = monthly_investment * ((((1 + monthly_rate)**12) - 1) / monthly_rate)
        future_value = fv_principal + fv_annuity
        projection_data.append({"Year": year, "Date": future_date.strftime("%Y-%m-%d"), "Projected Net Worth": future_value})

    df_projection = pd.DataFrame(projection_data)

    with st.container(border=True):
        st.subheader("Projected Growth")
        final_value = df_projection['Projected Net Worth'].iloc[-1]
        st.metric(f"Projected Net Worth in {years_to_project} years", f"{CURRENCY_SYMBOL}{format_inr(final_value)}")
        fig = px.line(df_projection, x='Date', y='Projected Net Worth', markers=True, title=f"Net Worth Projection over {years_to_project} Years")
        fig.update_layout(yaxis_title=f"Net Worth ({CURRENCY_SYMBOL})", xaxis_title="Year")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_projection, use_container_width=True, hide_index=True, column_config={ "Projected Net Worth": st.column_config.NumberColumn(format=f"{CURRENCY_SYMBOL} %d") })

# --- Main Application Logic ---

sheet_id = get_sheet_id_from_url()

if not sheet_id:
    # If no sheet_id is in the URL, show the setup page and stop.
    render_welcome_page()
else:
    # If we have a sheet_id, show the main app.
    with st.sidebar:
        st.sidebar.title("Pro Net Worth Tracker")
        page_selection = option_menu(
            menu_title="Navigation",
            options=["Dashboard", "Update Data", "Manage Data", "Projections"],
            icons=["house-door-fill", "pencil-square", "gear-fill", "graph-up-arrow"],
            menu_icon="compass-fill", default_index=0,
            key='navigation_menu'
        )
        st.sidebar.header("🎯 Goal Setting")
        if 'target_amount' not in st.session_state: st.session_state.target_amount = 5_000_000
        if 'target_date' not in st.session_state: st.session_state.target_date = date(2028, 1, 1)
        st.sidebar.number_input(f"Net Worth Target ({CURRENCY_SYMBOL})", key="target_amount", step=100_000, format="%d")
        st.sidebar.date_input("Target Date", key="target_date")

    # Load data using the ID from the URL
    df_main = load_data(sheet_id)

    # Route to the correct page, passing the sheet_id to pages that need it for saving
    if page_selection == "Dashboard":
        render_dashboard(df_main)
    elif page_selection == "Update Data":
        render_update_page(df_main, sheet_id)
    elif page_selection == "Manage Data":
        render_manage_page(df_main, sheet_id)
    elif page_selection == "Projections":
        render_projections_page(df_main)