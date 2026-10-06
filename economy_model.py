from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler


DATA_PATH = Path(__file__).with_name("Indian_Economy_Sectorwise_Dataset.csv")
MODEL_PATH = Path(__file__).with_name("lregression.pkl")

st.set_page_config(page_title="Indian Economy | Sector Dashboard", layout="wide")
st.markdown(
    """
    <style>
    :root { --ink: #172c27; --green: #176b52; --mint: #dcebe1; --paper: #f5f7f2; }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stSidebar"] { background: #e9f0e8; }
    h1, h2, h3 { color: var(--ink); font-family: Georgia, serif; }
    h1 { letter-spacing: 0; font-size: 2.45rem; }
    [data-testid="stMetric"] { background: white; border-top: 3px solid var(--green); padding: 14px 16px; }
    [data-testid="stMetricValue"] { color: var(--green); }
    [data-testid="stCaptionContainer"] { color: #52655e; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH.name}")
    st.stop()

economy = load_data()
required_columns = {
    "Year", "Sector", "GDP_Lakh_Crore", "Growth_%", "Exports_Crore", "Imports_Crore",
    "Employment_Million", "Inflation_%",
}
missing_columns = required_columns.difference(economy.columns)
if missing_columns:
    st.error(f"Dataset is missing required columns: {', '.join(sorted(missing_columns))}")
    st.stop()

min_year = int(economy["Year"].min())
max_year = int(economy["Year"].max())
sectors = sorted(economy["Sector"].dropna().unique().tolist())

st.sidebar.header("Explore the data")
year_range = st.sidebar.slider(
    "Year range", min_value=min_year, max_value=max_year, value=(min_year, max_year)
)
selected_sectors = st.sidebar.multiselect("Sectors", sectors, default=sectors)
forecast_year = st.sidebar.number_input(
    "Model prediction year", min_value=min_year, max_value=max_year + 10, value=max_year + 1
)

st.title("India's economy, by sector")
st.caption("Sector-level indicators, trade flows, and the saved GDP regression model")

if not selected_sectors:
    st.info("Select at least one sector in the sidebar to view the dashboard.")
    st.stop()

filtered = economy.loc[
    economy["Year"].between(year_range[0], year_range[1])
    & economy["Sector"].isin(selected_sectors)
].copy()
annual = filtered.groupby("Year", as_index=False).agg(
    GDP_Lakh_Crore=("GDP_Lakh_Crore", "sum"),
    **{
        "Growth_%": ("Growth_%", "mean"),
        "Exports_Crore": ("Exports_Crore", "sum"),
        "Imports_Crore": ("Imports_Crore", "sum"),
        "Employment_Million": ("Employment_Million", "mean"),
        "Inflation_%": ("Inflation_%", "mean"),
    },
)
latest = annual.iloc[-1]

overview_tab, sectors_tab, model_tab = st.tabs(["Overview", "Sector detail", "GDP model"])

with overview_tab:
    metric_columns = st.columns(4)
    metric_columns[0].metric("GDP in latest year", f"{latest['GDP_Lakh_Crore']:,.1f} Lakh Cr")
    metric_columns[1].metric("Average growth", f"{annual['Growth_%'].mean():.2f}%")
    metric_columns[2].metric("Exports in latest year", f"{latest['Exports_Crore']:,.0f} Cr")
    trade_balance = latest["Exports_Crore"] - latest["Imports_Crore"]
    metric_columns[3].metric("Latest trade balance", f"{trade_balance:,.0f} Cr")

    left, right = st.columns([1.5, 1])
    with left:
        st.subheader("GDP over time")
        st.line_chart(annual.set_index("Year")["GDP_Lakh_Crore"], color="#176b52")
    with right:
        st.subheader("Trade flows")
        st.line_chart(
            annual.set_index("Year")[["Exports_Crore", "Imports_Crore"]],
            color=["#176b52", "#d27a3e"],
        )

    st.subheader("Growth and employment")
    st.line_chart(
        annual.set_index("Year")[["Growth_%", "Employment_Million"]],
        color=["#176b52", "#d27a3e"],
    )

with sectors_tab:
    st.subheader("Sector contribution")
    sector_summary = filtered.groupby("Sector", as_index=False).agg(
        GDP_Lakh_Crore=("GDP_Lakh_Crore", "sum"),
        Growth_percent=("Growth_%", "mean"),
        Employment_Million=("Employment_Million", "mean"),
        Exports_Crore=("Exports_Crore", "sum"),
        Imports_Crore=("Imports_Crore", "sum"),
    ).sort_values("GDP_Lakh_Crore", ascending=False)
    st.bar_chart(sector_summary.set_index("Sector")["GDP_Lakh_Crore"], color="#176b52")
    st.dataframe(
        sector_summary.rename(columns={"Growth_percent": "Growth (%)"}),
        hide_index=True,
        width="stretch",
        column_config={
            "GDP_Lakh_Crore": st.column_config.NumberColumn(format="%.2f"),
            "Growth (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Employment_Million": st.column_config.NumberColumn(format="%.2f"),
            "Exports_Crore": st.column_config.NumberColumn(format="%.0f"),
            "Imports_Crore": st.column_config.NumberColumn(format="%.0f"),
        },
    )

with model_tab:
    st.subheader("Saved linear regression")
    st.caption(
        "This model predicts GDP per sector-quarter record. Its training feature is a binary year flag: "
        "2020 or earlier = 0; after 2020 = 1."
    )
    try:
        model = load_model()
        scaler = MinMaxScaler().fit(economy[["GDP_Lakh_Crore"]])
        year_flag = int(forecast_year > 2020)
        scaled_prediction = float(
            model.predict(pd.DataFrame({"year_new": [year_flag]})).reshape(-1)[0]
        )
        prediction = float(scaler.inverse_transform([[scaled_prediction]])[0][0])
        st.metric(
            f"Predicted GDP per record for {forecast_year}",
            f"{prediction:,.2f} Lakh Cr",
            help="The output is converted back from the MinMax-scaled GDP used during training.",
        )

        model_years = pd.DataFrame({"Year": sorted(economy["Year"].unique())})
        model_years["Observed mean GDP"] = model_years["Year"].map(
            economy.groupby("Year")["GDP_Lakh_Crore"].mean()
        )
        flags = (model_years["Year"] > 2020).astype(int)
        model_years["Model prediction"] = scaler.inverse_transform(
            model.predict(pd.DataFrame({"year_new": flags})).reshape(-1, 1)
        ).ravel()
        st.line_chart(model_years.set_index("Year"), color=["#d27a3e", "#176b52"])
        st.caption(
            "The flat segments reflect the model's two year groups; the prediction is not a year-specific trend forecast."
        )
    except (FileNotFoundError, OSError, ValueError, AttributeError) as error:
        st.error(f"Could not load or run lregression.pkl: {error}")