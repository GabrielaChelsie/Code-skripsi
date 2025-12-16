import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import visualization as viz


FILE_KONTRAK = Path("D:/DATA SKRIPSI/kontrak_sewa_bersih.xlsx")
FILE_PREDIKSI = Path("D:/DATA SKRIPSI/preprocessing/catboost_oof_pred_with_cluster_kfold_with_macro.xlsx")
PRED_MONTHLY = "Monthly_OOF"
PRED_DAILY = "Daily_OOF"
SHEET_KONTRAK = "Monthly_Fixed"
SHEET_DAILY = "Daily_Fixed"   

# Gambar hasil prediksi 
BASE_DIR = Path(__file__).parent
IMG_DAILY = BASE_DIR / "Daily_actual_vs_predicted.png"
IMG_MONTHLY = BASE_DIR / "Monthly_actual_vs_predicted.png"



@st.cache_data
def load_kontrak(path: Path, sheet_name: str) -> pd.DataFrame:
    if not path.exists():
        st.error(f"File kontrak tidak ditemukan: {path}")
        return pd.DataFrame()
    df = pd.read_excel(path, sheet_name=sheet_name)

    # rapikan RowID 
    
    if "RowID" in df.columns:
        df["RowID"] = df["RowID"].astype(str).str.strip()

    return df


@st.cache_data
def load_kontrak_daily(path: Path, sheet_name: str) -> pd.DataFrame:
    """Load sheet kontrak harian (struktur sama dengan monthly)."""
    if not path.exists():
        st.error(f"File kontrak DAILY tidak ditemukan: {path}")
        return pd.DataFrame()
    try:
        df = pd.read_excel(path, sheet_name=sheet_name)
    except Exception as e:
        st.error(f"Gagal membaca sheet daily '{sheet_name}': {e}")
        return pd.DataFrame()

    if "RowID" in df.columns:
        df["RowID"] = df["RowID"].astype(str).str.strip()

    return df


@st.cache_data
def load_prediksi(path: Path, sheet_name: str = None) -> pd.DataFrame:
    """
    Load file prediksi.
    - Jika sheet_name diberikan → baca sheet tersebut.
    - Jika sheet_name = None → baca sheet pertama.
    """

    if not path.exists():
        st.error(f"File prediksi tidak ditemukan: {path}")
        return pd.DataFrame()

    try:
        if sheet_name is None:
            df = pd.read_excel(path)
        else:
            df = pd.read_excel(path, sheet_name=sheet_name)

    except Exception as e:
        st.error(f"Gagal membaca sheet prediksi '{sheet_name}': {e}")
        return pd.DataFrame()

    # Rapikan RowID
    if "RowID" in df.columns:
        df["RowID"] = df["RowID"].astype(str).str.strip()

    return df



def merge_data(df_k: pd.DataFrame, df_p: pd.DataFrame) -> pd.DataFrame:
    """
    Merge berdasarkan RowID. Jika RowID tidak ada di salah satu,
    akan tampil warning dan merge dilakukan pakai kolom yang sama2 tersedia.
    """
    if "RowID" in df_k.columns and "RowID" in df_p.columns:
        on_cols = ["RowID"]
    else:
        # fallback kalau RowID tidak lengkap
        kandidat = ["Building", "UnitArea", "UnitFloor", "UnitNum", "CuryUnitPrice"]
        on_cols = [c for c in kandidat if c in df_k.columns and c in df_p.columns]
        st.warning(
            "Kolom RowID tidak lengkap di kedua file. "
            f"Merge menggunakan kolom: {', '.join(on_cols)}"
        )

    df_merged = pd.merge(df_k, df_p, on=on_cols, how="left")
    return df_merged


def compute_mape(df: pd.DataFrame, actual_col: str, pred_col: str):
    df_valid = df[[actual_col, pred_col]].replace([pd.NA, None], pd.NA).dropna()
    if df_valid.empty:
        return None
    df_valid = df_valid[df_valid[actual_col] != 0]
    if df_valid.empty:
        return None
    mape = (df_valid[pred_col] - df_valid[actual_col]).abs() / df_valid[actual_col]
    return float(mape.mean() * 100)


# MULAI APP
st.set_page_config(page_title="Dashboard Prediksi Tenant", layout="wide")
# Kurangi padding default halaman Streamlit
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# CSS chip
st.markdown("""
<style>

    /* ============================
       🎨 Ubah warna CHIP (tag selected)
       ============================ */
    span[data-baseweb="tag"] {
        background-color: #1e88e5 !important;   /* Biru */
        color: white !important;                /* Teks putih */
        border-radius: 6px !important;
    }

    /* Hover chip */
    span[data-baseweb="tag"]:hover {
        background-color: #1565c0 !important;
        color: white !important;
    }

    /* X (close icon) warna putih */
    span[data-baseweb="tag"] svg {
        fill: white !important;
    }

</style>
""", unsafe_allow_html=True)

# KONTEN HALAMAN DASHBOARD
st.title("Dashboard Prediksi Harga Sewa Tenant")
st.caption("Data: kontrak_sewa_bersih (Monthly & Daily) + catboost_oof_pred_with_cluster")

# Load data
df_kontrak = load_kontrak(FILE_KONTRAK, SHEET_KONTRAK)
df_kontrak_daily = load_kontrak_daily(FILE_KONTRAK, SHEET_DAILY)
df_prediksi = load_prediksi(FILE_PREDIKSI, PRED_MONTHLY)
df_dailypred = load_prediksi(FILE_PREDIKSI, PRED_DAILY )

if df_kontrak.empty or df_prediksi.empty:
    st.stop()

# Merge Monthly
df = merge_data(df_kontrak, df_prediksi)


if "RowID" in df_kontrak.columns and "RowID" in df_prediksi.columns:
    common_ids = set(df_kontrak["RowID"]) & set(df_prediksi["RowID"])
    st.write("Jumlah RowID di kontrak (Monthly):", df_kontrak["RowID"].nunique())
    # st.write("Jumlah RowID di prediksi:", df_prediksi["RowID"].nunique())
    # st.write("Jumlah RowID yang match:", len(common_ids))


# Standarkan nama kolom (Monthly)
df.columns = df.columns.str.strip()
df = df.rename(columns={
    "oof_pred": "PredictedRent",
    "CuryUnitPrice_x": "ActualRent",
    "cluster": "Cluster"
})
for col in ["ActualRent", "PredictedRent"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if df_kontrak_daily.empty:
    st.warning(
        f"Sheet daily ('{SHEET_DAILY}') kosong / tidak ditemukan. "
        "Level Daily akan memakai data Monthly sebagai fallback."
    )
    df_daily = df.copy()
else:
    df_daily = merge_data(df_kontrak_daily, df_dailypred)
    df_daily.columns = df_daily.columns.str.strip()
    df_daily = df_daily.rename(columns={
        "oof_pred": "PredictedRent",
        "CuryUnitPrice_x": "ActualRent",
        "cluster": "Cluster"
    })
    for col in ["ActualRent", "PredictedRent"]:
        if col in df_daily.columns:
            df_daily[col] = pd.to_numeric(df_daily[col], errors="coerce")


st.markdown("### 🔍 Filter Data")

# Tambah filter level data (Monthly / Daily)
level_col1, level_col2 = st.columns([1, 5])
with level_col1:
    level_data = st.radio(
        "Level data",
        ["Monthly", "Daily"],
        horizontal=False
    )

# Pilih dataframe dasar sesuai level
if level_data == "Monthly":
    df_base = df
else:
    df_base = df_daily

# Salinan untuk difilter
df_filtered = df_base.copy()

f1, f2, f3, f4 = st.columns([2, 2, 2, 3])

# Tipe Bisnis
if "BusinessType" in df_filtered.columns:
    with f1:
        options_bt = sorted(df_filtered["BusinessType"].dropna().unique())
        selected_bt = st.multiselect(
            "Tipe Bisnis",
            options=options_bt,
            default=[]  
        )
    if selected_bt:
        df_filtered = df_filtered[df_filtered["BusinessType"].isin(selected_bt)]

# Cluster
if "Cluster" in df_filtered.columns:
    with f2:
        options_cl = sorted(df_filtered["Cluster"].dropna().unique())
        selected_cl = st.multiselect(
            "Cluster",
            options=options_cl,
            default=[]  
        )
    if selected_cl:
        df_filtered = df_filtered[df_filtered["Cluster"].isin(selected_cl)]

# Building
if "Building" in df_filtered.columns:
    with f3:
        options_bld = sorted(df_filtered["Building"].dropna().unique())
        selected_bld = st.multiselect(
            "Building",
            options=options_bld,
            default=[]  
        )
    if selected_bld:
        df_filtered = df_filtered[df_filtered["Building"].isin(selected_bld)]

# Cari UnitID (kandung teks)
with f4:
    if "UnitID" in df_filtered.columns:
        q_unit = st.text_input("Cari UnitID (kandung teks)", "")
        if q_unit:
            df_filtered = df_filtered[
                df_filtered["UnitID"].str.contains(q_unit, case=False, na=False)
            ]



col1, col2, col3 = st.columns(3)

with col1:
    if "PredictedRent" in df_filtered.columns and not df_filtered.empty:
        avg_pred = df_filtered["PredictedRent"].mean()
        st.metric("Rata-rata Predicted Rent", f"Rp {avg_pred:,.0f}".replace(",", "."))
    else:
        st.metric("Rata-rata Predicted Rent", "-")

with col2:
    if "ActualRent" in df_filtered.columns and not df_filtered.empty:
        avg_actual = df_filtered["ActualRent"].mean()
        st.metric("Rata-rata Actual Rent", f"Rp {avg_actual:,.0f}".replace(",", "."))
    else:
        st.metric("Rata-rata Actual Rent", "-")

with col3:
    if not df_filtered.empty and "ActualRent" in df_filtered.columns and "PredictedRent" in df_filtered.columns:
        mape = compute_mape(df_filtered, "ActualRent", "PredictedRent")
    else:
        mape = None
    st.metric("MAPE (≈)", f"{mape:.2f}%" if mape is not None else "Tidak bisa dihitung")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Scatter",
    "📦 Distribusi Cluster",
    "📉 Tren Tahun",
    "🖼️ Cluster analysis"
])

with tab1:
    st.subheader("Predicted vs Actual Rent")

    if "ActualRent" in df_filtered.columns and "PredictedRent" in df_filtered.columns:
        df_valid = df_filtered.dropna(subset=["ActualRent", "PredictedRent"])
    else:
        df_valid = pd.DataFrame()

    if not df_valid.empty and level_data == "Monthly":
        fig = px.scatter(
            df_valid,
            x="ActualRent",
            y="PredictedRent",
            color="Cluster" if "Cluster" in df_valid.columns else None,
            color_continuous_scale="Viridis", 
            hover_data=[c for c in ["UnitID", "Building", "UnitArea", "UnitFloor"]
                        if c in df_valid.columns],
            trendline="ols",
            labels={"ActualRent": "Actual Rent", "PredictedRent": "Predicted Rent"}
        )
        st.plotly_chart(fig, use_container_width=True)

    elif level_data == "Daily":
        daily_actual_col = "ActualRent"
        daily_pred_col = "PredictedRent"

        if daily_actual_col in df_filtered.columns and daily_pred_col in df_filtered.columns:
            df_daily_valid = df_filtered.dropna(subset=[daily_actual_col, daily_pred_col])

            if not df_daily_valid.empty:
                fig_daily = px.scatter(
                    df_daily_valid,
                    x=daily_actual_col,
                    y=daily_pred_col,
                    color="Cluster" if "Cluster" in df_daily_valid.columns else None,
                    color_continuous_scale="Viridis",   
                    hover_data=[c for c in ["UnitID", "Building", "UnitArea", "UnitFloor"]
                                if c in df_daily_valid.columns],
                    trendline="ols",
                    labels={
                        daily_actual_col: "Actual Rent (Daily)",
                        daily_pred_col: "Predicted Rent (Daily)"
                    }
                )

                st.plotly_chart(fig_daily, use_container_width=True)
            else:
                st.info(
                    "Data Actual / Predicted untuk level **Daily** kosong setelah filter diterapkan."
                )
        else:
            st.info(
                "Kolom untuk daily scatter belum tersedia. "
                "Silakan sesuaikan nama kolom `daily_actual_col` dan `daily_pred_col` "
                "dengan struktur data harian Anda."
            )
    else:
        st.info("Data Actual atau Predicted kosong, tidak bisa membuat scatter.")

with tab2:
    st.subheader("Distribusi Predicted Rent per Cluster")
    if "Cluster" in df_filtered.columns and "PredictedRent" in df_filtered.columns and not df_filtered.empty:
        fig2 = px.box(
            df_filtered,
            x="Cluster",
            y="PredictedRent",
            points="all",
            labels={"Cluster": "Cluster", "PredictedRent": "Predicted Rent"}
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Kolom Cluster atau PredictedRent tidak tersedia / data kosong.")

    
with tab3:
    st.subheader("Rata-rata Actual vs Prediksi per Tahun Mulai Sewa")

    # Pastikan kolom tersedia
    needed_cols = {"LeaseYearStart", "PredictedRent", "ActualRent"}

    if needed_cols.issubset(df_filtered.columns):

        if not df_filtered.empty:

            # Hitung rata-rata per tahun
            df_trend = (
                df_filtered.groupby("LeaseYearStart", as_index=False)
                           .agg({
                               "ActualRent": "mean",
                               "PredictedRent": "mean"
                           })
                           .sort_values("LeaseYearStart")
            )

            if not df_trend.empty:
                import plotly.graph_objects as go

                fig3 = go.Figure()


                fig3.add_trace(go.Scatter(
                    x=df_trend["LeaseYearStart"],
                    y=df_trend["ActualRent"],
                    mode="lines+markers",
                    name="Actual",
                    line=dict(dash="solid", width=3, color="#1E88E5")
                ))


                fig3.add_trace(go.Scatter(
                    x=df_trend["LeaseYearStart"],
                    y=df_trend["PredictedRent"],
                    mode="lines+markers",
                    name="Predicted",
                    line=dict(dash="dash", width=3, color="#E53935")
                ))

                # Layout
                fig3.update_layout(
                    xaxis_title="Tahun Mulai Sewa",
                    yaxis_title="Nilai Rata-rata",
                    legend_title="Jenis Nilai",
                    hovermode="x unified",
                    template="plotly_white"
                )

                st.plotly_chart(fig3, use_container_width=True)

            else:
                st.info("Tidak ada data untuk membuat tren tahun.")

        else:
            st.info("Data kosong setelah filter diterapkan.")

    else:
        st.info("Kolom LeaseYearStart / ActualRent / PredictedRent tidak tersedia.")




with tab4:
    st.subheader("Hasil Prediksi")

    # st.markdown("---")
    st.subheader("Analisis Cluster vs Business Type")

    # Pilih sheet sesuai level_data
    if level_data == "Daily":
        sheet_bt = "Daily_Fixed_clustered"
    else:
        sheet_bt = "Monthly_Fixed_clustered"

        fig_bt_count, fig_bt_heat = viz.plot_cluster_business_type(sheet_bt)

        col1, col2 = st.columns(2)

        with col1:
            st.pyplot(fig_bt_count)

        with col2:
            st.pyplot(fig_bt_heat)


    st.markdown("---")
    st.subheader("Distribusi Tahun Mulai Kontrak (LeaseYearStart)")

    col_kecil, col_kosong = st.columns([1, 1])  # kolom kiri lebih sempit

    with col_kecil:
        fig_lease = viz.plot_lease_year_start(sheet_bt)
        st.pyplot(fig_lease, use_container_width=False)



st.markdown("### 📋 Data Detail per Unit")
show_cols = [c for c in [
    "UnitID", "Building", "UnitArea", "UnitFloor", "UnitNum",
    "BusinessType", "Cluster", "LeaseYearStart", "LeaseDurationMonths",
    "ActualRent", "PredictedRent"
] if c in df_filtered.columns]

if not df_filtered.empty and show_cols:
    st.dataframe(
        df_filtered[show_cols].sort_values("PredictedRent", ascending=False),
        use_container_width=True
    )
else:
    st.info("Data detail kosong atau kolom yang ingin ditampilkan tidak tersedia.")

st.download_button(
    label="💾 Download Data (Filtered) sebagai CSV",
    data=df_filtered.to_csv(index=False).encode("utf-8-sig") if not df_filtered.empty else "".encode("utf-8-sig"),
    file_name="prediksi_sewa_filtered.csv",
    mime="text/csv"
)
