import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


oof_path = Path("D:/DATA SKRIPSI/preprocessing/catboost_oof_pred_with_cluster_kfold_with_macro.xlsx")

df_monthly = pd.read_excel(oof_path, sheet_name="Monthly_OOF")
df_daily   = pd.read_excel(oof_path, sheet_name="Daily_OOF")

actual_col  = "CuryUnitPrice"
pred_col    = "oof_pred"
cluster_col = "cluster"

df_monthly["period"] = "Monthly"
df_daily["period"]   = "Daily"

df_all = pd.concat([df_monthly, df_daily], ignore_index=True)


def scatter_actual_vs_pred(df, period_label: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    for cl in sorted(df[cluster_col].dropna().unique()):
        sub = df[df[cluster_col] == cl]
        ax.scatter(sub[actual_col], sub[pred_col], alpha=0.6, label=f"Cluster {cl}")
    
    lo = min(df[actual_col].min(), df[pred_col].min())
    hi = max(df[actual_col].max(), df[pred_col].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--")

    ax.set_xlabel("Actual Rent (CuryUnitPrice)")
    ax.set_ylabel("Predicted Rent (oof_pred)")
    ax.set_title(f"Actual vs Predicted Rent - {period_label}")
    ax.legend()
    fig.tight_layout()

    return fig

def boxplot_pred_per_cluster(df, period_label: str):
    df = df.dropna(subset=[pred_col]).copy()
    clusters = sorted(df[cluster_col].dropna().unique())

    data = []
    labels = []

    for c in clusters:
        series = df.loc[df[cluster_col] == c, pred_col].dropna()
        if len(series) > 0:
            data.append(series.values)
            labels.append(str(c))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(data, labels=labels, showfliers=True)
    ax.set_xlabel("Cluster")
    ax.set_ylabel(f"Predicted Rent ({pred_col})")
    ax.set_title(f"Distribusi Prediksi Sewa per Cluster - {period_label}")
    fig.tight_layout()

    return fig


def bar_mean_actual_pred_per_cluster(df, period_label: str):
    grouped = (
        df.groupby(cluster_col)[[actual_col, pred_col]]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(cluster_col)
    )

    x = range(len(grouped))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - width/2 for i in x], grouped[actual_col], width=width, label="Actual")
    ax.bar([i + width/2 for i in x], grouped[pred_col],   width=width, label="Predicted")

    ax.set_xticks(list(x))
    ax.set_xticklabels(grouped[cluster_col].astype(str))

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Mean Rent")
    ax.set_title(f"Rata-rata Actual vs Predicted per Cluster - {period_label}")
    ax.legend()
    fig.tight_layout()

    return fig


def compare_error_monthly_daily():
    df_m = df_monthly.copy()
    df_d = df_daily.copy()

    for df in (df_m, df_d):
        df["abs_error"] = (df[pred_col] - df[actual_col]).abs()
        df["ape"] = df["abs_error"] / df[actual_col]

    df_m["period"] = "Monthly"
    df_d["period"] = "Daily"
    df_all_error = pd.concat([df_m, df_d], ignore_index=True)

    error_summary = (
        df_all_error.groupby("period")[["abs_error", "ape"]]
        .mean(numeric_only=True)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(error_summary["period"], error_summary["ape"])
    ax.set_ylabel("Mean Absolute Percentage Error")
    ax.set_title("Perbandingan Akurasi Model: Monthly vs Daily")
    fig.tight_layout()
    
    return fig, error_summary


# --- Mapping BusinessType ---
mapping = {
    1: "Food & Beverage",
    2: "Health & Beauty",
    3: "Fashion & Accessories",
    4: "Entertainment & Hobbies",
    5: "Departement Store",
    6: "Supermarket",
    7: "Bookstore",
    8: "Houseware & Home Furnishing",
    9: "Handphone & IT",
    10: "Church",
    11: "Bank",
    12: "School & Education",
    99: "Others"
}

# Path file yang berisi sheet Monthly_Fixed_clustered & Daily_Fixed_clustered
# (kalau beda file, ganti di sini)
file_path = Path("D:/DATA SKRIPSI/kontrak_sewa_bersih_clustered.xlsx")  # sesuaikan kalau perlu

def plot_cluster_business_type(sheet_name: str):
    """
    Return: fig_countplot, fig_heatmap
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    df["BusinessType"] = df["BusinessType"].astype(int)
    df["BusinessTypeName"] = df["BusinessType"].map(mapping)

    # --- Countplot Cluster vs BusinessType ---
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    sns.countplot(data=df, x="cluster", hue="BusinessTypeName", ax=ax1)
    ax1.set_title(f"Perbandingan Jumlah Tenant Berdasarkan Cluster dan Business Type\n({sheet_name})")
    ax1.set_xlabel("Cluster")
    ax1.set_ylabel("Jumlah Tenant")
    ax1.legend(title="Business Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    fig1.tight_layout()

    # --- Heatmap ---
    pivot = pd.crosstab(df["cluster"], df["BusinessTypeName"])

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="Blues", ax=ax2)
    ax2.set_title(f"Heatmap Frekuensi Cluster vs Business Type\n({sheet_name})")
    ax2.set_xlabel("Business Type")
    ax2.set_ylabel("Cluster")
    fig2.tight_layout()

    return fig1, fig2


def plot_lease_year_start(sheet_name: str):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    year_counts = df["LeaseYearStart"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6, 3.5))  # <<< ukuran diperkecil
    sns.barplot(x=year_counts.index, y=year_counts.values, ax=ax)
    ax.set_title(f"Jumlah Kontrak Mulai per Tahun ({sheet_name})", fontsize=10)
    ax.set_xlabel("LeaseYearStart")
    ax.set_ylabel("Jumlah")
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()

    return fig

