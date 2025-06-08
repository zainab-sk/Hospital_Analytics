# 📦 Required Libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import calendar

sns.set_theme(style="whitegrid", palette="pastel")

# 📁 Load Data
df = pd.read_csv("Patient's_Data.csv")
df
"# 🔧 Clean column names"
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(' ', '_')
    .str.replace('(', '', regex=False)
    .str.replace(')', '', regex=False)
)
df.rename(columns={'followup/new': 'followup_new'}, inplace=True)

# 📅 Date handling
df['date_visited'] = df['date_visited'].str.replace('-', '/', regex=False)
df['date_visited'] = pd.to_datetime(df['date_visited'], dayfirst=True, infer_datetime_format=True, errors='coerce')
df['month'] = df['date_visited'].dt.month
df['month_name'] = df['month'].apply(lambda x: calendar.month_name[int(x)] if pd.notnull(x) else 'Unknown')

# Clean 'payment_mode'
df['payment_mode'] = df['payment_mode'].replace({'Gpay': 'Upi'}).fillna("Unknown")

# Sidebar Filters
st.sidebar.title("🔍 Filters")

doctor_filter = st.sidebar.multiselect("Select Doctor(s)", options=sorted(df['dr_name_consultant'].dropna().unique()))
month_filter = st.sidebar.multiselect("Select Month(s)", options=sorted(df['month_name'].dropna().unique()))
visit_type_filter = st.sidebar.multiselect(
    "Select Visit Type(s)",
    options=sorted(df['followup_new'].dropna().unique()),
    default=None
)

# New filter for highest/lowest/all
hl_filter = st.sidebar.selectbox("Highlight", options=["All", "Highest", "Lowest"])

# Apply filters
if doctor_filter:
    df = df[df['dr_name_consultant'].isin(doctor_filter)]
if month_filter:
    df = df[df['month_name'].isin(month_filter)]
if visit_type_filter:
    df = df[df['followup_new'].isin(visit_type_filter)]

# Function to filter data based on hl_filter for value counts or grouped sums
def filter_highest_lowest(series_or_df, filter_type="All", agg_func='count'):
    """
    series_or_df can be a Series or DataFrame grouped data
    filter_type: "All", "Highest", "Lowest"
    agg_func: 'count' or 'sum' (for aggregation type if needed)
    """
    if filter_type == "All":
        return series_or_df
    elif filter_type == "Highest":
        max_val = series_or_df.max()
        return series_or_df[series_or_df == max_val]
    elif filter_type == "Lowest":
        min_val = series_or_df.min()
        return series_or_df[series_or_df == min_val]
    else:
        return series_or_df

# Dashboard Title
st.title("🏥 Hospital Dashboard – Visual Analysis")
st.markdown("Visuals extracted from hospital data with filters for doctor, month, visit type, and highlight.")

# 1️⃣ Gender Distribution
df['gender'] = df['gender'].fillna('Unknown').str.strip().str.lower()
gender_map = {
    'm': 'Male',
    'male': 'Male',
    'f': 'Female',
    'female': 'Female',
    'other': 'Other',
    'unknown': 'Unknown',
    'not specified': 'Unknown',
    '': 'Unknown',
    np.nan: 'Unknown'
}
df['gender'] = df['gender'].map(gender_map).fillna('Unknown')

st.subheader("1. Gender Distribution")
if 'gender' in df.columns:
    gender_counts = df['gender'].value_counts()
    gender_counts = filter_highest_lowest(gender_counts, hl_filter)
    
    fig1, ax1 = plt.subplots()
    ax1.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140)
    st.pyplot(fig1)

# 2️⃣ Top 10 Doctors by Number of Patients
st.subheader("2. Top 10 Doctors by Number of Patients")
top_doctors_by_count = df['dr_name_consultant'].value_counts()
top_doctors_by_count = filter_highest_lowest(top_doctors_by_count, hl_filter)
top_doctors_by_count = top_doctors_by_count.sort_values(ascending=True).tail(10)

fig9, ax9 = plt.subplots()
colors = sns.color_palette("Blues", len(top_doctors_by_count))
sns.barplot(x=top_doctors_by_count.values, y=top_doctors_by_count.index, palette=colors, ax=ax9)
ax9.set_title("Top 10 Doctors by Number of Patients")
ax9.set_xlabel("Number of Patients")
ax9.set_ylabel("Doctor")

for i, v in enumerate(top_doctors_by_count.values):
    ax9.text(v + 1, i, f'{int(v)}', va='center')

st.pyplot(fig9)

# 3️⃣ New vs Follow-up Patients
st.subheader("3. New vs Follow-up Patients")
df['followup_new'] = df['followup_new'].fillna('New')
followup_counts = df['followup_new'].value_counts()
followup_counts = filter_highest_lowest(followup_counts, hl_filter)

fig2, ax2 = plt.subplots()
colors = sns.color_palette("Blues", len(followup_counts))
sns.barplot(x=followup_counts.index, y=followup_counts.values, palette=colors, ax=ax2)
ax2.set_title("New vs Follow-up Patients")
ax2.set_xlabel("Visit Type")
ax2.set_ylabel("Number of Patients")

for bar in ax2.patches:
    height = bar.get_height()
    if height > 0:
        ax2.annotate(f'{int(height)}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5),
                     textcoords='offset points',
                     ha='center', va='bottom')

st.pyplot(fig2)

# 4️⃣ Daily Patient Visits Over Time
st.subheader("4. Daily Patient Visits")
daily_visits = df['date_visited'].value_counts().sort_index()
daily_visits_filtered = daily_visits
if hl_filter != "All":
    val = daily_visits.max() if hl_filter == "Highest" else daily_visits.min()
    daily_visits_filtered = daily_visits[daily_visits == val]

fig3, ax3 = plt.subplots(figsize=(12, 4))
sns.lineplot(x=daily_visits_filtered.index, y=daily_visits_filtered.values, marker='o', color='teal', linewidth=2, ax=ax3)
ax3.set_xlabel("Date")
ax3.set_ylabel("Visit Count")
ax3.set_title("Patient Visits Over Time")
ax3.tick_params(axis='x', rotation=45)

if not daily_visits_filtered.empty:
    max_date = daily_visits_filtered.idxmax()
    max_value = daily_visits_filtered.max()
    ax3.annotate(f'🔺 {max_value}', xy=(max_date, max_value), 
                 xytext=(0, 10), textcoords="offset points", ha='center', color='red', fontsize=9)

st.pyplot(fig3)

# 5️⃣ Patient Age Distribution
st.subheader("5. Patient Age Distribution")
df['age'] = pd.to_numeric(df['age'], errors='coerce')
fig4, ax4 = plt.subplots(figsize=(10, 4))

bins = list(range(0, 101, 10))  # bins 0-10, 11-20...
labels = [f"{i}-{i+9}" for i in bins[:-1]] + ["100+"]

sns.histplot(
    data=df,
    x='age',
    bins=bins,
    kde=True,
    color="#4db6ac",
    edgecolor="black",
    ax=ax4
)

ax4.set_xlabel("Age Group")
ax4.set_ylabel("Patient Count")
ax4.set_title("Patient Age Distribution")
ax4.set_xticks(bins)
ax4.set_xticklabels(labels)

st.pyplot(fig4)

# 6️⃣ Revenue by Payment Mode
st.subheader("6. Revenue by Payment Mode")
df['total_amount'] = pd.to_numeric(df['total_amount'], errors='coerce')
df = df.dropna(subset=['total_amount'])
revenue_by_payment = df.groupby('payment_mode')['total_amount'].sum().sort_values(ascending=False)
revenue_by_payment = filter_highest_lowest(revenue_by_payment, hl_filter)

fig5, ax5 = plt.subplots()
colors = sns.color_palette("BuGn", len(revenue_by_payment))
sns.barplot(x=revenue_by_payment.values, y=revenue_by_payment.index, palette=colors, ax=ax5)
ax5.set_title("Total Revenue by Payment Mode")
ax5.set_xlabel("Revenue (₹)")
ax5.set_ylabel("Payment Mode")

for i, v in enumerate(revenue_by_payment.values):
    ax5.text(v + 1000, i, f'₹{int(v):,}', va='center')

st.pyplot(fig5)

# 7️⃣ Top Doctors by Revenue
st.subheader("7. Top 10 Doctors by Revenue")
revenue_by_doctor = df.groupby('dr_name_consultant')['total_amount'].sum().sort_values(ascending=False)
revenue_by_doctor = filter_highest_lowest(revenue_by_doctor, hl_filter)
revenue_by_doctor = revenue_by_doctor.head(10)

fig6, ax6 = plt.subplots()
colors = sns.color_palette("YlGnBu", len(revenue_by_doctor))
sns.barplot(x=revenue_by_doctor.values, y=revenue_by_doctor.index, palette=colors, ax=ax6)
ax6.set_title("Top 10 Doctors by Revenue")
ax6.set_xlabel("Revenue (₹)")
ax6.set_ylabel("Doctor")

for i, v in enumerate(revenue_by_doctor.values):
    ax6.text(v + 1000, i, f'₹{int(v):,}', va='center')

st.pyplot(fig6)

# 8️⃣ Average Consultant Fee per Doctor
st.subheader("8. Top 10 Doctors by Avg Consultant Fee")
df['consultant_amount'] = pd.to_numeric(df['consultant_amount'], errors='coerce')
avg_consultant = df.groupby('dr_name_consultant')['consultant_amount'].mean().sort_values(ascending=False)
avg_consultant = filter_highest_lowest(avg_consultant, hl_filter)
avg_consultant = avg_consultant.head(10)

fig7, ax7 = plt.subplots()
colors = sns.color_palette("PuBuGn", len(avg_consultant))
sns.barplot(x=avg_consultant.values, y=avg_consultant.index, palette=colors, ax=ax7)
ax7.set_title("Top 10 Doctors by Avg Consultant Amount")
ax7.set_xlabel("Average Consultant Fee (₹)")
ax7.set_ylabel("Doctor")

for i, v in enumerate(avg_consultant.values):
    ax7.text(v + 50, i, f'₹{int(v):,}', va='center')

st.pyplot(fig7)

# 9️⃣ Age vs Total Amount Paid
st.subheader("9. Age vs Total Amount Paid")

fig8, ax8 = plt.subplots(figsize=(10, 4))
sns.scatterplot(
    data=df,
    x='age',
    y='total_amount',
    hue='age',
    palette='cool',
    alpha=0.7,
    ax=ax8
)

ax8.set_xlabel("Age")
ax8.set_ylabel("Total Amount Paid (₹)")
ax8.set_title("Age vs Total Amount Paid")
ax8.legend([], [], frameon=False)  # Remove legend

st.pyplot(fig8)
