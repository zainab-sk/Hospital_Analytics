#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np


# In[2]:


df = pd.read_csv("Patient's Data - Sheet1.csv")


# In[3]:


df.head()


# In[4]:


df.describe()


# In[6]:


df.info()


# In[7]:


df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(' ', '_')
    .str.replace('(', '', regex=False)
    .str.replace(')', '', regex=False)
)

# B. Drop unnamed columns
df.drop(columns=['unnamed:_13', 'unnamed:_14'], inplace=True, errors='ignore')

# C. Replace '-' with np.nan
df.replace('-', np.nan, inplace=True)

# D. Convert data types
df['date_visited'] = pd.to_datetime(df['date_visited'], errors='coerce')
for col in ['age', 'total_amount', 'consultant_amount', 'hospital_amount', 'ref_g.b']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Preview cleaned data
print(df.info())
print(df.head())


# In[8]:


# Standardize column name if not already done
df.rename(columns={'followup/new': 'followup_new'}, inplace=True)

# Replace NaN with 'New'
df['followup_new'] = df['followup_new'].fillna('New')

# Optional: standardize casing (e.g., capitalize first letter)
df['followup_new'] = df['followup_new'].str.strip().str.capitalize()


# In[9]:


# Standardize column name if needed
df.rename(columns={'gender': 'gender'}, inplace=True)  # just to be consistent

# Fill missing gender with 'Unknown'
df['gender'] = df['gender'].fillna('Unknown')

# Standardize gender values
df['gender'] = df['gender'].str.strip().str.lower()

# Map common variants to standard form
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


# In[11]:


# Replace any '-' or empty strings with NaN (if not already done)
df['age'] = df['age'].replace(['-', '', ' '], np.nan)

# Convert to numeric (integer)
df['age'] = pd.to_numeric(df['age'], errors='coerce')

# Optional: Check basic stats
print(df['age'].describe())


median_age = df['age'].median()
df['age'] = df['age'].fillna(median_age)


# In[12]:


# Fill missing payment modes with 'Unknown'
df['payment_mode'] = df['payment_mode'].fillna('Unknown')

# Standardize values (strip spaces and capitalize)
df['payment_mode'] = df['payment_mode'].str.strip().str.capitalize()

# Optional: Check unique payment modes to spot inconsistencies
print(df['payment_mode'].value_counts())


# In[13]:


amount_cols = ['total_amount', 'consultant_amount', 'hospital_amount']

for col in amount_cols:
    # Replace '-' or empty with NaN
    df[col] = df[col].replace(['-', '', ' '], np.nan)
    
    # Convert to numeric (float)
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Optional: Check summary stats
print(df[amount_cols].describe())


# In[15]:


print(df[['total_amount', 'consultant_amount', 'hospital_amount']].describe())


# In[16]:


print(df['gender'].value_counts())


# In[17]:


print(df['followup_new'].value_counts())



# In[18]:


print(df['dr_name_consultant'].value_counts().head(15))  # top 10 doctors by patient count


# In[19]:


revenue_by_payment = df.groupby('payment_mode')['total_amount'].sum()
print(revenue_by_payment)


# In[20]:


avg_consultant_per_doc = df.groupby('dr_name_consultant')['consultant_amount'].mean().sort_values(ascending=False)
print(avg_consultant_per_doc.head(10))


# In[21]:


import matplotlib.pyplot as plt

gender_counts = df['gender'].value_counts()
plt.figure(figsize=(6,6))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Gender Distribution')
plt.show()


# In[22]:


import seaborn as sns

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='followup_new', order=df['followup_new'].value_counts().index)
plt.title('New vs Followup Patients')
plt.xlabel('Patient Type')
plt.ylabel('Count')
plt.show()


# In[24]:


top_doctors = df['dr_name_consultant'].value_counts().head(15)

plt.figure(figsize=(10,5))
sns.barplot(x=top_doctors.values, y=top_doctors.index, palette='viridis')
plt.title('Top 10 Doctors by Number of Patients')
plt.xlabel('Number of Patients')
plt.ylabel('Doctor Name')
plt.show()


# In[25]:


revenue_by_payment = df.groupby('payment_mode')['total_amount'].sum().sort_values(ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x=revenue_by_payment.index, y=revenue_by_payment.values, palette='magma')
plt.title('Total Revenue by Payment Mode')
plt.xlabel('Payment Mode')
plt.ylabel('Total Revenue')
plt.show()


# In[26]:


avg_consultant_per_doc = df.groupby('dr_name_consultant')['consultant_amount'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,5))
sns.barplot(x=avg_consultant_per_doc.values, y=avg_consultant_per_doc.index, palette='coolwarm')
plt.title('Top 10 Doctors by Average Consultant Amount')
plt.xlabel('Average Consultant Amount')
plt.ylabel('Doctor Name')
plt.show()


# In[27]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))
sns.histplot(df['age'].dropna(), bins=20, kde=True, color='skyblue')
plt.title('Patient Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Boxplot by Gender
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='gender', y='age')
plt.title('Age Distribution by Gender')
plt.show()


# In[28]:


revenue_per_doctor = df.groupby('dr_name_consultant').agg({
    'total_amount': 'sum',
    'consultant_amount': 'sum',
    'hospital_amount': 'sum'
}).sort_values('total_amount', ascending=False).head(10)

print(revenue_per_doctor)


# In[29]:


# Convert 'Date Visited' to datetime if not already
df['date_visited'] = pd.to_datetime(df['date_visited'], errors='coerce')

# Count of New vs Followup patients per day
followup_trends = df.groupby(['date_visited', 'followup_new']).size().unstack(fill_value=0)

followup_trends.plot(kind='line', figsize=(12,6))
plt.title('New vs Followup Patients Over Time')
plt.xlabel('Date Visited')
plt.ylabel('Number of Patients')
plt.show()


# In[30]:


payment_trends = df.groupby(['date_visited', 'payment_mode']).size().unstack(fill_value=0)

payment_trends.plot(kind='line', figsize=(12,6))
plt.title('Payment Mode Trends Over Time')
plt.xlabel('Date Visited')
plt.ylabel('Number of Payments')
plt.show()


# In[31]:


referral_revenue = df.groupby('reference')['total_amount'].sum().sort_values(ascending=False).head(10)
print(referral_revenue)


# In[52]:


import matplotlib.pyplot as plt
import seaborn as sns

# Prepare the data
referral_revenue = df.groupby('reference')['total_amount'].sum().sort_values(ascending=False).head(8)

plt.figure(figsize=(10,6))
sns.barplot(x=referral_revenue.values, y=referral_revenue.index, palette='coolwarm')

plt.title('Top 10 Referral Sources by Total Revenue', fontsize=16)
plt.xlabel('Total Revenue', fontsize=14)
plt.ylabel('Referral Source', fontsize=14)

# Add value labels on bars
for i, v in enumerate(referral_revenue.values):
    plt.text(v + max(referral_revenue.values)*0.01, i, f"{v:.0f}", color='black', va='center')

plt.tight_layout()
plt.show()


# In[33]:


import matplotlib.pyplot as plt

# Ensure 'date_visited' is datetime
df['date_visited'] = pd.to_datetime(df['date_visited'], errors='coerce')

# Aggregate daily patient count and total revenue
daily_summary = df.groupby('date_visited').agg({
    'patient\'s_name': 'count',
    'total_amount': 'sum'
}).rename(columns={"patient's_name": 'patient_count', 'total_amount': 'total_revenue'})

# Plot
fig, ax1 = plt.subplots(figsize=(12,6))

color = 'tab:blue'
ax1.set_xlabel('Date Visited')
ax1.set_ylabel('Patient Count', color=color)
ax1.plot(daily_summary.index, daily_summary['patient_count'], color=color, marker='o')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second y-axis that shares the same x-axis

color = 'tab:green'
ax2.set_ylabel('Total Revenue', color=color)
ax2.plot(daily_summary.index, daily_summary['total_revenue'], color=color, marker='x')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Daily Patient Visits and Total Revenue Over Time')
plt.show()


# In[34]:


import matplotlib.pyplot as plt
import seaborn as sns

# Aggregate doctor-wise data
doctor_summary = df.groupby('dr_name_consultant').agg({
    'total_amount': 'sum',
    'consultant_amount': 'mean',
    'hospital_amount': 'mean',
    "patient's_name": 'count'
}).rename(columns={"patient's_name": 'patient_count'}).sort_values(by='total_amount', ascending=False)

# Show top 10 doctors by total revenue
top_doctors = doctor_summary.head(10)

print(top_doctors)


# In[35]:


fig, axes = plt.subplots(1, 2, figsize=(16,6))

sns.barplot(x=top_doctors['consultant_amount'], y=top_doctors.index, ax=axes[0], palette='coolwarm')
axes[0].set_title('Average Consultant Amount (Top 10 Doctors)')
axes[0].set_xlabel('Average Consultant Amount')
axes[0].set_ylabel('Doctor Name')

sns.barplot(x=top_doctors['hospital_amount'], y=top_doctors.index, ax=axes[1], palette='plasma')
axes[1].set_title('Average Hospital Amount (Top 10 Doctors)')
axes[1].set_xlabel('Average Hospital Amount')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()


# In[37]:


import seaborn as sns
import matplotlib.pyplot as plt

# Replace NaN in followup_new with 'New' (if not done)
df['followup_new'] = df['followup_new'].fillna('New')

# Boxplot comparing total_amount by patient type
plt.figure(figsize=(8,5))
sns.boxplot(x='followup_new', y='total_amount', data=df)
plt.title('Payment Amounts: Followup vs New Patients')
plt.ylabel('Total Amount Paid')
plt.xlabel('Patient Type')
plt.show()


# In[38]:


# Count patients per doctor and followup status
followup_counts = df.groupby(['dr_name_consultant', 'followup_new']).size().unstack(fill_value=0)

# Calculate followup rate = followup patients / total patients
followup_counts['followup_rate'] = followup_counts.get('Followup', 0) / (followup_counts.get('New', 0) + followup_counts.get('Followup', 0))

# Sort by followup rate descending
top_followup_rate = followup_counts.sort_values('followup_rate', ascending=False).head(10)

print(top_followup_rate[['New', 'Followup', 'followup_rate']])


# In[39]:


import missingno as msno
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
msno.matrix(df)
plt.show()

plt.figure(figsize=(12,6))
msno.heatmap(df)
plt.show()


# In[46]:


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# Load and clean data
df = pd.read_csv("Patient's Data - Sheet1.csv")  # <-- Update filename here

# Clean column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace('[^a-z0-9 ]', '', regex=True)
    .str.replace(' ', '_')
)

# Parse dates and convert amounts to numeric
df['date_visited'] = pd.to_datetime(df['date_visited'], errors='coerce')
df['total_amount'] = pd.to_numeric(df['total_amount'], errors='coerce')
df['Followup/New  '] = df['Followup/New'].fillna('New')

# Sidebar filters
st.sidebar.header("🔎 Filters")
doctor_list = df['dr_name_consultant'].dropna().unique()
selected_doctors = st.sidebar.multiselect("Doctor", options=doctor_list, default=doctor_list)

start_date = st.sidebar.date_input("Start Date", df['date_visited'].min())
end_date = st.sidebar.date_input("End Date", df['date_visited'].max())

# Filtered DataFrame
mask = (
    (df['date_visited'] >= pd.to_datetime(start_date)) &
    (df['date_visited'] <= pd.to_datetime(end_date)) &
    (df['dr_name_consultant'].isin(selected_doctors))
)
filtered_df = df[mask]

# KPI Metrics
st.title("🏥 Hospital Dashboard")
col1, col2, col3 = st.columns(3)
col1.metric("🧍 Total Patients", filtered_df.shape[0])
col2.metric("💰 Total Revenue", f"₹{filtered_df['total_amount'].sum():,.0f}")
col3.metric("👨‍⚕️ Doctors Active", filtered_df['dr_name_consultant'].nunique())

# Charts
st.markdown("## 📈 Daily Visits and Revenue")
daily_data = (
    filtered_df
    .groupby('date_visited')
    .agg(visits=('patients_name', 'count'), revenue=('total_amount', 'sum'))
    .reset_index()
)

fig, ax1 = plt.subplots(figsize=(10,4))
sns.lineplot(data=daily_data, x='date_visited', y='revenue', ax=ax1, label='Revenue', color='blue')
ax2 = ax1.twinx()
sns.barplot(data=daily_data, x='date_visited', y='visits', ax=ax2, alpha=0.3, color='orange')
ax1.set_title('Daily Revenue and Visits')
ax1.set_ylabel('Revenue')
ax2.set_ylabel('Visits')
st.pyplot(fig)

# Follow-up vs New
st.markdown("## 🔄 Follow-up vs New Patients")
followup_counts = filtered_df['followup_new'].value_counts()
fig2, ax3 = plt.subplots()
ax3.pie(followup_counts, labels=followup_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
ax3.set_title('Follow-up vs New Patients')
st.pyplot(fig2)

# Doctor performance
st.markdown("## 👩‍⚕️ Doctor-wise Revenue")
doctor_rev = (
    filtered_df
    .groupby('dr_name_consultant')['total_amount']
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

fig3, ax4 = plt.subplots(figsize=(10,4))
sns.barplot(data=doctor_rev, x='total_amount', y='dr_name_consultant', palette='viridis', ax=ax4)
ax4.set_xlabel("Total Revenue (₹)")
ax4.set_ylabel("Doctor")
ax4.set_title("Revenue by Doctor")
st.pyplot(fig3)

st.markdown("---")
st.markdown("✅ Built with ❤️ using Streamlit & Pandas")


# In[47]:


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# Load data
df = pd.read_csv("Patient's Data - Sheet1.csv")  # 👈 Change to your CSV filename

# Clean column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace('[^a-z0-9 ]', '', regex=True)
    .str.replace(' ', '_')
)

# Print columns to confirm renaming
# st.write(df.columns.tolist())  # Uncomment for debugging

# Handle missing or inconsistent columns safely
if 'followupnew' in df.columns:
    df['followupnew'] = df['followupnew'].fillna('New')
else:
    df['followupnew'] = 'New'

# Convert types
df['date_visited'] = pd.to_datetime(df['date_visited'], errors='coerce')
df['total_amount'] = pd.to_numeric(df['total_amount'], errors='coerce')

# Sidebar filters
st.sidebar.header("🔍 Filters")
doctor_list = df['dr_name_consultant'].dropna().unique().tolist()
selected_doctors = st.sidebar.multiselect("Doctor", options=doctor_list, default=doctor_list)
start_date = st.sidebar.date_input("Start Date", df['date_visited'].min())
end_date = st.sidebar.date_input("End Date", df['date_visited'].max())

# Apply filters
mask = (
    (df['date_visited'] >= pd.to_datetime(start_date)) &
    (df['date_visited'] <= pd.to_datetime(end_date)) &
    (df['dr_name_consultant'].isin(selected_doctors))
)
filtered_df = df[mask]

# KPIs
st.title("🏥 Hospital Dashboard")
col1, col2, col3 = st.columns(3)
col1.metric("👥 Total Patients", filtered_df.shape[0])
col2.metric("💸 Total Revenue", f"₹{filtered_df['total_amount'].sum():,.0f}")
col3.metric("🩺 Active Doctors", filtered_df['dr_name_consultant'].nunique())

# --- Daily Visits & Revenue ---
st.subheader("📅 Daily Visits & Revenue")
daily = (
    filtered_df
    .groupby('date_visited')
    .agg(visits=('patients_name', 'count'), revenue=('total_amount', 'sum'))
    .reset_index()
)
fig, ax1 = plt.subplots(figsize=(10, 4))
sns.lineplot(data=daily, x='date_visited', y='revenue', ax=ax1, label='Revenue', color='blue')
ax2 = ax1.twinx()
sns.barplot(data=daily, x='date_visited', y='visits', ax=ax2, alpha=0.3, color='gray', label='Visits')
ax1.set_ylabel("Revenue")
ax2.set_ylabel("Visits")
plt.xticks(rotation=45)
st.pyplot(fig)

# --- Follow-up vs New ---
st.subheader("🧾 Follow-up vs New Patients")
followup_counts = filtered_df['followupnew'].value_counts()
st.bar_chart(followup_counts)

# --- Doctor Performance ---
st.subheader("👨‍⚕️ Doctor-wise Revenue")
doc_perf = filtered_df.groupby('dr_name_consultant')['total_amount'].sum().sort_values(ascending=False)
st.bar_chart(doc_perf)


# In[ ]:




