# 🏥 Hospital Analytics Dashboard

An interactive **Streamlit dashboard** to explore and visualize hospital patient data.  
This project was built as part of our **On Job Training (OJT)** program to analyse patient visits, revenue, and doctor performance using Python and data visualization.

---

## 📌 Project Overview
This project provides a data-driven view of hospital operations.  
Key insights include:

- 📊 **Patient demographics** (Gender & Age distribution)
- 🆕 **New vs Follow-up patients** trends
- 📅 **Daily patient visits** over time
- 💳 **Revenue breakdown** by payment mode
- 🩺 **Top performing doctors** by revenue, consultant fee & patient count
- 💡 Dynamic **filters for Doctor, Month, Visit Type (New/Follow-up)** and a **Highest/Average/Lowest** selector to highlight key metrics

---

## 🛠️ Tech Stack
- **Python 3.12+**
- [Streamlit](https://streamlit.io/) – interactive dashboard
- [Pandas](https://pandas.pydata.org/) – data wrangling
- [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) – charts & visuals

---

## 📂 Project Structure
Hospital_dashboard/
│
├── app.py # Streamlit dashboard application
├── Patient's_Data.csv # Cleaned dataset used by the app
├── README.md # Project documentation
└── requirements.txt # Python dependencies (optional)

---

## ⚡ Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/zainab-sk/Hospital_Analytics.git
cd hospital-analytics-dashboard
```

### 2️⃣ Create a Virtual Environment (optional but recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
If requirements.txt is missing, install manually:
```bash
pip install streamlit pandas matplotlib seaborn
```
### 4️⃣ Run the App
```bash
streamlit run app.py
```
---
## 📝 Data Source
- The dataset was **manually digitized**: **1000+ records** were transcribed from handwritten hospital registers into Excel and then cleaned using **Python (Pandas)**.  
- All **personally identifiable information** has been removed to protect privacy.

---

## 🚀 Future Enhancements
- Add a **date-range filter** with a calendar picker  
- Enable **CSV export** of filtered data  
- Include **7-day rolling averages** in patient visit trends  
- Deploy the app on **Streamlit Community Cloud** or a similar hosting platform  

---

## 🌐 Live Dashboard
👉 **[View the Streamlit App](https://hospitalanalytics19.streamlit.app/)**  

---

## 🤝 Contributors
- **Zainab Shaikh** – Data Cleaning, Analysis & Dashboard Development  
- **OJT Team Members** – Data Collection & Manual Data Entry


