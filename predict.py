
# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================

import joblib
import pandas as pd


# ==========================================
# 2. LOAD TRAINED MODEL AND SCALER
# ==========================================

kmeans = joblib.load("model/kmeans_model.pkl")
scaler = joblib.load("model/scaler.pkl")

print("Model and scaler loaded successfully!")


# ==========================================
# 3. TAKE CUSTOMER INPUT
# ==========================================

annual_income = float(
    input("Enter Annual Income (in thousands): ")
)

spending_score = float(
    input("Enter Spending Score (1-100): ")
)


# ==========================================
# 4. CREATE CUSTOMER DATAFRAME
# ==========================================

new_customer = pd.DataFrame(
    [[annual_income, spending_score]],
    columns=[
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
)


# ==========================================
# 5. SCALE CUSTOMER DATA
# ==========================================

new_customer_scaled = scaler.transform(new_customer)


# ==========================================
# 6. PREDICT CUSTOMER CLUSTER
# ==========================================

cluster = kmeans.predict(new_customer_scaled)


# ==========================================
# 7. DISPLAY RESULT
# ==========================================

print("\n========== CUSTOMER SEGMENTATION ==========")
print("Annual Income:", annual_income, "k$")
print("Spending Score:", spending_score)
print("Customer belongs to Cluster:", int(cluster[0]))
print("============================================")
