# ===============================
# 📦 1. Import Libraries
# ===============================
import pandas as pd
import numpy as np



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import pickle
import warnings
warnings.filterwarnings("ignore")

# ===============================
# 📁 2. Load Data
# ===============================
data = pd.read_csv("/content/loan - loan.csv")
data.drop(['Loan_ID'], axis=1, inplace=True)

# ===============================
# 🔍 3. Exploratory Data Analysis (Optional)
# ===============================
# Count categorical variables
obj = (data.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:", len(object_cols))

# Plot categorical distributions
plt.figure(figsize=(18, 36))
for idx, col in enumerate(object_cols):
    plt.subplot(len(object_cols)//2 + 1, 2, idx+1)
    sns.countplot(data[col])
    plt.title(col)
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ===============================
# 🧹 4. Data Preprocessing
# ===============================
# Fill missing values
for col in data.columns:
    if data[col].dtype == 'object':
        data[col].fillna(data[col].mode()[0], inplace=True)
    else:
        data[col].fillna(data[col].mean(), inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
for col in object_cols:
    data[col] = label_encoder.fit_transform(data[col])

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ===============================
# ✂️ 5. Feature Selection
# ===============================
X = data.drop('Loan_Status', axis=1)
Y = data['Loan_Status']

# ===============================
# 🧪 6. Train-Test Split
# ===============================
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42, stratify=Y)

# Standardize (optional but good for SVM/KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# 🤖 7. Model Training & Evaluation
# ===============================
models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True),
    'Logistic Regression': LogisticRegression()
}

print("\n🔍 Model Performance on Test Set:\n")
for name, model in models.items():
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(Y_test, y_pred)
    print(f"{name} Accuracy: {acc*100:.2f}%")
    print(classification_report(Y_test, y_pred))

# ===============================
# 💾 8. Save the Best Model
# ===============================
best_model = models['Random Forest']
with open("loan_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Save scaler too (for future predictions)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

import streamlit as st
import numpy as np
import pickle

# Load model and scaler
with open("loan_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

st.title("🏦 Loan Approval Prediction")
st.write("Fill the details to check if your loan will be approved.")

# Input form
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term (in days)", min_value=0)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Map inputs to numbers
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0
dependents = 3 if dependents == "3+" else int(dependents)
property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

# Prepare input array
input_data = np.array([[gender, married, dependents, education, self_employed,
                        applicant_income, coapplicant_income, loan_amount,
                        loan_amount_term, credit_history, property_area]])

# Scale input
input_data_scaled = scaler.transform(input_data)

# Predict
if st.button("Check Loan Status"):
    prediction = model.predict(input_data_scaled)
    if prediction[0] == 1:
        st.success("✅ Loan is likely to be Approved.")
    else:
        st.error("❌ Loan is likely to be Rejected.")
