import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# --------------------- Helper Functions --------------------- #
def preprocess_data(data):
    categorical_cols = [col for col in data.columns if data[col].dtype == "object"]
    encoder = LabelEncoder()
    mapping_dict = {}
    for col in categorical_cols:
        data[col] = encoder.fit_transform(data[col].astype(str))
        mapping_dict[col] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))  # Save mapping
    return data, categorical_cols, mapping_dict

def fill_missing(data):
    for col in data.columns:
        if data[col].dtype == "object":
            data[col].fillna(data[col].mode()[0], inplace=True)
        else:
            data[col].fillna(data[col].mean(), inplace=True)
    return data

def remove_outliers(data):
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    return data

def display_correlation_heatmap(data):
    st.write("### Correlation Heatmap \U0001F4C9")
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# --------------------- Streamlit UI --------------------- #
st.title("\U0001F4CA AutoML - CSV Upload & ML Pipeline")

st.sidebar.header("Upload Your CSV File \U0001F4C2")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Preview of Dataset \U0001F4DC")
    st.dataframe(data.head())

    st.write("### Data Preprocessing \U0001F527")
    if st.checkbox("Show missing values"):
        st.write(data.isnull().sum())

    if st.checkbox("Fill missing values with mean/mode"):
        data = fill_missing(data)
        st.write(data.isnull().sum())

    if st.checkbox("Remove outliers using IQR method"):
        data = remove_outliers(data)
        st.success("Outliers removed successfully!")

    data, cat_cols, mapping_dict = preprocess_data(data)
    if cat_cols:
        st.write(f"Categorical Columns Encoded: {cat_cols}")

    display_correlation_heatmap(data)

    st.write("### Feature Selection \U0001F3AF")
    target_column = st.selectbox("Select Target Column", data.columns)
    feature_columns = st.multiselect("Select Features", [col for col in data.columns if col != target_column])
    model_list = st.selectbox("Select Models", ["Linear Regression","Logistic Regression","KNN","Decision Tree","K-means"])

    if st.button("Train Model 🚀"):
        if not feature_columns:
            st.error("Please select at least one feature column!")
        else:
            X = data[feature_columns]
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            with st.spinner("Training in progress..."):
                model = None
                
                if model_list == "Linear Regression":
                    model = LinearRegression()
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)

                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    st.write(f"### {model_list} Performance 📈")
                    st.write(f"🔹 MAE: {mae:.2f}")
                    st.write(f"🔹 MSE: {mse:.2f}")
                    st.write(f"🔹 R² Score: {r2:.2f}")

                elif model_list == "Logistic Regression":
                    model = LogisticRegression()
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    acc = accuracy_score(y_test, y_pred)

                    st.write(f"### {model_list} Accuracy: {acc * 100:.2f}%")
                elif model_list == "KNN":
                    model = KNeighborsRegressor()
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)

                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    st.write(f"### {model_list} Performance 📈")
                    st.write(f"🔹 MSE: {mse:.2f}")
                    st.write(f"🔹 R² Score: {r2:.2f}")

                elif model_list == "Decision Tree":
                    model = DecisionTreeRegressor()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    st.write(f"### {model_list} Performance 📈")
                    st.write(f"🔹 MSE: {mse:.2f}")
                    st.write(f"🔹 R² Score: {r2:.2f}")

                elif model_list == "Random Forest":
                    model = RandomForestRegressor()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    st.write(f"### {model_list} Performance 📈")
                    st.write(f"🔹 MSE: {mse:.2f}")
                    st.write(f"🔹 R² Score: {r2:.2f}")

                elif model_list == "K-means":
                    model = KMeans(n_clusters=3)
                    model.fit(X_train_scaled)
                    st.write(f"### {model_list} - Cluster Centers:")
                    st.write(model.cluster_centers_)

                if model is not None:
                    model_filename = model_list.lower().replace(" ", "_") + "_model.pkl"  # ensure model_filename is updated
                    with open(model_filename, "wb") as f:
                        pickle.dump({
                           "model": model,
                           "scaler": scaler,
                           "features": feature_columns,
                           "data": data,
                           "mapping": mapping_dict
                           }, f)
    
                    st.success(f"✅ {model_list} Trained & Saved!")
                    with open(model_filename, "rb") as f:
                        st.download_button(f"Download {model_list} Model", f, model_filename)
                    
# --------------------- Prediction Section --------------------- #
st.write("### Make Predictions \U0001F52E")
if uploaded_file:  
    model_filename = model_list.lower().replace(" ", "_") + "_model.pkl"  
    
    if os.path.exists(model_filename):  
        with open(model_filename, "rb") as f:
            model_data = pickle.load(f)
            model = model_data["model"]
            scaler = model_data["scaler"]
            feature_columns = model_data["features"]
            original_data = model_data.get("data")
            mapping_dict = model_data.get("mapping", {})
        
        input_data = []
            
        for feature in feature_columns:
            if feature in mapping_dict:
                options = list(mapping_dict[feature].keys())
                value = st.selectbox(f"Select {feature}", options)
                encoded_value = mapping_dict[feature][value]  # Map selected label to encoded number
                input_data.append(encoded_value)
            else:
                default_val = float(original_data[feature].mean()) if original_data is not None else 0.0
                value = st.number_input(f"Enter {feature}", value=default_val)
                input_data.append(value)

    if st.button("Predict Value \U0001F3E0"):
        if model is not None and scaler is not None:
            input_array = np.array(input_data).reshape(1, -1)

            # Optional: check input shape
            if input_array.shape[1] != scaler.n_features_in_:
                st.error(f"⚠️ Input shape mismatch: Expected {scaler.n_features_in_} features, got {input_array.shape[1]}")
                st.stop()

            scaled_input = scaler.transform(input_array)
            prediction = model.predict(scaled_input)
            st.success(f"\U0001F4CC Predicted Value: {prediction[0]:.2f}")
        else:
            st.error("Model or Scaler is not available. Please train and save the model first.")
