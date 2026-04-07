import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

st.set_page_config(page_title="Linear Regression Interactive", layout="wide")

st.title("Interactive Linear Regression Learning Journey")

st.sidebar.header("Controls")
dataset_type = st.sidebar.selectbox("Dataset", ["Clean", "Noisy", "Outliers", "Custom (Add manually)"])
n_samples = st.sidebar.slider("Number of samples", 10, 200, 50) if dataset_type != "Custom (Add manually)" else 0

# True parameters
true_m = 2.0
true_b = 1.0

# Generate Data
np.random.seed(42)

if dataset_type == "Custom (Add manually)":
    st.sidebar.write("Add your own data points below:")
    if "custom_data" not in st.session_state:
        st.session_state.custom_data = pd.DataFrame({"X": [1.0, 2.0, 3.0], "y": [2.5, 4.3, 6.1]})
    
    edited_df = st.sidebar.data_editor(st.session_state.custom_data, num_rows="dynamic")
    st.session_state.custom_data = edited_df
    
    X = edited_df["X"].values
    y = edited_df["y"].values
    n_samples = len(X) if len(X) > 0 else 1 # avoid division by zero
else:
    X = np.linspace(0, 10, n_samples)
    noise_level = st.sidebar.slider("Noise level", 0.0, 5.0, 1.0) if dataset_type != "Clean" else 0.0
    y = true_m * X + true_b + np.random.randn(n_samples) * noise_level

    if dataset_type == "Outliers":
        n_outliers = int(n_samples * 0.1)
        outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
        y[outlier_indices] += np.random.choice([-1, 1], n_outliers) * np.random.uniform(10, 20, n_outliers)

st.sidebar.markdown("---")
st.sidebar.header("Model Parameters")
m = st.sidebar.slider("Slope (m)", -5.0, 10.0, 0.0, 0.1)
b = st.sidebar.slider("Intercept (b)", -10.0, 10.0, 0.0, 0.1)

lr = st.sidebar.selectbox("Learning Rate (\u03B1)", [0.001, 0.01, 0.1, 0.5])

# Predictions & Error
y_pred = m * X + b
residuals = y - y_pred
mse = np.mean(residuals**2)

tab1, tab2, tab3, tab4 = st.tabs(["1. Data & Line Fit", "2. Error (MSE)", "3. Loss Surface", "4. Gradient Descent"])

# Combine data for native Streamlit charts
df_fit = pd.DataFrame({"X": X, "Actual Data": y, "Linear Fit": y_pred})

with tab1:
    st.header("Data Distribution & Line Fitting")
    st.write("Adjust the slope `m` and intercept `b` in the sidebar to fit the line to the data points. Notice how the line updates in real-time.")
    
    # Using Altair (built into Streamlit) to show points for data and a line for the fit
    base = alt.Chart(df_fit).encode(x='X')
    scatter = base.mark_circle(size=60, color='blue').encode(y='Actual Data', tooltip=['X', 'Actual Data'])
    line = base.mark_line(color='red', size=3).encode(y='Linear Fit', tooltip=['X', 'Linear Fit'])
    st.altair_chart(scatter + line, use_container_width=True)

with tab2:
    st.header("Error / Loss Function (MSE)")
    st.write(f"Mean Squared Error (MSE): **{mse:.4f}**")
    st.write("The chart below represents the squared residuals (errors) between the predictions and the actual data.")
    
    df_residuals = pd.DataFrame({"Squared Error": residuals**2}, index=X)
    st.bar_chart(df_residuals)

with tab3:
    st.header("Loss Curve (1D slice)")
    st.write("Since we are keeping it basic, here is a 1D slice of the MSE Loss if we vary `m` around your chosen value, keeping `b` constant.")
    
    m_range = np.linspace(-5, 10, 50)
    loss_slice = []
    for test_m in m_range:
        test_preds = test_m * X + b
        loss_slice.append(np.mean((y - test_preds)**2))
        
    df_loss = pd.DataFrame({"MSE": loss_slice}, index=m_range)
    st.line_chart(df_loss)

with tab4:
    st.header("Gradient Descent (Optimization Process)")
    epochs = st.slider("Number of Iterations", 1, 100, 20)
    if st.button("Run Gradient Descent"):
        curr_m, curr_b = np.random.randn(), np.random.randn()
        losses = [np.mean((y - (curr_m*X+curr_b))**2)]
        
        for _ in range(epochs):
            y_p = curr_m * X + curr_b
            dm = (-2/n_samples) * sum(X * (y - y_p))
            db = (-2/n_samples) * sum(y - y_p)
            curr_m = curr_m - lr * dm
            curr_b = curr_b - lr * db
            losses.append(np.mean((y - (curr_m*X+curr_b))**2))
            
        st.write("Learning Curve (Loss vs Iteration)")
        st.line_chart(pd.DataFrame({"MSE Loss": losses}))
        st.success(f"Final parameters after {epochs} iterations: m={curr_m:.2f}, b={curr_b:.2f}")
