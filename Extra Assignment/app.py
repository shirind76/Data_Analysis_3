import dash
from dash import dcc, html, Input, Output
import numpy as np
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import random

# Initialize the Dash dashboard
dashboard = dash.Dash(__name__)
dashboard.title = "MSE, Bias, and Variance Simulation Dashboard"

# Layout of the dashboard
dashboard.layout = html.Div([
    html.H1("MSE, Bias, and Variance Simulation for Stock Market Index Prediction"),
    html.P("Imagine you are a financial analyst tasked with predicting the future performance of a stock market index based on investor sentiment data."),
    html.P("Investor sentiment often influences market movements, but capturing this relationship accurately requires choosing the right complexity of the prediction model."),
    html.P("This simulation demonstrates the bias-variance tradeoff by allowing you to adjust the complexity of the model and see how it impacts the prediction accuracy, bias, and variance."),

    # Controls
    html.Div([
        html.Label('Polynomial Degree (Model Complexity)'),
        dcc.Slider(id='degree', min=1, max=10, step=1, value=1,
                   marks={i: str(i) for i in range(1, 11)})
    ], style={'margin': '20px'}),

    html.Div([
        html.Label('Sample Size (Number of Data Points)'),
        dcc.Slider(id='sample_size', min=250, max=3000, step=250, value=100,
                   marks={i: str(i) for i in range(250, 3001, 250)})
    ], style={'margin': '20px'}),

    html.Div([
        html.Label('Noise Level (Data Variability)'),
        dcc.Slider(id='noise_level', min=0, max=1, step=0.1, value=0.5,
                   marks={str(round(i,1)): str(round(i,1)) for i in np.arange(0, 2.1, 0.5)})
    ], style={'margin': '20px'}),

    # Output graphs and metrics
    dcc.Graph(id='data-plot'),
    dcc.Graph(id='prediction-plot'),
    html.Div(id='metrics-output', style={'padding': '10px', 'backgroundColor': '#f0f0f0', 'borderRadius': '5px'})
])

@dashboard.callback(
    [Output('data-plot', 'figure'),
     Output('prediction-plot', 'figure'),
     Output('metrics-output', 'children')],
    [Input('degree', 'value'),
     Input('sample_size', 'value'),
     Input('noise_level', 'value')]
)
def update_dashboard(degree, sample_size, noise_level, n_simulations=100):
    # Set seed for reproducibility
    np.random.seed(42)

    # Generate synthetic "true" data for investor sentiment in [0,10]
    X = np.linspace(0, 20, sample_size)
    y_true = 2 * np.sin(X) + X**0.5 + random.uniform(0,5)# True function
    X = X.reshape(-1, 1)
    
    # Split the data (without noise) into training and test sets.
    # The test set remains fixed to assess bias and variance.
    X_train, X_test, y_true_train, y_true_test = train_test_split(
        X, y_true, test_size=0.5, random_state=42
    )
    
    # Create a polynomial feature transformer.
    poly = PolynomialFeatures(degree=degree)
    # Transform test set features (used in every simulation)
    X_poly_test = poly.fit_transform(X_test)
    
    # Run multiple simulations: add noise to the training data each time,
    # train the model, and predict on the fixed test set.
    y_preds = []
    for _ in range(n_simulations):
        # Add noise to the training responses
        y_train_noisy = y_true_train + np.random.normal(scale=noise_level, size=y_true_train.shape[0])
        X_poly_train = poly.fit_transform(X_train)
        model = LinearRegression()
        model.fit(X_poly_train, y_train_noisy)
        y_pred = model.predict(X_poly_test)
        y_preds.append(y_pred)
    
    y_preds = np.array(y_preds)  # Shape: (n_simulations, n_test)
    
    # Compute the mean prediction and prediction variance for each test point.
    y_pred_mean = np.mean(y_preds, axis=0)
    y_pred_variance = np.var(y_preds, axis=0)
    
    # Direct MSE: average squared error between mean prediction and true test values.
    mse_direct = np.mean((y_pred_mean - y_true_test)**2)
    
    # Compute the squared bias (difference between mean prediction and true value)
    bias_squared = np.mean((y_pred_mean - y_true_test)**2)
    
    # Compute the variance: average of the prediction variances across test points.
    variance_val = np.mean(y_pred_variance)
    
    # MSE decomposition: MSE should equal Bias² + Variance.
    mse_decomposed = bias_squared + variance_val

    # Data Plot: Plot training data, test data, and the true function.
    # Training data is shown with added noise (one instance).
    y_train_example = y_true_train + np.random.normal(scale=noise_level, size=y_true_train.shape[0])
    data_fig = go.Figure()
    data_fig.add_trace(go.Scatter(
        x=X_train.flatten(), y=y_train_example, mode='markers',
        name='Training Data', marker=dict(color='blue')
    ))
    data_fig.add_trace(go.Scatter(
        x=X_test.flatten(), y=y_true_test, mode='markers',
        name='Test Data', marker=dict(color='red', symbol='diamond')
    ))
    data_fig.add_trace(go.Scatter(
        x=X.flatten(), y=y_true, mode='lines',
        name='True Function', line=dict(color='black', dash='dash')
    ))
    data_fig.update_layout(
        title="Training and Test Data with True Function",
        xaxis_title="Investor Sentiment",
        yaxis_title="Stock Market Index"
    )
    
    # Prediction Plot: Plot test true values and mean predictions.
    sorted_indices = np.argsort(X_test.flatten())
    pred_fig = go.Figure()
    pred_fig.add_trace(go.Scatter(
        x=X_test.flatten()[sorted_indices],
        y=y_true_test[sorted_indices],
        mode='markers', name='Test True Values', marker=dict(color='red')
    ))
    pred_fig.add_trace(go.Scatter(
        x=X_test.flatten()[sorted_indices],
        y=y_pred_mean[sorted_indices],
        mode='lines', name='Mean Prediction', line=dict(color='green')
    ))
    pred_fig.update_layout(
        title="Test Set: True Values vs. Mean Predictions",
        xaxis_title="Investor Sentiment",
        yaxis_title="Stock Market Index"
    )
    
    # Build the metrics output including formulas.
    tol = 0.05
    color_validation = 'green' if abs(mse_direct - mse_decomposed) < tol else 'red'
    metrics = html.Div([
        html.H3("Model Metrics"),
        html.P(f"MSE: {mse_direct:.2f}"),
        html.P("Formula: MSE = (1/N) Σ (ŷ - y)²"),
        html.Br(),
        html.P(f"Bias²: {bias_squared:.2f}"),
        html.P("Formula: Bias² = ( (1/N)Σ (ŷ - ȳ))²"),
        html.Br(),
        html.P(f"Variance: {variance_val:.2f}"),
        html.P("Formula: Variance = (1/N) Σ (ŷ-ȳ)²"),
        html.Br(),
        html.P(f" MSE =Bias² + Variance: {mse_decomposed:.2f}"),
        html.P("Validation: MSE ≈ Bias² + Variance", style={'color': color_validation})
    ])
    
    return data_fig, pred_fig, metrics

# Run the dashboard
if __name__ == '__main__':
    print("Dashboard is running...")
    dashboard.run_server(debug=True)
