import base64
import io
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import pickle
import warnings
import torch
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from pytorch_forecasting import TemporalFusionTransformer

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
MODEL_CHECKPOINT = "tft_model.ckpt"
METADATA_FILE = "tft_metadata.pkl"
MAX_FORECAST_HORIZON = 12
DEFAULT_FORECAST_HORIZON = 12  # 12 hours

print("Loading PyTorch Lightning model...")
try:
    loaded_model = TemporalFusionTransformer.load_from_checkpoint(MODEL_CHECKPOINT)
    loaded_model.eval()
    
    with open(METADATA_FILE, 'rb') as f:
        loaded_metadata = pickle.load(f)
    
    print("✓ Model and metadata loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    loaded_model = None
    loaded_metadata = None

# ==================== FEATURE ENGINEERING ====================
def add_features(df):
    """Add cyclic, lag, and rolling features for TFT model"""
    df = df.copy()
    df["Time"] = pd.to_datetime(
    df["Time"].astype(str).str.strip(),
    format="%d/%m/%Y %H:%M",
    errors="coerce"
    )

    df = df.dropna(subset=["Time"])

    df = df.set_index("Time")
    df = df.sort_index()

    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="H"
    )

    df = df.reindex(full_index)
    df = df.interpolate().ffill().bfill()

    # Cyclic features
    df['hour'] = df['Time'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    df['month'] = df['Time'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    df['dayofweek'] = df['Time'].dt.dayofweek
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # Lag features
    lags = [1, 2, 6, 12, 24, 48]
    for lag in lags:
        df[f'Power_lag{lag}'] = df['Power'].shift(lag)
        df[f'windspeed_100m_lag{lag}'] = df['windspeed_100m'].shift(lag)
    
    # Rolling features
    windows = [6, 12, 24]
    for window in windows:
        df[f'Power_roll_mean_{window}'] = df['Power'].rolling(window).mean()
        df[f'windspeed_100m_roll_mean_{window}'] = df['windspeed_100m'].rolling(window).mean()
        df[f'windspeed_100m_roll_std_{window}'] = df['windspeed_100m'].rolling(window).std()
    
    # Drop NaNs created by lags/rolling
    df = df.dropna().reset_index(drop=True)
    return df

# ==================== PREDICTION FUNCTION ====================
def predict_wind_power(df, forecast_hours, model, metadata, context_window=24):
    """Predict wind power using TFT with engineered features"""
    try:
        df_copy = add_features(df)
        
        # Remove duplicates and sort
        df_copy = df_copy.drop_duplicates(subset='Time', keep='first')
        df_copy = df_copy.sort_values('Time').reset_index(drop=True)
        df_copy = df_copy.set_index('Time')
        
        # Fill missing hours
        df_copy = df_copy.asfreq('h')
        df_copy = df_copy.interpolate(method='linear')
        df_copy = df_copy.fillna(method='bfill').fillna(method='ffill')
        
        if len(df_copy) < context_window:
            return {'error': f'Data requires {context_window} hours, got {len(df_copy)}'}
        
        # Get target and feature columns
        target_col = metadata.get('target_col', 'Power')
        feature_cols = [col for col in df_copy.columns if col != target_col]
        
        required_cols = [target_col] + feature_cols
        if not all(col in df_copy.columns for col in required_cols):
            return {'error': f'Missing required columns: {set(required_cols) - set(df_copy.columns)}'}
        
        # Scale numeric features dynamically
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(df_copy[required_cols])
        target_values = data_scaled[:, 0]
        feature_values = data_scaled[:, 1:]
        
        target_tensor = torch.FloatTensor(target_values).unsqueeze(-1)
        features_tensor = torch.FloatTensor(feature_values)
        
        # Context window
        context_data = target_tensor[-context_window:].unsqueeze(0)
        context_features = features_tensor[-context_window:].unsqueeze(0)
        
        # Prediction
        with torch.no_grad():
            predictions = model(context_data, context_features)
            if predictions.dim() > 2:
                forecast_scaled = predictions[0, :forecast_hours, 0].cpu().numpy()
            else:
                forecast_scaled = predictions[0, :forecast_hours].cpu().numpy()
        
        # Inverse scale
        forecast_values = scaler.inverse_transform(
            np.hstack([forecast_scaled.reshape(-1, 1), np.zeros((forecast_hours, len(feature_cols)))])
        )[:, 0]
        actual_values = scaler.inverse_transform(data_scaled)[:, 0]
        
        # Forecast timestamps
        last_timestamp = df_copy.index[-1]
        forecast_timestamps = pd.date_range(
            start=last_timestamp + timedelta(hours=1),
            periods=forecast_hours,
            freq='h'
        )
        
        return {
            'success': True,
            'forecast_values': forecast_values[:forecast_hours],
            'forecast_timestamps': forecast_timestamps,
            'actual_values': actual_values,
            'actual_timestamps': df_copy.index,
            'stats': {
                'forecast_min': float(np.min(forecast_values[:forecast_hours])),
                'forecast_max': float(np.max(forecast_values[:forecast_hours])),
                'forecast_mean': float(np.mean(forecast_values[:forecast_hours])),
                'actual_mean': float(np.mean(actual_values))
            }
        }
    
    except Exception as e:
        return {'error': str(e)}

# ==================== DASH APP ====================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    # Header
    dbc.Row([dbc.Col([html.H1("⚡ Wind Power Forecasting Dashboard", className="text-center mt-4 mb-2"),
                      html.P("Upload 24 hours of historical data to forecast the next 12 hours", 
                             className="text-center text-muted mb-4")])]),
    
    # Upload & Settings
    dbc.Row([
        dbc.Col([dbc.Card([dbc.CardBody([
            html.H5("📤 Upload Historical Data (CSV)", className="card-title"),
            html.P("Required columns: Time, Power, temperature_2m, relativehumidity_2m, dewpoint_2m, windspeed_10m, windspeed_100m, winddirection_10m, winddirection_100m, windgusts_10m", className="text-muted small"),
            html.P("✓ Required: 24 hours of hourly data", className="text-info small fw-bold"),
            dcc.Upload(id='upload-data', children=html.Div(['📁 Drag and Drop or ', html.A('Select Files')]),
                       style={'width': '100%','height': '60px','lineHeight': '60px','borderWidth': '2px','borderStyle': 'dashed','borderRadius': '5px','textAlign': 'center','cursor': 'pointer','backgroundColor': '#f8f9fa'},
                       multiple=False),
            html.Div(id='upload-status', className="mt-2")
        ])], className="mb-3")], lg=8),
        
        dbc.Col([dbc.Card([dbc.CardBody([
            html.H5("⚙️ Prediction Settings", className="card-title"),
            html.Label("Forecast Hours:", className="fw-bold"),
            dcc.Slider(id='forecast-slider', min=1, max=MAX_FORECAST_HORIZON, step=1, value=DEFAULT_FORECAST_HORIZON,
                       marks={i: f"{i}h" for i in range(1, MAX_FORECAST_HORIZON + 1, 3)},
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Label("Model Status:", className="fw-bold mt-4"),
            html.P("✓ PyTorch Lightning Ready" if loaded_model else "✗ Not Loaded", 
                   className="text-success" if loaded_model else "text-danger"),
            dbc.Button("📊 Generate Forecast", id='predict-btn', color="primary", className="mt-3 w-100", disabled=not loaded_model)
        ])])], lg=4)
    ], className="mb-4"),
    
    # Messages
    html.Div(id='error-message', className="mb-3"),
    html.Div(id='success-message', className="mb-3"),
    
    # Store
    dcc.Store(id='forecast-data-store'),
    
    # Results
    dbc.Row([dbc.Col([dbc.Card([dbc.CardBody([html.H5("📈 Forecast Chart", className="card-title mb-3"), dcc.Graph(id='forecast-graph')])], className="mb-3")])]),
    
    # Statistics & Table
    dbc.Row([
        dbc.Col([dbc.Card([dbc.CardBody([html.H5("📊 Forecast Statistics", className="card-title"),
                                         dbc.Row([dbc.Col([html.P("Max Power", className="text-muted small mb-0"), html.H4(id='stat-max', children="--")], md=6),
                                                  dbc.Col([html.P("Min Power", className="text-muted small mb-0"), html.H4(id='stat-min', children="--")], md=6)]),
                                         dbc.Row([dbc.Col([html.P("Mean Power", className="text-muted small mb-0"), html.H4(id='stat-mean', children="--")], md=6),
                                                  dbc.Col([html.P("Historical Avg", className="text-muted small mb-0"), html.H4(id='stat-hist-mean', children="--")], md=6)], className="mt-3")])], className="mb-3")], lg=4),
        dbc.Col([dbc.Card([dbc.CardBody([html.H5("📋 Detailed Forecast", className="card-title"), html.Div(id='forecast-table')])])], lg=8)
    ], className="mb-4"),
    
], fluid=True, style={'backgroundColor': '#f8f9fa', 'minHeight': '100vh', 'padding': '20px'})

# ==================== CALLBACKS ====================
@callback(
    [Output('forecast-data-store', 'data'),
     Output('upload-status', 'children')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def handle_upload(contents, filename):
    if contents is None:
        return None, None
    
    try:
        content_string = contents.split(',')[1]
        decoded = pd.read_csv(io.StringIO(base64.b64decode(content_string).decode('utf-8')))
        required_cols = ['Time', 'Power', 'temperature_2m', 'relativehumidity_2m','dewpoint_2m', 'windspeed_10m', 'windspeed_100m', 'winddirection_10m', 'winddirection_100m', 'windgusts_10m']
        if not all(col in decoded.columns for col in required_cols):
            return None, dbc.Alert("❌ Missing required columns!", color="danger")
        if len(decoded) < 24:
            return None, dbc.Alert(f"❌ Need 24 hours of data, got {len(decoded)}", color="danger")
        
        # Apply features before storing
        decoded_features = add_features(decoded)
        return decoded_features.to_json(date_format='iso', orient='split'), \
               dbc.Alert(f"✓ {filename} loaded ({len(decoded_features)} rows)", color="success")
    
    except Exception as e:
        return None, dbc.Alert(f"❌ Error: {str(e)}", color="danger")

@callback(
    [Output('forecast-graph', 'figure'),
     Output('stat-max', 'children'),
     Output('stat-min', 'children'),
     Output('stat-mean', 'children'),
     Output('stat-hist-mean', 'children'),
     Output('forecast-table', 'children'),
     Output('error-message', 'children')],
    Input('predict-btn', 'n_clicks'),
    State('forecast-data-store', 'data'),
    State('forecast-slider', 'value'),
    prevent_initial_call=True
)
def generate_forecast(n_clicks, stored_data, forecast_hours):
    if stored_data is None:
        empty_fig = go.Figure()
        return empty_fig, '--', '--', '--', '--', dbc.Alert("Upload data first", color="warning"), dbc.Alert("❌ No data uploaded", color="danger")
    
    df = pd.read_json(io.StringIO(stored_data), orient='split')
    results = predict_wind_power(df, forecast_hours, loaded_model, loaded_metadata)
    
    if 'error' in results:
        empty_fig = go.Figure()
        return empty_fig, '--', '--', '--', '--', dbc.Alert(f"Error: {results['error']}", color="danger"), dbc.Alert(f"❌ {results['error']}", color="danger")
    
    # Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results['actual_timestamps'], y=results['actual_values'], name='Historical', mode='lines', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=results['forecast_timestamps'], y=results['forecast_values'], name=f'{forecast_hours}-Hour Forecast', mode='lines+markers', line=dict(color='red', width=3, dash='dash'), marker=dict(size=8)))
    fig.update_layout(title=f"Wind Power Forecast ({forecast_hours} Hours)", xaxis_title="Time", yaxis_title="Power (MW)", hovermode='x unified', template='plotly_white', height=500)
    
    # Table
    table_data = [{'Time': ts.strftime('%Y-%m-%d %H:%M'), 'Power (MW)': f"{val:.2f}"} for ts, val in zip(results['forecast_timestamps'], results['forecast_values'])]
    table = dbc.Table.from_dataframe(pd.DataFrame(table_data), striped=True, bordered=True, hover=True, className="text-center")
    
    return fig, f"{results['stats']['forecast_max']:.2f} MW", f"{results['stats']['forecast_min']:.2f} MW", f"{results['stats']['forecast_mean']:.2f} MW", f"{results['stats']['actual_mean']:.2f} MW", table, None

# ==================== RUN APP ====================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Wind Power Dashboard Starting (PyTorch Lightning)...")
    print("📊 Open http://127.0.0.1:8050 in your browser")
    print("="*70 + "\n")
    app.run(debug=False, port=8050, host='127.0.0.1')
