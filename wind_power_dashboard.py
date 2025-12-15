import base64
import io
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import pickle
import warnings
import torch
from datetime import timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from pytorch_forecasting import TemporalFusionTransformer

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
MODEL_CHECKPOINT = "tft_model.ckpt"
METADATA_FILE = "tft_metadata.pkl"
DEFAULT_FORECAST_HORIZON = 12
CONTEXT_WINDOW = 24

# Load Model
loaded_model = None
model_status_msg = "Checking model..."
model_status_color = "warning"

try:
    print(f"Loading model from {MODEL_CHECKPOINT}...")
    loaded_model = TemporalFusionTransformer.load_from_checkpoint(MODEL_CHECKPOINT)
    loaded_model.eval()
    model_status_msg = "✅ Model Loaded Successfully"
    model_status_color = "success"
    print("Model loaded.")
except Exception as e:
    model_status_msg = f"❌ Model Load Failed: {str(e)}"
    model_status_color = "danger"
    print(f"Error: {e}")

# ==================== PROCESSING FUNCTIONS ====================
def add_features(df, fill_na=True):
    """Add cyclic, lag, and rolling features for TFT model"""
    df = df.copy()
    if df["Time"].dtype == 'object':
        df["Time"] = pd.to_datetime(
            df["Time"].astype(str).str.strip(),
            format="%d/%m/%Y %H:%M",
            errors="coerce"
        )
    
    # === Handle Missing Time Steps ===
    df = df.sort_values('Time').set_index('Time')
    df = df.resample('1H').asfreq()
    df = df.reset_index()
    
    # Interpolate
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate().ffill().bfill()
    
    # Time Index
    start_time = df['Time'].min()
    df['idx'] = (df['Time'] - start_time).dt.total_seconds() // 3600
    df['idx'] = df['idx'].astype(int)
    
    # Group
    if 'group' not in df.columns:
        df['group'] = '0'
    df['group'] = df['group'].fillna('0')

    # Cyclic
    df['hour'] = df['Time'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    df['month'] = df['Time'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    df['dayofweek'] = df['Time'].dt.dayofweek
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # Lags
    lags = [1, 2, 6, 12, 24, 48]
    for lag in lags:
        df[f'Power_lag{lag}'] = df['Power'].shift(lag)
        df[f'windspeed_100m_lag{lag}'] = df['windspeed_100m'].shift(lag)
    
    # Rolling
    windows = [6, 12, 24]
    for window in windows:
        df[f'Power_roll_mean_{window}'] = df['Power'].rolling(window).mean()
        df[f'windspeed_100m_roll_mean_{window}'] = df['windspeed_100m'].rolling(window).mean()
        df[f'windspeed_100m_roll_std_{window}'] = df['windspeed_100m'].rolling(window).std()
    
    if fill_na:
        cols = [c for c in df.columns if c != 'Power']
        num_cols = df.select_dtypes(include=[np.number]).columns
        cols_num = [c for c in cols if c in num_cols]
        df[cols_num] = df[cols_num].interpolate().ffill().bfill()
        df = df.fillna(0)
        
    return df

def run_prediction_with_interpretation(df, model):
    """Run model.predict and extract quantiles AND interpretation"""
    try:
        # 1. Predict
        raw_prediction = model.predict(df, mode="raw", return_x=True)
        preds = raw_prediction.output["prediction"] # (Batch, Time, Quantiles)
        
        # 2. Interpret (Feature Importance & Attention)
        # === FIX: Use reduction="none" to preserve batch dimension ===
        interpretation = model.interpret_output(raw_prediction.output, reduction="none")
        
        # Take the last prediction step (latest forecast)
        latest_pred = preds[-1] 
        
        # Extract quantiles
        n_q = latest_pred.shape[-1]
        res = {'quantiles': {}}
        
        if n_q >= 7:
            res['quantiles']['p50'] = latest_pred[:, 3].cpu().numpy()
            res['quantiles']['p10'] = latest_pred[:, 1].cpu().numpy()
            res['quantiles']['p90'] = latest_pred[:, 5].cpu().numpy()
            res['quantiles']['p02'] = latest_pred[:, 0].cpu().numpy()
            res['quantiles']['p98'] = latest_pred[:, 6].cpu().numpy()
        else:
            res['quantiles']['p50'] = latest_pred[:, 0].cpu().numpy()
            res['quantiles']['p10'] = res['quantiles']['p50']
            res['quantiles']['p90'] = res['quantiles']['p50']
            res['quantiles']['p02'] = res['quantiles']['p50']
            res['quantiles']['p98'] = res['quantiles']['p50']
        
        # Store Interpretation Data
        feat_importance = {}
        for key in ['encoder_variables', 'decoder_variables', 'static_variables']:
            if key in interpretation:
                # With reduction="none", shape is (Batch, Features)
                # We select [-1] for the last sample
                weights = interpretation[key][-1].cpu().numpy()
                
                # Normalize
                total = weights.sum() + 1e-9
                feat_importance[key] = (weights / total) * 100
                
        res['importance'] = feat_importance
        
        # Attention
        if 'attention' in interpretation:
            # Shape: (Batch, Prediction_Length, Encoder_Length)
            # We take the last sample -> (Prediction_Length, Encoder_Length)
            attn = interpretation['attention'][-1].cpu().numpy()
            
            # Check dimensions before averaging
            if attn.ndim == 2:
                # Average over the horizon to see "general importance of past steps"
                avg_attn = attn.mean(axis=0) 
                res['attention'] = avg_attn
            elif attn.ndim == 1:
                # Fallback if attention is already collapsed
                res['attention'] = attn
            else:
                res['attention'] = None
            
        return res, raw_prediction.x

    except Exception as e:
        # Graceful fallback if interpretation fails
        print(f"Interpretation Error: {e}")
        # Re-raise or return minimal results
        raise e

# ==================== DASH APP ====================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([dbc.Col([
        html.H2("⚡ TFT Wind Power Dashboard", className="text-center mt-4"),
        html.P("Forecast | Evaluation | Feature Attribution", className="text-center text-muted mb-4")
    ])]),
    
    # Upload & Controls
    dbc.Row([
        dbc.Col([
            dbc.Card([dbc.CardBody([
                html.H6("1. Upload Data (CSV)", className="card-subtitle text-muted mb-2"),
                dcc.Upload(
                    id='upload-data',
                    children=html.Div(['📂 Drag & Drop or Click to Upload']),
                    style={'border': '2px dashed #007bff', 'borderRadius': '10px', 'textAlign': 'center', 'padding': '15px', 'cursor': 'pointer', 'backgroundColor': '#f8f9fa'},
                    multiple=False
                ),
                html.Div(id='file-name-display', className="mt-2 text-primary small fw-bold")
            ])], className="h-100")
        ], md=8),
        
        dbc.Col([
            dbc.Card([dbc.CardBody([
                html.H6("2. System Status", className="card-subtitle text-muted mb-2"),
                dbc.Badge(model_status_msg, color=model_status_color, className="mb-2 p-2 w-100"),
                dbc.Button("🚀 Run Analysis", id='predict-btn', color="primary", className="w-100", disabled=(loaded_model is None))
            ])], className="h-100")
        ], md=4)
    ], className="mb-4"),
    
    # Main Content Tabs
    dbc.Tabs([
        dbc.Tab(label="📈 Forecast & Backtest", tab_id="tab-forecast"),
        dbc.Tab(label="🧠 Model Insights (XAI)", tab_id="tab-insights"),
        dbc.Tab(label="📊 Evaluation Metrics", tab_id="tab-metrics"),
    ], id="tabs", active_tab="tab-forecast", className="mb-3"),
    
    html.Div(id='tab-content')
    
], fluid=True, style={'backgroundColor': '#f4f6f9', 'minHeight': '100vh', 'padding': '20px'})

# --- Callbacks ---
@callback(Output('file-name-display', 'children'), Input('upload-data', 'filename'))
def display_name(name): return f"📄 {name}" if name else ""

@callback(
    Output('tab-content', 'children'),
    [Input('predict-btn', 'n_clicks'), Input('tabs', 'active_tab')],
    [State('upload-data', 'contents'), State('upload-data', 'filename')],
    prevent_initial_call=False
)
def render_content(n_clicks, active_tab, contents, filename):
    # Initial Load
    if not n_clicks and not contents:
        return dbc.Alert("Please upload a file and click 'Run Analysis' to begin.", color="info")
    
    if not contents:
        return dbc.Alert("⚠️ No file uploaded.", color="warning")

    try:
        # Process Data
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        
        if len(df) < (CONTEXT_WINDOW + DEFAULT_FORECAST_HORIZON):
             return dbc.Alert("❌ Not enough data. Need at least 48 hours.", color="danger")
        
        df_proc = add_features(df, fill_na=True)
        
        # === 1. Run Calculations (Validation & Future) ===
        
        # A. Validation (Backtest Last 12h)
        val_res, val_x = run_prediction_with_interpretation(df_proc, loaded_model)
        val_preds = val_res['quantiles']
        
        val_time = df_proc['Time'].iloc[-DEFAULT_FORECAST_HORIZON:]
        val_actual = df_proc['Power'].iloc[-DEFAULT_FORECAST_HORIZON:].values
        
        # Metrics Calculation
        mae = mean_absolute_error(val_actual, val_preds['p50'])
        rmse = np.sqrt(mean_squared_error(val_actual, val_preds['p50']))
        mape = mean_absolute_percentage_error(val_actual, val_preds['p50'])
        r2 = r2_score(val_actual, val_preds['p50'])

        # B. Future Forecast
        last_time = df_proc['Time'].iloc[-1]
        future_times = [last_time + timedelta(hours=x) for x in range(1, 13)]
        future_df = pd.DataFrame({'Time': future_times})
        df_extended = pd.concat([df_proc, future_df], ignore_index=True).ffill()
        df_extended.loc[df_extended.index[-12:], 'Power'] = np.nan
        df_fut_proc = add_features(df_extended, fill_na=True)
        
        fut_res, _ = run_prediction_with_interpretation(df_fut_proc, loaded_model)
        fut_preds = fut_res['quantiles']

        # === 2. Render Based on Tab ===
        
        if active_tab == "tab-forecast":
            # --- Forecast Plots ---
            # Validation Plot
            fig_val = go.Figure()
            fig_val.add_trace(go.Scatter(x=list(val_time)+list(val_time)[::-1], y=list(val_preds['p90'])+list(val_preds['p10'])[::-1], fill='toself', fillcolor='rgba(255,0,0,0.1)', line=dict(width=0), name='90% Conf'))
            fig_val.add_trace(go.Scatter(x=val_time, y=val_preds['p50'], name='Predicted', line=dict(color='red')))
            fig_val.add_trace(go.Scatter(x=val_time, y=val_actual, name='Actual', mode='markers+lines', line=dict(color='blue')))
            fig_val.update_layout(title="Backtest: Model vs Actual (Last 12h)", template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
            
            # Future Plot
            fig_fut = go.Figure()
            fig_fut.add_trace(go.Scatter(x=list(future_times)+list(future_times)[::-1], y=list(fut_preds['p98'])+list(fut_preds['p02'])[::-1], fill='toself', fillcolor='rgba(0,0,255,0.1)', line=dict(width=0), name='98% Conf'))
            fig_fut.add_trace(go.Scatter(x=list(future_times)+list(future_times)[::-1], y=list(fut_preds['p90'])+list(fut_preds['p10'])[::-1], fill='toself', fillcolor='rgba(0,0,255,0.2)', line=dict(width=0), name='90% Conf'))
            fig_fut.add_trace(go.Scatter(x=future_times, y=fut_preds['p50'], name='Future Forecast', line=dict(color='blue', width=3)))
            fig_fut.update_layout(title="Future Forecast (Next 12h)", template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
            
            return html.Div([
                dbc.Row([dbc.Col(dcc.Graph(figure=fig_val), lg=6), dbc.Col(dcc.Graph(figure=fig_fut), lg=6)])
            ])

        elif active_tab == "tab-insights":
            # --- Feature Importance & Attention ---
            # 1. Feature Importance Plot
            importance_data = []
            if 'encoder_variables' in fut_res['importance']:
                # Try to use model names if available, else indices
                try:
                    enc_names = loaded_model.encoder_variables
                except:
                    enc_names = [f"Enc_{i}" for i in range(len(fut_res['importance']['encoder_variables']))]
                    
                vals = fut_res['importance']['encoder_variables']
                if len(enc_names) == len(vals):
                    for n, v in zip(enc_names, vals): importance_data.append({'Feature': n, 'Importance': v, 'Type': 'Encoder'})
            
            if 'decoder_variables' in fut_res['importance']:
                try:
                    dec_names = loaded_model.decoder_variables
                except:
                    dec_names = [f"Dec_{i}" for i in range(len(fut_res['importance']['decoder_variables']))]

                vals = fut_res['importance']['decoder_variables']
                if len(dec_names) == len(vals):
                    for n, v in zip(dec_names, vals): importance_data.append({'Feature': n, 'Importance': v, 'Type': 'Decoder'})
            
            if importance_data:
                df_imp = pd.DataFrame(importance_data).sort_values('Importance', ascending=True)
                fig_imp = px.bar(df_imp, x='Importance', y='Feature', color='Type', orientation='h', title="Feature Importance (Variable Selection Network)")
                fig_imp.update_layout(template="plotly_white")
            else:
                fig_imp = go.Figure().add_annotation(text="No feature importance available", showarrow=False)
            
            # 2. Attention Plot
            if 'attention' in fut_res and fut_res['attention'] is not None:
                attn_weights = fut_res['attention']
                lookback_len = len(attn_weights)
                x_axis = np.arange(-lookback_len, 0)
                
                fig_attn = go.Figure()
                fig_attn.add_trace(go.Scatter(x=x_axis, y=attn_weights, fill='tozeroy', mode='lines', line=dict(color='purple')))
                fig_attn.update_layout(
                    title="Temporal Attention: Which past hours mattered most?",
                    xaxis_title="Time relative to Forecast Start (Hours)",
                    yaxis_title="Attention Weight",
                    template="plotly_white"
                )
            else:
                fig_attn = go.Figure().add_annotation(text="Attention weights not available", showarrow=False)
            
            return html.Div([
                dbc.Row([dbc.Col(dcc.Graph(figure=fig_imp), lg=6), dbc.Col(dcc.Graph(figure=fig_attn), lg=6)]),
                dbc.Alert("💡 'Attention' shows which historical time steps the model focused on to make the current prediction.", color="light", className="mt-2")
            ])

        elif active_tab == "tab-metrics":
            # --- Metrics Table & Residuals ---
            cards = dbc.Row([
                dbc.Col(dbc.Card([dbc.CardBody([html.H4(f"{mae:.4f}", className="text-primary"), html.P("MAE (Mean Abs Error)", className="small text-muted")])], className="text-center shadow-sm"), width=3),
                dbc.Col(dbc.Card([dbc.CardBody([html.H4(f"{rmse:.4f}", className="text-danger"), html.P("RMSE (Root Mean Sq Error)", className="small text-muted")])], className="text-center shadow-sm"), width=3),
                dbc.Col(dbc.Card([dbc.CardBody([html.H4(f"{mape:.2%}", className="text-warning"), html.P("MAPE (Mean Abs % Error)", className="small text-muted")])], className="text-center shadow-sm"), width=3),
                dbc.Col(dbc.Card([dbc.CardBody([html.H4(f"{r2:.4f}", className="text-success"), html.P("R² Score", className="small text-muted")])], className="text-center shadow-sm"), width=3),
            ], className="mb-4")
            
            residuals = val_actual - val_preds['p50']
            fig_res = go.Figure()
            fig_res.add_trace(go.Bar(x=val_time, y=residuals, name='Residuals', marker_color='gray'))
            fig_res.add_hline(y=0, line_dash="dash", line_color="black")
            fig_res.update_layout(title="Prediction Residuals (Actual - Predicted)", xaxis_title="Time", yaxis_title="Error (MW)", template="plotly_white")
            
            fig_scat = px.scatter(x=val_actual, y=val_preds['p50'], labels={'x': 'Actual Power', 'y': 'Predicted Power'}, title="Actual vs Predicted Scatter")
            fig_scat.add_shape(type="line", x0=val_actual.min(), y0=val_actual.min(), x1=val_actual.max(), y1=val_actual.max(), line=dict(color="Red", dash="dash"))
            fig_scat.update_layout(template="plotly_white")

            return html.Div([
                cards,
                dbc.Row([dbc.Col(dcc.Graph(figure=fig_res), lg=8), dbc.Col(dcc.Graph(figure=fig_scat), lg=4)])
            ])
            
    except Exception as e:
        import traceback
        return dbc.Alert([html.H5("Error Processing Data", className="alert-heading"), html.Pre(str(e)), html.Pre(traceback.format_exc())], color="danger")

if __name__ == '__main__':
    app.run(debug=True, port=8050)