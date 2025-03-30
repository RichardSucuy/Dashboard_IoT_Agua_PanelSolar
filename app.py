import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
import datetime
from sklearn.linear_model import LinearRegression

# Simulación de datos progresivos
np.random.seed(42)
time_series = pd.date_range(start=datetime.datetime.now(), periods=50, freq="10min")

base_turbidez = 3.0
base_filtros = 90.0
base_uv = 10.0
base_moringa = 80.0
base_ph = 7.0
base_tds = 250.0
base_temp = 25.0

data = pd.DataFrame({
    "Timestamp": time_series,
    "Turbidez (NTU)": base_turbidez + np.cumsum(np.random.normal(0, 0.2, len(time_series))),
    "Estado de Filtros (%)": base_filtros - np.cumsum(np.random.normal(0, 0.5, len(time_series))),
    "Intensidad UV (W/m²)": base_uv + np.cumsum(np.random.normal(0, 0.3, len(time_series))),
    "Estado Purificador Moringa (%)": base_moringa - np.cumsum(np.random.normal(0, 0.2, len(time_series))),
    "pH": base_ph + np.cumsum(np.random.normal(0, 0.05, len(time_series))),
    "TDS (ppm)": base_tds + np.cumsum(np.random.normal(0, 5, len(time_series))),
    "Temperatura (°C)": base_temp + np.cumsum(np.random.normal(0, 0.1, len(time_series)))
})

# Inicializar la app Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1("Dashboard de Purificación de Agua", className="text-center mt-4"),

    html.H2("📡 Monitoreo en Tiempo Real", className="text-center mt-4"),
    dbc.Row([
        dbc.Col([dcc.Graph(id="turbidez_graph"), dcc.Graph(id="estado_filtros_graph")], width=4),
        dbc.Col([dcc.Graph(id="intensidad_uv_graph"), dcc.Graph(id="estado_moringa_graph")], width=4),
        dbc.Col([dcc.Graph(id="ph_graph"), dcc.Graph(id="tds_graph"), dcc.Graph(id="temp_graph")], width=4),
    ]),

    html.Hr(),

    html.H2("🔮 Predicciones", className="text-center mt-4"),
    dbc.Row([
        dbc.Col([dcc.Graph(id="pred_estado_filtros")], width=4),
        dbc.Col([dcc.Graph(id="pred_calidad_agua")], width=4),
        dbc.Col([dcc.Graph(id="pred_intensidad_uv")], width=4),
    ]),

    dcc.Interval(id='interval_component', interval=5000, n_intervals=0)
])

@app.callback(
    Output("turbidez_graph", "figure"),
    Output("estado_filtros_graph", "figure"),
    Output("intensidad_uv_graph", "figure"),
    Output("estado_moringa_graph", "figure"),
    Output("ph_graph", "figure"),
    Output("tds_graph", "figure"),
    Output("temp_graph", "figure"),
    Output("pred_estado_filtros", "figure"),
    Output("pred_calidad_agua", "figure"),
    Output("pred_intensidad_uv", "figure"),
    Input("interval_component", "n_intervals")
)
def update_graphs(n_intervals):
    global data
    new_time = datetime.datetime.now()

    new_data = {
        "Timestamp": [new_time],
        "Turbidez (NTU)": [data["Turbidez (NTU)"].iloc[-1] + np.random.normal(0, 0.2)],
        "Estado de Filtros (%)": [data["Estado de Filtros (%)"].iloc[-1] - np.random.normal(0, 0.5)],
        "Intensidad UV (W/m²)": [data["Intensidad UV (W/m²)"].iloc[-1] + np.random.normal(0, 0.3)],
        "Estado Purificador Moringa (%)": [data["Estado Purificador Moringa (%)"].iloc[-1] - np.random.normal(0, 0.2)],
        "pH": [data["pH"].iloc[-1] + np.random.normal(0, 0.05)],
        "TDS (ppm)": [data["TDS (ppm)"].iloc[-1] + np.random.normal(0, 5)],
        "Temperatura (°C)": [data["Temperatura (°C)"].iloc[-1] + np.random.normal(0, 0.1)]
    }
    
    new_row = pd.DataFrame(new_data)
    data = pd.concat([data, new_row], ignore_index=True).tail(50)
    data = data.sort_values(by="Timestamp")

    # 📊 Gráficos en tiempo real
    turbidez_fig = px.line(data, x="Timestamp", y="Turbidez (NTU)", title="Turbidez del Agua", markers=True)
    estado_filtros_fig = px.line(data, x="Timestamp", y="Estado de Filtros (%)", title="Estado de los Filtros", markers=True)
    intensidad_uv_fig = px.line(data, x="Timestamp", y="Intensidad UV (W/m²)", title="Intensidad de Luz UV", markers=True)
    estado_moringa_fig = px.line(data, x="Timestamp", y="Estado Purificador Moringa (%)", title="Estado Purificador Moringa", markers=True)
    ph_fig = px.line(data, x="Timestamp", y="pH", title="Nivel de pH del Agua", markers=True)
    tds_fig = px.line(data, x="Timestamp", y="TDS (ppm)", title="Sólidos Disueltos Totales (TDS)", markers=True)
    temp_fig = px.line(data, x="Timestamp", y="Temperatura (°C)", title="Temperatura del Agua", markers=True)

    # 📈 Predicciones (en rojo)
    model = LinearRegression()
    X = np.array(range(len(data))).reshape(-1, 1)

    model.fit(X, data["Estado de Filtros (%)"])
    pred_filtros = model.predict(X)

    model.fit(X, data["Turbidez (NTU)"])
    pred_calidad = model.predict(X)

    model.fit(X, data["Intensidad UV (W/m²)"])
    pred_uv = model.predict(X)

    pred_filtros_fig = px.line(data, x="Timestamp", y=pred_filtros, title="Predicción del Estado de los Filtros",
                               color_discrete_sequence=["red"], markers=True)
    pred_calidad_fig = px.line(data, x="Timestamp", y=pred_calidad, title="Predicción de Calidad del Agua",
                                color_discrete_sequence=["red"], markers=True)
    pred_uv_fig = px.line(data, x="Timestamp", y=pred_uv, title="Predicción de Intensidad UV",
                           color_discrete_sequence=["red"], markers=True)

    return (
        turbidez_fig, estado_filtros_fig, intensidad_uv_fig, estado_moringa_fig, 
        ph_fig, tds_fig, temp_fig, pred_filtros_fig, pred_calidad_fig, pred_uv_fig
    )

if __name__ == '__main__':
    app.run_server(debug=True)


