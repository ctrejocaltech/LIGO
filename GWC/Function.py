from math import log
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_plotly_events import plotly_events
import plotly.graph_objects as go
import altair as alt
import pycbc
from pycbc.waveform import get_td_waveform, fd_approximants
import pylab
import openpyxl
import requests
import plotly.io as pio
from urllib.parse import parse_qs, urlparse
from matplotlib.ticker import FuncFormatter, AutoLocator

#other imports
import matplotlib.pyplot as plt
from scipy import signal, datasets, fft
from scipy.signal import get_window
import gwpy

import requests, os
from gwpy.timeseries import TimeSeries
from gwosc.locate import get_urls
from gwosc import datasets
from gwosc.datasets import event_gps
from gwosc.api import fetch_event_json

from copy import deepcopy
import base64

from scipy.io import wavfile
import io

# Use the non-interactive Agg backend, which is recommended as a
# thread-safe backend.
# See https://matplotlib.org/3.3.2/faq/howto_faq.html#working-with-threads.
import matplotlib as mpl                
mpl.use("agg")

##############################################################################
# Workaround for the limited multi-threading support in matplotlib.
# Per the docs, we will avoid using `matplotlib.pyplot` for figures:
# https://matplotlib.org/3.3.2/faq/howto_faq.html#how-to-use-matplotlib-in-a-web-application-server.
# Moreover, we will guard all operations on the figure instances by the
# class-level lock in the Agg backend.
##############################################################################
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock

# --Set page config
apptitle = 'GWTC Global View'
st.set_page_config(page_title=apptitle, layout="wide")

#Title the app
st.title('Gravitational-wave Transient Catalog Dashboard')
st.write('The Gravitational-wave Transient Catalog (GWTC) is a cumulative set of gravitational wave transients maintained by the LIGO/Virgo/KAGRA collaboration. The online GWTC contains confidently-detected events from multiple data releases. For further information, please visit https://gwosc.org')

# Get the current page URL
url = st.experimental_get_query_params()

# Get specific parameter values, e.g., event_name
event_url = url.get("event_name", ["default_event"])[0]

# Fetch the data from the URL and load it into a DataFrame
@st.cache_data
def load_and_group_data():
    df = pd.read_csv("https://gwosc.org/eventapi/csv/allevents/")
    
    catalogs = ['GWTC', 'GWTC-1-confident', 'GWTC-2', 'GWTC-2.1-confident', 'GWTC-3-confident', 'O3_Discovery_Papers']
    
    grouped_data = {}
    
    for catalog in catalogs:
        if catalog == 'GWTC':
            event_data = df[df['catalog.shortName'].str.contains('confident', case=False)]
        else:
            event_data = df[(df['catalog.shortName'] == catalog)]
        grouped_data[catalog] = event_data

    return grouped_data

grouped_data = load_and_group_data()
st.divider()
#create top row columns for selectbox and charts
col1, col2, col3 = st.columns(3)

with col1:
    selected_cat = st.selectbox('Select an Event Catalog (Defaults to GWTC)', grouped_data.keys())
    if selected_cat in grouped_data:
        event_df = grouped_data[selected_cat]

col1.write('Each catalog contains a collection of events observed during a LIGO/Virgo observation run.')

# Eliminate rows with missing mass_1_source or mass_2_source
event_df = event_df.dropna(subset=['mass_1_source', 'mass_2_source'])
event_df['total_mass_source'] = event_df['mass_1_source'] + event_df['mass_2_source'] #fix missing mass issue
event_df.to_excel('updated_GWTC.xlsx', index=False)
updated_excel = 'updated_GWTC.xlsx'

# Loads df to use for the rest of the dash
df = pd.read_excel(updated_excel)

# Count for total observations 
count = event_df.commonName.unique().size
col2.metric(label="Total Observations in the Catalog",
    value=(count),    
)
col2.write('This is the number of confident observations for the catalog selected, for a complete list of all events please visit: https://gwosc.org/eventapi' )

# Sort mass for event type distribution
def categorize_event(row):
    if row['mass_1_source'] < 3 and row['mass_2_source'] < 3:
        return 'Binary Neutron Star'
    elif row['mass_1_source'] >= 3 and row['mass_2_source'] >= 3:
        return 'Binary Black Hole'
    else:
        return 'Neutron Star - Black Hole'
df['Event'] = df.apply(categorize_event, axis=1)

# Group data by event type and count occurrences
grouped_df = df.groupby('Event').size().reset_index(name='Count')

# Function to filter event options based on input prefix
def filter_event_options(prefix):
    return df[df['commonName'].str.startswith(prefix)]['commonName'].tolist()

# Get the list of event options
event_options = filter_event_options("")

#MAIN CHART FOR USER INPUT
event_chart = px.scatter(df, x="mass_1_source", y="mass_2_source", color="network_matched_filter_snr", labels={
    "network_matched_filter_snr": "Network SNR",
    "luminosity_distance": "Luminosity Distance (Mpc)",
    "commonName": "Name",
    "mass_1_source": "Mass 1",
    "mass_2_source": "Mass 2", 
}, title= "Event Catalog of source-frame component masses m<sub>(i)</sub>", color_continuous_scale = "dense", hover_data=["commonName"])

event_chart.update_traces(
    marker=dict(size=10,
    symbol="circle",
    )
)
event_chart.update_layout(
    hovermode='x unified',
    width=900,
    height=450,
    xaxis_title="Mass 1 (M<sub>☉</sub>)",  # Add the smaller Solar Mass symbol using <sub> tag
    yaxis_title="Mass 2 (M<sub>☉</sub>)", 
    hoverdistance=-1,
)
event_chart.update_xaxes(
    title_standoff=10,
    title_font = {"size": 15},
)
event_chart.update_yaxes(
    title_standoff=10,
    title_font = {"size": 15},
)

# Create the selectbox with options
event_input = st.selectbox(
    "If you want to look up a specific Event, type the name below or click on an event in the chart below to populate more information.",
    [""] + event_options,  # Add an empty option as the default
    key="event_input",
)

# Initialize select_event as an empty list
select_event = []
#User Selection
select_event = plotly_events(event_chart, click_event=True)

#lets user select an event by input or click
if event_input:  # Check if event_input is not empty
    selected_event_name = event_input  # Update selected_event_name based on user input
    selected_event_row = df[df['commonName'] == selected_event_name]

    if not selected_event_row.empty:
        selected_x = selected_event_row['mass_1_source'].values[0]
        selected_y = selected_event_row['mass_2_source'].values[0]
        select_event = [{'x': selected_x, 'y': selected_y}]
    else:
        selected_event_name = ("Click on an Event")

if select_event:
    # Retrieve clicked x and y values
    clicked_x = select_event[0]['x']
    clicked_y = select_event[0]['y']

    # Find the row in the DataFrame that matches the clicked x and y values
    selected_row = df[(df["mass_1_source"] == clicked_x) & (df["mass_2_source"] == clicked_y)]

    if not selected_row.empty:
        selected_common_name = selected_row["commonName"].values[0]
        event_name = selected_common_name
        if gps_info := event_gps(event_name):
            mass_1 = selected_row['mass_1_source'].values[0]
            mass_2 = selected_row['mass_2_source'].values[0]
            dist = selected_row['luminosity_distance'].values[0]
            total_mass_source = selected_row['total_mass_source'].values[0]
            snr = selected_row['network_matched_filter_snr'].values[0]
            chirp = selected_row['chirp_mass'].values[0]
        else:
            st.write("GPS Information not available for the selected event.")    

#CHARTS WITH USER INPUT
if select_event:    
    st.divider()
    st.markdown('### EVENT METRICS for the selected event: ' + event_name)
    st.write("GPS Time:", gps_info, "is the end time or merger time of the event in GPS seconds.")

    ##Gauge Indicators
    total_mass_lower = selected_row['total_mass_source_lower'].values[0] + selected_row['total_mass_source'].values[0] 
    total_mass_upper = selected_row['total_mass_source_upper'].values[0] + selected_row['total_mass_source'].values[0]    
    total_mass = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = total_mass_source,
    number = {"suffix": "M<sub>☉</sub>"},
    title = {'text': "Total Mass (M<sub>☉</sub>)"},
    gauge = {'axis': {'range': [None, 250]},
            'bar': {'color': "#4751a5"},             
            'steps' : [
                {'range': [total_mass_source, total_mass_upper], 'color': "lightskyblue"},
                {'range': [total_mass_source, total_mass_lower], 'color': "lightskyblue"}],             
            'bgcolor': "white",
            'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 181}},
    domain = {'x': [0, 1], 'y': [0, 1]}
    ))
    total_mass.update_layout(
        autosize=False,
        width=400,
        height=400,
    )

st.write('To learn more about Gravitational waves please visit the [Gravitational Wave Open Science Center Learning Path](https://gwosc.org/path/)')
