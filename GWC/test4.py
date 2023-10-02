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

# Custom color scale for events
event_colors = alt.Scale(
    domain=['Binary Black Hole', 'Neutron Star - Black Hole', 'Binary Neutron Star'],  # Replace with event names
    range=['#201e66', '#504eca', '#bdbceb']  # Replace with desired colors
)
# Create the pie chart
pie_chart = alt.Chart(grouped_df).mark_arc().encode(
    theta=alt.Theta(field='Count', type='quantitative'),
    color=alt.Color(field='Event', type='nominal', scale=event_colors),
    tooltip=['Event', 'Count']
).properties(
    width=300,
    height=300,
    title='Merger Type Distribution'
)
col3.altair_chart(pie_chart, use_container_width=True)
col3.write('The observed events are mergers of neutron stars and/or black holes.')
st.divider()    
#mass chart for Dashboard
mass_chart = alt.Chart(df, title="Total Mass Histogram in Solar Masses").mark_bar().encode(
    x=alt.X('total_mass_source:N', title='Total Source Frame Mass ', bin=True),
    y=alt.Y('count()', title='Count'),
    #tooltip=['commonName', 'GPS']
)

#Histogram for SNR
snr = alt.Chart(df, title="Network SNR Histogram").mark_bar().encode(
    x=alt.X('network_matched_filter_snr:Q', title='SNR', bin=True),
    y=alt.Y('count()', title='Count')
)

#Histogram for Distance
dist = alt.Chart(df, title="Luminosity Distance Histogram").mark_bar().encode(
    x=alt.X('luminosity_distance:Q', title='Distance in Mpc', bin=alt.Bin(maxbins=10)),
    y=alt.Y('count()', title='Count')
)
#SECOND ROW COLUMNS
col4, col5, col6 = st.columns(3)
col4.altair_chart(mass_chart, use_container_width=True)
col4.write('Shows the distribution of mass for objects contained in the Catalog selected.')
col5.altair_chart(dist, use_container_width=True)
col5.write('Shows the distribution of luminosity distance in megaparsec (3.26 million lightyears) for objects contained in the Catalog selected.')
col6.altair_chart(snr, use_container_width=True)
col6.write('This network signal to noise ratio (SNR) is the quadrature sum of the individual detector SNRs for all detectors involved in the reported trigger. ')
#cite from https://journals.aps.org/prx/pdf/10.1103/PhysRevX.9.031040
st.divider()
st.markdown('### Select an event from the catalog to learn more.')

# Function to filter event options based on input prefix
def filter_event_options(prefix):
    return df[df['commonName'].str.startswith(prefix)]['commonName'].tolist()

# Get the current page URL
url = st.experimental_get_query_params()
# Get specific parameter values, e.g., event_name
event_url = url.get("event_name", [""])[0]

# Get the list of event options
event_options = filter_event_options("")
event_input = ""

# Initialize event_input with event_url if it exists in the list of event options
if event_url in df['commonName'].values:
    event_input = event_url
else:
    st.error("Error: event_name parameter not found in URL.")
    event_input = ""

# Set select_event to event_input if event_url is not found in the list of event options
if not event_input:
    select_event = event_input

# Create the selectbox with options
selected_event = st.selectbox(
    "If you want to look up a specific Event, type the name below or click on an event in the chart below to populate more information.",
    [event_input] + [""] + event_options,
    key="event_input",
)

# Update event_input based on user selection
if select_event != event_input:
    event_input = select_event

#MAIN CHART FOR USER INPUT
event_chart = px.scatter(df, x="mass_1_source", y="mass_2_source", color="network_matched_filter_snr", labels={
    "network_matched_filter_snr": "Network SNR",
    "luminosity_distance": "Luminosity Distance (Mpc)",
    "commonName": "Name",
    "mass_1_source": "Mass 1",
    "mass_2_source": "Mass 2", 
}, title= "Event Catalog of source-frame component masses m<sub>(i)</sub>", color_continuous_scale = "dense", hover_data=["commonName"])

# Initialize select_event as an empty list
select_event = []
#User Selection
select_event = plotly_events(event_chart, click_event=True)

selected_event_name = ""

st.write('Compare the masses between both sources, along with the strength in Network SNR. A mass above 3 solar masses is considered a black hole, a mass with less than 3 solar masses is a neutron star. ')

# If an event_input is selected or an event_url exists, update selected_event_name
if event_input:  # Check if event_input is not empty
    selected_event_name = event_input
elif event_url:  # Check if event_url exists
    selected_event_name = event_url

# Define a function to handle the selection logic
def handle_event_selection():
    global selected_event_name
    global select_event

    if event_input:
        selected_event_name = event_input
    elif selected_event_name:
        pass  # Use the existing selected_event_name
    elif select_event:
        # Retrieve clicked x and y values
        clicked_x = select_event[0]['x']
        clicked_y = select_event[0]['y']

        # Find the row in the DataFrame that matches the clicked x and y values
        selected_row = df[(df["mass_1_source"] == clicked_x) & (df["mass_2_source"] == clicked_y)]

        if not selected_row.empty:
            selected_common_name = selected_row["commonName"].values[0]
            selected_event_name = selected_common_name
        else:
            selected_event_name = "Click on an Event"

    # Use selected_event_name to populate the charts and data
    selected_event_row = df[df['commonName'] == selected_event_name]

    if not selected_event_row.empty:
        selected_x = selected_event_row['mass_1_source'].values[0]
        selected_y = selected_event_row['mass_2_source'].values[0]
        select_event = [{'x': selected_x, 'y': selected_y}]
    else:
        selected_event_name = "Click on an Event"

# Call the function to handle event selection
if event_input or event_url or select_event:handle_event_selection()

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
            # Continue with the code that uses gps_info here
            st.write("GPS Time:", gps_info, "is the end time or merger time of the event in GPS seconds.")
        else:
            st.write("GPS Information not available for the selected event.")

#CHARTS WITH USER INPUT
if select_event:    
    st.divider()
    st.markdown('### EVENT METRICS for the selected event: ' + event_input)
    st.write("GPS Time:", gps_info, "is the end time or merger time of the event in GPS seconds.")
    st.write('The :red[red line |] indicates the largest value found to date for each category.')
    st.write('The :blue[[blue area]] indicates the margin of error for each source.')
    st.write('Note: Some events may not have error information.')
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
