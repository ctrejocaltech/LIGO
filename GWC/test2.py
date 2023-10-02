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
if selected_event != event_input:
    if selected_event:
        event_input = selected_event
    elif select_event:
        # Retrieve clicked x and y values
        clicked_x = select_event[0]['x']
        clicked_y = select_event[0]['y']

        # Find the row in the DataFrame that matches the clicked x and y values
        selected_event_name = df[(df['mass_1_source'] == clicked_x) & (df['mass_2_source'] == clicked_y)]['commonName'].values[0]
        event_input = selected_event_name

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
st.write(
"""
The chart allows the following interactivity:
- Pan and Zoom
- Box Selection
- Download chart as a PNG
"""
)
# Initialize select_event as an empty list
select_event = []
#User Selection
select_event = plotly_events(event_chart, click_event=True)

selected_event_name = ""

st.write('Compare the masses between both sources, along with the strength in Network SNR. A mass above 3 solar masses is considered a black hole, a mass with less than 3 solar masses is a neutron star. ')

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
    total_mass.update_layout(
        autosize=False,
        width=400,
        height=400,
    )
    #mass 1 gauge
    m1_lower = selected_row['mass_1_source_lower'].values[0] + selected_row['mass_1_source'].values[0] 
    m1_upper = selected_row['mass_1_source_upper'].values[0] + selected_row['mass_1_source'].values[0]    
    m1 = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = mass_1,
    number = {"suffix": "M<sub>☉</sub>"},
    title = {'text': "Mass of source 1 (M<sub>☉</sub>)"},
    gauge = {'axis': {'range': [None, 200]},
            'bar': {'color': "#4751a5"},             
            'steps' : [
                {'range': [mass_1, m1_upper], 'color': "lightskyblue"},
                {'range': [mass_1, m1_lower], 'color': "lightskyblue"}],             
            'bgcolor': "white",
            'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 105}},
    domain = {'x': [0, 1], 'y': [0, 1]}
    ))
    m1.update_layout(
        autosize=False,
        width=400,
        height=400,
    )
    #mass 2 gauge
    m2_lower = selected_row['mass_2_source_lower'].values[0] + selected_row['mass_2_source'].values[0] 
    m2_upper = selected_row['mass_2_source_upper'].values[0] + selected_row['mass_2_source'].values[0]    
    m2 = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = mass_2,
    number = {"suffix": "M<sub>☉</sub>"},
    title = {'text': "Mass of source 2 (M<sub>☉</sub>)"},
    gauge = {'axis': {'range': [None, 180]},  
            'bar': {'color': "#4751a5"},         
            'steps' : [
                {'range': [mass_2, m2_upper], 'color': "lightskyblue"},
                {'range': [mass_2, m2_lower], 'color': "lightskyblue"}],
            'bgcolor': "white",           
            'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 76}},
    domain = {'x': [0, 1], 'y': [0, 1]}
    ))
    m2.update_layout(
        autosize=False,
        width=400,
        height=400,
    )
    #lum dist gauge
    lum_dist_lower = selected_row['luminosity_distance_lower'].values[0] + selected_row['luminosity_distance'].values[0] 
    lum_dist_upper = selected_row['luminosity_distance_upper'].values[0] + selected_row['luminosity_distance'].values[0]        
    #Convert lum_dist from Mpc to Gpc 
    dist = dist/1000
    lum_dist_lower = lum_dist_lower/1000
    lum_dist_upper = lum_dist_upper/1000 
    lum_dist = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = dist,
    number = {"suffix": "Gpc"},
    title = {'text': "Luminosity Distance (Gpc)"},
    gauge = {'axis': {'range': [None, 18]},
            'bar': {'color': "#4751a5"},
            'steps' : [
                {'range': [dist, lum_dist_upper], 'color': "lightskyblue"},
                {'range': [dist, lum_dist_lower], 'color': "lightskyblue"}],             
            'bgcolor': "white",
            'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 8.28}},
    domain = {'x': [0, 1], 'y': [0, 1]}
    ))
    lum_dist.update_layout(
        autosize=False,
        width=400,
        height=400,
    )
    #snr gauge
    snr_lower = selected_row['network_matched_filter_snr_lower'].values[0] + selected_row['network_matched_filter_snr'].values[0] 
    snr_upper = selected_row['network_matched_filter_snr_upper'].values[0] + selected_row['network_matched_filter_snr'].values[0]
    snr_chart = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = snr, 
    title = {'text': "Network Matched Filter SNR"},
    gauge = {'axis': {'range': [None, 40]},
            'steps' : [
                {'range': [snr, snr_upper], 'color': "lightskyblue"},
                {'range': [snr, snr_lower], 'color': "lightskyblue"}],
            'bar': {'color': "#4751a5"},
            'bgcolor': "white",
            'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 33}},      
    ))
    snr_chart.update_layout(
    autosize=False,
    width=400,
    height=400,
    )
    #Ridgeline plots
    ridge_mass = go.Figure()
    ridge_mass.add_trace(go.Violin(x=df['total_mass_source'], line_color='#808080', name = ''))
    ridge_mass.add_shape(
        dict(
            type="line",
            x0=total_mass_source,
            x1=total_mass_source,
            y0=0,
            y1=2,  # Adjust the y1 value as needed to cover the violin plot height
            line=dict(color="#4751a5", width=3),
        )
    )
    ridge_mass.update_traces(orientation='h', side='positive', width=4, points=False)
    ridge_mass.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)
    ridge_mass.update_layout(
        autosize=False,
        width=400,
        height=300,
        annotations=[
            dict(
                text= "Total mass of " + event_name + " in relation to the catalogs mass distribution in solar mass.",
                xref="paper",
                yref="paper",
                x=0.5,  # Adjust the x position for centering
                y=-0.5,  # Adjust the y position for distance from the chart
                showarrow=False,
                font=dict(size=10),
            )
        ]
    )
    #mass1 plot
    ridge_mass1 = go.Figure()
    ridge_mass1.add_trace(go.Violin(x=df['mass_1_source'], line_color='#808080', name = ''))
    ridge_mass1.add_shape(
        dict(
            type="line",
            x0=mass_1,
            x1=mass_1,
            y0=0,
            y1=2,  # Adjust the y1 value as needed to cover the violin plot height
            line=dict(color="#4751a5", width=3),
        )
    )
    ridge_mass1.update_traces(orientation='h', side='positive', width=4, points=False)
    ridge_mass1.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)
    ridge_mass1.update_layout(
        autosize=False,
        width=400,
        height=300,
        annotations=[
            dict(
                text= "The mass of source 1 in relation to the catalogs mass 1 distribution in solar mass.",
                xref="paper",
                yref="paper",
                x=0.5,  # Adjust the x position for centering
                y=-0.5,  # Adjust the y position for distance from the chart
                showarrow=False,
                font=dict(size=10),
            )
        ]
    )
    #mass2 plot
    ridge_mass2 = go.Figure()
    ridge_mass2.add_trace(go.Violin(x=df['mass_2_source'], line_color='#808080', name = ''))
    ridge_mass2.add_shape(
        dict(
            type="line",
            x0=mass_2,
            x1=mass_2,
            y0=0,
            y1=2,  # Adjust the y1 value as needed to cover the violin plot height
            line=dict(color="#4751a5", width=3),
        )
    )
    ridge_mass2.update_traces(orientation='h', side='positive', width=4, points=False)
    ridge_mass2.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)
    ridge_mass2.update_layout(
        autosize=False,
        width=400,
        height=300,
        annotations=[
            dict(
                text= "The mass of source 2 in relation to the catalogs mass 2 distribution in solar mass.",
                xref="paper",
                yref="paper",
                x=0.5,  # Adjust the x position for centering
                y=-0.5,  # Adjust the y position for distance from the chart
                showarrow=False,
                font=dict(size=10),
            )
        ]
    )

    #lum_dist  plot
    ridge_dist = go.Figure()
    ridge_dist.add_trace(go.Violin(x=df['luminosity_distance'], line_color='#808080', name = ''))
    ridge_dist.add_shape(
        dict(
            type="line",
            x0=dist,
            x1=dist,
            y0=0,
            y1=2,  # Adjust the y1 value as needed to cover the violin plot height
            line=dict(color="#4751a5", width=3),
        )
    )
    ridge_dist.update_traces(orientation='h', side='positive', width=4, points=False)
    ridge_dist.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)
    ridge_dist.update_layout(
        autosize=False,
        width=400,
        height=300,
        annotations=[
            dict(
                text= "The luminosity distance in relation to the catalogs range in Gpc.",
                xref="paper",
                yref="paper",
                x=0.5,  # Adjust the x position for centering
                y=-0.5,  # Adjust the y position for distance from the chart
                showarrow=False,
                font=dict(size=10),
            )
        ]
    )
    
    #snr ridge plot
    ridge_snr = go.Figure()
    ridge_snr.add_trace(go.Violin(x=df['network_matched_filter_snr'], line_color='#808080', name = ''))
    ridge_snr.add_shape(
        dict(
            type="line",
            x0=snr,
            x1=snr,
            y0=0,
            y1=2,  # Adjust the y1 value as needed to cover the violin plot height
            line=dict(color="#4751a5", width=3),
        )
    )
    ridge_snr.update_traces(orientation='h', side='positive', width=4, points=False)
    ridge_snr.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)
    ridge_snr.update_layout(
        autosize=False,
        width=400,
        height=300,
        annotations=[
            dict(
                text= "The network SNR in relation to the catalogs distribution.",
                xref="paper",
                yref="paper",
                x=0.5,  # Adjust the x position for centering
                y=-0.5,  # Adjust the y position for distance from the chart
                showarrow=False,
                font=dict(size=10),
            )
        ]
    )
    #Columns for Gauges
    st.write('Largest Total Mass found to date is for Event GW190426_190642 at :red[181.5 solar masses], with the largest mass of object 1 at :red[105.5 solar masses], and the largest mass of object 2 at :red[76.5 solar masses].')
    col7, col8, col9 = st.columns(3)
    col7.write(total_mass)
    col7.write(ridge_mass)
    col8.write(m1)
    col8.write(ridge_mass1) 
    col9.write(m2)
    col9.write(ridge_mass2)
    st.divider()
    #second column
    col10, col11, = st.columns(2)
    col10.write(lum_dist)
    col10.write('The furthest merger observed to date is for Event GW190403_051519 at :red[8.28 Gpc].')
    col10.write(ridge_dist)
    col11.write(snr_chart)
    col11.write('The highest SNR observed to date is for Event: GW170817 at :red[33].')
    col11.write(ridge_snr)
    st.divider()
    #have users select a detector
    detectorlist = ['H1', 'L1', 'V1']
    detector = st.selectbox("Select a Detector, (Note: Not all events available for all detectors.)", detectorlist)

    #generate waveform
    hp, hc = get_td_waveform(approximant="IMRPhenomD",
                            mass1=mass_1,
                            mass2=mass_2,
                            delta_t=1.0/16384,
                            f_lower=45,
                            distance=dist)
    
    # Convert the TimeSeries data to a numpy array
    hp_array = np.array(hp)
    # Scale the data to 16-bit integer values
    hp_scaled = np.int16(hp_array / np.max(np.abs(hp_array)) * 32767)
    # Save the waveform as a WAV file
    wavfile.write("waveform.wav", 44100, hp_scaled)
    
    #Zoom in near the merger time
    wave = plt.figure(figsize=pylab.figaspect(0.4))
    plt.plot(hp.sample_times, hp, label='Plus Polarization')
    plt.plot(hp.sample_times, hc, label='Cross Polarization')
    plt.xlabel('Time (s)')
    plt.ylabel('Strain')
    plt.xlim(-.5, .5)
    plt.legend()
    plt.grid()
    
    #Fetch Time Series Data
    def fetch_time_series(detector, segment):
        try:
            return TimeSeries.fetch_open_data(detector, *segment, verbose=True, cache=True)
        except Exception as e:
            st.error(f"Please select a valid detector: {str(e)}")
            return None

    bns = False  # Initialize bns to a default value

    if gps_info:
        # Define the segment based on GPS info
        segment = (int(gps_info) - 5, int(gps_info) + 5)

        # Fetch time series data for the selected detector
        ldata = fetch_time_series(detector, segment)

        chirp_mass = selected_row['chirp_mass_source'].values[0]   
        if chirp_mass < 5:
            bns = True
        else:
            print('failed to find chirp mass')

    if bns:
        dt = 2
    else:
        dt = 0.3

    t0 = datasets.event_gps(event_name)
    q_center = 100*(1/chirp_mass)
    if q_center < 5:
        q_center = 5
    qrange = (int(q_center*0.8), int(q_center*1.2))  
    
    outseg = (t0-dt, t0+dt)
    hq = ldata.q_transform(outseg=outseg, qrange=qrange)
    x_values = hq.times.value - t0  # Calculate the time relative to t0
    fig4 = hq.plot()
    ax = fig4.gca()
    fig4.colorbar(label="Normalised energy", vmax=25, vmin=0)
    ax.grid(False)
    ax.set_yscale('log')
    ax.set_ylim(ymin=20, ymax=1024)
    # Set the new x-axis limits and labels
    #ax.set_xlim(x_values.min(), x_values.max())  # Set limits based on the new x values
    
    # Define a custom formatting function to display two decimal places
    #def custom_format(x, pos):
    #    return f"{x:.2f}"

    # Apply the custom formatting function to the x-axis
    #ax.xaxis.set_major_formatter(FuncFormatter(custom_format))
    
    # Specify the tick locator (AutoLocator)
    #ax.xaxis.set_major_locator(AutoLocator())
    #ax.set_xlabel("Time from Merger (s)")  # Update the x-axis label
        
    #last column
    col12, col13 = st.columns(2)
    col12.subheader('Q-transform')            
    col12.pyplot(fig4, clear_figure=True)
    col12.write("""
    A Q-transform plot shows how a signal’s frequency changes with time.
    * The x-axis shows time
    * The y-axis shows frequency

    The color scale shows the amount of “energy” or “signal power” in each time-frequency pixel.
    
    """)
    col13.subheader('Waveform')
    col13.write(wave)
    col13.write('Listen to what the waveform sounds like')
    col13.audio("waveform.wav")
    col13.write('The waveform is a simplified example of the gravitational waveform radiated during a compact binary coalescence using basic parameters. ')
else:
    st.write("Click on a event to view more details")

st.write('To learn more about Gravitational waves please visit the [Gravitational Wave Open Science Center Learning Path](https://gwosc.org/path/)')
