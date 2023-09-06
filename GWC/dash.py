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

#create top row columns for selectbox and charts
col1, col2, col3 = st.columns(3)

with col1:
    selected_event = st.selectbox('Select Event (Defaults to GWTC)', grouped_data.keys())
    if selected_event in grouped_data:
        event_df = grouped_data[selected_event]

col1.write('Select an Event Catalog to learn more about each individual release')

# Eliminate rows with missing mass_1_source or mass_2_source
event_df = event_df.dropna(subset=['mass_1_source', 'mass_2_source'])

#fix missing mass issue
event_df['total_mass_source'] = event_df['mass_1_source'] + event_df['mass_2_source']

event_df.to_excel('updated_GWTC.xlsx', index=False)

updated_excel = 'updated_GWTC.xlsx'

# Loads df to use for the rest of the dash
df = pd.read_excel(updated_excel)

# Count for total observations 
count = event_df.commonName.unique().size

col2.metric(label="Total Observations in the Catalog",
    value=(count),    
)
col2.write('This is the number of confident observations for the catalog selected, for a complete list of all events please visit: https://gwosc.org/eventapi/html/allevents/' )

# Sort mass for event type distribution
def categorize_event(row):
    if row['mass_1_source'] < 3 and row['mass_2_source'] < 3:
        return 'Binary Neutron Star'
    elif row['mass_1_source'] >= 3 and row['mass_2_source'] >= 3:
        return 'Binary Black Hole'
    else:
        return 'Neutron Star - Black Hole'

# sourcery skip: assign-if-exp
df['Event'] = df.apply(categorize_event, axis=1)

# Group data by event type and count occurrences
grouped_df = df.groupby('Event').size().reset_index(name='Count')

# Create the pie chart
pie_chart = alt.Chart(grouped_df).mark_arc().encode(
    theta=alt.Theta(field='Count', type='quantitative'),
    color=alt.Color(field='Event', type='nominal'),
    tooltip=['Event', 'Count']
).properties(
    width=300,
    height=300,
    title='Merger Type Distribution'
)

col3.altair_chart(pie_chart, use_container_width=True)
col3.write('Breakdown of the type of events we have detected so far')

#mass chart for Dashboard
mass_chart = alt.Chart(df, title="Total Mass Histogram").mark_bar().encode(
    x=alt.X('total_mass_source:N', title='Total Mass', bin=True),
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
col6.write('This network SNR is the quadrature sum of the individual detector SNRs for all detectors involved in the reported trigger. ')
#cite from https://journals.aps.org/prx/pdf/10.1103/PhysRevX.9.031040

st.markdown('### Select an event from the catalog to learn more')
#MAIN CHART FOR USER INPUT
event_chart = px.scatter(event_df, x="mass_1_source", y="mass_2_source", color="network_matched_filter_snr", labels={
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

# Function to filter event options based on input prefix
def filter_event_options(prefix):
    return event_df[event_df['commonName'].str.startswith(prefix)]['commonName'].tolist()

event_input = st.multiselect(
    "Type the Event name below or click on an event in the chart to populate more information.",
    filter_event_options(""),
    default=[],
    key="event_input",
)
# Initialize select_event as an empty list
select_event = []
#User Selection
select_event = plotly_events(event_chart, click_event=True)

expander = st.expander("Expand for more information regarding the Event Catalog Chart")
expander.write('Compare the masses between both sources, along with the strength in Network SNR. A mass above 3 solar masses is considered a black hole, a mass with less than 3 solar masses is a neutron star. ')
expander.write(
"""
The chart allows the following interactivity:
- Pan and Zoom
- Box Selection
- Download chart as a PNG
"""
)
#lets user select an event by input or click
if event_input:
    selected_event_name = event_input[0]
    selected_event_row = df[df['commonName'] == selected_event_name]
    if not selected_event_row.empty:
        selected_x = selected_event_row['mass_1_source'].values[0]
        selected_y = selected_event_row['mass_2_source'].values[0]
        select_event = [{'x': selected_x, 'y': selected_y}]
    else:
        st.write("Selected event not found in the dataset.")

if select_event:
    # Retrieve clicked x and y values
    clicked_x = select_event[0]['x']
    clicked_y = select_event[0]['y']

    # Find the row in the DataFrame that matches the clicked x and y values
    selected_row = df[(df["mass_1_source"] == clicked_x) & (df["mass_2_source"] == clicked_y)]

    if not selected_row.empty:
        selected_common_name = selected_row["commonName"].values[0]
        event_name = selected_common_name
        st.markdown('### Selected Event: ' + event_name)
        if gps_info := event_gps(event_name):
            st.write("GPS Time:", gps_info, "is the end time or merger time of the event in GPS seconds.")

            mass_1 = selected_row['mass_1_source'].values[0]
            mass_2 = selected_row['mass_2_source'].values[0]
            dist = selected_row['luminosity_distance'].values[0]
            total_mass_source = selected_row['total_mass_source'].values[0]
            snr = selected_row['network_matched_filter_snr'].values[0]
        else:
            st.write("GPS Information not available for the selected event.")
            
#CHARTS WITH USER INPUT
if select_event:    
    
    st.markdown('### EVENT METRICS')
    st.write('The :red[red line |] indicates the largest value found to date for each category')
    st.write('The :blue[[blue area]] indicates the margin of error for each source')

    ##Gauge Indicators
    total_mass_lower = selected_row['total_mass_source_lower'].values[0] + selected_row['total_mass_source'].values[0] 
    total_mass_upper = selected_row['total_mass_source_upper'].values[0] + selected_row['total_mass_source'].values[0]    
    total_mass = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = total_mass_source,
    number = {"suffix": "M<sub>☉</sub>"},
    title = {'text': "Total Mass (M<sub>☉</sub>)"},
    gauge = {'axis': {'range': [None, 200]},
            'bar': {'color': "black"},             
            'steps' : [
                {'range': [total_mass_source, total_mass_upper], 'color': "lightskyblue"},
                {'range': [total_mass_source, total_mass_lower], 'color': "lightskyblue"}],             
            'bgcolor': "white",
            'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 181}},
    domain = {'x': [0, 1], 'y': [0, 1]}
    ))
    total_mass.update_layout(
        autosize=False,
        width=300,
        height=300,
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
            'bar': {'color': "black"},             
            'steps' : [
                {'range': [mass_1, m1_upper], 'color': "lightskyblue"},
                {'range': [mass_1, m1_lower], 'color': "lightskyblue"}],             
            'bgcolor': "white",
            'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 105}},
    domain = {'x': [0, 1], 'y': [0, 1]}
    ))
    m1.update_layout(
        autosize=False,
        width=300,
        height=300,
    )
    #mass 2 gauge
    m2_lower = selected_row['mass_2_source_lower'].values[0] + selected_row['mass_2_source'].values[0] 
    m2_upper = selected_row['mass_2_source_upper'].values[0] + selected_row['mass_2_source'].values[0]    
    m2 = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = mass_2,
    number = {"suffix": "M<sub>☉</sub>"},
    title = {'text': "Mass of source 2 (M<sub>☉</sub>)"},
    gauge = {'axis': {'range': [None, 200]},           
            'steps' : [
                {'range': [mass_2, m2_upper], 'color': "lightskyblue"},
                {'range': [mass_2, m2_upper], 'color': "lightskyblue"}],
            'bgcolor': "white",
            'bar': {'color': "black"},              
            'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 76}},
    domain = {'x': [0, 1], 'y': [0, 1]}
    ))
    m2.update_layout(
        autosize=False,
        width=300,
        height=300,
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
    gauge = {'axis': {'range': [None, 10]},
            'bar': {'color': "black"},
            'steps' : [
                {'range': [dist, lum_dist_upper], 'color': "lightskyblue"},
                {'range': [dist, lum_dist_lower], 'color': "lightskyblue"}],             
            'bgcolor': "white",
            'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 8.28}},
    domain = {'x': [0, 1], 'y': [0, 1]}
    ))
    lum_dist.update_layout(
        autosize=False,
        width=300,
        height=300,
    )
    #snr gauge
    snr_lower = selected_row['network_matched_filter_snr_lower'].values[0] + selected_row['network_matched_filter_snr'].values[0] 
    snr_upper = selected_row['network_matched_filter_snr_upper'].values[0] + selected_row['network_matched_filter_snr'].values[0]
    snr = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = snr, 
    title = {'text': "Network Matched Filter SNR"},
    gauge = {'axis': {'range': [None, 40]},
            'steps' : [
                {'range': [snr, snr_upper], 'color': "lightskyblue"},
                {'range': [snr, snr_lower], 'color': "lightskyblue"}],
            'bar': {'color': "black"},
            'bgcolor': "white",
            'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 33}},      
    ))
    snr.update_layout(
    autosize=False,
    width=300,
    height=300,
    )

    #Columns for Gauges
    col7, col8, col9 = st.columns(3)
    col7.write(total_mass)
    col8.write(m1)
    col9.write(m2)
    expander = st.expander('Total Mass before merger and individual mass for both objects shown')
    expander.write("The largest combined mass found so far is for Event: GW190426_190642 with a combined mass of :red[181.5 solar masses], Mass of object 1 is :red[105.5 solar masses] and for object 2 it is :red[76.5 solar masses].")
    #second column
    col10, col11, = st.columns(2)
    col10.write(lum_dist)
    expdr = col10.expander('Luminosity Distance is how far, in Gpc, the merger is from the sun. This is a good indicator of the distance between the two sources')
    expdr.write('More info!')
    col11.write(snr)
    expdr = col11.expander('The Network Matched Filter SNR is a measure of the quality of the data, with a higher SNR giving us better data')
    expdr.write('More info!')

    #have users select a detector
    detectorlist = ['H1', 'L1', 'V1']
    detector = st.selectbox("Select a Detector, (Note: Not all events available for all detectors)", detectorlist)

    # Q-transform and other charts
    #st.subheader('Q-transform')
    
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

    if gps_info:
        # Define the segment based on GPS info
        segment = (int(gps_info) - 5, int(gps_info) + 5)

        # Fetch time series data for the selected detector
        ldata = fetch_time_series(detector, segment)

        if ldata is not None:  # Check if ldata is not None before proceeding
            #setup for q transform
            hq = None
            chirp_mass = event_df['chirp_mass_source']
            bns = chirp_mass < 5
            t0 = datasets.event_gps(event_name)
            q_center = 100*(1/chirp_mass)
            q_center[q_center > 5] = 5
            qrange_min = float((q_center*0.8).min())
            qrange_max = float((q_center*1.2).max())
            qrange = (qrange_min, qrange_max)

            if bns.any():
                dt = 2
            else:
                dt = 0.3
            #q transform
            hq = ldata.q_transform(outseg=(t0-dt, t0+dt), qrange=qrange)
            fig4 = hq.plot()
            ax = fig4.gca()
            fig4.colorbar(label="Normalised energy", vmax=25, vmin=0)
            ax.grid(False)
            ax.set_yscale('log')
            ax.set_ylim(ymin=20, ymax=1024)
            #last column
            col12, col13 = st.columns(2)
            col12.subheader('Q-transform')            
            col12.pyplot(fig4, clear_figure=True)
            expdr = col12.expander('Q-transform')
            expdr.write('More info!')
            col13.subheader('Waveform')
            col13.write(wave)
            col13.write('Listen to what the waveform sounds like')
            col13.audio("waveform.wav")
            expdr = col13.expander('Waveform')
            expdr.write('more info!')
else:
    st.write("Click on a event to view more details")

