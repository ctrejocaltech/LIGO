import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_plotly_events import plotly_events
import plotly.graph_objects as go
from PIL import Image
import altair as alt
import pycbc
from pycbc.waveform import get_td_waveform, fd_approximants
import pylab
import openpyxl

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
st.title('GWTC Dashboard')

# Fetch the data from the URL and load it into a DataFrame
@st.cache_data
def get_data() -> pd.DataFrame:
    return pd.read_csv("https://gwosc.org/eventapi/csv/GWTC/")

old_df = get_data()

#add missing mass values
old_df["mass_1_source"].fillna(0, inplace=True)
old_df["mass_1_source"] = old_df["mass_1_source"].astype(int)
old_df["mass_2_source"].fillna(0, inplace=True)
old_df["mass_2_source"] = old_df["mass_2_source"].astype(int)

old_df['total_mass_source'] = old_df['mass_1_source'] + old_df['mass_2_source']

old_df.to_excel('updated_GWTC.xlsx', index=False)

updated_excel = 'updated_GWTC.xlsx'

df = pd.read_excel(updated_excel)

#num of unique events
count = df.commonName.unique().size

#mass chart for Dashboard
mass_chart = alt.Chart(df, title="Mass 1 vs Mass 2").mark_circle().encode(
    x=alt.X('mass_1_source:Q', title='Source of Mass 1'),
    y=alt.Y('mass_2_source:Q', title='Source of Mass 2'),
    tooltip=['commonName', 'GPS']
)

#Histogram for SNR
snr = alt.Chart(df, title="Histogram of Network SNR").mark_bar().encode(
    x=alt.X('network_matched_filter_snr:Q', title='SNR', bin=True),
    y=alt.Y('count()', title='Count')
)

#Histogram for Distance
dist = alt.Chart(df, title="Histogram of Distance").mark_bar().encode(
    x=alt.X('luminosity_distance:Q', title='Distance in Mpc'),
    y=alt.Y('count()', title='Count')
)

#pie chart breakdown
source = pd.DataFrame(
    {"Merger Type": ["BNS", "NSBH", "BBH"], "Value": [20, 40, 60]}
)

base = alt.Chart(source).mark_arc(innerRadius=75).encode(
    theta = alt.Theta(field="Value", type="quantitaive"),
    color = alt.Color(field="Merger Type", type="nominal"),
)

#Top Dashboard Elements
st.markdown('### Metrics')

col1, col2, col3 = st.columns(3)

col1.metric(label="Total Observations to Date",
    value=(count),    
)
expdr = col1.expander('Show more info in column!')
expdr.write('More info!')

col2.metric(
    label="Total Obvs Time",
    value=("Get Value"),
)
expdr = col2.expander('Show more info in column!')
expdr.write('More info!')

col3.altair_chart(base, use_container_width=True)
expdr = col3.expander('Show more info in column!')
expdr.write('More info!')

#second row of columns
col4, col5, col6 = st.columns(3)

col4.altair_chart(mass_chart, use_container_width=True)
expdr = col4.expander('Show more info in column!')
expdr.write('More info!')


col5.altair_chart(dist, use_container_width=True)
expdr = col5.expander('Show more info in column!')
expdr.write('More info!')

col6.altair_chart(snr, use_container_width=True)
expdr = col6.expander('Show more info in column!')
expdr.write('More info!')

st.markdown('### Select an event, default is GW150914')

#set default event
default_event = "GW150914"
selected_gwc_event = [default_event]

#create scatter chart 
event_chart = px.scatter(df, x="total_mass_source", y="commonName")

event_chart.update_traces(
    marker=dict(size=15, symbol="circle-dot"),
)

event_chart.update_layout(
    hovermode='y',
    autosize=True,
    xaxis_title="Total Mass",
    yaxis_title="Event",
    yaxis=dict(
        tickmode='array',
        title='Event',  # Set your y-axis title here
    ),
    margin=dict(l=150),  # Adjust the left margin value to create more space for the y-axis labels
)
event_chart.update_xaxes(range=[0,200],
    title_font = {"size": 20},
    title_standoff = 20,
)
event_chart.update_yaxes(range=[-1,18],
    title_font = {"size": 20},
    title_standoff = 50,
    
)

#get user input
select_event = plotly_events(event_chart, click_event=True)
if select_event:
    selected_gwc_event = [point['y'] for point in select_event]

# Display the selected points
st.write("Selected Event:", selected_gwc_event)

# Load GPS information corresponding to the selected event
if selected_gwc_event:
    event_name = selected_gwc_event[0]
    gps_info = event_gps(event_name)
    if gps_info:
        st.write("GPS Information:", gps_info)
        selected_data = df[df['commonName'] == event_name]
        if not selected_data.empty:
            mass_1 = selected_data['mass_1_source'].values[0]
            mass_2 = selected_data['mass_2_source'].values[0]
            dist = selected_data['luminosity_distance'].values[0]
    else:
        st.write("GPS Information not available for the selected event.")
else:
    st.write("Select an event by clicking on the plot.")

detectorlist = ['H1', 'L1', 'V1']
detector = st.selectbox("Pick the Detector", detectorlist)
st.write('Detector: ', detector)

#generate waveform
hp, hc = get_td_waveform(approximant="IMRPhenomD",
                         mass1=mass_1,
                         mass2=mass_2,
                         delta_t=1.0/16384,
                         f_lower=30,
                         distance=dist)

#Zoom in near the merger time
wave = plt.figure(figsize=pylab.figaspect(0.4))
plt.plot(hp.sample_times, hp, label='Plus Polarization')
plt.plot(hp.sample_times, hc, label='Cross Polarization')
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.xlim(-.01, .01)
plt.legend()
plt.grid()

#print timeseries and gps info to confirm
segment = (int(gps_info)-5, int(gps_info)+5)

ldata = TimeSeries.fetch_open_data(detector, *segment, verbose=True, cache=True)

#Spectrogram to confirm data is feeding through
specgram = ldata.spectrogram(2, fftlength=1, overlap=.5)**(1/2.)

plot = specgram.imshow(norm='log', vmin=5e-24, vmax=1e-19)
ax = plot.gca()
ax.set_yscale('log')
ax.set_ylim(10, 2000)
ax.colorbar(
    label=r'Gravitational-wave amplitude'
)

col7, col8 = st.columns(2)

col7.write(wave, use_container_width=True)

col8.pyplot(plot, use_container_width=True)

#--Add
#toggle between confirmed and marginal
#pie chart of BNS/NSBH/BBH

#--Fix errors
#0 mass error
#startup gps error
#detector error
#presistance issue with graph