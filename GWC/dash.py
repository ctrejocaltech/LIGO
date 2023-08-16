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

sub = 'confident'

old_df = old_df[old_df['catalog.shortName'].str.contains('confident', case=False)]

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
mass_chart = alt.Chart(df, title="Total Mass Histogram").mark_bar().encode(
    x=alt.X('total_mass_source:N', title='Total Mass', bin=True),
    y=alt.Y('count()', title='Count'),
    #tooltip=['commonName', 'GPS']
)

#Histogram for SNR
snr = alt.Chart(df, title="Network SNR Histogram").mark_bar().encode(
    x=alt.X('network_matched_filter_snr:Q', title='SNR'),
    y=alt.Y('count()', title='Count')
)

#Histogram for Distance
dist = alt.Chart(df, title="Distance Histogram").mark_bar().encode(
    x=alt.X('luminosity_distance:Q', title='Distance in Mpc', bin=alt.Bin(maxbins=10)),
    y=alt.Y('count()', title='Count')
)

#pie chart breakdown
source = pd.DataFrame(
    {"Merger Type": ["BNS", "NSBH", "BBH"], "Value": [20, 40, 60]}
)

base = alt.Chart(source).mark_arc(innerRadius=75).encode(
    theta = alt.Theta("Value:Q").stack(True),
    color = alt.Color("Merger Type:N").legend()
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

#create scatter chart 
event_chart = px.scatter(df, x="mass_1_source", y="mass_2_source", hover_data=["commonName"])

event_chart.update_traces(
    marker=dict(size=10, symbol="circle-dot"),
)
event_chart.update_layout(
   xaxis_title="Mass 1",
    yaxis_title="Mass 2",
)
event_chart.update_xaxes(
    title_font = {"size": 20},
)
event_chart.update_yaxes(
    title_font = {"size": 20},
)

select_event = plotly_events(event_chart, click_event=True)

if select_event:
    # Retrieve clicked x and y values
    clicked_x = select_event[0]['x']
    clicked_y = select_event[0]['y']

    # Find the row in the DataFrame that matches the clicked x and y values
    selected_row = df[(df["mass_1_source"] == clicked_x) & (df["mass_2_source"] == clicked_y)]

    if not selected_row.empty:
        selected_common_name = selected_row["commonName"].values[0]
        st.write("Selected Event:", selected_common_name)
        event_name = selected_common_name
        gps_info = event_gps(event_name)
        if gps_info:
            st.write("GPS Information:", gps_info)
            mass_1 = selected_row['mass_1_source'].values[0]
            mass_2 = selected_row['mass_2_source'].values[0]
            dist = selected_row['luminosity_distance'].values[0]
        else:
            st.write("GPS Information not available for the selected event.")
        

detectorlist = ['H1', 'L1', 'V1']
detector = st.selectbox("Pick the Detector", detectorlist)
st.write('Detector: ', detector)

if select_event:
    #generate waveform
    hp, hc = get_td_waveform(approximant="IMRPhenomD",
                             mass1=mass_1 if 'mass_1' in locals() else default_mass_1,
                             mass2=mass_2 if 'mass_2' in locals() else default_mass_2,
                             delta_t=1.0/16384,
                             f_lower=30,
                             distance=dist)

    #Zoom in near the merger time
    wave = plt.figure(figsize=pylab.figaspect(0.4))
    plt.plot(hp.sample_times, hp, label='Plus Polarization')
    plt.plot(hp.sample_times, hc, label='Cross Polarization')
    plt.xlabel('Time (s)')
    plt.ylabel('Strain')
    plt.xlim(-.5, .5)
    plt.legend()
    plt.grid()


        #get timeseries and gps info to confirm
    segment = (int(gps_info)-5, int(gps_info)+5)
    ldata = TimeSeries.fetch_open_data(detector, *segment, verbose=True, cache=True)

    #Q Transform GPS data
    gps_time_start = st.slider('Select GPS Start Range', -5.0, 0.1, (-1.0))
    gps_time_end = st.slider('Select GPS End Range', 0.1, 5.0, (1.0))

    #st.subheader('Q-transform')
    t0 = datasets.event_gps(event_name)
    dtboth = st.slider('Time Range (seconds)', 0.1, 8.0, 1.0)  # min, max, default
    dt = dtboth / 2.0
    vmax = st.slider('Colorbar Max Energy', 10, 500, 25)  # min, max, default
    qcenter = st.slider('Q-value', 5, 120, 5)  # min, max, default
    qrange = (int(qcenter*0.8), int(qcenter*1.2))

    hq = ldata.q_transform(outseg=(t0-dt, t0+dt), qrange=qrange)

    with _lock:
        fig4 = hq.plot()
        ax = fig4.gca()
        fig4.colorbar(label="Normalised energy", vmax=vmax, vmin=0)
        ax.grid(False)
        ax.set_yscale('log')
        ax.set_ylim(bottom=15)
        st.pyplot(fig4, clear_figure=True)

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

    col7.write(wave)

    col8.pyplot(plot)

else:
    st.write("Waiting for user to click on event...")




#--Add

#--Fix errors
#startup gps error
#detector error