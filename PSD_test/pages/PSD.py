import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, datasets, fft
from scipy.signal import get_window
import gwpy

import requests, os
from gwpy.timeseries import TimeSeries
from gwosc.locate import get_urls
from gwosc import datasets
from gwosc.api import fetch_event_json


from copy import deepcopy
import base64


from helper import make_audio_file

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


# -- Set page config
apptitle = 'PSD Test'

st.set_page_config(page_title=apptitle, page_icon=":eyeglasses:")

# -- Default detector list
detectorlist = ['H1','L1', 'V1']

# Title the app
st.title('PSD Test')

st.markdown("""
 * Use the menu at left to select data and set plot parameters
 * Your plots will appear below
""")

@st.cache_data(max_entries=5)   #-- Magic command to cache data
def load_gw(t0, detector, fs=4096):
    strain = TimeSeries.fetch_open_data(detector, t0-14, t0+14, sample_rate = fs, cache=False)
    return strain

@st.cache_data(max_entries=10)   #-- Magic command to cache data
def get_eventlist():
    allevents = datasets.find_datasets(type='events')
    eventset = set()
    for ev in allevents:
        name = fetch_event_json(ev)['events'][ev]['commonName']
        if name[0:2] == 'GW':
            eventset.add(name)
    eventlist = list(eventset)
    eventlist.sort()
    return eventlist
    
st.sidebar.markdown("## Select Data Time and Detector")

# -- Get list of events
eventlist = get_eventlist()

#-- Set time by GPS or event
select_event = st.sidebar.selectbox('How do you want to find data?',
                                    ['By event name', 'By GPS'])

if select_event == 'By GPS':
    # -- Set a GPS time:        
    str_t0 = st.sidebar.text_input('GPS Time', '1126259462.4')    # -- GW150914
    t0 = float(str_t0)

    st.sidebar.markdown("""
    Example times in the H1 detector:
    * 1126259462.4    (GW150914) 
    * 1187008882.4    (GW170817) 
    * 1128667463.0    (hardware injection)
    * 1132401286.33   (Koi Fish Glitch) 
    """)

else:
    chosen_event = st.sidebar.selectbox('Select Event', eventlist)
    t0 = datasets.event_gps(chosen_event)
    detectorlist = list(datasets.event_detectors(chosen_event))
    detectorlist.sort()
    st.subheader(chosen_event)
    st.write('GPS:', t0)
    
    # -- Experiment to display masses
    try:
        jsoninfo = fetch_event_json(chosen_event)
        for name, nameinfo in jsoninfo['events'].items():        
            st.write('Mass 1:', nameinfo['mass_1_source'], 'M$_{\odot}$')
            st.write('Mass 2:', nameinfo['mass_2_source'], 'M$_{\odot}$')
            st.write('Network SNR:', int(nameinfo['network_matched_filter_snr']))
            eventurl = 'https://gw-osc.org/eventapi/html/event/{}'.format(chosen_event)
            st.markdown('Event page: {}'.format(eventurl))
            st.write('\n')
    except:
        pass

    
#-- Choose detector as H1, L1, or V1
detector = st.sidebar.selectbox('Detector', detectorlist)

# -- Select for high sample rate data
fs = 4096
maxband = 1200
high_fs = st.sidebar.checkbox('Full sample rate data')
if high_fs:
    fs = 16384
    maxband = 2000


# -- Create sidebar for plot controls
st.sidebar.markdown('## Set Plot Parameters')
dtboth = st.sidebar.slider('Time Range (seconds)', 0.1, 8.0, 1.0)  # min, max, default
dt = dtboth / 2.0


#-- Create a text element and let the reader know the data is loading.
strain_load_state = st.text('Loading data...this may take a minute')
try:
    strain_data = load_gw(t0, detector, fs)
except:
    st.warning('{0} data are not available for time {1}.  Please try a different time and detector pair.'.format(detector, t0))
    st.stop()
    
strain_load_state.text('Loading data...done!')

#-- Make a time series plot

cropstart = t0-0.2
cropend   = t0+0.1

cropstart = t0 - dt
cropend   = t0 + dt

st.subheader('Raw data')
center = int(t0)
strain = deepcopy(strain_data)

with _lock:
    fig1 = strain.crop(cropstart, cropend).plot()
    #fig1 = cropped.plot()
    st.pyplot(fig1, clear_figure=True)


#--Make a PSD Plot

window = get_window('hann', strain_data.size)
lwin = strain_data*window

fftamp = lwin.fft().abs()

with _lock:
    fig5 = fftamp.plot(xscale="log", yscale="log")
    st.pyplot(fig5, clear_figure=True)


#--Make a ASD Plot

fft = strain_data.fft()

window = get_window('hann', strain_data.size)
lwin = strain_data*window

fftamp = lwin.fft().abs()

asd = strain_data.asd(fftlength=2, method="median")

with _lock:
    fig6 = asd.plot(xscale="log", yscale="log")
    st.pyplot(fig6, clear_figure=True)




st.subheader("About this app")
st.markdown("""
This app displays data from LIGO, Virgo, and GEO downloaded from
the Gravitational Wave Open Science Center at https://gwosc.org .


You can see how this works in the [Quickview Jupyter Notebook](https://github.com/losc-tutorial/quickview) or 
[see the code](https://github.com/jkanner/streamlit-dataview).

""")