import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import requests, os
from gwpy.timeseries import TimeSeries
from gwosc.locate import get_urls
from gwosc import datasets
from gwosc.api import fetch_event_json

from copy import deepcopy
import base64
#new imports
import altair as alt
from vega_datasets import data

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
apptitle = 'Quickview GW Catalog'

st.set_page_config(page_title=apptitle, page_icon=":eyeglasses:")

# Title the app
st.title('Quickview GW Catalog')

st.markdown("""
 * Use the menu at left to select data and set plot parameters
 * Your plots will appear below
""")


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

# --Make Histogram of events by mass
import altair as alt
from vega_datasets import data

source = eventlist

alt.Chart(source).mark_bar().encode(
    alt.X("By Mass", bin=True),
    y='count()',
)



#-- Create a text element and let the reader know the data is loading.
strain_load_state = st.text('Loading data...this may take a minute')
try:
    strain_data = load_gw(t0, detector, fs)
except:
    st.warning('{0} data are not available for time {1}.  Please try a different time and detector pair.'.format(detector, t0))
    st.stop()
    
strain_load_state.text('Loading data...done!')




st.subheader("About this app")
st.markdown("""
This app displays data from LIGO, Virgo, and GEO downloaded from
the Gravitational Wave Open Science Center at https://gwosc.org .


You can see how this works in the [Quickview Jupyter Notebook](https://github.com/losc-tutorial/quickview) or 
[see the code](https://github.com/jkanner/streamlit-dataview).

""")