import pandas as pd
import altair as alt
import numpy as np
import streamlit as st


# Fetch the data from the URL and load it into a DataFrame
url = 'https://gwosc.org/eventapi/csv/GWTC/'
data = pd.read_csv(url)

#try search field
dropdown = alt.binding_select(
    options=['total_mass_source', 'mass_1_source', 'mass_2_source'],
    name='Mass Source '

)

xcol_param = alt.param(
    value='total_mass_source',
    bind=dropdown
)

#tooltip = ['total_mass_source', 'commonName', 'GPS', 'final_mass_source'] #check why this scales the y-axis down
# Create the Altair histogram
hist = alt.Chart(data, title='Histogram of Total Mass').mark_bar().encode(
    #alt.X('total_mass_source:Q', bin=alt.Bin(step=1), title='Total Mass'),
    alt.X('x:Q').title(''),
    alt.Y('count()', scale=alt.Scale(domain=[0, 5]), title='Count'),
    ).transform_calculate(
        x=f'datum[{xcol_param.name}]'
    ).add_params(
        xcol_param
    ).interactive()


# Show the chart
st.altair_chart(hist)

