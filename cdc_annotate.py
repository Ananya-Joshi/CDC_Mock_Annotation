import streamlit as st
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
import pandas as pd
import numpy as np
import helper_st

#only allow scrolling, do not allow zoom.
#select points as up, down, or flat
#change the buttons on modebar to only be one of three things
#select points in range
# a, b = st.columns([0.2, 0.8])
# with a:
st.write('Instructions: Select a regime and highlight the region on the graph that corresponds to that regime.')
regime = st.radio("Pick a Regime: ", ['Up', 'Down', 'Flat', "None"], horizontal=True)

df_list, dates_list, titles, natl_ref, captions = helper_st.generate_inputs()

title = titles[1]
df_old= df_list[1][['received_date', 'all']]
df_old['label'] = "None"
if 'points' not in st.session_state:
    st.session_state['points'] = df_old
    st.session_state['state'] = "None"
df = st.session_state['points']
from plotly.subplots import make_subplots
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.update_layout(title=title, showlegend=True,selectdirection='h', xaxis_title='Date', yaxis_title=title, dragmode='select')
df_gray = df
fig.add_trace(go.Scatter(x=df_gray['received_date'], y=df_gray['all'],marker=dict(
        color='gray'), name='Indicator Value', mode='markers+lines'))

df_up = df.query('label =="Up"')
fig.add_trace(go.Scatter(x=df_up['received_date'], y=df_up['all'],marker=dict(
        color='red'), name='Up', mode='markers'))


df_down = df.query('label =="Down"')
fig.add_trace(go.Scatter(x=df_down['received_date'], y=df_down['all'],marker=dict(
        color='blue'), name='Down', mode='markers'))

df_flat = df.query('label =="Flat"')
fig.add_trace(go.Scatter(x=df_flat['received_date'], y=df_flat['all'],marker=dict(
        color='black'), name='Flat', mode='markers'))

fig.update_xaxes(fixedrange=True)
fig.update_yaxes(fixedrange=True)
selected_points = plotly_events(fig, select_event=True, override_width="100%")

if st.session_state['state'] != selected_points:
    st.session_state['state'] = selected_points
    if len(selected_points) > 0:
        selected_dates = pd.date_range(min([x['x'] for x in selected_points]), max([x['x'] for x in selected_points]))
        df['label'] = df[['received_date', 'label']].apply(
            lambda x: regime if x.received_date in selected_dates else x.label, axis=1)
        st.write(df)
        st.session_state['points'] = df
        st.experimental_rerun()

st.button("Submit")