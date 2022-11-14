import streamlit as st
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
import pandas as pd

#only allow scrolling, do not allow zoom.
#select points as up, down, or flat
#change the buttons on modebar to only be one of three things
#select points in range
# a, b = st.columns([0.2, 0.8])
# with a:
st.set_page_config(layout="wide")
#
# st.title("Label Regimes in Flu Data")
st.write('### Instructions:')
st.write('##### Select a regime and highlight the region on the graph that corresponds to that regime.')
regime = st.radio("", ['Up', 'Down', 'Flat', "None"], horizontal=True)

title = "CA FluSurv Hospitalization Rate"
df_old = pd.read_csv("ca_flusurvdata.csv")[['epiweek', 'rate_overall']]

df_old.columns = ['received_date', 'all']
df_old['label'] = "None"

df_old['received_date'] = df_old['received_date'].astype(int)
#gap is defined in the same year
dates_subset = [f"{x}{str(y).zfill(2)}" for x in range(2013, 2022) for y in range(1, 50)]
dates_df = pd.DataFrame()
dates_df['received_date'] = dates_subset
dates_df['received_date'] = dates_df['received_date'].astype(int)

#st.write(dates_df)
df_old = df_old.merge(dates_df, left_on='received_date', right_on='received_date', how='outer').sort_values('received_date')#.reset_index().drop_duplicates(keep="first")
df_old['old_date'] = df_old['received_date'].copy()
df_old['received_date']  = df_old['received_date'].astype(str)+ "-1"
df_old['received_date'] = pd.to_datetime(df_old['received_date'], format="%Y%W-%w")
#st.write(df_old)
if 'points' not in st.session_state:
    st.session_state['points'] = df_old
    st.session_state['state'] = "None"
    st.session_state['fig'] = go.Figure()
    st.session_state['range'] = [df_old['received_date'].iloc[40],
                        df_old['received_date'].iloc[-40]]

df = st.session_state['points']
# st.write(df_old)
st.write("##### Select Epiweeks to Zoom:")
range_dates = st.select_slider(, df_old['old_date'],
                        value = [df_old['old_date'].iloc[40],
                        df_old['old_date'].iloc[-40]]
                        )#, value=[df_old['received_date'].min(), df_old['received_date'].max()])

if range_dates != st.session_state['range']:
    st.session_state['range']= range_dates
    st.experimental_rerun()
#
range_dates_transform = [pd.to_datetime(str(range_dates[0])+"-1", format="%Y%W-%w"),
                        pd.to_datetime(str(range_dates[-1])+"-1", format="%Y%W-%w"),
                         ]
#df = df.query("received_date > @range_dates[0] and  received_date < @range_dates[1]")

#
# st.write(df['received_date'])
# st.write(df)
from plotly.subplots import make_subplots
fig = make_subplots(specs=[[{"secondary_y": True}]])

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

#fig.update_xaxes(fixedrange=True)
fig.update_yaxes(fixedrange=True).update_traces(connectgaps=False)
# st.write(df['received_date'])

fig.update_layout(xaxis=dict(
        range=[range_dates_transform[0], range_dates_transform[1]],
        tickformat="%Y-%W",
# rangeslider=dict(
#             visible=True
#         )
    ),


    title=title, showlegend=True,selectdirection='h', xaxis_title='Epiweek',   yaxis_title="Hospital admissions rate (per 100,000)", dragmode='select')
#fig.update_xaxes(range=(pd.to_datetime("11/1/2014"), pd.to_datetime("11/1/2017")))
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
# print(fig.layout.xaxis.range)