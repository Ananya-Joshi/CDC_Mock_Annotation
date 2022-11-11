from streamlit_plotly_events import plotly_events
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import random

states = ['ak', 'al', 'ar', 'as', 'az', 'ca', 'co', 'ct', 'dc', 'de', 'fl', 'ga',
          'gu', 'hi', 'ia', 'id', 'il', 'in', 'ks', 'ky', 'la',
          'ma', 'md', 'me', 'mi', 'mn', 'mo', 'mp', 'ms', 'mt', 'nc',
          'nd', 'ne', 'nh', 'nj', 'nm', 'nv', 'ny', 'oh', 'ok',
          'or', 'pa', 'pr', 'ri', 'sc', 'sd', 'tn', 'tx', 'ut', 'va', 'vi', 'vt',
          'wa', 'wi', 'wv', 'wy', 'us'
          ]

holiday_ranges = [
    # pd.date_range("11/24/2021", "11/29/2021"),
    # pd.date_range("12/22/2021", "1/2/2022"),
    pd.date_range("5/27/2022", "5/30/2022"),
    # pd.date_range("7/1/2022", "7/4/2022")
]
holiday_dates = []
[holiday_dates.extend(list(x)) for x in holiday_ranges]

def generate_inputs():
        df_list = []
        captions = []
        natl_ref = []

        updated_data = pd.read_csv('data_input.csv', index_col=[0], parse_dates=[0])
        dates_list = [[], [], [], [], [], []]
        for i, name in enumerate(['US', 'TX', 'Loving', 'Missouri', 'LA County, CA', 'NY']):
            tmp_df = updated_data.iloc[:, i].reset_index()
            tmp_df['weeknum'] = tmp_df.Date.dt.weekday
            tmp_df['state'] = name
            tmp_df.columns = ['received_date', 'all', 'weeknum', 'state']
            df_list.append(tmp_df)
            natl_ref.append(updated_data.iloc[:, 0])
            captions.append('')
        return df_list, dates_list, updated_data.columns, natl_ref, captions


def graph_setup(df, title):

    return df, None


def points_setup2(p_str, dates, title):
    # st.write(dates)
    if p_str not in st.session_state:
        g_df = pd.DataFrame()
        g_df['Date'] = pd.to_datetime(dates)
        g_df['Title'] = title
        g_df['Category'] = "Confident"
        g_df['Original Category'] = "Confident"
        st.session_state[p_str] = g_df.copy()
        st.experimental_rerun()


def points_setup1(p_str, dates, title):
    if p_str not in st.session_state:
        st.session_state[p_str] = pd.DataFrame(columns=['Date', 'Value', 'Title', 'Category'])


def create_fig(title, natl_ref, df_old, df, points_setup, p_str, dates=None):
    ax_color = 'darkmagenta'
    ax_color2 = 'darkgreen'

    points_setup2(p_str, dates, title)
    points = st.session_state[p_str]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(title=title, showlegend=True, xaxis_title='Date', yaxis_title=title)
    #
    # if togvals[0]:
    #     fig.add_trace(go.Scatter(x=df_old['received_date'], y=df_old['all'].rolling(7).sum() / 7, name='Weekly Average',
    #                              marker=dict(
    #                                  color=ax_color), opacity=0.3))
    st.write(df)
    fig.add_trace(go.Scatter(x=df['received_date'], y=df['all'], name='Indicator Value', marker=dict(
        color=ax_color), mode='markers+lines', text=pd.to_datetime(df['received_date']).dt.day_name(),
                             hovertemplate="Date: %{x}<br> Weekday: %{text} <br> Value%{y}"))
    name_col = 'all'
    if not points.empty:
        points['Date'] = pd.to_datetime(points['Date'])
        points_1 = points.query('Category=="Mildly Confident"')
        x = points_1['Date']
        y = [float(df_old.query("@z==received_date")['all'].values[0]) for z in x]
        fig.add_trace(go.Scatter(x=x, y=y, marker=dict(size=10,
                                                       color='orange'), name='Mildly Confident', mode='markers',
                                 text=points_1['Date'].dt.day_name(),
                                 hovertemplate="Date: %{x}<br> Weekday: %{text} <br> Value%{y}"))
        points_2 = points.query('Category=="Confident"')
        x = points_2['Date']
        y = [float(df_old.query("@z==received_date")['all'].values[0]) for z in x]

        fig.add_trace(go.Scatter(x=x, y=y, marker=dict(size=10,
                                                       color='red'), name='Confident', mode='markers',
                                 text=points_2['Date'].dt.day_name(),
                                 hovertemplate="Date: %{x}<br> Weekday: %{text} <br> Value%{y}"))
        points_3 = points.query('Category=="Unevaluated"')
        x = points_3['Date']
        y = [float(df_old.query("@z==received_date")['all'].values[0]) for z in x]

        fig.add_trace(go.Scatter(x=x, y=y, marker=dict(size=10,
                                                       color='black'), name='Unevaluated', mode='markers',
                                 text=points_3['Date'].dt.day_name(),
                                 hovertemplate="Date: %{x}<br> Weekday: %{text} <br> Value%{y}"))

    # if 'us' not in title:
    #     if togvals[1]:
    #         fig.add_trace(go.Scatter(x=natl_ref['received_date'], y=natl_ref['all'].rolling(7).sum() / 7,
    #                                  name='Weekly Average National', marker=dict(
    #                 color=ax_color2), visible=True, opacity=0.3), secondary_y=True)

    # To DO: Make it so that you can't select points in that range- you can tooltip over them in gray, they will go down as holiday
    for i, range_dates in enumerate(holiday_ranges):
        if i == 0:
            fig.add_trace(go.Scatter(x=[pd.to_datetime(range_dates[0]), pd.to_datetime(range_dates[0])],
                                     y=[df_old[name_col].min(), df_old[name_col].max()], mode='lines',
                                     marker=dict(size=0,
                                                 color='gray'), name='holidays', legendgroup=1))
            fig.add_trace(go.Scatter(x=[pd.to_datetime(range_dates[-1]), pd.to_datetime(range_dates[-1])],
                                     y=[df_old[name_col].min(), df_old[name_col].max()], mode='lines',
                                     marker=dict(size=10,
                                                 color='gray'), showlegend=False, legendgroup=1))
        else:
            fig.add_trace(go.Scatter(x=[pd.to_datetime(range_dates[0]), pd.to_datetime(range_dates[0])],
                                     y=[df_old[name_col].min(), df_old[name_col].max()], mode='lines',
                                     marker=dict(size=10,
                                                 color='gray'), showlegend=False, legendgroup=1))
            fig.add_trace(go.Scatter(x=[pd.to_datetime(range_dates[-1]), pd.to_datetime(range_dates[-1])],
                                     y=[df_old[name_col].min(), df_old[name_col].max()], mode='lines',
                                     marker=dict(size=10,
                                                 color='gray'), showlegend=False, legendgroup=1))

        fig.add_vrect(
            x0=pd.to_datetime(range_dates[0]), x1=pd.to_datetime(range_dates[-1]),
            fillcolor="Gray", opacity=0.6, line_width=0, name='Holiday'
        )

        df_gray = df.query('received_date in @range_dates')
        fig.add_trace(
            go.Scatter(x=df_gray['received_date'], y=df_gray[name_col], name='holiday', showlegend=False, marker=dict(
                color='gray'), mode='markers+lines', text=pd.to_datetime(df_gray['received_date']).dt.day_name(),
                       hovertemplate="Date: %{x}<br> Weekday: %{text} <br> Value%{y}"))

    if 'sel_dates' not in st.session_state:
        st.session_state['sel_dates'] = {'dates': [], 'pval':{}}
        st.session_state['sel_dates2'] = {'dates': [], 'pval':{}}

    x = st.session_state['sel_dates2']['dates']
    y = [float(df_old.query("@z==received_date")['all'].values[0]) for z in x]
    fig.add_trace(go.Scatter(x=x, y=y, name='selected date', marker=dict(
        color='green'), mode='markers'))


    fig.update_yaxes(title_text="<b>Natl</b> Weekly Avg Claims", secondary_y=True)
    fig.update_layout(yaxis=dict(color=ax_color, tickfont=dict(color=ax_color)),
                      yaxis2=dict(color=ax_color2, tickfont=dict(color=ax_color2)))

    if 'state' not in st.session_state:
        st.session_state['state'] = pd.DataFrame()


    x = st.session_state['sel_dates']['dates']
    y = [float(df_old.query("@z==received_date")['all'].values[0]) for z in x]
    fig.add_trace(go.Scatter(x=x, y=y, name='selected date',marker=dict(
                color='blue'), mode='markers' ))


    selected_points = plotly_events(fig, select_event=True, override_width="100%")
    if len(x) > 0:
        st.write("Rank Green Points")
        for x_i in x:
            value = 0.5
            if st.session_state['sel_dates']['pval'].get(x_i, None) != None:
                value = st.session_state['sel_dates']['pval'].get(x_i, None)
            st.session_state['sel_dates']['pval'][x_i] = st.slider(f'Date: {x_i}',  0.0, 1.0, value=value, key=f"{x_i}_ck")

    return points, selected_points, fig


def highlight(s):
    if pd.to_datetime(s.Date) in holiday_dates:
        return ['background-color: gray'] * len(s)
    elif s.Category == "Mildly Confident":
        return ['background-color: orange'] * len(s)
    else:
        return ['background-color: red'] * len(s)


def task_1(df_old, title, natl_ref):
    df = df_old.copy()
    p_str = 'points1'
    points, selected_points, fig = create_fig(title, natl_ref, df_old, df, points_setup1, p_str)
    name_col = 'all'
    state = st.session_state['state']
    sp_df = pd.DataFrame.from_records(selected_points).drop_duplicates(subset=['x'])
    # print(sp_df)
    if not state.equals(sp_df):
        if sp_df.shape[0] > 0:
            for i, point in sp_df.iterrows():
                # print("iterant", i, point)
                date = pd.to_datetime(point['x'])
                val = float(df_old.query("received_date==@date")[name_col].values[0])
                # print(df_old.query("received_date==@date"), points)
                if not points.empty:
                    # print("POINTS ARE **CRUCIAL",points.query('Date==@date'))
                    if points.query('Date==@date').empty:
                        st.session_state[p_str] = pd.concat([points, pd.DataFrame([{'Date': date, 'Value': val,
                                                                                    'Title': title,
                                                                                    'Category': 'Mildly Confident'}])]).reset_index(
                            drop=True)
                    elif (points.query('Date==@date').Category == "Mildly Confident").all():
                        points = points[points.Date != date]
                        st.session_state[p_str] = pd.concat([points, pd.DataFrame(
                            [{'Date': date, 'Value': val, 'Title': title, 'Category': 'Confident'}])]).reset_index(
                            drop=True)
                    else:
                        # print('remove state')
                        st.session_state[p_str] = points[points.Date != date]

                else:
                    st.session_state[p_str] = pd.DataFrame(
                        [{'Date': date, 'Value': val, 'Title': title, 'Category': "Mildly Confident"}])
                    selected_points[i]['curveNumber'] = 'NA'
                points = st.session_state[p_str]
                # print(points, 'PE')
        st.session_state['state'] = sp_df
        st.experimental_rerun()
    st.write("Points Identified")
    t_list = []
    if points.empty:
        st.write("⚠️ No Points Identified")
    else:

        points['Date'] = points['Date'].dt.strftime('%b %d, %Y')

        with st.expander('Points'):
            for i, row in points[['Date', 'Value', 'Category']].reset_index(drop=True).iterrows():
                a, b = st.columns(2)
                with a:
                    repl_val = st.select_slider(f"Flag: {pd.to_datetime(row.Date).strftime('%b %d, %Y')}",
                                                ["Disagree", "Mildly Confident", "Confident"], value=row.Category,
                                                key=f"{i}_check")
                    st.write(points.query('Date==@row.Date').Category.values[0])
                    if points.query('Date==@row.Date').Category.values[0] != repl_val:
                        points.loc[points.query('Date==@row.Date').index, 'Category'] = repl_val
                        st.experimental_rerun()
                with b:
                    t_list.append(
                        st.text_area("Additional Comments (may not save between refreshes):", key=f"{i}_text"))

        # st.dataframe(points[['Date', 'Value', 'Category']].reset_index(drop=True).style.apply(highlight, axis=1))
    if not st.session_state[p_str].equals(points):
        st.session_state[p_str] = points
    # print(datetime.now(), 'END METH')
    return points, pd.DataFrame(t_list)


def task_2(df_old, dates, title, natl_ref):
    # checkboxes

    df, togvals = graph_setup(df_old.copy(), title)

    p_str = 'points2'
    # st.write(dates)
    points, selected_points, fig = create_fig(title, natl_ref, df_old, df, points_setup2, p_str, togvals, dates)
    selected_dates = pd.Series([pd.to_datetime(x['x']) for x in selected_points])
    sel_dates = [pd.to_datetime(x) for x in selected_dates[selected_dates.isin(dates)]]
    # change the level of the sel_dates points %3

    points2 = points.copy()
    #st.write(points2)
    if st.session_state.get('points3', None) is None:
        points2.Category = 'Unevaluated'

        # print(sel_dates, points2.Date.isin(sel_dates))
    points2.apply(lambda x: "Mildly Confident" if (pd.to_datetime(x.Date) in sel_dates) else None, axis=1)
    points2['Category'] = points2.apply(
        lambda x: "Confident" if (x.Category == "Mildly Confident" and x.Date in sel_dates)
        else "Disagree" if (x.Category == "Confident" and x.Date in sel_dates)
        else "Mildly Confident" if (x.Date in sel_dates)
        else x.Category, axis=1)
    # print(points2)
    # points2['Category'] = points2.apply(
    #     lambda x: "Disagree" if (x.Category == "Confident" and x in sel_dates) else x.Category, axis=1)
    # points2['Category'] = points2.apply(
    #     lambda x: "Mildly Confident" if (x.Category in ["Disagree", "Unevaluated"] and x in sel_dates) else x.Category, axis=1)

    # if "pd.unique(points2.Category):
    # add mechanism for unevaluated and evaluated points
    # unevaluated
    for i, col in points2.query('Date.isin(@dates) and Category=="Unevaluated"').iterrows():
        points2.iloc[i, 2] = st.select_slider(f"Flag: {pd.to_datetime(col.Date).strftime('%b %d, %Y')}",
                                              ["Disagree", "Mildly Confident", "Confident", 'Unevaluated'],
                                              value='Unevaluated',
                                              key=f"{i}_check")
    with st.expander("Evaluated Points"):
        for i, col in points2.query('Date.isin(@dates) and Category!="Unevaluated"').iterrows():
            points2.iloc[i, 2] = st.select_slider(f"Flag: {pd.to_datetime(col.Date).strftime('%b %d, %Y')}",
                                                  ["Disagree", "Mildly Confident", "Confident", 'Unevaluated'],
                                                  value=col.Category,
                                                  key=f"{i}_check_2")
    # st.write(points2)
    state = st.session_state[p_str]
    if not points2.equals(state):
        st.session_state[p_str] = points2
        st.session_state['points3'] = points2.Category
        st.experimental_rerun()
    return points2, []


def pretask1(args):
    df_list = args[0].reset_index(drop=True)
    df_list.iloc[34, 1] = 50000
    natl_ref = args[1]
    st.header("Pre-Test 1/2:")
    st.write('No data will be collected from this question.')
    st.write(f'Complete all the following tasks  on the interactive graph below to move on to the survey. \
        Please watch [this video](https://youtu.be/rUS6gTbqnrU) if you have difficulty with the interface   \n ')
    if 'symbols' not in st.session_state:
        st.session_state['symbols'] = [False, False, False, False]
    symb_count = 0
    for symb, capt in zip(st.session_state['symbols'], ['1. Wait for plots to load.',
                                                        '2. Add point with mildly confident label.',
                                                        '3. Add point with confident label.',
                                                        '4. Toggle Day of week visibility. '
                                                        ]):
        if symb:
            symb_count += 1
            st.write(f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ✅ {capt}  ')
        else:
            st.write(f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ❌ {capt} ')

    st.markdown("""---""")
    if 'state_demo' not in st.session_state:
        st.session_state['state_demo'] = pd.DataFrame()
    if 'wk_demo' not in st.session_state:
        st.session_state['wk_demo'] = []

    task_1(df_list, "Demo Example [Counts]", natl_ref)
    pts = st.session_state['points1']
    wklb = st.session_state['dow_vals']
    if (not pts.equals(st.session_state['state_demo']) or (wklb != st.session_state['wk_demo'])):
        # st.write(not pts.query('Category =="Mildly Confident"').empty)
        st.session_state['symbols'][0] = True
        if not st.session_state['symbols'][1]:
            st.session_state['symbols'][1] = not pts.query('Category =="Mildly Confident"').empty
        if not st.session_state['symbols'][2]:
            st.session_state['symbols'][2] = not pts.query('Category =="Confident"').empty
        if not st.session_state['symbols'][3]:
            st.session_state['symbols'][3] = (sum(wklb) != 7)

        st.session_state['state_demo'] = pts

        st.session_state['wk_demo'] = wklb
        st.experimental_rerun()
    st.markdown("""---""")
    placeholder = st.empty()

    if symb_count == 4:
        placeholder.write('Pretest 1/2 completed!')
        if st.button('Continue'):
            st.session_state['symbols'] = [False, False, False]
            return True
    else:
        placeholder.write(f"⚠️ *{symb_count}/4 tasks completed.*  \n  \
     4/4 needed to move on to Pre-Test 2 ")
    return False


def pretask2(args):
    df_list = args[0]
    natl_ref = args[1]
    st.header("Pre-Test 2/2:")
    st.write('No data will be collected from this question.')
    st.write(f'Complete all the following tasks  on the interactive graph below to move on to the survey. \
        Please watch [this video](https://youtu.be/rUS6gTbqnrU) if you have difficulty with the interface   \n ')
    symb_count = 0
    for symb, capt in zip(st.session_state['symbols'], ['1. Wait for plots to load.',
                                                        '2. Toggle Day of week visibility.',
                                                        '3 Change Slider Value. '
                                                        ]):
        if symb:
            symb_count += 1
            st.write(f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ✅ {capt}  ')
        else:
            st.write(f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ❌ {capt} ')

    st.markdown("""---""")
    if 'state_demo2' not in st.session_state:
        st.session_state['state_demo2'] = pd.DataFrame()
    if 'wk_demo2' not in st.session_state:
        st.session_state['wk_demo2'] = []

    task_2(df_list, np.array(["04/05/2022"]), "Demo Example", natl_ref)
    pts = st.session_state['points2']
    wklb = st.session_state['dow_vals']
    if (not pts.equals(st.session_state['state_demo2']) or (wklb != st.session_state['wk_demo2'])):
        st.session_state['symbols'][0] = True
        if not st.session_state['symbols'][1]:
            st.session_state['symbols'][1] = (sum(wklb) != 7)
        st.session_state['symbols'][2] = not pts['Original Category'].equals(pts['Category'])
        st.session_state['state_demo2'] = pts
        st.session_state['wk_demo2'] = wklb
        st.experimental_rerun()
    st.markdown("""---""")
    placeholder = st.empty()

    if symb_count == 3:
        placeholder.write('Pretest 2/2 completed!')
        if st.button('Continue'):
            return True
    else:
        placeholder.write(f"⚠️ *{symb_count}/3 tasks completed.*  \n  \
     3/3 needed to move on to Survey ")
    return False


def instructions():
    st.markdown(open('instructions.md').read(), unsafe_allow_html=False)
    st.image('demo_pic.png')
    st.markdown(open('instructions2.md').read(), unsafe_allow_html=False)

    y4 = st.checkbox('I have read and understood the instructions.')
    if y4:
        return True
    if st.button('Exit'):
        st.session_state['current_loc'] = 'exit'
        st.experimental_rerun()
    return False



