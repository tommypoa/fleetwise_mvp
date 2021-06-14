# !pip install xlrd
# !pip install plotly
# !pip install jupyter_dash
# !git clone https://github.com/jhochs/Fleetwise.git


import pandas as pd
import numpy as np
import pickle

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from jupyter_dash import JupyterDash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import base64
import io
import os


import json
from dash.exceptions import PreventUpdate

pd.options.mode.chained_assignment = None # default='warn'

### FUNCTIONS

def import_subWO_pull(filename):
    df = pd.read_excel(filename, na_values='')
    return df

def WO_from_subWO(subWO):
    WO = subWO.groupby(['Work Order Id']).agg({'Asset Id':'first', 'Item Desc':'first', 'Priority Cd':'first', 'Sub Work Order State Cd':'first', 'Actual Labor Cost':sum, 'Actual Labor Hours':sum, 'Actual Non-Labor Cost':sum, 'Est Labor Cost':sum, 'Est Labor Hours':sum, 'Est Non-Labor Cost':sum, 'Estbd Dt/Time':min, 'Last Tran Dt/Time':max,  'Asset LIN/TAMCN':'first', 'Closed Dt':max, 'Maint Team Desc':pd.Series.mode, 'Service End Dt':max, 'Downtime days':max})
    return WO

def calculate_downtime(WO):
    today = pd.Timestamp('today')
    # today = pd.to_datetime('2021-05-05 00:00:00') #for development
    if type(WO['Estbd Dt/Time'].loc[0]) == str:
        WO['Estbd Dt/Time'] = WO['Estbd Dt/Time'].map(lambda x: pd.Timestamp(x))
        WO['Service End Dt'] = WO['Service End Dt'].map(lambda x: pd.Timestamp(x))
        WO['Closed Dt'] = WO['Closed Dt'].map(lambda x: pd.Timestamp(x))
    WO['Downtime days'] = (today - WO['Estbd Dt/Time']).astype('timedelta64[D]').astype(int)
    return WO

def import_MVR_pull(filename):
    MVR = pd.read_excel(filename, na_values='')
    MVR.dropna(axis=1, how='all', inplace=True) #drop empty columns
    MVR.dropna(axis=0, how='all', inplace=True) #drop empty rows
    MVR = MVR[~MVR['Reg Number'].str.contains('Vac')] # drop vacancy rows
    
    MVR_types = MVR.groupby(['Mgmt Cd']).agg({'Auth Qty':sum, 'Asset Quantity':sum, 'Vacant Authorizations':sum, 'VEH Type Name':pd.Series.mode, 'VEH Cat':pd.Series.mode})

    # Create dict connecting reg number and unit assigned
    reg_unit_dict = pd.Series(MVR['Unit'].values, index=MVR['Mgmt Cd'].values).to_dict()
    # reg_unit_dict = {'AF'+str(k): v for k, v in reg_unit_dict.items()} # add AF to beginning of reg number to match sub-WO pull

    # Create dict connecting MGMT code and description
    type_desc_dict = pd.Series(MVR_types['VEH Type Name'], index=MVR_types.index).to_dict()

    # Create dict connecting MGMT code and category
    type_cat_dict = pd.Series(MVR_types['VEH Cat'], index=MVR_types.index).to_dict()

    return MVR, MVR_types, reg_unit_dict, type_desc_dict, type_cat_dict

def calculate_MCR(open_WO, MVR): # todo: totals = 0 validation
  # CALCULATE MC RATE BY VEH CATEGORY

  totals = []
  NMCS = []
  NMCM = []

  cats = MVR['VEH Cat'].unique()
  
  for cat in cats:
      # Sum total number of vehicles on base with this category:
      totals.append(len(MVR.loc[MVR['VEH Cat'] == cat]))

      # NMCS has state code AWSM:
      NMCS_WO = open_WO.loc[np.logical_and((open_WO['VEH Cat'] == cat), \
                                          (open_WO['Sub Work Order State Cd'] == 'AWSM-Apprvd-in shop awtng mtrls, wrk stop'))]
      NMCS.append(len(NMCS_WO))

      # NMCM has state codes AIPR, CAWI, IIPR:
      NMCM_WO = open_WO.loc[np.logical_and((open_WO['VEH Cat'] == cat), \
                                          (open_WO['Sub Work Order State Cd'].str.contains('AIPR|CAWI|IIPR', regex=True)))]
      NMCM.append(len(NMCM_WO))

      # QUESTION: different state codes for sub work orders?

  # Calculate percentage rates:
  MCR = 100 * np.divide(np.subtract(totals, np.add(NMCM, NMCS)), totals)
  MCR_overall = 100 * (1 - (np.sum(NMCM)+np.sum(NMCS)) / sum(totals) )
  NMCSR = 100 * np.divide(NMCS, totals)
  NMCMR = 100 * np.divide(NMCM, totals)

  return MCR, MCR_overall, NMCSR, NMCMR, cats

def calculate_overview_numbers(open_WO, WO):
  # CALCULATE OVERVIEW NUMBERS (opened, closed, count by ETIC):

  today = pd.Timestamp('today')
  # today = pd.to_datetime('2021-05-04 00:00:00') # for development, since DPAS pull is not up to date

  week_ago = today - pd.Timedelta(value=7, unit='days')
  month_ago = today - pd.Timedelta(value=30, unit='days')
  ranges = [week_ago, month_ago]

  open = len(open_WO)
  ETIC_expired = len(open_WO[open_WO['Service End Dt'] < today])
  ETIC_1week = len(open_WO[np.logical_and((open_WO['Service End Dt'] > today), (open_WO['Service End Dt'] < today + pd.Timedelta(value=7, unit='days')))])

  opened = []
  closed = []
  for t0 in ranges:
    opened.append(len(WO[WO['Estbd Dt/Time'] > t0]))
    closed.append(len(WO[WO['Closed Dt'] > t0]))

  return open, opened, closed, ETIC_expired, ETIC_1week

def downtime_report(WO, intervals):
  # GENERATE NUMBERS FOR DOWNTIME BAR CHART:

  WO = WO.loc[WO['Sub Work Order State Cd'].str.contains('AWSM|AIPR|CAWI|IIPR', regex=True)]
  MGMTs = WO['Asset LIN/TAMCN'].unique()
  counts = np.zeros((len(MGMTs), len(intervals)))

  for i, mgmt in enumerate(MGMTs):
    # Get the open WOs for this MGMT code only:
    WO_mgmt = WO.loc[WO['Asset LIN/TAMCN'] == mgmt]
    for j in range(len(intervals)-1):
      counts[i,j] = len(WO_mgmt.loc[np.logical_and(WO_mgmt['Downtime days'] > intervals[j], WO_mgmt['Downtime days'] <= intervals[j+1])])
    # The final column is for number of vehicles with downtime greater than the highest interval:
    counts[i,j+1] = len(WO_mgmt.loc[WO_mgmt['Downtime days'] > intervals[j+1]])

  # From intervals, record column names for legend:
  col_names = []
  for j in range(len(intervals)-1):
    col_names.append(str(intervals[j]) + '-' + str(intervals[j+1]) + ' days')
  col_names.append(str(intervals[j+1]) + '+ days')

  # Create new dataframe:
  downtime_counts = pd.DataFrame(data=counts.astype(int), columns=col_names)

  return MGMTs, downtime_counts

def subWO_table(subWO_df, reg_unit_dict):
  # CREATE SUBWO DATAFRAME FOR TABLE DISPLAY:

  # Delete the columns not of interest from the open sub WO dataframe:
  subWO_df_tab = subWO_df.drop(subWO_df.columns.difference(['Work Order Id', 'Sub Work Order Id', 'Asset Id', 'Sub Work Order State Cd', 'Asset LIN/TAMCN', 'Unit', 'Estbd Dt/Time', 'Maint Team Desc', 'Remarks', 'Service Performed', 'Service End Dt', 'Downtime days']), 1)

  #Drop any rows with state codes other than AWSM|AIPR|CAWI|IIPR:
  subWO_df_tab = subWO_df_tab[subWO_df_tab['Sub Work Order State Cd'].str.contains('AWSM|AIPR|CAWI|IIPR', regex=True)]

  # Add column for unit:
  subWO_df_tab.insert(loc=3, column='Unit', value=subWO_df_tab['Asset LIN/TAMCN'].map(reg_unit_dict))
  subWO_df_tab['Unit'] = subWO_df_tab['Unit'].map(lambda x: "Unknown" if pd.isna(x) else x)

  # Convert ETIC from datetime to date:
  subWO_df_tab['ETIC'] = subWO_df_tab['Service End Dt'].dt.strftime('%Y-%m-%d') #.dt.date
  subWO_df_tab.drop(columns=['Service End Dt'], inplace=True)

  # Add column for downtime:
  today = pd.Timestamp('today')
  # today = pd.to_datetime('2021-05-04 00:00:00') #for development
  subWO_df_tab['Downtime days'] = (today - subWO_df_tab['Estbd Dt/Time']).astype('timedelta64[D]').astype(int)
  subWO_df_tab.drop(columns=['Estbd Dt/Time'], inplace=True)

  # Reorder columns:
  subWO_df_tab = subWO_df_tab[['Work Order Id', 'Asset Id', 'Asset LIN/TAMCN', 'Unit', 'ETIC', 'Downtime days', 'Sub Work Order Id', 'Sub Work Order State Cd', 'Maint Team Desc', 'Service Performed', 'Remarks']]

  # Sort by WO ID (with secondary sort by sub WO ID):
  subWO_df_tab.sort_values(by=['Work Order Id', 'Sub Work Order Id'], inplace=True)

  return subWO_df_tab

def create_plots(MCR, MCR_overall, NMCSR, NMCMR,  MGMTs, downtime_counts, open, opened, closed, ETIC_expired, ETIC_1week, cats):
  # MAKE DASHBOARD PLOTS:

  today = pd.Timestamp('today')
  # today = pd.to_datetime('2021-05-04 00:00:00') #for development

  fig = make_subplots(
      rows=4, cols=6,
      row_heights=[0.16, 0.37, 0.10, 0.37],
      vertical_spacing=0.15,
      specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"},  None],
            [{"type": "bar", "colspan": 3}, None,  None, {"type": "bar", "colspan": 3},  None,  None],
            [{"colspan": 6}, None,  None,  None,  None,  None], # empty row because bar chart xlabels extend down
            [{"colspan": 6}, None,  None,  None,  None,  None]],
      subplot_titles=("Open work orders", "Opened in last month", "Closed in last month", "ETIC expired", "ETIC due in next 7 days", "MCR - Current  (Overall: " + str(MCR_overall)[0:4] + "%)", "NMCR - Current", "", "Downtime report (open work orders)"))

  # Numbers overview:
  fig.add_trace(go.Indicator(mode = "number", value = open), row=1, col=1)
  fig.add_trace(go.Indicator(mode = "number", value = opened[1]), row=1, col=2)
  fig.add_trace(go.Indicator(mode = "number", value = closed[1]), row=1, col=3)
  fig.add_trace(go.Indicator(mode = "number", value = ETIC_expired), row=1, col=4)
  fig.add_trace(go.Indicator(mode = "number", value = ETIC_1week), row=1, col=5)

  # MCR graph:
  target = 90
  fig.add_trace(go.Scatter(name='Target', x=cats, marker_color="red", y=target*np.ones(cats.shape), mode='lines', legendgroup = '1', hoverinfo='skip'), row=2, col=1)
  fig.add_trace(go.Bar(name='MCR', x=cats, y=MCR, marker_color="royalblue", legendgroup = '1', hovertemplate='<b>%{x}</b>: %{y:.2f}%'), row=2, col=1)

  # NMCR graph:
  fig.add_trace(go.Bar(name='NMCS', x=cats, y=NMCSR, marker_color="gold", legendgroup = '2', hovertemplate='<b>%{x}</b>: %{y:.2f}%'), row=2, col=4)
  fig.add_trace(go.Bar(name='NMCM', x=cats, y=NMCMR, marker_color="coral", legendgroup = '2', hovertemplate='<b>%{x}</b>: %{y:.2f}%'), row=2, col=4)

  # Downtime report graph:
  colors = ['gold', 'orange', 'coral', 'orangered']
  descs = []
  for mgmt in MGMTs:
    try:
      descs.append(type_desc_dict[mgmt])
    except:
      descs.append('Unknown')
  for i, col in enumerate(downtime_counts.columns):
    fig.add_trace(go.Bar(name=col, x=MGMTs, y=downtime_counts[col], customdata=descs, marker_color=colors[i], legendgroup = '3', hovertemplate='<b>%{x} (%{customdata})</b>: %{y:.0f}'), row=4, col=1)

  '''
  # Sub-WO table:
  fig.add_trace(
      go.Table(
          header=dict(
              values=["<b>Work Order ID</b>", "<b>Sub Work Order ID</b>", "<b>Asset ID</b>", "<b>Sub Work Order<br>State Code</b>",
                      "<b>Maint Team</b>", "<b>ETIC</b>", "<b>Service</b>"],
              font=dict(size=10),
              align="left"
          ),
          cells=dict(
              values=[open_subWO[k].tolist() for k in open_subWO.columns],
              font=dict(size=10),
              align = "left")
      ),
      row=4, col=1
  )
  '''
  fig.update_layout({"barmode":"stack"})
  fig.update_layout(height=600, title_text="<b>Fleet Overview - " + today.strftime("%d %b %Y") + "</b>")

  return fig

def create_dashboard(subWO):
  # Calculate donwtime days:
  subWO = calculate_downtime(subWO)

  # Create new dataframe with only open work orders:
  open_subWO = subWO.loc[subWO['Work Order Status Cd'] == 'O-Open']

  # Create new dataframes dropping repetitive sub work orders
  open_WO = WO_from_subWO(open_subWO)
  WO = WO_from_subWO(subWO)

  # Add category columns:
  open_WO['VEH Cat'] = open_WO['Asset LIN/TAMCN'].map(type_cat_dict)
  open_WO['VEH Cat'] = open_WO['VEH Cat'].fillna('Unknown') # handles any MGMTs missing from MVR

  MCR, MCR_overall, NMCSR, NMCMR, cats = calculate_MCR(open_WO, MVR)
  open, opened, closed, ETIC_expired, ETIC_1week = calculate_overview_numbers(open_WO, WO)
  MGMTs, downtime_counts = downtime_report(open_WO, [0, 30, 60, 90])
  open_subWO_tab = subWO_table(open_subWO, reg_unit_dict)
  fig = create_plots(MCR, MCR_overall, NMCSR, NMCMR, MGMTs, downtime_counts, open, opened, closed, ETIC_expired, ETIC_1week, cats)

  return fig, open_subWO_tab

def create_MEL_table(subWO):
  MEL = subWO.loc[(subWO['Work Order Status Cd'] == 'O-Open') & (subWO['Sub Work Order State Cd'].str.contains('AWSM|AIPR|CAWI|IIPR', regex=True))]
  MEL = MEL.groupby('Asset LIN/TAMCN')['Asset Id'].nunique().to_frame('In Shop').reset_index().rename(columns={'Asset LIN/TAMCN': 'Mgmt Cd'})
  MEL = MVR_types.reset_index().merge(MEL, how='left').fillna(0).drop(columns=['Vacant Authorizations'])
  MEL = MEL.merge(vprl.groupby('Mgmt Cd').agg({'MEL':sum}).reset_index(), how='left')
  MEL['MEL +/-'] = MEL['Asset Quantity'] - MEL['In Shop'] - MEL['MEL']
  MEL['MEL +/-'] = MEL['MEL +/-'].fillna(MEL['Asset Quantity'] - MEL['In Shop'])
  MEL_columns = MEL.columns.tolist()
  MEL = MEL[MEL_columns[:3] + MEL_columns[-3:] + MEL_columns[3:5]] # Reordering
  return MEL

### STYLESHEET

stylesheet = {'logo': {
                'height': '9vh',
                'paddingLeft': '2vw'
              },
              'H3': {
                'fontSize': '22px'
              },
              'header_bar': {
                'height': '10vh',
                'display': 'flex',
                'flex-direction': 'row',
                'align-items': 'center',
                'backgroundColor': '#060D3D',
                'color': 'white',
                'fontFamily': '"Open Sans", verdana, arial, sans-serif'
              },
              'header_bar_L': {
                'width': '9vw'
              },
              'header_bar_C': {
                'width': '60vw'
              },
              'header_bar_R': {
                'width': '29vw'                
              },
              'upload': {
                'height': '90%',
                'borderWidth': '3px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'padding': '10px',
                'textAlign': 'center',
                'margin': '6px',
                'fontSize': '14px',
                'white-space': 'pre'
               },
              'hyperlink': {
                'cursor': 'pointer'
                },
              'H4':{
                'fontSize': '18px',
                'color': '#717171',
                'fontFamily': '"Open Sans", verdana, arial, sans-serif',

              },
              'filters': {
                'width' : '300px',
                'padding': '3px',
                'fontFamily': '"Open Sans", verdana, arial, sans-serif'
                  },
              'graph' : {
                'height': '70vh', 
                'width': '100vw', 
                'marginLeft': 'auto', 
                'marginRight': 'auto'
              },

              'table-cell': {
                'whiteSpace': 'normal',
                'textAlign': 'left',
                'fontSize': 11, 
                'font-family':'"Open Sans", verdana, arial, sans-serif',
                'padding': '4px',
                'minWidth': 95, 
                'maxWidth': 95, 
                'width': 95
              },
              'table-header': {
                'whiteSpace': 'normal',
                'fontWeight': 'bold'
              },
              'table-style': {
                'height': '100vh', 
                'width': '90vw',
                'overflowY': 'scroll', 
                'marginLeft': 'auto', 
                'marginRight': 'auto'
              },

              'filter-container': {
                'margin': '20px',
                'display': 'flex',
                'flex-direction': 'row',
                # 'justify-content': 'space-between',
              }
              }


### MAIN


# IMPORT FILES
# subWO = import_subWO_pull(os.getcwd() + '/static/2020-2021 Sub Work Order Inquiry.xls')
MVR, MVR_types, reg_unit_dict, type_desc_dict, type_cat_dict = import_MVR_pull(os.getcwd() + '/static/Fleet_Posture_(Current) SJAFB 8 Apr 2021.xls')
logo_base64 = base64.b64encode(open(os.getcwd() + '/static/logo.png', 'rb').read()).decode('ascii')

# VPRL / MEL
vprl = pd.read_excel(os.getcwd() + '/static/2021 MEL VPRL VETTING.xls', skiprows=7)
vprl.columns = vprl.columns.str.title()
# vprl = vprl.drop(columns=['Substitute Vehicles', '  Pri Recall','Unnamed: 8','Unnamed: 9', 'Remarks', 'Detail Document Number'])
# vprl = vprl.drop(index=0)
vprl = vprl.rename(columns={'Vehicle Nomenclature': 'Vehicle Type', 'Asg\'D': 'ASSN', 'Mgt Code': 'Mgmt Cd', 'Mel': 'MEL'})

# fig, open_subWO_tab = create_dashboard(subWO)

app = JupyterDash(__name__)
server = app.server

app.layout = html.Div([
    ### Data store
    dcc.Store(id='table-data'),

    ### Header bar with upload button
    html.Div([
      html.Div([html.Img(src='data:image/png;base64,{}'.format(logo_base64), style=stylesheet['logo'])], style=stylesheet['header_bar_L']),
      html.Div([html.H3('4th Logistics Readiness Squadron', style=stylesheet['H3'])], style=stylesheet['header_bar_C']),
      html.Div([dcc.Upload(
              id='upload-data',
              children=html.Div(
                  html.A('Upload DPAS Sub-Work Order Pull Excel\n(Drag and Drop or Click to Select Files)', style=stylesheet['hyperlink']), 
                  style=stylesheet['upload']),
                  multiple=True)], 
              style=stylesheet['header_bar_R']),
    ], style=stylesheet['header_bar']),

    html.Br(),

    ### Dashboard graphs
    dcc.Loading(
     id = 'dashboard-loading',
     type='circle',
     children=[
      html.Div(
      id = 'dashboard-container',
      children = [
        dcc.Graph(id='dashboard',
                  style=stylesheet['graph'])
      ])
     ]),

    html.Div(
      id = 'dashboard-prompt',
      children = [
        html.H4("Upload data to generate graphs", style=stylesheet['H4'])
      ]
    ),

    ### Filters
    # html.H3('Filters'),
    html.Div(
        children=[
                  html.H4("Open sub work orders (AWSM, AIPR, CAWI, IIPR only):", style=stylesheet['H4'])
                  ]
             ),

    html.Div([
      html.Div(dcc.Dropdown(id='unit_filters',
                            multi=True,
                            # options=[{'label': unit, 'value': unit} for unit in open_subWO_tab['Unit'].unique()],
                            placeholder="Units"),
                style=stylesheet['filters']),


      html.Div(dcc.Dropdown(id='status_filters',
                            multi=True,
                            # options=[{'label': status, 'value': status} for status in open_subWO_tab['Sub Work Order State Cd'].unique()],
                            placeholder='Vehicle status'),
                 style=stylesheet['filters']),

      html.Div(dcc.Dropdown(id='veh_type_filters',
                            multi=True,
                            # options=[{'label': veh_type, 'value': veh_type} for veh_type in open_subWO_tab['Asset LIN/TAMCN'].unique()],
                            placeholder='Vehicle types'),
                style=stylesheet['filters'])
    ],
      style=stylesheet['filter-container']),

    ### Work order table
    dash_table.DataTable(
        id='datatable-interactivity',
        # columns=[
        #     {"name": i, "id": i, "deletable": False, "selectable": True} for i in open_subWO_tab.columns #["Work Order ID", "Sub Work Order ID", "Asset ID", "Sub Work Order State Code", "Maint Team", "ETIC", "Service"]
        # ],
        # data=open_subWO_tab.to_dict('records'),
        style_cell=stylesheet['table-cell'],
        style_header=stylesheet['table-header'],
        style_cell_conditional=[{
        'if': {'column_id': 'Sub Work Order State Code'},
        'leftBorder': 'rgb(30, 30, 30)',
        'color': 'white'
        }],
        style_header_conditional=[{
        'if': {'column_editable': False},
        'backgroundColor': 'rgb(30, 30, 30)',
        'color': 'white'
        }],
        style_data_conditional=[
        {
            'if': {
                'row_index': 'odd'
            },
            'backgroundColor': 'rgb(248, 248, 248)'
        },
        {
            'if': {
                'column_id': 'Downtime days',
                'filter_query': '{Downtime days} < 30'
            },
            'backgroundColor': 'gold',
            'color': 'white'
        }, 
                {
            'if': {
                'column_id': 'Downtime days',
                'filter_query': '{Downtime days} >= 30 && {Downtime days} < 60'
            },
            'backgroundColor': 'orange',
            'color': 'white'
        }, 
                {
            'if': {
                'column_id': 'Downtime days',
                'filter_query': '{Downtime days} >= 60 && {Downtime days} < 90'
            },
            'backgroundColor': 'coral',
            'color': 'white'
        }, 
                {
            'if': {
                'column_id': 'Downtime days',
                'filter_query': '{Downtime days} > 90'
            },
            'backgroundColor': 'orangered',
            'color': 'white'
        }, 
        ],
        style_table=stylesheet['table-style'],
        editable=True,
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        row_deletable=False,
        selected_columns=[],
        selected_rows=[],
        fixed_rows={'headers': True},
        page_action="none",
        export_format="csv",
        virtualization=True # makes it load faster, but for some reason this prevents text wrapping
    ),
      html.Div(
        children=[
                  html.H4("MEL Table", style=stylesheet['H4'])
                  ]
             ),
    dash_table.DataTable(
      id='MEL-data',
      style_cell={
      'whiteSpace': 'normal',
      'textAlign': 'left',
      'fontSize':11, 
      'font-family':'arial',
      'padding': '4px'
      },
      style_header={
      'whiteSpace': 'normal',
      'fontWeight': 'bold'
      },
      style_table={
      'height': '500px', 
      'width': '100%',
      'overflowY': 'auto', 
      'marginLeft': 'auto', 
      'marginRight': 'auto'
      },
      style_data_conditional=([
          { 'if': {
              'filter_query': '{MEL +/-} > 0',
              'column_id': 'MEL +/-'
          },
          'backgroundColor': '#3D9970',
          'color': 'white',
      }                    
      ] + 
      [
          { 'if': {
              'filter_query': '{MEL +/-} = 0',
              'column_id': 'MEL +/-'
          },
          'backgroundColor': 'yellow',
          'color': 'white',
      }                    
      ] +
      [
          { 'if': {
              'filter_query': '{MEL +/-} < 0',
              'column_id': 'MEL +/-'
          },
          'backgroundColor': 'tomato',
          'color': 'white',
      }                    
      ]) ,
      editable=True,
      filter_action="native",
      sort_action="native",
      sort_mode="multi",
      export_format="csv"
    )
])
# ['gold', 'orange', 'coral', 'orangered']
# Callback: Hides dashboard graphs when store is empty / no data is being uploaded
@app.callback(
    [Output('dashboard-container','style'),
     Output('dashboard-prompt', 'style')],
    [Input('dashboard','figure')]
)
def graph_hide(fig):
    if fig is None:
        return dict(display='none'), dict(margin='10px')
    else:
        return dict(), dict(display='none')

# Callback: Triggered by file import, updates dashboard and data store with table-data
@app.callback([Output('dashboard', 'figure'),
               Output('table-data', 'data'),
               Output('MEL-data', 'data'), 
               Output('MEL-data', 'columns')],
              #  Output('unit_filters', 'options'),
              #  Output('status_filters', 'options'),
              #  Output('veh_type_filters', 'options')],
              [Input('upload-data', 'contents'),
              Input('upload-data', 'filename')], prevent_initial_call=True)
def update_dashboard(list_of_contents, list_of_filename):
    if list_of_contents is not None:
      contents = list_of_contents[0]
      filename = list_of_filename[0]
      content_type, content_string = contents.split(',')
      decoded = base64.b64decode(content_string)

      try:
          if 'csv' in filename:
              # Assume that the user uploaded a CSV file
              df = pd.read_csv(
                  io.StringIO(decoded.decode('utf-8')))
          elif 'xls' in filename:
              # Assume that the user uploaded an excel file
              df = pd.read_excel(io.BytesIO(decoded))
      except Exception as e:
          print(e)
          return html.Div([
              'There was an error processing this file.'
          ])
      fig, open_subWO_tab = create_dashboard(df)
      MEL_data = create_MEL_table(df)
      # unit_filters = [{'label': unit, 'value': unit} for unit in open_subWO_tab['Unit'].unique()]
      # status_filters = [{'label': status, 'value': status} for status in open_subWO_tab['Sub Work Order State Cd'].unique()]
      # veh_type_filters = [{'label': veh_type, 'value': veh_type} for veh_type in open_subWO_tab['Asset LIN/TAMCN'].unique()]
      return fig, open_subWO_tab.to_json(), MEL_data.to_dict('records'), [{"name": i, "id": i, "deletable": False, "selectable": True} for i in MEL_data.columns]#, unit_filters, status_filters, veh_type_filters


# Callback: Triggered by data store change or filters, updates data table
@app.callback([Output('datatable-interactivity', 'data'),
              Output('datatable-interactivity', 'columns'),
              Output('unit_filters', 'options'),
              Output('status_filters', 'options'),
              Output('veh_type_filters', 'options')],
              [Input('unit_filters', 'value'),
               Input('status_filters', 'value'),
               Input('veh_type_filters', 'value'),
               Input('table-data', 'data')], prevent_initial_call=True)
def update_table(unit_filters, status_filters, veh_type_filters, store):
    if store:
      open_subWO_tab = pd.read_json(store)

      unit_filters_options = [{'label': unit, 'value': unit} for unit in open_subWO_tab['Unit'].unique()] # Want to see all units at all times

      if unit_filters:
        open_subWO_tab = open_subWO_tab[open_subWO_tab['Unit'].isin(unit_filters)]

      if status_filters:
        open_subWO_tab = open_subWO_tab[open_subWO_tab['Sub Work Order State Cd'].isin(status_filters)]
        unit_filters_options = [{'label': unit, 'value': unit} for unit in open_subWO_tab['Unit'].unique()]

      if veh_type_filters:
        open_subWO_tab = open_subWO_tab[open_subWO_tab['Asset LIN/TAMCN'].isin(veh_type_filters)]
        unit_filters_options = [{'label': unit, 'value': unit} for unit in open_subWO_tab['Unit'].unique()]
        
      columns = [
            {"name": i, "id": i, "deletable": False, "selectable": True} for i in open_subWO_tab.columns #["Work Order ID", "Sub Work Order ID", "Asset ID", "Sub Work Order State Code", "Maint Team", "ETIC", "Service"]
        ]
      status_filters_options = [{'label': status, 'value': status} for status in open_subWO_tab['Sub Work Order State Cd'].unique()]
      veh_type_filters_options = [{'label': veh_type, 'value': veh_type} for veh_type in open_subWO_tab['Asset LIN/TAMCN'].unique()]
      return open_subWO_tab.to_dict('records'), columns, unit_filters_options, status_filters_options, veh_type_filters_options
    else:
      raise PreventUpdate

# Run app and display result inline in the notebook
if __name__ == '__main__':
    app.run_server(mode='external')