import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import gams.transfer as gt
import pybalmorel as pyb
import gams
import os
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from io import StringIO
from typing import Union
import matplotlib.cm as cm
import plotly.express as px 

#%% ------------------------------- ###
###          0. Hard coded          ###
### ------------------------------- ###

RRR_to_CCC = {
    'DK1': 'DENMARK','DK2': 'DENMARK','FIN': 'FINLAND','NO1': 'NORWAY','NO2': 'NORWAY','NO3': 'NORWAY','NO4': 'NORWAY','NO5': 'NORWAY','SE1': 'SWEDEN',
    'SE2': 'SWEDEN', 'SE3': 'SWEDEN','SE4': 'SWEDEN','UK': 'UNITED_KINGDOM','EE': 'ESTONIA','LV': 'LATVIA','LT': 'LITHUANIA','PL': 'POLAND','BE': 'BELGIUM',
    'NL': 'NETHERLANDS','DE4-E': 'GERMANY','DE4-N': 'GERMANY','DE4-S': 'GERMANY','DE4-W': 'GERMANY','FR': 'FRANCE','IT': 'ITALY','CH': 'SWITZERLAND',
    'AT': 'AUSTRIA','CZ': 'CZECH_REPUBLIC','ES': 'SPAIN','PT': 'PORTUGAL','SK': 'SLOVAKIA','HU': 'HUNGARY','SI': 'SLOVENIA','HR': 'CROATIA','RO': 'ROMANIA',
    'BG': 'BULGARIA','GR': 'GREECE','IE': 'IRELAND','LU': 'LUXEMBOURG','AL': 'ALBANIA','ME': 'MONTENEGRO', 'MK': 'NORTH_MACEDONIA',
    'BA': 'BOSNIA_AND_HERZEGOVINA','RS': 'SERBIA','TR': 'TURKEY','MT': 'MALTA','CY': 'CYPRUS'
}

Parameters_names = [ "CO2_TAX","CO2_EFF","ELYS_ELEC_EFF","H2S_INVC","SMR_CCS_INVC","PV_INVC","ONS_WT_INVC","H2_OandM","SMR_CCS_OandM","H2_TRANS_INVC","ELEC_TRANS_INVC",
                    "IMPORT_H2_P","DH2_DEMAND_EAST","DH2_DEMAND_SOUTH","DH2_DEMAND_NORTH","DH2_DEMAND_WEST",
                    "DE_DEMAND_EAST","DE_DEMAND_SOUTH","DE_DEMAND_NORTH","DE_DEMAND_WEST","PV_LIMIT_NORTH","PV_LIMIT_SOUTH","PV_LIMIT_EAST","PV_LIMIT_WEST",
                    "ONS_LIMIT_EAST","ONS_LIMIT_WEST","ONS_LIMIT_NORTH","ONS_LIMIT_SOUTH",
                    "TRANS_DEMAND_NORTH","TRANS_DEMAND_SOUTH", "TRANS_DEMAND_EAST", "TRANS_DEMAND_WEST", "NATGAS_P" ]

results_selections = ["Elec RE Capacity","Elec PV Capacity", "Elec ONSHORE Capacity", "Elec OFFSHORE Capacity",
                      "Elec RE Production","Elec PV Production", "Elec ONSHORE Production", "Elec OFFSHORE Production", 
                      "H2 Green Capacity", "H2 Blue Capacity", "H2 Green Production", "H2 Blue Production", 
                      "H2 Import Capacity", "H2 Import Production", "H2 Storage", 
                      "H2 Transmission Capacity", "H2 Transmission Flow"]

input_data_selections = ["NATGAS_P", "CO2_TAX", "SMR_CCS_INVC", "PV_LIMIT_NORTH","PV_LIMIT_SOUTH","PV_LIMIT_EAST","PV_LIMIT_WEST","ONS_LIMIT_EAST","ONS_LIMIT_WEST","ONS_LIMIT_NORTH","ONS_LIMIT_SOUTH"]

#%% ------------------------------- ###
###            1. Class             ###
### ------------------------------- ###

class MainResults_GSA:
    def __init__(self, path: str,
               scenario_files: list[str],
               baseline_files: list[str] = [],
               model_names: list[str] = [],
               input_files: list[str] = None
               ):
        
        # Verify that there is the same number of baseline and scenario files or 0 baseline files
        if len(scenario_files) != len(baseline_files) and len(baseline_files) != 0:
            raise ValueError("The number of baseline files should be the same as the number of scenario files or should be 0")
        
        self.path = path
        self.scenario_files = scenario_files
        self.baseline_files = baseline_files
        self.model_names = model_names
        
        # Verify that there is the same number of model names as the number of scenario files
        if len(model_names) != len(scenario_files) and len(model_names) != 0:
            raise ValueError("The number of model names should be the same as the number of scenario files or should be 0")
        # If no model names, assign one by default
        if len(model_names) == 0:
            model_names = ["Model " + str(i) for i in range(1, len(scenario_files) + 1)]
            
        self.nb_models = len(scenario_files)
            
        # Verify that we have two files in the input files list
        if input_files is not None:
            if len(input_files) != 2:
                raise ValueError("The input files should contain 2 files, the gdx input data baseline file and the csv sample file.")
        
        if input_files is not None:
            self.input_files = input_files
            
    ### ------------------------------- ###
    ###         1.1 Import data         ###
    ### ------------------------------- ###
    
    # Import all the results from the scenario and baseline files into a dictionnary used in the other functions
    def import_results(self) -> None:
        
        # Creating the data dictionnary
        dict_results = {}
        
        # Import the scenario data in the DataFrame
        for i, scenarios in enumerate(self.scenario_files):
            scenarios_path = os.path.join(os.path.abspath(self.path), scenarios) # Path to the scenario file
            df = gt.Container(scenarios_path) # Import the scenario file
            # Find the data in the scenario file
            if "PRO_YCRAGF" in df.data.keys():
                df_PRO = pd.DataFrame(df.data["PRO_YCRAGF"].records)
                df_PRO.columns = ['Scenarios', 'Y', 'C', 'RRR', 'AAA', 'G', 'FFF', 'COMMODITY', 'TECH_TYPE', 'UNITS', 'value']
                df_PRO["Table"] = "Generation Production"
                df_PRO['Scenarios'] = df_PRO['Scenarios'].str.extract(r'(\d+)').astype(int)
            if "G_CAP_YCRAF" in df.data.keys():
                df_CAP = pd.DataFrame(df.data["G_CAP_YCRAF"].records)
                df_CAP.columns = ['Scenarios', 'Y', 'C', 'RRR', 'AAA', 'G', 'FFF', 'COMMODITY', 'TECH_TYPE', 'VARIABLE_CATEGORY', 'UNITS', 'value']
                df_CAP["Table"] = "Generation Capacity"
                df_CAP['Scenarios'] = df_CAP['Scenarios'].str.extract(r'(\d+)').astype(int)
            if "XH2_CAP_YCR" in df.data.keys():
                df_XH2_CAP = pd.DataFrame(df.data["XH2_CAP_YCR"].records)
                df_XH2_CAP.columns = ['Scenarios', 'Y', 'C', 'IRRRE', 'IRRRI', 'VARIABLE_CATEGORY', 'UNITS', 'value']
                df_XH2_CAP["Table"] = "Hydrogen Transmission Capacity"
                df_XH2_CAP['Scenarios'] = df_XH2_CAP['Scenarios'].str.extract(r'(\d+)').astype(int)
            if "XH2_FLOW_YCR" in df.data.keys():
                df_XH2_FLOW = pd.DataFrame(df.data["XH2_FLOW_YCR"].records)
                df_XH2_FLOW.columns = ['Scenarios', 'Y', 'C', 'IRRRE', 'IRRRI', 'UNITS', 'value']
                df_XH2_FLOW["Table"] = "Hydrogen Transmission Flow"
                df_XH2_FLOW['Scenarios'] = df_XH2_FLOW['Scenarios'].str.extract(r'(\d+)').astype(int)
            if "G_STO_YCRAF" in df.data.keys():
                df_PRO_STO = pd.DataFrame(df.data["G_STO_YCRAF"].records)
                df_PRO_STO.columns = ['Scenarios', 'Y', 'C', 'RRR', 'AAA', 'G', 'FFF', 'COMMODITY', 'TECH_TYPE', 'VARIABLE_CATEGORY', 'UNITS', 'value']
                df_PRO_STO["Table"] = "Storage"
                df_PRO_STO['Scenarios'] = df_PRO_STO['Scenarios'].str.extract(r'(\d+)').astype(int)
            # Concatenate all the dataframe together and add the model name
            df_scenarios = pd.concat([df_PRO, df_CAP, df_XH2_CAP, df_XH2_FLOW, df_PRO_STO])
            
            if len(self.baseline_files) != 0:
                # Import the baseline data in the DataFrame
                baseline = self.baseline_files[i]
                baseline_path = os.path.join(os.path.abspath(self.path), baseline) # Path to the baseline file
                df = gt.Container(baseline_path) # Import the baseline file
                # Find the data in the baseline file
                if "PRO_YCRAGF" in df.data.keys():
                    df_PRO = pd.DataFrame(df.data["PRO_YCRAGF"].records)
                    df_PRO["Table"] = "Generation Production"
                if "G_CAP_YCRAF" in df.data.keys():
                    df_CAP = pd.DataFrame(df.data["G_CAP_YCRAF"].records)
                    df_CAP["Table"] = "Generation Capacity"
                if "XH2_CAP_YCR" in df.data.keys():
                    df_XH2_CAP = pd.DataFrame(df.data["XH2_CAP_YCR"].records)
                    df_XH2_CAP["Table"] = "Hydrogen Transmission Capacity"
                if "XH2_FLOW_YCR" in df.data.keys():
                    df_XH2_FLOW = pd.DataFrame(df.data["XH2_FLOW_YCR"].records)
                    df_XH2_FLOW["Table"] = "Hydrogen Transmission Flow"
                if "G_STO_YCRAF" in df.data.keys():
                    df_PRO_STO = pd.DataFrame(df.data["G_STO_YCRAF"].records)
                    df_PRO_STO["Table"] = "Storage"
                # Concatenate all the dataframe together and add the model name
                df_baseline = pd.concat([df_PRO, df_CAP, df_XH2_CAP, df_XH2_FLOW, df_PRO_STO])
                df_baseline["Scenarios"] = 0
                df_scenarios = pd.concat([df_scenarios, df_baseline])
                
            # Add the dataframe to the output dictionary
            dict_results[f"{self.model_names[i]}"] = df_scenarios
        
        self.dict_results = dict_results
        print("Data imported successfully")
        
    def import_input_data(self) -> None:
        
        # Check that input data files have been provided
        if not hasattr(self, "input_files"):
            raise ValueError("The input data files have not been provided when calling the class. Please provide them.")
        
        # Path of the input data files
        baseline_input_data_path = os.path.join(os.path.abspath(self.path), self.input_files[0])
        scenarios_sample_path = os.path.join(os.path.abspath(self.path), self.input_files[1])
        
        # Retrieve the parameters from the input data file ### Hard coded for the moment, could link it to GSA_parameters class
        df = gt.Container(baseline_input_data_path)
        df_CCS = pd.DataFrame(df.data["CCS_CO2CAPTEFF_G"].records)
        df_DE = pd.DataFrame(df.data["DE"].records)
        df_EMIPOL = pd.DataFrame(df.data["EMI_POL"].records)
        df_FUELPRICE = pd.DataFrame(df.data["FUELPRICE"].records)
        df_GDATA_categorical = pd.DataFrame(df.data["GDATA_categorical"].records)
        df_GDATA_numerical = pd.DataFrame(df.data["GDATA_numerical"].records)
        df_HYDROGEN_DH2 = pd.DataFrame(df.data["HYDROGEN_DH2"].records)
        df_SUBTECHGROUPKPOT = pd.DataFrame(df.data["SUBTECHGROUPKPOT"].records)
        df_XH2INVCOST = pd.DataFrame(df.data["XH2INVCOST"].records)
        df_XINVCOST = pd.DataFrame(df.data["XINVCOST"].records)

        baseline_input_data = {'CCS_CO2CAPTEFF_G': df_CCS, 'DE': df_DE, 'EMI_POL': df_EMIPOL, 'FUELPRICE': df_FUELPRICE, 'GDATA_numerical': df_GDATA_numerical, 'GDATA_categorical': df_GDATA_categorical,
                               'HYDROGEN_DH2': df_HYDROGEN_DH2, 'SUBTECHGROUPKPOT': df_SUBTECHGROUPKPOT, 'XH2INVCOST': df_XH2INVCOST, 'XINVCOST': df_XINVCOST}

        # Retrieve the sample data, which contains the values of each parameters in the scenarios
        with open(scenarios_sample_path, 'r') as file:
            sample_raw = file.read()
        df_scenarios_sample = pd.read_csv(StringIO(sample_raw), header=None)
        df_scenarios_sample.columns = Parameters_names
        
        self.baseline_input_data = baseline_input_data
        self.df_scenarios_sample = df_scenarios_sample
        print("Input data imported successfully")    
    
    ### ------------------------------- ###
    ###      1.2 Extract information    ###
    ### ------------------------------- ###
           
    # Extract the results of a specific information, for specific countries, on a specific year
    def get_results(self, model: str, selection: str, Countries: list[str], YEAR: int) -> pd.DataFrame:
        
        # Check that the data has been imported
        if not hasattr(self, "dict_results"):
            raise ValueError("The data has not been imported yet. Please use the import_results() function.")
        # Check that the selection is correct
        if selection not in results_selections:
            raise ValueError("The selection is not correct.")
        
        # identify the table needed
        if selection in ["Elec RE Capacity","Elec PV Capacity", "Elec ONSHORE Capacity", "Elec OFFSHORE Capacity",
                         "H2 Green Capacity", "H2 Blue Capacity", "H2 Import Capacity"] :
            Table = "Generation Capacity"
        if selection in ["Elec RE Production","Elec PV Production", "Elec ONSHORE Production", "Elec OFFSHORE Production",
                         "H2 Green Production", "H2 Blue Production", "H2 Import Production"] :
            Table = "Generation Production"
        if selection == "H2 Storage" :
            Table = "Storage"
        if selection == "H2 Transmission Capacity" :
            Table = "Hydrogen Transmission Capacity"
        if selection == "H2 Transmission Flow" :
            Table = "Hydrogen Transmission Flow"
            
        # Get the data for the correct model, the selected countries, the correct year and the correct table
        df = self.dict_results[model]
        df = df[df["Table"] == Table]
        df = df[df["C"].isin(Countries)]
        df = df[df["Y"] == YEAR]
        
        # Filter and transform the data as needed
        # Commodity filtering
        if selection in ["Elec RE Capacity","Elec PV Capacity", "Elec ONSHORE Capacity", "Elec OFFSHORE Capacity",
                         "Elec RE Production","Elec PV Production", "Elec ONSHORE Production", "Elec OFFSHORE Production"]:
            df = df[df['COMMODITY']=='ELECTRICITY']
        if selection in ["H2 Green Capacity", "H2 Blue Capacity", "H2 Green Production", "H2 Blue Production", "H2 Storage"]:
            df = df[(df['COMMODITY']=='HYDROGEN') & (df['FFF'] != 'IMPORT_H2')]
        elif selection in ["H2 Import Capacity", "H2 Import Production"]:
            df = df[(df['COMMODITY']=='HYDROGEN') & (df['FFF'] == 'IMPORT_H2')]
        
        # Tech type filtering
        if selection in ["Elec RE Capacity","Elec RE Production"]:
            df = df[df['TECH_TYPE'].isin(['WIND-ON', 'WIND-OFF', 'SOLAR-PV'])]
        elif selection in ["Elec PV Capacity", "Elec PV Production"]:
            df = df[(df['TECH_TYPE']=='SOLAR-PV')]
        elif selection in ["Elec ONSHORE Capacity", "Elec ONSHORE Production"]:
            df = df[(df['TECH_TYPE']=='WIND-ON')]
        elif selection in ["Elec OFFSHORE Capacity", "Elec OFFSHORE Production"]:
            df = df[(df['TECH_TYPE']=='WIND-OFF')]
        if selection in ["H2 Green Capacity", "H2 Green Production"]:
            df[(df['TECH_TYPE']=='ELECTROLYZER')]
        elif selection in ["H2 Blue Capacity", "H2 Blue Production"]:
            df = df[(df['G'].str.contains('CCS')) & ((df['TECH_TYPE']=='STEAMREFORMING'))]
        
        # Import and export regions filtering
        if selection in ["H2 Transmission Capacity", "H2 Transmission Flow"]:
            df['CI'] = df['IRRRI'].map(RRR_to_CCC)
            df = df[~df["CI"].isin(Countries)]
        
        if selection in ["Elec RE Capacity","Elec PV Capacity", "Elec ONSHORE Capacity", "Elec OFFSHORE Capacity",
                         "Elec RE Production","Elec PV Production", "Elec ONSHORE Production", "Elec OFFSHORE Production",
                         "H2 Green Capacity", "H2 Blue Capacity", "H2 Green Production", "H2 Blue Production", "H2 Import Capacity", "H2 Import Production", "H2 Storage"]:
            df = df.groupby('Scenarios')['value'].sum().reset_index()
        elif selection in ["H2 Transmission Capacity", "H2 Transmission Flow"]:
            df = df.groupby(['Scenarios', 'CI'])['value'].sum().reset_index()
        
        return df
    
    # Sammple the input data for a specific parameter
    def sample_input_data(self, selection: str, Countries: list[str], YEAR: int) -> pd.DataFrame:
        
        # Check that the data has been imported
        if not hasattr(self, "baseline_input_data"):
            raise ValueError("The input data have not been imported yet. Please use the import_input_data() function.")
        # Check that the selection is correct
        if selection not in input_data_selections:
            raise ValueError("The selection is not correct.")
        
        # Natural gas price 
        if selection == "NATGAS_P":
            df = self.baseline_input_data["FUELPRICE"]
            df = df[(df["FFF"]=="NATGAS") & (df["YYY"]==YEAR)] ### For now the fuel price is the same in all countries for a given year, so we take the first value
        # CO2 tax
        elif selection == "CO2_TAX":
            df = self.baseline_input_data["EMI_POL"]
            df = df[(df["EMIPOLSET"]=="TAX_CO2") & (df["YYY"]==YEAR)] ### For now the co2 tax is the same in all countries for a given year, so we take the first value
        # SMR CCS investment cost
        elif selection == "SMR_CCS_INVC":
            df = self.baseline_input_data["GDATA_numerical"]
            df = df[(df['GGG'].str.contains("GNR_STEAM-REFORMING-CCS")) & (df['GDATASET']=='GDINVCOST0')] 
            
        # KPOT of the subtechnology group
        if selection in ["PV_LIMIT_NORTH","PV_LIMIT_SOUTH","PV_LIMIT_EAST","PV_LIMIT_WEST","ONS_LIMIT_EAST","ONS_LIMIT_WEST","ONS_LIMIT_NORTH","ONS_LIMIT_SOUTH"]:
            Regions = [key for key, value in RRR_to_CCC.items() if value in Countries] # To get the regions associated to the countries
            df = self.baseline_input_data["SUBTECHGROUPKPOT"]
            if selection in ["PV_LIMIT_NORTH","PV_LIMIT_SOUTH","PV_LIMIT_EAST","PV_LIMIT_WEST"]:
                df = df[(df["TECH_GROUP"]=="SOLARPV") & (df['CCCRRRAAA'].isin(Regions))]
            elif selection in ["ONS_LIMIT_EAST","ONS_LIMIT_WEST","ONS_LIMIT_NORTH","ONS_LIMIT_SOUTH"]:
                df = df[(df["TECH_GROUP"]=="WINDTURBINE_ONSHORE") & (df['CCCRRRAAA'].isin(Regions))]
            df_baseline = df['value'].sum()
        else :
            df_baseline = df.iloc[0]['value']
        
        # Sample the data
        df = df_baseline * self.df_scenarios_sample[selection] #Natural gas price all scenarios
        df.index = df.index + 1
        df.loc[0] = df_baseline
        df = df.sort_index().reset_index(drop=True)
        df = df.to_frame(name=selection)
        df.rename(columns={selection: 'value'}, inplace=True)
        df['Scenarios'] = df.index
        
        return df

    ### ------------------------------- ###
    ###      1.3 Plotting functions     ###
    ### ------------------------------- ###
    
    # Plot the violin plot of a specific information, for specific countries, on a specific year
    def violin_plot(self, selection: str, Countries: Union[list[str], dict], YEAR: int,
                    model_filter: list[str] = None, show_baseline = False) -> go.Figure:
        
        # Color of the violin plots
        if selection in ["H2 Green Capacity", "H2 Green Production"]:
            color = 'green'
        elif selection in ["H2 Blue Capacity", "H2 Blue Production"]:
            color = 'blue'
        elif selection in ["H2 Storage", "H2 Import Capacity", "H2 Import Production"]:
            color = 'orange'
            unit = 'TWh'
        elif selection in ["H2 Transmission Capacity"]:
            color = 'red'
        elif selection in ["H2 Transmission Flow"]:
            color = 'red'
        # Unit of the violin plots
        if "Capacity" in selection:
            unit = 'GW'
        elif "Production" in selection or "Flow" in selection:
            unit = 'TWh'
            
        # Create plot
        fig = go.Figure()
        
        # Iterate over the countries if defined as a dictionnary
        if type(Countries) == list :
            if len(Countries) == 0:
                raise ValueError("The list of countries is empty.")
            elif len(Countries) == 1:
                Countries = {Countries[0] : Countries}
            else :
                Countries = {f"{len(Countries)} countries" : Countries}   
        
        for country in Countries.keys():
            # Filter the models we want to plot
            if model_filter is not None:
                for model_name in model_filter:
                    if model_name not in self.model_names:
                        raise ValueError(f"The model {model_name} is not in the list of models.")
                self.model_names = model_filter
                self.nb_models = len(model_filter)
            # Iterate over the model names
            for model in self.model_names:
                # Choose name adequatly
                model_name = model # Name to be put on the graph
                if len(Countries.keys()) != 1:
                    violin_name = f"{model_name} {country}"
                else:
                    violin_name = model_name
                
                try :
                    # Get the data
                    df = self.get_results(model, selection, Countries[country], YEAR)
                    ### For now we don't deal with the case where we look at different countries to import transmission data
                    df = df.groupby('Scenarios')['value'].sum().reset_index()
                    # Extract the baseline value
                    if show_baseline:
                        baseline = df[df['Scenarios'] == 0]['value'].values[0]
                    df = df[df['Scenarios'] != 0]
                    # Plot the data
                    fig.add_trace(go.Violin(y=df['value'], name=violin_name,
                                            box_visible=True, line_color=color))
                    if show_baseline:
                        fig.add_trace(go.Scatter(x=[violin_name], y=[baseline], mode='markers',
                                                marker=dict(color='#3C3D37', size=10), name=violin_name + ' Baseline'))
                except :
                    print(f"No {selection} data for {model_name} - {country}")
                
            
        fig.update_layout(
            title={
                'text': f'Violin Plot: Value Distribution of {selection} ({YEAR})',
                'font': {'size': 16}
            },
            yaxis_title=f"{selection} [{unit}]",
            yaxis_range=[0, None],
            height=600,
            width=200*self.nb_models*len(Countries),
            legend=dict(orientation='h', x=0.5, y=1.04, xanchor='center', yanchor='bottom'),  # Adjust legend position
            margin=dict(t=150, b=50),  # Adjust top and bottom margins to accommodate title and legend
            plot_bgcolor='white',  # Set background color to white
            xaxis=dict(
                tickfont=dict(size=15)  # Set font size for x-axis title
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='black',
                gridwidth=1,
                tickfont=dict(size=16),
                title_font=dict(size=16)
            )
        )
            
        return fig
    
    def correlation_plot(self, model: str, selection: list[str], Countries: list[str], YEAR: int, color_selection: str = None,
                         show_baseline = False, show_regression = True) -> go.Figure:
        
        # Verify that the selection list has two elements
        if len(selection) != 2:
            raise ValueError("The selection list should contain two elements. One for the x-axis and one for the y-axis.")
        
        # Get the data needed, find the unit and calculate regression
        x_selection = selection[0]
        if x_selection in input_data_selections:
            df_x = self.sample_input_data(x_selection, Countries, YEAR)
            x_unit = ""
        elif x_selection in results_selections:
            df_x = self.get_results(model, x_selection, Countries, YEAR)
            if "Capacity" in x_selection:
                x_unit = 'GW'
            elif "Production" in x_selection or "Flow" in x_selection:
                x_unit = 'TWh'
        
        y_selection = selection[1]
        if y_selection in input_data_selections:
            df_y = self.sample_input_data(y_selection, Countries, YEAR)
            y_unit = ""
        elif y_selection in results_selections:
            df_y = self.get_results(model, y_selection, Countries, YEAR)
            if "Capacity" in y_selection:
                y_unit = 'GW'
            elif "Production" in y_selection or "Flow" in y_selection:
                y_unit = 'TWh'
                
        if color_selection is not None:
            if color_selection in input_data_selections:
                df_color = self.sample_input_data(color_selection, Countries, YEAR)
                color_unit = ""
            elif color_selection in results_selections:
                df_color = self.get_results(model, color_selection, Countries, YEAR)
                if "Capacity" in color_selection:
                    color_unit = 'GW'
                elif "Production" in color_selection or "Flow" in color_selection:
                    color_unit = 'TWh'
        
        # Extract the baseline value
        if show_baseline:
            baseline_x = df_x[df_x['Scenarios'] == 0]['value'].values[0]
            baseline_y = df_y[df_y['Scenarios'] == 0]['value'].values[0]
        df_x = df_x[df_x['Scenarios'] != 0]
        df_y = df_y[df_y['Scenarios'] != 0]
        
        # Do the linear regression if needed
        if show_regression:
            X = df_x['value'].values.reshape(-1, 1)
            y = df_y['value'].values
            reg = LinearRegression().fit(X, y)
            y_pred = reg.predict(X)
            r2 = r2_score(y, y_pred)
        
        # Create plot
        fig = go.Figure()
        
        # Plot the data
        if color_selection is None:
            fig.add_trace(go.Scatter(x=df_x['value'], y=df_y['value'], mode='markers',
                                    showlegend=False))
        else :
            fig.add_trace(go.Scatter(x=df_x['value'], y=df_y['value'], mode='markers',
                                    marker=dict(color=df_color['value'], colorscale='Portland', size=8, showscale=True,
                                                colorbar=dict(title=f"{color_selection} [{color_unit}]", len=0.6)),
                                    showlegend=False))
        
        if show_regression:
            fig.add_trace(go.Scatter(x=df_x['value'], y=y_pred, mode='lines', name='Trendline', line=dict(color='red'),showlegend=False))
            
        if show_baseline:
            fig.add_trace(go.Scatter(x=[baseline_x], y=[baseline_y], mode='markers',
                                    marker=dict(color='#3C3D37', size=10), name='Baseline'))
        
        fig.update_layout(
            title=f'Correlation between {x_selection} and {y_selection} ({YEAR})',
            xaxis_title=f"{x_selection} [{x_unit}]",
            yaxis_title=f"{y_selection} [{y_unit}]",
            width=800,
            height=600,
        )
        
        return fig



#%% ------------------------------- ###