import Functions_analysis as FA
import os

YEAR = '2030'

Nordics = ['DENMARK', 'NORWAY', 'SWEDEN', 'FINLAND']

MainResults_path = f"../DATA/MainResults_{YEAR}.gdx"
MainResults_path = os.path.abspath(MainResults_path)
MonteCarlo_path = f"../DATA/MonteCarlo_DK_{YEAR}.gdx"
MonteCarlo_path = os.path.abspath(MonteCarlo_path)

def Main_Analysis(MainResults_path, MonteCarlo_path, YEAR, Countries_from, Countries_to) :
    # Import results of the Main Result file
    df_PRO_BASE, df_CAP_BASE, df_XH2_CAP_BASE, df_XH2_FLOW_BASE = FA.Import_MainResults(MainResults_path)
    
    # Export all hydrogen results
    df_H2_CAP_BASE, df_H2_CAP_GREEN_BASE, df_H2_CAP_GREEN_tot_BASE, df_H2_CAP_BLUE_BASE, df_H2_CAP_BLUE_tot_BASE, df_H2_CAP_STO_BASE, df_H2_CAP_STO_tot_BASE = FA.H2_CAP(df_CAP_BASE, Countries_from, YEAR)
    df_H2_PRO_BASE, df_H2_PRO_GREEN_BASE, df_H2_PRO_GREEN_tot_BASE, df_H2_PRO_BLUE_BASE, df_H2_PRO_BLUE_tot_BASE, df_H2_PRO_STO_BASE, df_H2_PRO_STO_tot_BASE = FA.H2_PRO(df_PRO_BASE, df_CAP_BASE, Countries_from, YEAR)
    dict_df_XH2_CAP_TO_BASE = {}
    dict_df_XH2_FLOW_TO_BASE = {}
    for country in Countries_to :
        dict_df_XH2_CAP_TO_BASE[country] = FA.XH2(df_XH2_CAP_BASE, Countries_from, [country], YEAR)
        dict_df_XH2_FLOW_TO_BASE[country] = FA.XH2(df_XH2_FLOW_BASE, Countries_from, [country], YEAR)
        
    # Import resuls of the MonteCarlo file
    df_PRO_scen, df_CAP_scen, df_XH2_CAP_scen, df_XH2_FLOW_scen, scen = FA.Import_MonteCarlo(MonteCarlo_path)
    
    # Export all hydrogen results
    df_H2_CAP_scen, df_H2_CAP_GREEN_scen, df_H2_CAP_BLUE_scen, df_H2_CAP_STO_scen = FA.H2_CAP_scen(df_CAP_scen, scen, Countries_from, YEAR)
    df_H2_PRO_scen, df_H2_PRO_GREEN_scen, df_H2_PRO_BLUE_scen, df_H2_PRO_STO_scen = FA.H2_PRO_scen(df_PRO_scen, df_CAP_scen, scen, Countries_from, YEAR)
    dict_df_XH2_CAP_TO_scen = {}
    dict_df_XH2_FLOW_TO_scen = {}
    for country in Countries_to :
        df_XH2_CAP_tot_scen, dict_df_XH2_CAP_TO_scen[country] = FA.XH2_scen(df_XH2_CAP_scen, scen, Countries_from, [country], YEAR)
        df_XH2_FLOW_tot_scen, dict_df_XH2_FLOW_TO_scen[country] = FA.XH2_scen(df_XH2_FLOW_scen, scen, Countries_from, [country], YEAR)
    
    ECDF_Hist_PRO = FA.ECDF_Hist_PRO(df_H2_PRO_GREEN_scen, df_H2_PRO_BLUE_scen, df_H2_PRO_STO_scen, df_H2_PRO_GREEN_tot_BASE, df_H2_PRO_BLUE_tot_BASE, df_H2_PRO_STO_tot_BASE, ' '.join(Countries_from), YEAR)
    ECDF_Hist_CAP = FA.ECDF_Hist_CAP(df_H2_CAP_GREEN_scen, df_H2_CAP_BLUE_scen, df_H2_CAP_GREEN_tot_BASE, df_H2_CAP_BLUE_tot_BASE, ' '.join(Countries_from), YEAR)
    Violin_PRO = FA.Violin_PRO(df_H2_PRO_GREEN_scen, df_H2_PRO_BLUE_scen, df_H2_PRO_STO_scen, df_H2_PRO_GREEN_tot_BASE, df_H2_PRO_BLUE_tot_BASE, df_H2_PRO_STO_tot_BASE, ' '.join(Countries_from), YEAR)
    Violin_CAP = FA.Violin_CAP(df_H2_CAP_GREEN_scen, df_H2_CAP_BLUE_scen, df_H2_CAP_STO_scen, df_H2_CAP_GREEN_tot_BASE, df_H2_CAP_BLUE_tot_BASE, df_H2_CAP_STO_tot_BASE, ' '.join(Countries_from), YEAR)
    BoxPlot_CAP = FA.BoxPlot_CAP(dict_df_XH2_CAP_TO_BASE, dict_df_XH2_CAP_TO_scen, ' '.join(Countries_from), YEAR)
    BoxPlot_FLOW = FA.BoxPlot_FLOW(dict_df_XH2_FLOW_TO_BASE, dict_df_XH2_FLOW_TO_scen, ' '.join(Countries_from), YEAR)
    
    ECDF_Hist_PRO.write_image(f"Plots/ECDF_Hist_PRO_{' '.join(Countries_from)}_{YEAR}.png", scale=2)
    ECDF_Hist_CAP.write_image(f"Plots/ECDF_Hist_CAP_{' '.join(Countries_from)}_{YEAR}.png", scale=2)
    Violin_PRO.write_image(f"Plots/Violin_PRO_{' '.join(Countries_from)}_{YEAR}.png", scale=2)
    Violin_CAP.write_image(f"Plots/Violin_CAP_{' '.join(Countries_from)}_{YEAR}.png", scale=2)
    BoxPlot_CAP.write_image(f"Plots/BoxPlot_CAP_{' '.join(Countries_from)}_{YEAR}.png", scale=2)
    BoxPlot_FLOW.write_image(f"Plots/BoxPlot_FLOW_{' '.join(Countries_from)}_{YEAR}.png", scale=2)
    

Main_Analysis(MainResults_path, MonteCarlo_path, YEAR, ['DENMARK'], ['SWEDEN', 'GERMANY', 'NETHERLANDS'])