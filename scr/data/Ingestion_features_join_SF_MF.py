#!/usr/bin/env python
# coding: utf-8

# From ph2_approach_a-Copy5_v2_For_GitHub.ipynb:  Master, split into the following 6 notebooks
# 
# ph2_approach_a_SF_MF_data_ingestion_join.ipynb:  √ Data ingestion, prep, feature engineering & selection, joining
# ph2_approach_a_SF_MF_EDA.ipynb: (need to make)
# 
# ph2_approach_a_SF_model_table.ipynb: 
# ph2_approach_a_MF_model_table.ipynb:
# 
# ph2_approach_a_SF_visualization_map.ipynb:  
# ph2_approach_a_MF_visualization_map.ipynb:  
# 
#    
#     Part 1: Prepare & Join the parcel, shapefiles, fire incident & google ratings data
#     Part 2: The Model & Fire Risk Prediction Table
#     Part 3: Gathering data for visualization in Tableau
#     Part 4: Splitting input csvs to share in GitHub
#     References & Links
# 
# 
# ###  Code Contributors: Margaret Catherman, Yvonne Zhang,  Sanal Shivaprasad & Chiyoung Lee for ATLytiCS & DeKalb County Fire Rescue (DCFR)
#     
# 
# ## Required Input csvs:
# 
#     1. Shapefiles: see next chunk
#     2. Parcel data: 
#         a. 'fire-comm3_owndat.csv' OR 
#         b. 'fire-comm3_owndat_00.csv', & fire-comm3_owndat_01.csv'
#     3. Fire data: 
#         a. 'ATLyticsFiredatacombined.csv', & 'ATLyticsFiredatacombined_2.csv' OR 
#         b. 'ATLyticsFiredatacombined_00.csv' to 'ATLyticsFiredatacombined_13.csv' (14 csvs)
#     4. Google Ratings API:'Motels Near Dekalb Atlanta version 1.csv', 'Apts Near Dekalb Atlanta version 6.csv'
#     
# ## Generated csvs:
# 
#     1. Fire Prediction Table: 'fire_prediction_table_1_client.csv'
#     3. Data rejoined with Table for viz, 'all_viz_5R2_v2.csv'
#     
# 

# # Prepare the environment
# 
# ## Shapefiles: Full set  required on OS.
# 

# In[1]:


#A. DeKalb County Parcel 
#FILES from DCFR: Updated 11/2022

#Tax_Parcels_Nov2022.cpg;
#Tax_Parcels_Nov2022.dbf; # OR Tax_Parcels_Nov2022.dbf_00.csv to Tax_Parcels_Nov2022.dbf_39.csv
#Tax_Parcels_Nov2022.sbn;
#Tax_Parcels_Nov2022.sbx;
#Tax_Parcels_Nov2022.shx;
#Tax_Parcels_Nov2022.shp; #OR Tax_Parcels_Nov2022.shp_00 to Tax_Parcels_Nov2022.shp_09
#Tax_Parcels_Nov2022.shp.xml; 
#Tax_Parcels_Nov2022.prj;

#C. DeKalb County Station & Battalion Boundaries
#From DCFR 2022

#Station_Territory_Boundaries.cpg
#Station_Territory_Boundaries.dbf
#Station_Territory_Boundaries.prj
#Station_Territory_Boundaries.sbn
#Station_Territory_Boundaries.sbx
#Station_Territory_Boundaries.shp.xml
#Station_Territory_Boundaries.shp
#Station_Territory_Boundaries.shx


#B. Census 2021
#Downloaded GA from: https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2022&layergroup=Blocks+%282020%29
#Possible API option.

#tl_2022_13_tabblock20.shp.ea.iso.xml
#tl_2022_13_tabblock20.shp.iso.xml
#tl_2022_13_tabblock20.dbf
#tl_2022_13_tabblock20.shp
#tl_2022_13_tabblock20.shx
#tl_2022_13_tabblock20.cpg
#tl_2022_13_tabblock20.prj



# In[2]:


import os
import sys
import pandas as pd
import numpy as np
import requests
import math
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point
import ast
import re
#from dataprep.clean import clean_address
import seaborn as sns
from datetime import datetime, date
from sklearn import metrics, ensemble, preprocessing
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix, brier_score_loss,  f1_score, log_loss,
                             precision_score, recall_score,roc_auc_score, accuracy_score,confusion_matrix, 
                             classification_report,cohen_kappa_score, make_scorer)
from sklearn.feature_selection import mutual_info_classif   
from scipy import stats
from sklearn import calibration
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
import folium #for muti-maps
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from lazypredict.Supervised import LazyClassifier

import random
import time
import censusgeocode as cg  #MODEL_CONFIG
import censusdata
import csv


pd.options.display.max_columns = None
pd.options.display.max_rows = None


# In[3]:


import pkg_resources
import types
def get_imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            # Split ensures you get root package, 
            # not just imported function
            name = val.__name__.split(".")[0]

        elif isinstance(val, type):
            name = val.__module__.split(".")[0]
            
        # Some packages are weird and have different
        
        # imported names vs. system/pip names. Unfortunately,
        # there is no systematic way to get pip names from
        # a package's imported name. You'll have to add
        # exceptions to this list manually!
        poorly_named_packages = {
            "PIL": "Pillow",
            "sklearn": "scikit-learn"
        }
        if name in poorly_named_packages.keys():
            name = poorly_named_packages[name]
            
        yield name
imports = list(set(get_imports()))

# The only way I found to get the version of the root package
# from only the name of the package is to cross-check the names 

# of installed packages vs. imported packages
requirements = []
for m in pkg_resources.working_set:
    if m.project_name in imports and m.project_name!="pip":
        requirements.append((m.project_name, m.version))

for r in requirements:
    print("{}=={}".format(*r))


# In[3]:


# See https://github.com/atlytics/FireExplorer_Learner/blob/main/models/learner_preprocessor.py 
#for ideas


# In[4]:


def unique (df):
    dfun = df.nunique(axis=0)
    dfun_df = pd.DataFrame(dfun) 
    column_names=["Distinct"]
    dfun_df.columns = column_names
    dfun_df_sorted = dfun_df.sort_values('Distinct', ascending=False)
    return (dfun_df_sorted)

def check_missing_values (df):
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'percent_missing': percent_missing})
    missing_value_df.sort_values('percent_missing', inplace=True)
    if missing_value_df['percent_missing'].any():
        return missing_value_df
    else: 
        return "No missing values"


def not_yet_joined (df1, joined_df, variabe_df1):  
    need_to_join= df1[~df1[variabe_df1].isin(joined_df[variabe_df1])]
    #need_to_join.reset_index(drop=True, inplace=True)
    return need_to_join

 
filepath = "/Users/margaretcatherman"
epsg = "4326"  #EPSG correct


# # Part 1: Prepare 
# 
# ### Stations & Census

# In[35]:


#filepath = "/Users/margaretcatherman"
def station_shp_converter(filepath, epsg):
    station_shp = gpd.read_file(filepath + '/Station_Territory_Boundaries.shp')
    station_shp = station_shp[station_shp.geometry.notnull()]
    station_shp_geo = station_shp.to_crs(epsg=epsg)  #epsg = "4326"  #EPSG correct
    station_shp = station_shp_geo.loc[:, ('Station_Na','ID1','geometry')] 
    station_shp= station_shp.rename(columns={'Station_Na': 'Station', 'ID1':'Battalion', 'geometry': 'geometry_stion'})
    station_shp['Station'] = station_shp.Station.astype('category')
    return station_shp


# In[36]:


stations = station_shp_converter(filepath, epsg)
#stations.info() #'Addresses_of_Smoke_Alarms.csv'
#stations.head(2)


# In[8]:


#>>>> Fiter to omit AT
def tiger_shp_stations_converter(filepath, epsg, stations):
    tiger_shp = gpd.read_file(filepath + '/tl_2022_13_tabblock20.shp')
    tiger_shp = tiger_shp[tiger_shp.geometry.notnull()]
    tiger_shp_geo = tiger_shp.to_crs(epsg=epsg)
    tiger_shp_geo = tiger_shp_geo[tiger_shp_geo['COUNTYFP20']=='089']
    new = tiger_shp_geo['GEOID20'].str.split("130890", n = 1, expand = True)
    tiger_shp_geo["BG_ID_9"]= new[1]
    tiger_shp_geo_2 = tiger_shp_geo[tiger_shp_geo['BG_ID_9'].notna()]
    tiger_shp_geo_2["BG_ID_6"] =tiger_shp_geo_2["BG_ID_9"].str[:-3]
    tiger_shp_geo_2['geometry_9'] = tiger_shp_geo_2['geometry']
    df = tiger_shp_geo_2.loc[:, ('BG_ID_6','geometry')]
    dfa = df.dissolve(by='BG_ID_6')
    dfa = dfa.rename(columns={'geometry': 'geometry_6'})
    dfa.reset_index(drop=False, inplace= True)
    tgr = pd.merge(tiger_shp_geo_2,dfa,left_on = ['BG_ID_6'],right_on = ['BG_ID_6'],how= 'left') 
    #stations = station_shp_converter(filepath, epsg)
    stations['geometry'] = stations['geometry_stion']
    tiger_2 = tgr.sjoin(stations, how="inner", predicate='within')
    stations.drop('geometry',axis=1, inplace=True)
    tiger_2.drop_duplicates(inplace=True, ignore_index=True) 
    tiger_2 = tiger_2.loc[:, ('BG_ID_6','BG_ID_9','ALAND20', 'AWATER20', 'POP20', 'HOUSING20', 'Station','Battalion', 'geometry_6','geometry_9', 'geometry_stion')] 
    tiger_2.reset_index(drop= True, inplace=True)
    tiger_2 = tiger_2.add_suffix('_tg')
    cols = ['Station_tg','Battalion_tg', 'geometry_stion_tg']
    tiger =  tiger_2.drop(cols,axis=1) 
    return tiger, tiger_2  



# In[9]:


tiger, tiger_2, = tiger_shp_stations_converter(filepath, epsg,stations)
#tiger_2.info()
#tiger_2.head(2)


# In[10]:


#tgr_2 = tiger_2.copy()
#tgr_2['geometry'] = tgr_2['geometry_6_tg']  
#tgr_2.plot()


# # Dekalb Shapefile: Prepare for join 

# In[11]:


#2
def parcel_shp_converter(filepath, epsg):
    df_shp = gpd.read_file(filepath + '/Tax_Parcels_Nov2022.shp')
    df_shp = df_shp[df_shp.geometry.notnull()]
    df_shp_geo = df_shp.to_crs(epsg=epsg)
    df_shp_geo['centerpoint'] = df_shp_geo['geometry'].centroid
    df_shp_geo["x"] = df_shp_geo.geometry.centroid.x
    df_shp_geo["y"] = df_shp_geo.geometry.centroid.y
    df_shp_geo = df_shp_geo.loc[:, ('PARCELID', 'geometry','centerpoint', 'x', 'y')]  #PSTLADDRES, FULL_STREET
    df_shp_geo = df_shp_geo[df_shp_geo["PARCELID"].str.contains("<New parcel>|None") == False] 
    df_shp_geo.reset_index(drop= True, inplace=True) 
    df_shp_geo['geometry_polygon'] = df_shp_geo['geometry']  
    df_shp_geo = df_shp_geo.add_suffix('_sh')  
    return df_shp_geo
    


# In[12]:


df_shp_geo = parcel_shp_converter(filepath, epsg)  #dekalb_crs_3
#df_shp_geo.info()  
#df_shp_geo.head(2)  #tax_2


# # Parcel Data: Prepare for join
# ### Very important to group by parcel id & get sum/mean of values BEFORE dropin parcel id.
# ### Join Shapefiles + Parcel data on Parcel id, SF & MF

# In[13]:


def shape_parcel_geoid_processor_both (df_shp_geo, parcel_raw, tiger):
    df_shp_geo=df_shp_geo.astype({'PARCELID_sh':str})
    parcel_raw=parcel_raw.astype({'parid_tp':str})
    parcel_df = pd.merge(df_shp_geo,parcel_raw,left_on = ['PARCELID_sh'],right_on = ['parid_tp'],how= 'inner') 
    parcel_df.reset_index(drop = True, inplace=True)
    parcel_df = parcel_df.loc[:, parcel_df.isnull().mean() < .02]
    parcel_df.drop_duplicates(keep=False, inplace=True) #drop any duplicate rows
    parcel_df['centerpoint_sh_2'] = parcel_df['centerpoint_sh']
    parcel_df = parcel_df.rename(columns={'centerpoint_sh_2':'geometry'})  
    parcel_geo = gpd.GeoDataFrame(parcel_df)   
    tiger['geometry_9_tg_2'] = tiger['geometry_9_tg']
    tiger = tiger.rename(columns={'geometry_9_tg_2':'geometry'})
    #Sjoin
    parcel_geodf = parcel_geo.sjoin(tiger, how="inner", predicate='within')
    parcel_geodf.reset_index(drop = True, inplace=True)
    parcel_geodf.drop_duplicates(keep=False, inplace=True) #drop any duplicate rows
    cols = [col for col in parcel_geodf.columns if 'index' in col]
    parcel_geodf.drop(cols,axis=1, inplace=True)
    parcel_geodf = parcel_geodf.sort_index(axis=1)#sorts alphabeticaly by co
    return parcel_geodf #parcel_geodf_sf parcel_geodf_mf


# In[14]:


def parcel_csv_processor_sf(filepath, df_shp_geo, tiger):
    """
    In numerous previous similar analysis of fire ris probabiity usin
    """    
    parcel_raw_sf = gpd.read_file(filepath + '/TAX_ASSESSOR_DATA.csv')
    parcel_raw_sf = parcel_raw_sf.loc[:, parcel_raw_sf.isnull().mean() < .02]
    parcel_raw_sf.drop_duplicates(keep=False, inplace=True)
    parcel_raw_sf=parcel_raw_sf.astype({'parid':str})
    parcel_raw_sf = parcel_raw_sf[parcel_raw_sf["parid"].str.contains("<New parcel>|None") == False]   
    co = ['2019_landval','2015_landval', '2019_bldgval', '2015_bldgval', '2019_totval', '2015_totval'] 
    parcel_raw_sf[co] = parcel_raw_sf[co].apply(pd.to_numeric, errors='coerce')
    parcel_raw_sf['landval_change'] = parcel_raw_sf['2019_landval'] - parcel_raw_sf['2015_landval']
    parcel_raw_sf['bldgval_change'] = parcel_raw_sf['2019_bldgval'] - parcel_raw_sf['2015_bldgval']
    parcel_raw_sf['totval_change'] = parcel_raw_sf['2019_totval'] - parcel_raw_sf['2015_totval']
    encoder = OrdinalEncoder()
    parcel_raw_sf['grade'] = encoder.fit_transform(np.asarray(parcel_raw_sf['grade']).reshape(-1,1))
    parcel_raw_sf = parcel_raw_sf.loc[:, ('_id','parid', 'class', 'extwall', 'cdu', 'heat','nbhd', 'style',
       'luc', 'sfla', 'grade', 'yrblt', 'rmbed', 'acres',
       'stories', 'area', 'fp_wood_burning', '2019_landval',
       '2019_bldgval', '2019_totval', 'landval_change',
       'bldgval_change', 'totval_change', 'card', 'fixaddl')]
    cat_vars = ['class', 'luc', 'extwall', 'cdu', 'heat']
    df = parcel_raw_sf[cat_vars].apply(lambda x: x.fillna(x.mode()[0]).astype('category').cat.codes)
    parcel_raw_sf.drop(cat_vars,axis=1, inplace=True)
    parcel_raw_sf =pd.concat([df, parcel_raw_sf],axis=1) 
    df_5 = parcel_raw_sf.groupby("parid").mean()
    df_5_med = df_5.add_suffix('_mean')
    df_6 = parcel_raw_sf.groupby('parid').sum()   
    df_6_sum = df_6.add_suffix('_sum')
    parcel_raw_sf =pd.concat([df_6_sum, df_5_med],axis=1)
    parcel_raw_sf.reset_index(inplace=True) 
    parcel_raw_sf = parcel_raw_sf.sort_index(axis=1)#sorts alphabeticaly by co
    parcel_raw_sf = parcel_raw_sf.add_suffix('_tp') 
    parcel_geodf_sf = shape_parcel_geoid_processor_both (df_shp_geo, parcel_raw_sf, tiger)
    return parcel_geodf_sf  #parcel_df


# In[15]:


parcel_geodf_sf = parcel_csv_processor_sf(filepath, df_shp_geo, tiger)
#parcel_geodf_sf.head(2)


# In[16]:


parcel_raw_mf = pd.read_csv('fire-comm3_owndat.csv', low_memory=False) #from DF 11/21/22
def parcel_csv_processor_mf(parcel_raw_mf, df_shp_geo, tiger):
    #parcel_raw_mf = gpd.read_file(filepath + '/fire-comm3_owndat.csv')
    parcel_raw_mf.drop_duplicates(keep=False, inplace=True)
    parcel_raw_mf = parcel_raw_mf.loc[:, ('ADJRCN', 'ADRDIR_1', 'ADRNO_1', 'ADRSTR_1', 'ADRSUF2_1', 'ADRSUF_1',
           'AIR', 'APRTOT', 'AREA', 'AREASUM', 'BASERATE', 'BUILDING', 'CALCACRES',
           'CITYNAME_1', 'CLASS_1', 'CONSTR', 'CUBICFT', 'DEPR', 'FEATVAL',
           'FLRFROM', 'FUNCTUTIL', 'HEAT', 'HEATRATE', 'IASW_ID_1',
           'INCUSE', 'LINEVAL', 'LLINE', 'LUC', 'MSCLASS', 'MSHEAT',
           'MSHEATPRICE', 'MSRANK', 'MSSECT', 'NBHD', 'OCCUPANCY', 'OFCARD',
            'PARID', 'PERIM', 'PHYCOND', 'PRICE', 'RATE',
           'STATECODE_1', 'STATUS_2', 'STORIES', 'STORIES (FLRTO)', 'USETYPE',
           'YR BUILT', 'ZIP1_1', 'ZIP2_1')]
    parcel_raw_mf[['ADRDIR_1','ADRSUF_1', 'ADRSUF2_1']] = parcel_raw_mf[['ADRDIR_1','ADRSUF_1', 'ADRSUF2_1']].fillna('') 
    parcel_raw_mf['Address'] = parcel_raw_mf['ADRNO_1'].astype(str) + " " + parcel_raw_mf['ADRDIR_1'].astype(str) + " " + parcel_raw_mf['ADRSTR_1'].astype(str) + " " + parcel_raw_mf['ADRSUF_1'].astype(str) + " " + parcel_raw_mf['ADRSUF2_1'].astype(str)+ "., " + parcel_raw_mf['CITYNAME_1'].astype(str) + ", " + "GA" + " " + parcel_raw_mf['ZIP1_1'].astype(str) 
    parcel_raw_mf['STAddress'] = parcel_raw_mf['ADRNO_1'].astype(str) + " " + parcel_raw_mf['ADRSTR_1'].astype(str)     #+ " " + tax['ADRSUF_1'].astype(str) + " " + tax['ADRSUF2_1'].astype(str)+ "., " + tax['CITYNAME_1'].astype(str) + ", " + "GA" + " " + tax['ZIP1_1'].astype(str) 
    cols_2 = ["ADRNO_1", "ADRDIR_1", "ADRSTR_1", "ADRSUF_1", "ADRSUF2_1", "CITYNAME_1", "ZIP2_1"]      
    parcel_raw_mf.drop(cols_2,axis=1, inplace=True)
    parcel_raw_mf.Address = parcel_raw_mf.Address.str.title() #convert to upper case, first word
    parcel_raw_mf.STAddress = parcel_raw_mf.STAddress.str.title() #convert to upper case, first word
    parcel_raw_mf = parcel_raw_mf.rename(columns={'PARID': 'parid'})
    parcel_raw_mf.reset_index(drop = True, inplace=True)
    parcel_raw_mf = parcel_raw_mf.add_suffix('_tp') 
    parcel_raw_mf = parcel_raw_mf.sort_index(axis=1)#sorts alphabeticaly by co
    parcel_raw_multi = parcel_raw_mf.copy()
    parcel_geodf_mf = shape_parcel_geoid_processor_both (df_shp_geo, parcel_raw_multi, tiger)
    return parcel_geodf_mf  
  


# In[17]:


parcel_geodf_mf = parcel_csv_processor_mf(parcel_raw_mf, df_shp_geo, tiger)
#parcel_geodf_mf.head(2)


# In[18]:


parcel_geodf_mf.shape


# In[19]:


#pred_mf_stion.head(2)  #Latitude_adr_mf:  33.81;   Longitude_adr_mf: -84.20
parcel_geodf_mf.head(2)  #  x_sh: -84.27;   y_sh: 33.93


# # Fire Incident Data: Feature Engineering & Selection for Join
# 

# In[20]:


def nfirs_processor_geodf_both(filepath, epsg):
    """
    NFIRS data: get latest data set from Dung Nguyen with geocode for each fire incident
    https://drive.google.com/file/d/1PTwMIHvV-GB9_eun7wkO5rlUhXNIbDzi/view
    filter callType column, **only keep entries which start with ‘1’ (e.g. 100, 103, 150)  keep these rows
    **MC: Keep all call types and use as features to predict "Had Fire" (0,1)
    filter Basic Property Use (FD1.46) column, only keep:
    A. SF: ‘1 or 2 family dwelling’ and ‘Multifamily dwelling’
    B. MF: half of the rows have latitude and longitude flipped
    drop rows without longitude and latitude  #don't need to do this
    cleaned data set is saved in clean_data_3 folder, named as ATLyticsFiredata_cleaned.csv
    """
    fire1_raw = gpd.read_file(filepath + '/ATLyticsFiredatacombined.csv', encoding= 'unicode_escape', low_memory=False) 
    fire2_raw = gpd.read_file(filepath + '/ATLyticsFiredatacombined_2.csv', encoding= 'unicode_escape', low_memory=False)
    fire_rawS = pd.concat([fire1_raw,fire2_raw])
    fire_0 = fire_rawS.drop_duplicates() 
    fire = fire_0.loc[:, ('IncidentDate', 'IncidentID', 'CallType', 'DeKalbOrNot','Basic Property Use (FD1.46)', 
                          'Basic Property Use Code (FD1.46)', 'Basic Incident Geocoded Latitude', 'Basic Incident Geocoded Longitude','DISTRICTID',
                          'Basic Incident Street Number (FD1.10)', 'Basic Incident Street Name (FD1.12)')]
    ### Filter
    fire['FireIndicator']=fire['CallType'].apply(lambda x: 1 if str(x).startswith('1') else 0)
    fire['4Code']=fire['Basic Property Use Code (FD1.46)'].apply(lambda x: str(x)[0])
    fire_2=fire[(fire['4Code']=='4') & (fire['DeKalbOrNot']=='DeKalb')]   #&(fire['Basic Property Use (FD1.46)']!='1 or 2 family dwelling')]  #)&(fire['FireIndicator']==1)]
    #get rid of NA, so it won't appear in constructed "Address"
    fire_2[['Basic Incident Street Number (FD1.10)', 'Basic Incident Street Name (FD1.12)']] = fire_2[['Basic Incident Street Number (FD1.10)',
             'Basic Incident Street Name (FD1.12)']].fillna('') 
    fire_2['STAddress']= fire_2['Basic Incident Street Number (FD1.10)'].fillna('').map(str)+ ' ' + fire_2['Basic Incident Street Name (FD1.12)'].fillna('').map(str)        #+ ' ' + fire_2['Basic Incident Street Type (FD1.13)'].fillna('').map(str)+ ' ' + fire_2['Basic Incident Street Suffix (FD1.14)'].fillna('').map(str)+ '., ' + fire_2['Basic Incident City Name (FD1.16)'].fillna('').map(str)+ ', ' + fire_2['Basic Incident State (FD1.18)'].fillna('').map(str) + ' ' + fire_2['Basic Incident Postal Code (FD1.19)'].fillna('').map(str)
    fire_2.STAddress = fire_2.STAddress.str.title() #convert to upper case, first word
    fire_2['Longitude'] = fire_2.loc[:, 'Basic Incident Geocoded Longitude'].fillna('0')
    fire_2['Latitude'] = fire_2.loc[:, 'Basic Incident Geocoded Latitude'].fillna('0')
    fire_2['Longitude'] = pd.to_numeric(fire_2['Longitude'],errors='coerce')
    fire_2['Latitude'] = pd.to_numeric(fire_2['Latitude'],errors='coerce')
    # Create IncidentID_date & IncidentID_date_dup_count
    fire_2["IncidentID_date"] = fire_2["IncidentID"].astype(str) + fire_2["IncidentDate"].astype(str)
    #Call type
    fire_2['CallType_2'] = fire_2['CallType']
    fire_2['CallType'] = fire_2['CallType'].astype(str).fillna('0')
    fire_2['CallType_Catagory'] = fire_2['CallType'].str[:1] #keep only first v in str, starting from ledt
    fire_2['CallType_Catagory'] = fire_2['CallType_Catagory'].astype(str) + '00'
    values = ['N00', 'U00']
    fire_2 = fire_2[~fire_2['CallType_Catagory'].isin(values)]
    ct_dummies=pd.get_dummies(fire_2['CallType_Catagory'],prefix='Call_Cat')
    fire_2c=pd.concat([fire_2,ct_dummies],axis=1)
    fire_2_clean_2_copy = fire_2c.copy()
    #Get count of IncidentID_date
    c = ['IncidentID_date']
    fire_2a = fire_2_clean_2_copy[fire_2_clean_2_copy.duplicated(c)].groupby(c).size().reset_index(name='IncidentID_date_dup_count')
    #Join w/ existing df
    fire_3_raw_s = pd.merge(fire_2_clean_2_copy, fire_2a, left_on =  ['IncidentID_date'],right_on = ['IncidentID_date'],how= 'left') 
    fire_3 = fire_3_raw_s.drop_duplicates()
    fire_3['IncidentID_date_dup_count'] = fire_3['IncidentID_date_dup_count'].fillna(0)
    #drop duplicates IncidentID_date or spread
    #cols = ['Basic Incident Street Number (FD1.10)','Basic Incident Street Name (FD1.12)','DeKalbOrNot','Basic Incident Geocoded Latitude', 'Basic Incident Geocoded Longitude']             
    cols = ['DeKalbOrNot','Basic Incident Geocoded Latitude', 'Basic Incident Geocoded Longitude']             
    
    fire_3.drop(cols,axis=1, inplace=True) 
    #Must sort this way before dropping dupes, to keep all 'had fire' '1's in (0,1).  
    fire_3.sort_values(['IncidentID_date', 'CallType_Catagory'], ascending=[True, True], inplace=True)
    fire_3.drop_duplicates(subset=['IncidentID_date'],inplace=True,ignore_index=True)
    #activate "date"
    fire_3['IncidentDate'] = pd.to_datetime(fire_3['IncidentDate'])
    fire_3 = fire_3.add_suffix('_fi') 
    # GeoDataFrame
    nfirs_geodf_both = gpd.GeoDataFrame(fire_3, geometry=gpd.points_from_xy(fire_3.Longitude_fi, fire_3.Latitude_fi))
    nfirs_geodf_both = nfirs_geodf_both.set_crs(epsg=epsg)
    nfirs_geodf_both = nfirs_geodf_both.sort_index(axis=1)#sorts alphabeticaly by co
    #return nfirs_geodf_both
    nfirs_geodf_sf = nfirs_geodf_both[nfirs_geodf_both["Basic Property Use (FD1.46)_fi"].isin(['Multifamily dwelling', '1 or 2 family dwelling'])]
    nfirs_geodf_sf.drop("Basic Property Use (FD1.46)_fi",axis=1, inplace=True)
    #return nfirs_geodf_sf  
    nfirs_geodf_mf = nfirs_geodf_both[nfirs_geodf_both["Basic Property Use (FD1.46)_fi"]!='1 or 2 family dwelling']   
    nfirs_geodf_mf.drop("Basic Property Use (FD1.46)_fi",axis=1, inplace=True)
    return nfirs_geodf_sf, nfirs_geodf_mf
    
#pred_ash_geoid.plot(column='Predicted_Fire_Risk', legend=True, cmap='autumn', figsize=(20,10));


# In[21]:


nfirs_geodf_sf, nfirs_geodf_mf = nfirs_processor_geodf_both(filepath, epsg)
nfirs_geodf_sf.shape,  nfirs_geodf_mf.shape #nfirs_geodf_sf['IncidentID_date_fi'].nunique(),   


# In[22]:


nfirs_geodf_sf.FireIndicator_fi.value_counts()
#0    289999
#1      5128


# In[23]:


nfirs_geodf_mf.FireIndicator_fi.value_counts()
#0    289999
#1      5128


# In[24]:


#nfirs_geodf_sf.CallType_Catagory_fi.value_counts()


# In[25]:


#nfirs_geodf_sf.info()  #    return parcel_geodf #parcel_geodf_sf parcel_geodf_mf
#nfirs_geodf_sf.head(2)


# In[26]:


#nfirs_geodf_mf.info()  #    return parcel_geodf #parcel_geodf_sf parcel_geodf_mf
#nfirs_geodf_mf.head(2)


# In[27]:


#census_all.to_csv('census_all_2.csv', na_rep='NA')  #index=True
#census_all = pd.read_csv('census_all.csv', index_col='BG_ID_6')  #set_index
census_all = pd.read_csv('census_all.csv')  #set_index
#census_all.set_index('BG_ID_6', inplace=True)
#census_all.info()
#census_all.head(2)


# In[28]:


check_missing_values(census_all)


# # Prepare for Join of census to nfirs_parcel_merged bot
# 

# In[29]:


def nfirs_parcel_census_processor_both(nfirs_parcel_merged, census_all):
        joined_df_raw = pd.merge(census_all,nfirs_parcel_merged,left_on = ['BG_ID_6'],right_on = ['BG_ID_6_tg'],how= 'inner') 
        if 'Address_tp' in joined_df_raw.columns:
            joined_df_raw.set_index(['Address_tp'], inplace=True)
            joined_df_raw.rename(columns={'FireIndicator_fi': 'FireIndicator'},inplace = True)
        else: joined_df_raw.set_index(['BG_ID_9_tg'], inplace=True)
        return joined_df_raw           


# # Join tiger, nfirs_df & parcel  SF
# 

# In[30]:


def nfirs_parcel_bgid_processor_sf(tiger, nfirs_geodf_sf, parcel_geodf_sf, census_all):
    #Part 1: join tiger + nfirs_geodf
    tiger['geometry_9_tg_2'] = tiger['geometry_9_tg']
    tiger = tiger.rename(columns={'geometry_9_tg_2':'geometry'})
    #Join nfirs_geodf_sf + tiger
    round_1 = nfirs_geodf_sf.sjoin(tiger, how="inner", predicate='within')
    round_1.drop_duplicates(inplace=True, ignore_index=True) 
    round_1.reset_index(drop=True, inplace=True)#.
    sum_cols = ['Call_Cat_100_fi','Call_Cat_200_fi', 'Call_Cat_300_fi', 'Call_Cat_400_fi','Call_Cat_500_fi','Call_Cat_600_fi',  'Call_Cat_700_fi', 'Call_Cat_800_fi', 'Call_Cat_900_fi']
    round_1_g = round_1.groupby(['BG_ID_9_tg'])[sum_cols].aggregate('sum')
    round_1_g = round_1_g.add_suffix('_sum')
    #round_1_all = pd.concat([round_1_b, round_1_g], axis=1)
    round_1_all = round_1_g.copy()   #pd.concat([round_1_b, round_1_g], axis=1)
    cols_tg = [col for col in parcel_geodf_sf.columns if '_tg' in col]
    df4 =  parcel_geodf_sf.loc[:, (cols_tg)]  
    df4.drop_duplicates(inplace=True)# ,ignore_index=True)
    df4.set_index(['BG_ID_9_tg', 'BG_ID_6_tg'], inplace=True)
    df4l = list(df4.columns)
    parcel_geodf_sf_s = parcel_geodf_sf.drop(df4l,axis=1)
    df_5 = parcel_geodf_sf_s.groupby(["BG_ID_9_tg","BG_ID_6_tg"]).mean()
    df_5_mean = df_5.add_suffix('_mean')
    df_6 = parcel_geodf_sf_s.groupby(["BG_ID_9_tg","BG_ID_6_tg"]).sum()   
    df_6_sum = df_6.add_suffix('_sum')
    round_2_all = pd.concat([df4, df_5_mean, df_6_sum], axis=1)
    round_2_all.reset_index(inplace = True)    
    round_1_all.reset_index(inplace = True)
    #round_2_all.reset_index(inplace = True)
    nfirs_parcel_merged_sf = pd.merge(round_1_all,round_2_all,left_on = ['BG_ID_9_tg'],right_on = ['BG_ID_9_tg'],how= 'right')#.merge(df4,on='BG_ID_9_tg')
    nfirs_parcel_merged_sf = nfirs_parcel_merged_sf.fillna('0.00')
    nfirs_parcel_merged_sf['Call_Cat_100_fi_sum'] = nfirs_parcel_merged_sf['Call_Cat_100_fi_sum'].astype(float)
    nfirs_parcel_merged_sf['FireIndicator']=nfirs_parcel_merged_sf['Call_Cat_100_fi_sum'].apply(lambda x: 0 if x == 0.0 else 1)
    nfirs_parcel_merged_sf = nfirs_parcel_merged_sf.sort_index(axis=1)#sorts alphabeticaly by co
    joined_df_raw_sf  = nfirs_parcel_census_processor_both(nfirs_parcel_merged_sf, census_all)
    return joined_df_raw_sf



# In[31]:


joined_df_raw_sf = nfirs_parcel_bgid_processor_sf(tiger, nfirs_geodf_sf, parcel_geodf_sf, census_all)
joined_df_raw_sf.shape
#(4925, 72)  #(4925, 228) (3392, 425)


# In[32]:


joined_df_raw_sf.FireIndicator.value_counts()
#0    3191
#1    1734
#0    2084
#1    1308


# In[33]:


#joined_df_raw_sf.info()
#joined_df_raw_sf.head(2)


# # Join tiger, nfirs_df & parcel  MF
# 

# In[34]:


def not_yet_joined (df1, joined_df, variabe_df1):  
    need_to_join= df1[~df1[variabe_df1].isin(joined_df[variabe_df1])]
    need_to_join.reset_index(drop=True, inplace=True)
    return need_to_join


# In[35]:


#A
def nfirs_parcel_processor_mf(nfirs_geodf_mf, parcel_geodf_mf): 
    # Round 1: sjoin 
    round_1_join = nfirs_geodf_mf.sjoin(parcel_geodf_mf, how="inner", predicate='within')
    round_1_join.drop_duplicates(subset=['IncidentID_date_fi'], inplace= True) #Can drop now, has been tallied
    r1_needed_nfirs_geodf_mf = not_yet_joined (nfirs_geodf_mf, round_1_join, 'IncidentID_date_fi')  
    # Round 2: Join On address 
    round_2_join =pd.merge(parcel_geodf_mf,r1_needed_nfirs_geodf_mf,left_on = 'STAddress_tp', right_on = 'STAddress_fi', how = 'inner')
    round_2_join.drop_duplicates(subset=['IncidentID_date_fi'], inplace= True) #Can drop now, has been tallied
    # Round 3: 1+ 2
    round_3_join = pd.concat([round_1_join, round_2_join])
    round_3_join.drop_duplicates(subset=['IncidentID_date_fi'], inplace= True) #Can drop now, has been tallied
    # join_nearest, after what was joined/what is still needed
    r3_needed_nfirs_geodf_mf = not_yet_joined (nfirs_geodf_mf, round_3_join, 'IncidentID_date_fi')  #, 'Address_tp')
    join_nearest = gpd.sjoin_nearest(r3_needed_nfirs_geodf_mf, parcel_geodf_mf, distance_col="distances", how="inner")
    join_nearest.sort_values(['IncidentID_date_fi', 'distances'], ascending=[True, True], inplace=True) 
    join_nearest.drop_duplicates('IncidentID_date_fi', inplace=True) #.reset_index(drop=True, inplace=True)  
    #Concat above: round_1_join + round_2_join + join_nearest
    nfirs_parcel_merged_mf_r = pd.concat([round_3_join,join_nearest])
    nfirs_parcel_merged_mf_r.drop_duplicates(subset=['IncidentID_date_fi'],inplace=True) #Can drop now, has been tallied
    #See rows need_to_add from parcel_geodf_mf
    need_to_add = not_yet_joined (parcel_geodf_mf, nfirs_parcel_merged_mf_r, 'parid_tp')  #, 'Address_tp')
    #Final Join
    nfirs_parcel_merged_mf = pd.concat([need_to_add, nfirs_parcel_merged_mf_r])
    nfirs_parcel_merged_mf.sort_index(axis=1, inplace=True)#sorts alphabeticaly 
    nfirs_parcel_merged_mf.reset_index(drop=True, inplace=True)
    cols = [col for col in nfirs_parcel_merged_mf.columns if 'index' in col]
    nfirs_parcel_merged_mf.drop(cols,axis=1, inplace=True)
    fi_cols = [col for col in nfirs_parcel_merged_mf.columns if '_fi' in col]
    nfirs_parcel_merged_mf[fi_cols] = nfirs_parcel_merged_mf[fi_cols].fillna(0)
    nfirs_parcel_merged_mf = nfirs_parcel_merged_mf.sort_index(axis=1)#sorts alphabeticaly by co
    return nfirs_parcel_merged_mf


# In[36]:


start = datetime.now()
nfirs_parcel_merged_mf = nfirs_parcel_processor_mf(nfirs_geodf_mf, parcel_geodf_mf)
end = datetime.now()
print("Elapsed", (end - start).total_seconds() * 10**6, "µs")
nfirs_parcel_merged_mf.shape  #(121357, 81)


# In[37]:


#nfirs_parcel_merged_mf.info()
#nfirs_parcel_merged_mf.head(2)


# In[38]:


nfirs_parcel_merged_mf.FireIndicator_fi.value_counts()
#0    289999
#1      5128
#0.00    119201 <
#1.00      2174


# In[39]:


nfirs_parcel_merged_mf['IncidentID_date_fi'].nunique(), nfirs_parcel_merged_mf['Address_tp'].nunique(), nfirs_parcel_merged_mf['parid_tp'].nunique()
#((108636, 75), 107734, 1046, 1166)
#(120443, 1040, 1357)


# In[40]:


def nfirs_parcel_merged_feature_engineering_mf(nfirs_parcel_merged_mf, census_all):
    #Enineer boundary of apartment compex from parce boundaries:
    df0 = nfirs_parcel_merged_mf.loc[:, ('Address_tp','geometry_polygon_sh')]
    df0.rename(columns={'geometry_polygon_sh': 'geometry'}, inplace = True)
    df0b = df0.dissolve(by='Address_tp')
    df0b['centerpoint_ad'] = df0b['geometry'].centroid
    df0b["longitude_ad"] = df0b.geometry.centroid.x  #Latitude_fi	Longitude_fi
    df0b["latitude_ad"] = df0b.geometry.centroid.y
    #Features: mean, sum
    df2 = nfirs_parcel_merged_mf.loc[:, ('ALAND20_tg', 'AWATER20_tg', 'POP20_tg', 'HOUSING20_tg','ADJRCN_tp', 
                                         'APRTOT_tp', 'AREASUM_tp', 'AREA_tp', 'Address_tp',
           'BASERATE_tp', 'BUILDING_tp','Basic Property Use Code (FD1.46)_fi', 'CALCACRES_tp', 'CLASS_1_tp',
           'CONSTR_tp', 'CUBICFT_tp', 'CallType_Catagory_fi',
           'DEPR_tp', 'DISTRICTID_fi','FEATVAL_tp', 'FLRFROM_tp', 'FUNCTUTIL_tp',
           'INCUSE_tp', 'LINEVAL_tp', 'LLINE_tp','LUC_tp', 'MSCLASS_tp', 'MSRANK_tp',
           'MSSECT_tp', 'NBHD_tp', 'OCCUPANCY_tp', 'OFCARD_tp', 'PERIM_tp', 'PHYCOND_tp', 'PRICE_tp', 'RATE_tp', 
           'STATUS_2_tp',  'STORIES (FLRTO)_tp','STORIES_tp', 'USETYPE_tp', 'YR BUILT_tp')] 
    df2a = df2.loc[:, df2.isnull().mean() < .10]
    df2a_mean = df2a.groupby("Address_tp").mean()
    df2a_mean = df2a_mean.add_suffix('_mean')    
    df2a_sum = df2a.groupby("Address_tp").sum()
    df2a_sum = df2a_sum.add_suffix('_sum')    
    df5 = nfirs_parcel_merged_mf.loc[:, ('Address_tp','Call_Cat_100_fi', 'Call_Cat_200_fi', 'Call_Cat_300_fi',
           'Call_Cat_400_fi', 'Call_Cat_500_fi', 'Call_Cat_600_fi','Call_Cat_700_fi', 'Call_Cat_800_fi', 'Call_Cat_900_fi')]
    df5a = df5.loc[:, df5.isnull().mean() < .10]   
    df5a_sum = df5a.groupby("Address_tp").sum()
    df5a_sum = df5a_sum.add_suffix('_sum')  
    #To ensure all Fire "1" are saved
    df3 = nfirs_parcel_merged_mf.loc[:, ('Address_tp','FireIndicator_fi')]
    df3.sort_values(['Address_tp', 'FireIndicator_fi'], inplace=True, ascending=[False, False]) 
    df3.drop_duplicates(['Address_tp'], inplace = True)  #drop_duplicates('Address_tp') # dropping duplicates keeps first
    df3.set_index(['Address_tp'], inplace = True)   #set_index('Address_tp')
    #Save smallest distance
    df4 = nfirs_parcel_merged_mf.loc[:, ('Address_tp','BG_ID_6_tg', 'geometry_6_tg')]
    df4.drop_duplicates(['Address_tp'], inplace = True) # dropping duplicates keeps first
    df4.set_index(['Address_tp'], inplace = True)
    #concat above
    nfirs_parcel_merged_mf_2_so =pd.concat([df0b, df2a_mean, df2a_sum, df5a_sum,df3, df4], axis=1)
    nfirs_parcel_merged_mf_2 = nfirs_parcel_merged_mf_2_so.fillna(nfirs_parcel_merged_mf_2_so.mean())
    nfirs_parcel_merged_mf_2.reset_index(drop=False, inplace=True)    
    nfirs_parcel_merged_mf_2.sort_index(axis=1, inplace=True)#sorts alphabeticaly by co
    joined_df_raw_mf  = nfirs_parcel_census_processor_both(nfirs_parcel_merged_mf_2, census_all)
    return joined_df_raw_mf




# In[41]:


joined_df_raw_mf = nfirs_parcel_merged_feature_engineering_mf (nfirs_parcel_merged_mf, census_all)
joined_df_raw_mf.info()
joined_df_raw_mf.head(2)



# In[42]:


joined_df_raw_mf.shape  #(1040, 93)  #(1040, 249)
#(647, 957)


# In[43]:


joined_df_raw_mf.FireIndicator.value_counts()
#544 5608
#502  486
#0.00    531 <
#1.00    509
#1.00    420
#0.00    227


# In[44]:


### joined_df_mf.to_csv('joined_df_mf.csv', na_rep='NA' )
#joined_df_mf_2 = pd.read_csv('joined_df_mf.csv')
#joined_df_mf_2.set_index(['Address_tp'], inplace = True)   
#joined_df_mf_2.head(2)

#joined_df_sf_2 = pd.read_csv('joined_df_sf.csv')
#joined_df_sf_2.set_index(['BG_ID_9_tg'], inplace = True)   #set_index('Address_tp')
#joined_df_sf_2.head(2)


# In[45]:


#joined_df_sf = correlation_processor (joined_df_raw_sf)
#joined_df_sf.shape  4925, 172)


# In[46]:


#joined_df_mf= correlation_processor (joined_df_raw_mf)
#joined_df_mf.shape    (1040, 186)


# In[47]:


###   joined_df_raw_mf.to_csv('joined_df_raw_mf_2.csv', na_rep='NA' )
####  joined_df_raw_sf.to_csv('joined_df_raw_sf_2.csv', na_rep='NA' )


# In[48]:


#df.to_file('MyGeometries.shp', driver='ESRI Shapefile')
#gdf.to_file('dataframe.shp') 


# # >>>>>>>>>>>>>>>>>>>>>>

# In[ ]:




