import folium
import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
import geopandas as gpd#
from shapely.geometry import Point
from shapely import wkt
import branca
from folium import plugins
import branca.colormap as cm   # colormap = cm.StepColormap sow leend
from shapely import wkt
epsg = "4326" 

st.write("# DeKalb County Fire Risk Probability Multifamily")
st.markdown('''A colaboration between Dekalb County Fire & Rescue & ATLytiCS''')

@st.cache_data
def csv_geo_processor (csv):
    df = pd.read_csv(csv, low_memory=False) 
    df.drop(['Unnamed: 0'], axis = 1, inplace=True)
    df = df.dropna(axis=1, how='all')
    df= df.rename(columns={'Station_tg': 'Station', 
                                        'Battalion_tg':'Battalion',
                                            'Predicted_Fire_Risk_6_mf': 'Fire_Risk_BG6',
                                            'BG_ID_6': 'BGID_6'})
    df['geometry'] = df['geometry_6'].apply(wkt.loads)
    df['BGID_6'] = df['BGID_6'].astype(str)  
    df['BG_ID_6_tg'] = df['BG_ID_6_tg'].astype(str)  
    geodf= gpd.GeoDataFrame(df)
    geodf = geodf.set_crs(epsg=epsg)
    bg_6 = geodf.loc[:, ('BGID_6','Fire_Risk_BG6','BG_ID_6_tg','geometry')] 
    bg_6 = bg_6.dropna(axis=1, how='all')
    bg_6 = bg_6.set_index('BGID_6')
    bg_6_geojs = bg_6.to_json()
    adr_df = df.loc[:, ('Address_mf','Predicted_Fire_Risk_adr_mf','Predicted_Fire_Probability_adr_mf','Latitude_adr_mf','Longitude_adr_mf')] 
    adr_df = adr_df.dropna(axis=1, how='all')
    adr_df['Fire_Risk_Prob_%'] = round(adr_df['Predicted_Fire_Probability_adr_mf'].multiply(100),0)
    adr_df['Fire_Risk_Prob_%'] = adr_df['Fire_Risk_Prob_%'].astype('int')
    df2 = df.loc[:, ('Battalion','Station', 'Fire_Risk_BG6','BGID_6','geometry')] 
    df2.drop_duplicates(inplace = True)
    df2.sort_values(by=["Battalion", "Station", "Fire_Risk_BG6"], inplace =True)

    return  bg_6_geojs, adr_df, df2

mf_bg_6_geojs, adr_df, df2 =  csv_geo_processor ('data/processed/pred_mf_stion.csv')

@st.cache_data
def bg6_risk_by_bat_stion (df2):
    df2=df2.astype({'Battalion':str, 'Station':str})
    df3 = (df2.groupby(["Battalion", "Station", "Fire_Risk_BG6"])
            .agg({'BGID_6': lambda x: ",".join(x)})
            .reset_index())
    df3=df3.astype({'Fire_Risk_BG6':'category', 'Station':int})
    df3["Fire_Risk_BG6"] = df3["Fire_Risk_BG6"].cat.reorder_categories(["High Risk", "Moderate Risk", "Low Risk"])
    df3.sort_values(by=["Fire_Risk_BG6","Battalion", "Station"], inplace =True)
    df3.reset_index(drop=True,inplace = True)
    return df3
df3 = bg6_risk_by_bat_stion (df2)

st.write("Show YOY % Change Table:") #Ad df as a table
if st.checkbox('by Individual BGID_6 with Boundary Coordinates'):
    st.dataframe(df2,hide_index=True, height= 150)

if st.checkbox('by Battalion/Station'):
    st.dataframe(df3,hide_index=True, height= 150)

# Base map:  #33.80	-84.23	
m1 = folium.Map(location=[33.80, -84.23], zoom_start=10.5)

######### MAP LAYER A: MF BGID 6; Color by Risk levels: high, moderate, & low; Label BGID_6
@st.cache_data
def field_type_colour(feature):
    if feature['properties']['Fire_Risk_BG6'] == 'High Risk':
        return '#e6194b'
    elif feature['properties']['Fire_Risk_BG6'] == 'Moderate Risk':
        return 'gold'
    elif feature['properties']['Fire_Risk_BG6'] == 'Low Risk':
        return '#19e6b4'    

cm1 = folium.GeoJson(mf_bg_6_geojs,
                style_function= lambda feature: {'fillColor':field_type_colour(feature), 
                                                'fillOpacity':0.9, 'weight':0},
                tooltip=folium.GeoJsonTooltip(fields=['Fire_Risk_BG6','BG_ID_6_tg'],
                                            labels=True,
                                            sticky=True),show=False,
                ).add_to(m1)

########### MAP LAYER B: MF Apartment Complex
@st.cache_data
def min_max_proecssor():
    '''This creates the color bar for fire risk by apartment complex, as well as the colormap
    referenced in the code, below.
    '''
    min_prob=adr_df["Fire_Risk_Prob_%"].min()
    max_prob=adr_df["Fire_Risk_Prob_%"].max()

    colormap = cm.StepColormap(colors=['white', 'darkgray',  'black' ] ,#renkler
                                index=[min_prob,33.3,66.6, max_prob], #eşik değerler
                                vmin= min_prob,
                                vmax=max_prob)

    colormap.caption = 'Predicted Fire Probability % by Apt Complex Address'
    colormap.style = 'font-size: 20px'
    return adr_df, colormap

adr_df, colormap, = min_max_proecssor()
colormap.add_to(m1)  

#### Add a tooltip w/ text re: Complex address, MF Fire Risk, & lat/lng coordinates.

for loc, p, cd, tt in zip(zip(adr_df["Latitude_adr_mf"],adr_df["Longitude_adr_mf"]),
                        adr_df["Fire_Risk_Prob_%"],adr_df["Address_mf"],adr_df["Fire_Risk_Prob_%"]):
            folium.Circle(
            location=loc,
            radius=4, #yarıçap
            fill=True, 
            color=colormap(p),
            tooltip="Fire Probability {}%; {}".format(tt,cd),
            tooltip_kwds=dict(labels=True),  #legend_kwds=dict(colorbar=False),  # 
            legend=True, #loc='lower left', # show legend
            legend_kwds={"loc": 'center right'},##
            name="delta",  # name of the layer in the map

).add_to(m1)
    
st_data = st_folium(m1, width=725)


 ##  $ streamlit run scr/apps/Fire_Risk_Map_MF.py

