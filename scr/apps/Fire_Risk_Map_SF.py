###>>>> MONDAY 10:10 AM  SF  REVISED SNADBOX <<<<<
import folium
from folium import plugins
import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
import geopandas as gpd#
from shapely.geometry import Point
from shapely import wkt
import branca
#import streamlit.components.v1 as components
import branca.colormap as cm   # colormap = cm.StepColormap sow leend
epsg = "4326" 

#st.set_page_config(
   # page_title="Dekalb County Fire Risk Probability Single Family", 
    
    #page_icon="👋",
#)

st.write("# DeKalb County Fire Risk Probability Single Family")
st.markdown(
    '''
    A colaboration between Dekalb County Fire & Rescue & ATLytiCS
    '''
)

# Access & preprocess data
@st.cache_data
def csv_geo_processor_sf2 (csv, co):
    df = pd.read_csv(csv, low_memory=False) 
    df.drop(['Unnamed: 0'], axis = 1, inplace=True)
    df = df.dropna(axis=1, how='all')
    df['geometry'] = df['geometry_6'].apply(wkt.loads)
    df['BG_ID_6'] = df['BG_ID_6'].astype(str)  
    df['BG_ID_6_tg'] = df['BG_ID_6_tg'].astype(str)  
    df= df.rename(columns={'Station_tg': 'Station', 
                                        'Battalion_tg':'Battalion',
                                            'Predicted_Fire_Risk_6_sf': 'Fire_Risk_BG6',
                                            'BG_ID_6': 'BGID_6'})    
    
    geodf= gpd.GeoDataFrame(df)
    geodf = geodf.set_crs(epsg=epsg)
    bg_6 = geodf.loc[:, ('BGID_6','Fire_Risk_BG6','BG_ID_6_tg','geometry')] 
    bg_6 = bg_6.dropna(axis=1, how='all')
    bg_6 = bg_6.set_index('BGID_6')
    bg_6_geojs = bg_6.to_json()    
    
    # Create SF BG_9 layer from geodf:
    sf_df = geodf.loc[:, ('BG_ID_9','Predicted_Fire_Risk_9_sf','Predicted_Fire_Probability_9_sf',
    'geometry_9','geometry')] 
    sf_df = sf_df.dropna(axis=1, how='all')
    sf_df['Fire_Risk_Prob_%'] = round(sf_df['Predicted_Fire_Probability_9_sf'].multiply(100),0)
    sf_df['Fire_Risk_Prob_%'] = sf_df['Fire_Risk_Prob_%'].astype('int')  
    
    # SF BG_9 lat/lng info for BG_9
    sf_df.drop('geometry',axis=1, inplace=True)
    sf_df['geometry'] = sf_df['geometry_9'].apply(wkt.loads)
    sf_df['centerpoint'] = sf_df['geometry'].centroid
    sf_df["lng_bg9"] = sf_df.geometry.centroid.x
    sf_df["lat_bg9"] = sf_df.geometry.centroid.y
    
    # The next step means the df is no longer a geodf; however, as we are using
    # folium, we no longer need a geodf. Benefit: less memory needed
    sf_df = sf_df.loc[:, ('BG_ID_9','Predicted_Fire_Risk_9_sf','Fire_Risk_Prob_%',
        'lng_bg9', 'lat_bg9')] 
    # Clean up df to be used w/ map as table
    #df.drop('geometry',axis=1, inplace=True)
    df2 = df.loc[:, ('Battalion','Station', 'Fire_Risk_BG6','BGID_6','geometry')] 
    df2.drop_duplicates(inplace = True)
    df2.sort_values(by=["Battalion", "Station", "Fire_Risk_BG6"], inplace =True)

    return bg_6_geojs, sf_df, df2


sf_bg_6_geojs, sf_df, df2 =  csv_geo_processor_sf2 ('data/processed/pred_sf_stion.csv', 'Fire_Risk_BG6')                            
# BG_ID_6_tg	Station_tg	Battalion_tg Predicted_Fire_Risk_6_sf geometry_6

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

st.write("Show Fire Risk Table:") #Ad df as a table
if st.checkbox('by Individual BGID_6 with Boundary Coordinates'):
    st.dataframe(df2,hide_index=True, height= 150)

if st.checkbox('by Battalion/Station'):
    st.dataframe(df3,hide_index=True, height= 150)


#df, sf_bg_6_geojs, sf_df =  csv_geo_processor_sf ('pred_sf_stion.csv', 'Fire_Risk_BG6')                            

# Create SF Fire Risk Map by BG 6 & 9
#m2 = folium.Map(location=[33.92, -84.29], zoom_start=10.5)
m2 = folium.Map(location=[33.80, -84.23], zoom_start=10.5)

#### MAP LAYER A: SF BG 6; Color by Risk levels: high, moderate, & low; Label BG_ID_6
@st.cache_data
def field_type_colour(feature):
    if feature['properties']['Fire_Risk_BG6'] == 'High Risk':
        return '#e6194b'
    elif feature['properties']['Fire_Risk_BG6'] == 'Moderate Risk':
        return 'gold'
    elif feature['properties']['Fire_Risk_BG6'] == 'Low Risk':
        return '#19e6b4'    

cm1 = folium.GeoJson(sf_bg_6_geojs,
                style_function= lambda feature: {'fillColor':field_type_colour(feature), 
                                                'fillOpacity':0.9, 'weight':0},
                tooltip=folium.GeoJsonTooltip(fields=['Fire_Risk_BG6','BG_ID_6_tg'],
                                            labels=True,
                                            sticky=True),show=False,
                ).add_to(m2)
# sf_bg_6_geojs, sf_df =  csv_geo_processor ('pred_sf_stion.csv', 'Fire_Risk_BG6')

########### MAP LAYER B: SF BG_9

@st.cache_data
def min_max_proecssor_sf():
    '''This creates the color bar for fire risk for SF BG_9, as well as the colormap
    referenced in the code, below.
    '''
    min_prob=sf_df["Fire_Risk_Prob_%"].min()
    max_prob=sf_df["Fire_Risk_Prob_%"].max()

    colormap = cm.StepColormap(colors=['white', 'darkgray',  'black' ] ,#rank
                                index=[min_prob,33.3,66.6, max_prob], #degre of risk
                                vmin= min_prob,
                                vmax=max_prob)

    colormap.caption = 'SF Predicted Fire Probability % by BG_9'
    colormap.style = 'font-size: 20px'
    return sf_df, colormap

sf_df, colormap, = min_max_proecssor_sf()
colormap.add_to(m2) 

#### Add a tooltip w/ text re: BG_9, SF Fire Risk, & lat/lng coordinates.

for loc, p, cd, tt in zip(zip(sf_df["lat_bg9"],sf_df["lng_bg9"]),
                        sf_df["Fire_Risk_Prob_%"],sf_df["BG_ID_9"],sf_df["Fire_Risk_Prob_%"]):
            folium.Circle(
            location=loc,
            radius=4, #yarıçap
            fill=True, 
            color=colormap(p),
            #tooltip="Fire Probability {}%; BG9: {}".format(tt,cd),
            tooltip="Fire Probability {}%; BG9: {}; {}".format(tt,cd,loc),
            tooltip_kwds=dict(labels=True),  #legend_kwds=dict(colorbar=False),  # 
            legend=True, #loc='lower left', # show legend
            legend_kwds={"loc": 'center right'},##
            name="delta",  # name of the layer in the map

).add_to(m2)

st_data = st_folium(m2, width=725)

##  $ streamlit run scr/apps/Fire_Risk_Map_SF.py

