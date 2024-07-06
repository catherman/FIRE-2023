# from ph2_approach_a.ipynb
#Alternatively, this may be accessed via API

import pandas as pd
import censusdata
import re

def get_census_data (types):
    type_df = pd.DataFrame(types)
    type_var = type_df[type_df.columns[0]].values.tolist()
    type_tbl = type_df[type_df.columns[1]].values.tolist()
    type_lab = type_df[type_df.columns[2]].values.tolist()
    if 'Estimate!!Total!!Never married!!Male!!In labor force!!Unemployed' in type_lab:
         #type_var = type_var[:B27011_017E]+type_var[C18120_006E:]
        type_var = type_var[:86]+type_var[94:]
        type_tbl = type_tbl[:86]+type_tbl[94:]
        type_lab = type_lab[:86]+type_lab[94:]

    elif 'Estimate!!Total!!Educational services, and health care and social assistance' in type_lab:
        type_var = type_var[:282]   
        type_tbl = type_tbl[:282]
        type_lab = type_lab[:282]   
    else: 
        pass
    census_type = censusdata.download('acs5', 2021,
                                   censusdata.censusgeo([('state', '13'),
                                                         ('county', '089'),
                                                         ('block group', '*')]),
                                                          type_var)
    census_type.columns = [type_lab]
    census_type_2=census_type.loc[:, census_type.isnull().mean() < .05] 
    return census_type_2

def census_processor():
    housing = [] 
    unemploy = []       
    poverty = [] 
    health = [] 
    education = [] 
    population = [] 
    rooms = []  #
    rent = []       
    fuel = [] 
    occupied  = [] 
    structure = [] 
    income = [] 

    typ_list = [("housing", "census_housing"),
                    ("unemploy", "census_unemployment"),
                    ("population", "census_population"),       
                    ("poverty", "census_poverty"),
                    ("health", "census_health"),   
                    ("education","census_education"),
                    ("rooms", "census_rooms"),
                    ("rent", "census_rent"),
                    ("fuel", "census_fuel"),       
                    ("occupied", "census_occupied"),
                    ("structure", "census_structure"),   
                    ("income","census_income")
    ]
    census = [] 
    
    for i, (typ, name) in enumerate(typ_list):
        types = censusdata.search('acs5', 2015, 'label', typ)  
        names = get_census_data (types)
        census.append(names)
    census_all = pd.concat(census, axis =1)      
    census_all = census_all.iloc[:,~census_all.columns.duplicated()] 
    census_all = census_all.rename_axis('GEOID_long').reset_index()
    census_all.columns = census_all.columns.map('_'.join) #rid of multi index
    census_all['GEOID_long'] = census_all['GEOID_long'].astype(str)  #.str.replace(r'\D', '', regex=True)
    census_all['GEOID_long'] = census_all['GEOID_long'].str.replace(r'\D', '', regex=True) #et rid of etters
    new = census_all["GEOID_long"].str.split("130890", n = 1, expand = True)
    census_all["BG_ID_6"]= new[1]
    census_all = census_all.drop(['GEOID_long'],axis=1)
    census_all.set_index(['BG_ID_6'], inplace = True)   
    census_all=census_all.loc[:, census_all.isnull().mean() == 0.0] 
    #census_all=census_all.loc[:, census_all.isnull().mean() < .05] 
    return census_all

census_all = census_processor()
census_all.shape  ##(536, 160)
#census_all.head(2)
#(536, 356)
#536, 1445)
#(536, 871)