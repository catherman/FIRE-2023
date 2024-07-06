
# FIRE 2024

An urban fire risk prediction model using machine learning. A joint project with ATLytiCS and DeKalb County Fire and Rescue (DCFR).  (2022-2023)

Business Challenge: Allocation of DCFR’s limited Community Risk Reduction resources to properties at greatest risk of fire.

Solution: Engineer a machine learning model in Python to identify fire risk probability and ranking of multifamily properties.

Domain Knowledge: Consulted with a domain expert and researched similar urban fire risk analysis, reviewing summary reports, approach & code for inspiration and strategies to improve model’s performance. 

Data:  Conducted statistical & spatial analytics using large microdata sets such as: local government geospatial, parcel, NFIRS (EMS calls), land use, socio-economic, demographic, google reviews, and shape files (county and TIGER).  Collaborated with team on formulating exploratory data analysis, data preparation, visualization, feature selection & engineering, joining of large “messy” & complex data sets.

Model selection, fitting and parameter tuning of Random Forest and AdaBoost to optimize probability metrics (Brier loss and Log loss). Random Forest Classification Model accuracy: 83%.

Deliverables: An interactive [Tableau Dashboard](https://public.tableau.com/app/profile/margaret.catherman/viz/FIREIIMultifamilyFireRiskAnalytics/SummaryPublic2), Streamlit app, and [Summary Report](https://github.com/catherman/FIRE-2024/blob/main/reports/FIRE%20II%20Summary.pdf), ranking 1046 multifamily properties in DeKalb County, identifying property risk levels, address, geo coordinates, fire district and station, to be used by DCFR in their Community Risk Assessment (CRA). Presented strategy and findings to industry professionals.

### How to reproduce the Streamlit App  
<img align="right" src="https://github.com/catherman/FIRE-II/assets/43255276/37eeb3a9-9a6a-46ff-ae38-106c8f83eb7e" alt="Screen Shot 2024-05-05 at 1 57 02 PM">
The Streamlit library allows for app creation using only Python. While there are limitations to the format, it is useful to quickly generate apps for internal review. 

1. Move the required code & csvs to your OS, either:
   
     a. Clone this repository to GitHub or VSCode;  OR
  
     b. Copy/download to your OS & Python interpreter only files needed for the multifamily or single family app, noted below:
  
       -Data required:  “data/processed/pred_mf_stion.csv” &/or “data/processed/pred_sf_stion.csv”
    
       -Code required: “src/apps/Fire_Risk_Map_MF.py” &/or “src/apps/Fire_Risk_Map_MF.py”
    
3. Install required libraries, if not already installed: "!pip install streamlit"
4. Run either the MF or SF code. The app will appear in a few minutes on a new browser.

<br clear="right"/> 

For more info, see 
[Streamlit documentation](https://docs.streamlit.io/get-started)

### Contents of Repository
Note: 
Complete raw data not provided, as this is an internal project for DCFR.  Rather, this repository is meant to provide a reference for other similar projects.
Production pipeline: work in progress, incomplete files noted with “add”. 


![Screen Shot 2024-07-06 at 9 34 48 AM](https://github.com/catherman/Fire_2/assets/43255276/6efc5e92-eba0-489a-8cfd-4d0536b5bdc3)

![Screen Shot 2024-07-06 at 9 35 06 AM](https://github.com/catherman/Fire_2/assets/43255276/6454dcdd-17b1-439c-9bde-06ea79a02e7a)

![Screen Shot 2024-07-06 at 9 35 20 AM](https://github.com/catherman/Fire_2/assets/43255276/b2770d4b-e1dd-483f-8418-368d317a27d2)
