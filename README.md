{ .FIRE_2}/
├── configs
│   └── cloudbuild.yaml  # *add*
├── data
    ├── external
    │   ├── get_census.py
    │   ├── get_shapefiles.py
    ├── processed
    │   ├── pred_mf_stion.py
    │   ├── pred_sf_stion.py
    ├── raw
    │       ├── parcel.csv  *#add?*
    │       ├── NFRS.csv  *#add?*             
├── notebooks
│   	    ├── 0_Complete.ipynb
│   		├── 1_Data_ingestion_prep_join_SF_MF.ipynb
│   		├── 2_Data_Profile_EDA_MF_SF.ipynb   *#add*
│   		├── 3a_Model_table_SF.ipynb
│   		└── 3b_Model_table_MF.ipynb
│   		└── 4a_Viz_Map_SF.ipynb
│   		└── 4b_Viz_Map_MF.ipynb
├── references
│   		└── NFIRS Incident Type Cheat Sheet (PDF).pdf
│   		└── OrionCAMADataDictionary.xlsx
├── reports
│   		└── FIRE II Summary.pdf
│   		└── sf_summary.png *#add*
│   		└── mf_summary.png│
└── src
    ├── _ init _.py  *#add*
    ├── apps
    │   ├── Fire_Risk_Map_MF.py
    │   ├── Fire_Risk_Map_SF.py
    ├── data
    │   ├──ingestion_feature_join_SF_MF.py
    ├── models
    │   └── MF_w_table.py
    │   └── SF_w_table.py
    └── visualizations
    		├── evaluation.py    #add
        	└── exploration.py  #add
├── tools
    └── utils.py
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
![Screen Shot 2024-07-06 at 9 34 48 AM](https://github.com/catherman/Fire_2/assets/43255276/6efc5e92-eba0-489a-8cfd-4d0536b5bdc3)
![Screen Shot 2024-07-06 at 9 35 06 AM](https://github.com/catherman/Fire_2/assets/43255276/6454dcdd-17b1-439c-9bde-06ea79a02e7a)
![Screen Shot 2024-07-06 at 9 35 20 AM](https://github.com/catherman/Fire_2/assets/43255276/b2770d4b-e1dd-483f-8418-368d317a27d2)
