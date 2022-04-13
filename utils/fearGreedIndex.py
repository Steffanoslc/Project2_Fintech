import requests
import pandas as pd

def feargreedindex(url):
    limit=365*8
    feer_greed_index_url= f"{url}limit={limit}&format=json&date_format=us"
    response= requests.get(feer_greed_index_url).json()

    feer_greed_df= []
    for row in response["data"]:
        feer_greed_df.append(pd.Series(row))

    feer_greed_df= pd.DataFrame(data= feer_greed_df)
    feer_greed_df= feer_greed_df.set_index("timestamp")
    feer_greed_df= feer_greed_df.drop(columns=["value_classification","time_until_update"])
    
    feer_greed_df= feer_greed_df.reset_index()
    feer_greed_df["timestamp"]= pd.to_datetime(
    feer_greed_df["timestamp"],
    infer_datetime_format= True,
    utc= True)
    feer_greed_df["timestamp"]= feer_greed_df["timestamp"]+ pd.DateOffset(hours=6)
    feer_greed_df= feer_greed_df.set_index("timestamp")
    feer_greed_df["value"]= feer_greed_df["value"].astype("float")
    feer_greed_df.sort_index(inplace=True)

    return feer_greed_df