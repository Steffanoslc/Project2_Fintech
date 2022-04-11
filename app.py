
import pandas as pd
import utils.alpacaConnect
import utils.fearGreedIndex

ticker= "BTCUSD"
url= "https://api.alternative.me/fng/?"



















def run():
    btcusd_df= utils.alpacaConnect.get_data(ticker)
    feer_greed_df= utils.fearGreedIndex.feargreedindex(url)
    all_data= pd.concat(btcusd_df,feer_greed_df, axis=1)

    return all_data



if __name__ == "__main__":
    run()