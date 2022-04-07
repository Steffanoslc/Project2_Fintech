from matplotlib import ticker
import utils.alpacaConnect






















def run():
    data= utils.alpacaConnect.get_data(ticker)

    btcusd_df= utils.alpacaConnect.data_cleaning(data)





if __name__ == "__main__":
    run()