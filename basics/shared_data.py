import pandas as pd

class _SharedData:
    _instance = None

    transactions = None
    content = None

    txns_file_name = 'https://raw.githubusercontent.com/kavurisrikanth/news-recommender-capstone/master/data/consumer_transanctions.csv'
    cnt_file_name = 'https://raw.githubusercontent.com/kavurisrikanth/news-recommender-capstone/master/data/platform_content.csv'

    data_path = '../data/'

    def __init__(self):
        if not self.transactions:
            self.transactions = pd.read_csv(self.txns_file_name)
        if not self.content:
            self.content = pd.read_csv(self.cnt_file_name)

    def get_transactions(self):
        return self.transactions

    def get_content(self):
        return self.content

    def set_content(self, content):
        self.content = content

def SharedData():
    if not _SharedData._instance:
        _SharedData._instance = _SharedData()
    return _SharedData._instance
