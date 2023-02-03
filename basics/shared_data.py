import pandas as pd

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class SharedData(metaclass=Singleton):
    transactions = None
    content = None

    def __init__(self):
        super.__init__()

    def store_transactions(self, txns):
        self.transactions = txns

    def store_content(self, cnt):
        self.content = cnt

def get_data():
    return SharedData()