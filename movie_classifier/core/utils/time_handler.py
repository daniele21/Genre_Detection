import datetime
from time import time


def timestamp_now():
    return time()


def readable_timestamp():
    timestamp = timestamp_now()
    date_obj = datetime.datetime.fromtimestamp(timestamp)

    return date_obj.strftime('%Y-%m-%d_%H-%M-%S')
