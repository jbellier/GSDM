from datetime import datetime, timedelta
import pandas as pd


def yyyy(yyyymmddhh):
    return yyyymmddhh // 1000000
def yyyymm(yyyymmddhh):
    return yyyymmddhh // 10000
def yyyymmdd(yyyymmddhh):
    return yyyymmddhh // 100
def mm(yyyymmddhh):
    yyyymm = yyyymmddhh // 10000
    return yyyymm - 100*yyyy(yyyymmddhh)
def dd(yyyymmddhh):
    yyyymmdd = yyyymmddhh // 100
    return yyyymmdd - 100*yyyymm(yyyymmddhh)
def hh(yyyymmddhh):
    return yyyymmddhh - 100*yyyymmdd(yyyymmddhh)
    
def date_to_yyyymmddhh(dt_time):
    return 1000000*dt_time.year + 10000*dt_time.month + 100*dt_time.day + dt_time.hour

def yyyymmddhh_to_calendarDay(yyyymmddhh):
    return int(datetime(yyyy(yyyymmddhh), mm(yyyymmddhh), dd(yyyymmddhh), hh(yyyymmddhh)).strftime('%j'))

def yyyymmddhh_to_time(yyyymmddhh, time_unit='hours since 1900-01-01 00:00:00', time_calendar='gregorian'):
#     assert yyyymmddhh.ndim == 1
    dates = pd.to_datetime(yyyymmddhh.astype(str), format='%Y%m%d%H')
    return date2num(dates.tolist(), time_unit, time_calendar)

def yyyymmddhh_to_date(yyyymmddhh):
#     assert yyyymmddhh.ndim == 1
    return pd.to_datetime(yyyymmddhh.astype(str), format='%Y%m%d%H')
