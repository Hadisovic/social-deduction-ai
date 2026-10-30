"""Observer clock only. Windows unbiased time excludes sleep/hibernation."""
import os,time


def clock():
    if os.name=='nt':
        import ctypes
        counter=ctypes.c_ulonglong()
        precise=getattr(ctypes.windll.kernel32,'QueryUnbiasedInterruptTimePrecise',None)
        if precise is not None:precise(ctypes.byref(counter));return counter.value/10_000_000
        if ctypes.windll.kernel32.QueryUnbiasedInterruptTime(ctypes.byref(counter)):return counter.value/10_000_000
    return time.perf_counter()


SOURCE='Windows QueryUnbiasedInterruptTime[Precise], excluding suspend; perf_counter fallback outside Windows'
