"""Dependency-free Windows process-tree CPU/working-set measurement for benchmarks."""
import ctypes
from ctypes import wintypes
import os
import threading
from time import perf_counter


class Resources:
    def __init__(self, include_parent_startup=False):
        self.handles = {}
        self.mutex = threading.Lock()
        self.stop = threading.Event()
        self.peak_rss = self.peak_private = 0
        self.supported = os.name == 'nt'
        self.include_parent_startup = include_parent_startup
        if not self.supported:
            return
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.psapi = ctypes.WinDLL('psapi', use_last_error=True)
        self.kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        self.kernel.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4

        class Memory(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage')]
        self.Memory = Memory
        self.psapi.GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.POINTER(Memory), wintypes.DWORD)

    def _cpu(self, handle):
        times = [wintypes.FILETIME() for _ in range(4)]
        if not self.kernel.GetProcessTimes(handle, *[ctypes.byref(t) for t in times]):
            return 0.
        return sum((t.dwHighDateTime << 32) | t.dwLowDateTime for t in times[2:]) / 1e7

    def watch(self, pid):
        if not self.supported:
            return
        handle = self.kernel.OpenProcess(0x410, False, pid)
        if not handle:
            raise OSError('Cannot measure our benchmark process')
        with self.mutex:
            self.handles[pid] = (handle, self._cpu(handle) if pid == os.getpid() and not self.include_parent_startup else 0.)

    def sample(self):
        if not self.supported:
            return
        rss = private = 0
        with self.mutex:
            for handle, _ in self.handles.values():
                memory = self.Memory()
                memory.cb = ctypes.sizeof(memory)
                if self.psapi.GetProcessMemoryInfo(handle, ctypes.byref(memory), memory.cb):
                    rss += memory.WorkingSetSize
                    private += memory.PrivateUsage
        self.peak_rss = max(self.peak_rss, rss)
        self.peak_private = max(self.peak_private, private)

    def __enter__(self):
        self.started = perf_counter()
        self.watch(os.getpid())
        def observe():
            while not self.stop.wait(.1):
                self.sample()
        self.thread = threading.Thread(target=observe, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.sample()
        self.stop.set()
        self.thread.join(timeout=1)
        seconds = perf_counter() - self.started
        cpu = sum(self._cpu(h) - baseline for h, baseline in self.handles.values()) if self.supported else None
        self.result = dict(measurement='Windows native process-tree counters, 100ms samples' if self.supported else 'unavailable',
                           cpu_seconds=cpu, average_cpu_cores=cpu / seconds if cpu is not None else None,
                           average_machine_cpu_percent=100 * cpu / seconds / os.cpu_count() if cpu is not None else None,
                           peak_summed_working_set_mib=self.peak_rss / 2**20 if self.supported else None,
                           peak_summed_private_commit_mib=self.peak_private / 2**20 if self.supported else None,
                           memory_note='Working-set sum counts shared pages repeatedly; sampled peaks can miss brief spikes.')
        if self.supported:
            for handle, _ in self.handles.values():
                self.kernel.CloseHandle(handle)


def hardware():
    result = dict(logical_cpus=os.cpu_count())
    if os.name == 'nt':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            result['processor'] = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        class Status(ctypes.Structure):
            _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD)] + [
                (name, ctypes.c_ulonglong) for name in ('total', 'available', 'page_total', 'page_available', 'virtual_total', 'virtual_available', 'extended')]
        status = Status()
        status.length = ctypes.sizeof(status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        result['ram_gib'] = status.total / 2**30
    return result
