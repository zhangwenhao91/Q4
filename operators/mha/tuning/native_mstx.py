"""Host MSTX range via the public CANN C ABI, independent of torch profiler state."""
import ctypes
import os
from pathlib import Path


class NativeMstx:
    def __init__(self,library,description):
        self.library=library
        self.description=description
        # CANN ms_tools_ext.h: uint64_t mstxRangeStartA(const char*, aclrtStream);
        # void mstxRangeEnd(uint64_t). A null stream marks a host range.
        self.start=library.mstxRangeStartA
        self.end=library.mstxRangeEnd
        self.start.argtypes=[ctypes.c_char_p,ctypes.c_void_p]
        self.start.restype=ctypes.c_uint64
        self.end.argtypes=[ctypes.c_uint64]
        self.end.restype=None

    def range_start(self,message,stream=None):
        if stream is not None:
            raise ValueError('Native MSTX adapter uses a host range with a null stream')
        return int(self.start(message.encode('utf-8'),None))

    def range_end(self,range_id):
        self.end(range_id)


def load_native_mstx():
    # Honor a collector's preload/interposed symbols before opening another library.
    process=ctypes.CDLL(None)
    try:
        return NativeMstx(process,dict(symbol_scope='RTLD_DEFAULT',library='process/preloaded symbols'))
    except AttributeError:
        pass
    candidates=[]
    home=os.environ.get('ASCEND_HOME_PATH')
    if home:
        candidates.extend([Path(home)/'lib64/libms_tools_ext.so',
                           Path(home)/'tools/mstx/lib64/libms_tools_ext.so'])
    candidates.extend(Path(entry)/'libms_tools_ext.so'
                      for entry in os.environ.get('LD_LIBRARY_PATH','').split(os.pathsep) if entry)
    errors=[]
    for path in dict.fromkeys(candidates):
        if not path.is_file():
            continue
        try:
            loaded=ctypes.CDLL(str(path),mode=getattr(ctypes,'RTLD_GLOBAL',0))
            # Resolve through global scope again, preserving a collector's hooks.
            adapter=NativeMstx(ctypes.CDLL(None),dict(symbol_scope='RTLD_DEFAULT',library=str(path)))
            adapter.loaded_library=loaded  # Keep dlopen handle alive for the entire range.
            return adapter
        except (OSError,AttributeError) as error:
            errors.append(f'{path}: {error}')
    raise RuntimeError('Cannot resolve CANN mstxRangeStartA/mstxRangeEnd. '
                       +' | '.join(errors or ['No matching CANN library found']))
