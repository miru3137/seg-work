import ctypes

lib = ctypes.cdll.LoadLibrary('build/libsegwork.so')
lib.say_hello()
