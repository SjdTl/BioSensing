#%%
import numpy as np
import pandas as pd
import sys
import json
import time
from telnetlib import Telnet


tn=Telnet('localhost',13854)

start=time.perf_counter()

i=0
tn.write(str.encode('{"enableRawOutput": true, "format": "Json"}'))

eSenseDict={'attention':0, 'meditation':0}
waveDict={'lowGamma':0, 'highGamma':0, 'highAlpha':0, 'delta':0, 'highBeta':0, 'lowAlpha':0, 'lowBeta':0, 'theta':0}
signalLevel=0

while time.perf_counter() - start < 30:
	blinkStrength=0
	line=tn.read_until(str.encode('\r'))
	if len(line) > 20:
		timediff=time.perf_counter()-start
		# dict=json.loads(line)
		print(line)
tn.close()
	


# %%
