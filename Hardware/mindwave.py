#%%
import numpy as np
import pandas as pd
import sys
import json
import time
from telnetlib import Telnet

# Initializing the arrays required to store the data.
attention_values = np.array([])
meditation_values = np.array([])
delta_values = np.array([])
theta_values = np.array([])
lowAlpha_values = np.array([])
highAlpha_values = np.array([])
lowBeta_values = np.array([])
highBeta_values = np.array([])
lowGamma_values = np.array([])
highGamma_values = np.array([])
blinkStrength_values = np.array([])
time_array = np.array([])

tn=Telnet('localhost',13854)

start=time.perf_counter()

i=0
tn.write(str.encode('{"enableRawOutput": true, "format": "Json"}'))


outfile="null"
if len(sys.argv)>1:
	outfile=sys.argv[len(sys.argv)-1]
	outfptr=open("EEG.txt",'w')

eSenseDict={'attention':0, 'meditation':0}
waveDict={'lowGamma':0, 'highGamma':0, 'highAlpha':0, 'delta':0, 'highBeta':0, 'lowAlpha':0, 'lowBeta':0, 'theta':0}
signalLevel=0

while time.perf_counter() - start < 30:
	blinkStrength=0
	line=tn.read_until(str.encode('\r'))
	if len(line) > 20:	
		timediff=time.perf_counter()-start
		dict=json.loads(str(line))
		if "poorSignalLevel" in dict:
			signalLevel=dict['poorSignalLevel']
		if "blinkStrength" in dict:
			blinkStrength=dict['blinkStrength']
		if "eegPower" in dict:
			waveDict=dict['eegPower']
			eSenseDict=dict['eSense']
		outputstr=str(timediff)+ ", "+ str(signalLevel)+", "+str(blinkStrength)+", " + str(eSenseDict['attention']) + ", " + str(eSenseDict['meditation']) + ", "+str(waveDict['lowGamma'])+", " + str(waveDict['highGamma'])+", "+ str(waveDict['highAlpha'])+", "+str(waveDict['delta'])+", "+ str(waveDict['highBeta'])+", "+str(waveDict['lowAlpha'])+", "+str(waveDict['lowBeta'])+ ", "+str(waveDict['theta'])
		time_array = np.append(time_array, [timediff])
		blinkStrength_values = np.append(blinkStrength_values, [blinkStrength])
		lowGamma_values = np.append(lowGamma_values, [waveDict['lowGamma']])
		highGamma_values = np.append(highGamma_values, [waveDict['highGamma']])
		highAlpha_values = np.append(highAlpha_values, [waveDict['highAlpha']])
		delta_values = np.append(delta_values, [waveDict['delta']])
		lowBeta_values = np.append(lowBeta_values, [waveDict['lowBeta']])
		highBeta_values = np.append(highBeta_values, [waveDict['highBeta']])
		theta_values = np.append(theta_values, [waveDict['theta']])
		lowAlpha_values = np.append(lowAlpha_values, [waveDict['lowAlpha']])
		attention_values = np.append(attention_values, [eSenseDict['attention']])
		meditation_values = np.append(meditation_values, [eSenseDict['meditation']])
		print (outputstr)
		if outfile!="null":	
			outfptr.write(outputstr+"\n")		

person_name = input('Enter the name of the person: ')
blink_label = input('Enter left or right eye blink(1 for left, 2 for right): ')
time_starting = input('When does TGC start: ')
lefty_righty = input('Is the person left-handed or right-handed: ')
time_blinking = input('The time of the blink: ')

# Data Recorded for a single person
data_row = pd.DataFrame({'Name': person_name, 'attention': [attention_values], 'meditation': [meditation_values], 'delta': [delta_values], 'theta': [theta_values], 'lowAlpha': [lowAlpha_values], 'highAlpha': [highAlpha_values], 'lowBeta': [lowBeta_values], 'highBeta': [highBeta_values],
                          'lowGamma':[lowGamma_values] , 'highGamma': [highGamma_values], 'blinkStrength': [blinkStrength_values], 'time': [time_array], 'LOR': blink_label})

# Reading the data stored till now
dataset = pd.read_csv('data_eeg.csv')

from numpy import nan as Nan

dataset = dataset.append(pd.Series([blink_label, person_name, [attention_values], [blinkStrength_values], [delta_values]
	, [highAlpha_values], [highBeta_values], [highGamma_values], [lowAlpha_values], [lowBeta_values], [lowGamma_values], [meditation_values], 
	[theta_values], [time_array], time_starting, lefty_righty, time_blinking], index=['LOR', 'Name', 'attention', 'blinkStrength', 'delta', 'highAlpha', 'highBeta', 'highGamma', 'lowAlpha', 'lowBeta', 'lowGamma', 'meditation', 'theta', 'time', 'time_start', 'LTYRTY', 'time_blink']),  ignore_index = True)

#Appending and storing the data in the same csv
#dataset.append(data_row)
dataset.to_csv('data_eeg.csv')      
    
tn.close()
#outfptr.close()

# %%
