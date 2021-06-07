from pint import toa
from pint.models import get_model
from astropy.time import Time
from astropy import log
import astropy.units as u
import pint.observatory.topo_obs
import pint
import time
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec
import pytesseract
import cv2
from datetime import datetime
from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline
import logging

def extract_data(image):
	# Load image
	im = cv2.imread(image, cv2.IMREAD_COLOR)
	text = pytesseract.image_to_string(im, config='-l eng --oem 1 --psm 3') #1,3
	text = text.replace(' ', '\n')

	# Fetch PSR name
	psr_name = text.split('PSR_')[1].split('\n')[0]

	# Fetch obs date
	date_obs = text.split('mops\n')[1].split('\n')[0]
	date_obs = ''.join([i if ord(i) < 128 else '-' for i in date_obs]).replace('--', '-').replace('JulI', 'Jul')
	date_obs = (datetime.strptime(date_obs, '%d-%b-%Y') - datetime(1970,1,1)).total_seconds()
	date_obs = date_obs/86400.0 + 40587

	# Fetch best period
	P0_obs = text.split('Period\n')[1].split('\n(ms)')[0]
	P0_obs = float(''.join(i for i in P0_obs if i.isdigit() or i == '.'))

	# Return data
	return [psr_name, date_obs, P0_obs, text]

#print('psr_name:'+str(img_data[0]))
#print('date_obs:'+str(img_data[1]))
#print('P0_obs:'+str(img_data[2]))
#print(img_data[3])
for filename in os.listdir('data'):
	if ('329+54' in filename or '529_54' in filename) and (filename.endswith('.jpg') or filename.endswith('.png')):
		img_data = extract_data('data/'+filename)
		break

logging.getLogger('pint.observatory.topo_obs').setLevel(logging.ERROR)
logging.getLogger('pint').setLevel(logging.ERROR)
logging.getLogger('toa').setLevel(logging.ERROR)
logging.getLogger('get_model').setLevel(logging.ERROR)
log.setLevel('ERROR')

plt.rcParams.update({'font.size': 12})

# Define pulsar name
#psr_name = '0531+21' #'B0531+21' #'J0332+5434'

# Define observatory location
observatory = 'GBO'

if observatory == 'GBO':
	pint.observatory.topo_obs.TopoObs(
		observatory,
		aliases=['GBT'],
		itrf_xyz=[882589.638, -4924872.319, 3943729.355]
	)
elif observatory == 'ATA':
        pint.observatory.topo_obs.TopoObs(
                observatory,
                aliases=['HCRO'],
                itrf_xyz=[-2524263.18,-4123529.78,4147966.36]
        )

# Define center frequency
frequency = 1420 # Must be in MHz

# Load .par file
os.system('psrcat -db_file psrcat.db -e2 '+img_data[0]+' |grep -v ELONG|grep -v ELAT >psr.par 2>/dev/null')
par_file = 'psr.par'

# Obtain current datetime
epoch = time.time()
date = epoch/86400.0 + 40587 #Current datetime in MJD

# Compute topocentric spin period
#dt_array = []
#P0_array = []

dt_offset = []
P0_offset = []

model = get_model(par_file)

dt_array = np.loadtxt(model.PSR.value+'_dt.txt', dtype=int).tolist()
P0_array = np.loadtxt(model.PSR.value+'_P0.txt', dtype=float).tolist()

dt_obs = np.loadtxt(model.PSR.value+'_dt_obs.txt', dtype=int).tolist()
P0_obs = np.loadtxt(model.PSR.value+'_P0_obs.txt', dtype=float).tolist()

step = 1
"""
for dt in range(56658, 59215, step): #range(int(date)-1825, int(date)-1825, step):
	print(59215-dt)
	dates = [Time(dt,format='mjd')]
	toa_entry = toa.TOA(dates[0], obs=observatory, freq=frequency)
	toas = toa.get_TOAs_list([toa_entry])
	F0_apparent = model.d_phase_d_toa(toas)
	dt_array.append(dt)
	P0_array.append(1000/F0_apparent.value[0])

np.savetxt(model.PSR.value+'_dt.txt', dt_array, fmt='%.9g')
np.savetxt(model.PSR.value+'_P0.txt', P0_array, fmt='%.9g')
"""
# Barycentric spin frequency
F0 = float(model.F0.quantity*u.s)

# Plot data
fig = plt.figure(figsize=(10, 7))
gs = GridSpec(2, 1)

ax1 = fig.add_subplot(gs[0, 0])
ax1_secondary = ax1.twinx()
ax2 = fig.add_subplot(gs[1, 0])

ax1.plot(dt_array, P0_array, linewidth=2, c='C0', label='Predicted (Step = '+str(step)+')')
#ax1.axhline(y=1/F0, label='Barycentric', color='brown', linestyle='--', linewidth=2)
ax1.set_title('Topocentric Spin Period for PSR '+model.PSR.value+' ('+observatory+')')
ax1.set_ylabel('Topocentric Period (ms)')
ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.9g'))
ax1.get_shared_x_axes().join(ax1, ax2)
ax1.set_xticklabels([])

ax1_secondary.set_ylabel('Topocentric Frequency (Hz)', labelpad=5)
ax1_secondary.tick_params(axis='y')
ax1_secondary.set_ylim(1000/np.min(P0_array), 1000/np.max(P0_array))
ax1_secondary.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.9g'))
ax1.grid(alpha=.4,linestyle='--')

ax2.set_xlim(np.min(dt_array), np.max(dt_array))
ax2.set_xlabel('Date (MJD)')
ax2.set_ylabel('Spin Offset (ppm)')
ax2.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.9g'))
ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.9g'))
ax2.grid(alpha=.4,linestyle='--')
"""
dt_obs = []
P0_obs = []

flag = True
for filename in os.listdir('data'):
	if ('329+54' in filename or '329_54' in filename) and (filename.endswith('.jpg') or filename.endswith('.png')):
		print('[*] Currently analyzing '+filename+'...')
		try:
			img_data = extract_data('data/'+filename)
			if flag:
				ax1.plot(img_data[1], img_data[2], linewidth=2, marker='o', markersize=5, c='C1', label='Observed')
				flag = False
			else:
				ax1.plot(img_data[1], img_data[2], linewidth=2, marker='o', markersize=5, c='C1')
			#index = dt_array.index(int(img_data[1]))
			dt_obs.append(img_data[1])
			P0_obs.append(img_data[2])
		except Exception as e:
			print(e)
			print('[-] *********************** IMAGE ANALYSIS FAILED. SKIPPING... ***********************')
"""
#P0_obs[2] = np.nan
#print(P0_obs)
try:
        np.savetxt(model.PSR.value+'_dt_obs.txt', np.array(dt_obs), fmt='%.9g')
        np.savetxt(model.PSR.value+'_P0_obs.txt', np.array(P0_obs), fmt='%.9g')
except Exception as e:
        print(e)

ax1.plot(dt_obs, P0_obs, '.', linewidth=2, marker='o', markersize=5, c='C1', label='Observed (NRAO 20m)')
P0_offset = []
for date in dt_obs:
	try:
		index = dt_array.index(date)
		P0_offset.append(((P0_array[index]/P0_obs[dt_obs.index(date)]) - 1) * 1e6)
	except Exception as e:
		print(e)
###P0_offset.append(((P0_array[index]/img_data[2]) - 1) * 1e6)


"""
#P0_array[:] = [x / 1000 for x in P0_array]
#dt_array = list(set(dt_array).intersection(dt_offset))
mismatch = []
print('size of dt_array:')
print(np.size(dt_array))
print('size of dt_offset:')
print(np.size(dt_offset))
for d in dt_array:
	for i in range(dt_offset.count(d)):
		mismatch.append(dt_offset.index(d))
#print(mismatch)
print('--- size of mismatch')
print(np.size(mismatch))
print('--- mismatch:')
print(mismatch)

P0_array_new = []
for i in mismatch:
	#for j in [k for k, x in enumerate(mismatch) if x == i]:
	P0_array_new.append(P0_array[mismatch.index(i)])
print(P0_array_new)
#P0_array_new = np.delete(P0_array, mismatch)

print('--- size of P0_array:')
print(np.size(P0_array))
print('--- size of P0_array_new:')
print(np.size(P0_array_new))
print('--- size of P0_offset:')
print(np.size(P0_offset))

#	if dt_array
#	if P0_offset[index]
#		del a[1]

ax1.plot(dt_offset, ((1000000)*np.array(P0_array_new))/(np.array(P0_offset)+(1000000)))
ax1.plot(dt_offset, [x/y for x, y in zip([i * 1e6 for i in P0_array], [j+1e6 for j in P0_offset])], linewidth=2, marker='o', markersize=5, c='C1', label='Observed')
"""

ax2.plot(dt_obs, P0_offset, '-', linewidth=2, marker='o', markersize=5, c='C2')
#ax1.set_ylim(714.45, 714.6)
#ax2.set_ylim(-2.5, 2.5)

# Add Legend
ax1.legend()

# Output plot
#plt.show()
plt.tight_layout()
plt.subplots_adjust(hspace=0.05)
plt.savefig('predict.png')
