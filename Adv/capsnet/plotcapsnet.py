import os
import argparse
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument('-d','--download', dest='download',action='store_true')
cli_args = parser.parse_args()


if 'screenlog.0' not in os.listdir('./'):
	print "File Missing, Initiaing Downlaod."
	cli_args.download = True

if cli_args.download :
	server = 'ec2-54-148-70-252.us-west-2.compute.amazonaws.com'
	servDir = '/home/ubuntu/capsnet/screenlog.0'
	localDir = os.getcwd()
	os.chdir('/Users/muditverma/AWSexperiments')
	os.system('scp -i "deep_learning_ami.pem" ubuntu@'+server+':'+servDir+' '+localDir)
	os.chdir(localDir)
else :
	pass

with open('screenlog.0','r') as f:
	r = f.read()


r= r.split('Train Epoch:')
data = {}
epochtmp = {}
for rr in r[1:]:

	if '%' in rr:
		# print 'PERC'
		epoch = int(rr.split(' ')[1])
		per = int(rr.split(' ')[3][1:-3])
		loss = float(rr.split(' ')[6][:9])
		epochvals = [per,loss]

		try :
			epochtmp[epoch].append(epochvals)
		except:
			epochtmp[epoch] = [epochvals]

	if 'Time Taken' in rr.split('\n')[1]:
		# print 'TT'
		epoch = int(rr.split('\n')[1].split(' ')[4])
		time = float(rr.split('\n')[1].split(' ')[6])

		epochtmp[epoch].append(time)

		acc = int(rr.split('Set:')[1].split('Accuracy')[1].split(' ')[-1][1:-4])
		# print rr.split('Set:')[1]
		epochtmp[epoch].append(acc)
	# if 'Set:' in rr.split('\n')[2]:
	# 	pass
	# 	# print 'TS'
	# print '---------'
		# print rr[:-2]

# pprint(epochtmp)



from matplotlib import pyplot as plt

x = []
y = []
xa = []
acc = []
ttime = []
for k,v in epochtmp.iteritems():
	for vv in v:
		try:
			val = k*100+vv[0]
			x.append(val)
			y.append(vv[1])
		except :
			pass
	if type(v[-2]) is not list and type(v[-1]) is not list:
		xa.append(k)
		acc.append(v[-1])
		ttime.append(v[-2])
	else:
		continue



fig,(ax1,ax2,ax3) = plt.subplots(1,3)



ax1.plot(x,y)

# ax1.title='Loss'

ax2.plot(xa,acc)
# ax2.title='Acc'

ax3.plot(range(len(ttime)),ttime)
# ax3.title='Time'

plt.title('LOSS-ACCURACY-TIME TAKEN')
plt.show()

















