
import pdb 
import pandas as pd 
import os 
import numpy as np 

#from time_series import elan_to_dataframe

import pympi
#text_dir = '/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/TPOT/ElanFiles_fromFTP/'
text_dir ='/run/user/1435515045/gvfs/smb-share:server=terracotta.psychology.pitt.edu,share=projects/Sanchayan/ELAN_output/'

text_dir='/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/TPOT/elan_csv_output/'

#import xml.etree.ElementTree as ET 

def elan_to_df (filename):

	try:
		eaf = pympi.Eaf (filename,'utf-8')
		tiers = list (eaf.get_tier_names())
	except:
		print (filename)

	for idx, name in enumerate(tiers):
		
		data = eaf.get_annotation_data_for_tier (name)
		

		if name != 'default':
			if name.startswith('Parent'):
				print (data)

		'''
		try:
			print (data, name)
		except:
			im,jm,km=data[1]
			continue
		'''
		#if name == 'Parent':
		#	print (data)
		
	#print (tiers)
	return 0


def xmlread(filename):
	

	data=pd.read_csv (filename,sep='\t',header=None)
	
	subject = data[0].values 
	start_time = data[3].values
	end_time = data[5].values 
	duration = data[7].values 
	text = data[8].values 

	try:
		print (text)
		return [subject,start_time,end_time,duration,text]
	except:

		return []

def get_data ():
	txt_files = [ x for x in os.listdir (text_dir) if x.endswith('.txt')]


	family = []
	start_list=[]
	end_list=[]
	duration_list=[]
	subject_list=[]
	text_list=[]
	for files in txt_files:


		filename = text_dir + files 
		#filename = text_dir + '01250_02_ELAN_JR.eaf'
		try:
			data=xmlread (os.path.join(text_dir,filename))
		except:
			pass
		if len(data) < 1:
			pass 
		else:
			[subject, start_time, end_time, duration, text] = data 

			fam = files[0:4]
			subject = np.array(['2' if x=='Parent' else '1' for x in subject])	
			indices = np.argsort(start_time)

			start_time= start_time[indices]
			end_time = end_time [indices]
			duration = duration [indices]
			text = text [indices]
			subject = subject[indices]
			
			start_list.append(start_time)
			end_list.append(end_time)
			duration_list.append(duration)
			text_list.append(text)
			subject_list.append(subject)
			family.append(fam)

	data = {}
	data ['family'] = np.array(family)
	data ['start_time']= np.array(start_list)
	data ['end_time'] = np.array(end_list)
	data ['duration'] = np.array(duration_list)
	data ['subject'] = np.array(subject_list)
	data ['text']= np.array(text_list) 


	return data 

data = get_data()
pdb.set_trace()
np.save ('text_data1.npy', data)

