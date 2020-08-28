import pdb 
import pandas as pd 
import os 
import numpy as np
from scipy.io import loadmat
from scipy import signal 
from scipy.stats import iqr 


from joblib import Parallel, delayed
import multiprocessing




text_dir = '/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/TPOT/ElanFiles_fromFTP/'

life_code_dir='/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/TPOT/LIFE/LIFE Coding Stop Frame Constructs/PSI Task/TXT Files/StartFrame/'
zface_dir ='/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/TPOT/20191111_Converted_pipeline_output/'
au_dir	='/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/TPOT/20191111_Converted_pipeline_output/AU_input/formatted/occurrence/'
au_intensity_dir = '/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/TPOT/20191111_Converted_pipeline_output/AU_input/formatted/intensity/'
covarep_dir='/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/TPOT/audio_processed/covarep/'
opensmile_egemaps_dir='/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/TPOT/audio_processed/opensmile_eGeMAPSv01a/'
opensmile_vad_dir='/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/TPOT/audio_processed/opensmile_vad_opensource/'
opensmile_prosody_dir='/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/TPOT/audio_processed/opensmile_prosodyAcf/'
volume_dir='/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/TPOT/audio_processed/volume'
upsampled_dir= '/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/TPOT/upsampled/'

zface_list=np.array([name for name in os.listdir(zface_dir) if name.endswith('fit.mat')])
au_list=np.array([name for name in os.listdir(au_dir) if name.endswith('.mat')])
au_intensity_list=np.array([name for name in os.listdir(au_intensity_dir) if name.endswith ('.mat')])
covarep_list=np.array([name for name in os.listdir(covarep_dir) if name.endswith('.hdf')])
opensmile_egemaps_list=np.array([name for name in os.listdir(opensmile_egemaps_dir) if name.endswith('.hdf')])
opensmile_vad_list=np.array([name for name in os.listdir(opensmile_vad_dir) if name.endswith('.hdf')])
opensmile_prosody_list= np.array([name for name in os.listdir(opensmile_prosody_dir) if name.endswith('.hdf')])
volume_list=np.array([name for name in os.listdir(volume_dir) if name.endswith ('.hdf')])
upsampled_list=np.array([name for name in os.listdir(upsampled_dir) if name.endswith('.hdf')])

def load_text_data():
	data = np.load ('text_data1.npy',allow_pickle=True).item()
	
	family= data ['family'] 
	start_time=data ['start_time']
	end_time=data ['end_time'] 
	duration=data ['duration'] 
	subject=data ['subject'] 
	text=data ['text'] 

	return [family, start_time, end_time, duration, subject, text]


def get_unique_filenames(files):
	lookuptable={}
	unique_list=[]
	sanity_list=[]
	for name in files:
		try:
			fam,task,coder= name.split('.')[0].split('_')		
		except:
			print("Exception",name)

		if len(fam) > 6: 
			fam_id=fam[2:6]
			sub_id=fam[6]
		else:
			
			fam_id=fam[1:5]
			sub_id=fam[5]

		if fam+task in lookuptable:
			#print (coder, lookuptable[fam+task])
			pass
		else:
			lookuptable[fam+task]=coder
			unique_list.append(name)
		#lookuptable[fam+task]=1
			
		#sanity_list.append(fam+task)
	
	#sanity_list=np.array(sanity_list)

	#print (len(sanity_list),len(unique_list),len(np.unique(sanity_list)))

	return unique_list

def does_zface_exist(filename):
	name = zface_list[zface_list==filename]
	if len(name):
		return True
	return False 

def does_AU_exist (filename):
	name = au_list[au_list==filename]
	if len(name):
		return True
	return False

def does_covarep_exist(filename):
	name =covarep_list[covarep_list==filename]

	if len(name):
		return True
	return False
def does_au_intensity_exist(filename):
	'''
	for intensity_files in au_intensity_list:
		

		family_subject,task,video = intensity_files.split('.mat')[0].split('_')

		print (family,subject,video)
		input ('')
	'''
	name = au_intensity_list[au_intensity_list==filename]
	
	if len(name):
		return True
	return False
def does_opensmile_egemaps_exist (filename):
	name = opensmile_egemaps_list[opensmile_egemaps_list==filename]
	

	if len(name):
		return True
	return False
def does_opensmile_prosody_exist (filename):
	name = opensmile_prosody_list[opensmile_prosody_list==filename]
	
	
	if len(name):
		return True
	return False
def does_opensmile_vad_exist (filename):
	name = opensmile_vad_list[opensmile_vad_list==filename]
	
	if len(name):
		return True
	return False
def does_volume_exist (filename):
	name= volume_list [ volume_list == filename]
	if len(name):
		return True 
	return False 
def does_upsampled_exist(filename):
	name= upsampled_list[ upsampled_list == filename]
	if len(name):
		return True 
	return False  





def expand_filenames (family_list,gender_list,filelist):

	[turn_family, turn_start_time, turn_end_time, turn_duration, turn_subject, turn_text]= load_text_data()

	parent = '2'
	child  = '1'

	item_list=[]
	for idx, family in enumerate(family_list):

		parent_name = family + parent 
		child_name  = family + child

		check_parent = np.array([x.find(parent_name) for x in unique_list])
		check_child  = np.array([x.find(child_name) for x in unique_list])
	
		check_parent , check_child = np.where(check_parent>-1)[0], np.where(check_child>-1)[0]
	
		if len(check_parent) == 1 and len(check_child)==1 :

			parent_life = unique_list[check_parent[0]] #-Getting life codes
			child_life  = unique_list[check_child[0]]
		
			 

			p_zface_filename= family + parent +'_02_01_fit.mat'
			p_au_filename=    family + parent + '_02_01_au_out.mat'
			p_au_intensity_filename= family + parent + '_02_01.mat'
			p_covarep_filename= 'TPOT_' + family + '_' + parent + '_2.hdf'
			p_opensmile_filename_lld= 'TPOT_' + family + '_' + parent + '_2_lldcsvoutput.hdf'
			p_opensmile_filename_csv= 'TPOT_'  + family + '_' + parent + '_2_csvoutput.hdf'

			p_opensmile_prosody_file= 'TPOT_'  + family + '_' + parent + '_2_csvoutput.hdf'
			p_opensmile_vad_file    = 'TPOT_'  + family + '_' + parent + '_2_csvoutput.hdf'
			p_volume_file 		  = 'TPOT_'  + family + '_' + parent + '_2.hdf'

			p_upsampled_file 		  = 'TPOT_'  + family + '_' + parent + '_2.hdf'

			parent_items = [p_zface_filename,p_au_filename,p_au_intensity_filename,p_covarep_filename,p_opensmile_filename_lld,p_opensmile_filename_csv,p_opensmile_prosody_file,\
							p_opensmile_vad_file, p_volume_file, p_upsampled_file]

			c_zface_filename= family + child +'_02_01_fit.mat'
			c_au_filename=    family + child + '_02_01_au_out.mat'
			c_au_intensity_filename= family + child + '_02_01.mat'
			c_covarep_filename= 'TPOT_' + family + '_' + child + '_2.hdf'
			c_opensmile_filename_lld= 'TPOT_' + family + '_' + child + '_2_lldcsvoutput.hdf'
			c_opensmile_filename_csv= 'TPOT_'  + family + '_' + child + '_2_csvoutput.hdf'

			c_opensmile_prosody_file= 'TPOT_'  + family + '_' + child + '_2_csvoutput.hdf'
			c_opensmile_vad_file    = 'TPOT_'  + family + '_' + child + '_2_csvoutput.hdf'
			c_volume_file 		  = 'TPOT_'  + family + '_' + child + '_2.hdf'

			c_upsampled_file 		  = 'TPOT_'  + family + '_' + child + '_2.hdf'
			
			child_items = [c_zface_filename,c_au_filename,c_au_intensity_filename,c_covarep_filename,c_opensmile_filename_lld,c_opensmile_filename_csv,c_opensmile_prosody_file,\
							c_opensmile_vad_file,c_volume_file,c_upsampled_file]

			p_zface_exist= does_zface_exist(p_zface_filename)
			p_au_exist=    does_AU_exist(p_au_filename)		
			p_au_intensity_exist= does_au_intensity_exist (p_au_intensity_filename)
			p_covarep_exist= does_covarep_exist (p_covarep_filename)
			p_opensmile_lld_exist= does_opensmile_egemaps_exist (p_opensmile_filename_lld)
			p_opensmile_csv_exist= does_opensmile_egemaps_exist (p_opensmile_filename_csv)
			p_opensmile_prosody_exist= does_opensmile_prosody_exist (p_opensmile_prosody_file)
			p_opensmile_vad_exist =  does_opensmile_vad_exist (p_opensmile_vad_file)
			p_volume_exist        =  does_volume_exist (p_volume_file)
			p_upsampled_exist     =  does_upsampled_exist (p_upsampled_file)

			c_zface_exist= does_zface_exist(c_zface_filename)
			c_au_exist=    does_AU_exist(c_au_filename)		
			c_au_intensity_exist= does_au_intensity_exist (c_au_intensity_filename)
			c_covarep_exist= does_covarep_exist (c_covarep_filename)
			c_opensmile_lld_exist= does_opensmile_egemaps_exist (c_opensmile_filename_lld)
			c_opensmile_csv_exist= does_opensmile_egemaps_exist (c_opensmile_filename_csv)
			c_opensmile_prosody_exist= does_opensmile_prosody_exist (c_opensmile_prosody_file)
			c_opensmile_vad_exist =  does_opensmile_vad_exist (c_opensmile_vad_file)
			c_volume_exist        =  does_volume_exist (c_volume_file)
			c_upsampled_exist     =  does_upsampled_exist (c_upsampled_file)

	
			
			if family in turn_family and p_zface_exist and c_zface_exist and p_au_exist and c_au_exist and p_au_intensity_exist and c_au_intensity_exist and p_covarep_exist and c_covarep_exist \
			and p_opensmile_vad_exist and c_opensmile_vad_exist and p_opensmile_prosody_exist and c_opensmile_prosody_exist and p_opensmile_csv_exist and c_opensmile_csv_exist \
			and p_opensmile_lld_exist and c_opensmile_lld_exist and p_volume_exist and c_volume_exist and p_upsampled_exist and c_upsampled_exist:

				
				
				turn_idx= list(turn_family).index(family)
				turn_items= [family,turn_start_time[turn_idx], turn_end_time[turn_idx], turn_duration[turn_idx], turn_subject[turn_idx],turn_text[turn_idx] ]
				
				items = [parent_life,child_life,parent_items,child_items, turn_items]
				item_list.append(items)

	return item_list

def read_codes(life_code_file,fps=30):


	agg=0
	dys=0
	pos=0
	other=0

	
		
	filename= life_code_dir + life_code_file

	f=open (filename,"r")
	lines=f.readlines()

	df= pd.read_table(filename,header=None,usecols=[0,2,3],names=['category','start_time','onset_seconds'])
	
	category=np.array(df['category'])
	onset_seconds=np.array(df['onset_seconds'])

	sorted_indices=np.argsort(onset_seconds)

	sorted_onset_seconds=onset_seconds[sorted_indices]
	#onset_seconds_shifted
	#np.pad(sorted_onset_seconds,(1,0),'constant',constant_values=(0))[:-1]
	
	event_duration=sorted_onset_seconds[1:] - sorted_onset_seconds[:-1]     # Duration of the event . This is in seconds 
	
	event_start=(sorted_onset_seconds * fps).astype('int')[:-1]           	#Start frame of the event . We are eliminating the last construct since it has no ened 
	event_end=(np.multiply(sorted_onset_seconds[1:], fps) -0).astype('int') #End Frame of the event which is 1 frame before the start frame of next event
	
	sorted_category=category[sorted_indices][:-1]

	start_time_in_seconds= sorted_onset_seconds [:-1]
	end_time_in_seconds= sorted_onset_seconds[1:]
	
	#assert event_start.shape == event_end.shape
	#print (event_start[-2:],event_end[-2:],event_duration[-2:]*fps)
	#input ('')

	'''
	"Compute count of the constructs"
	agg += np.sum(category=='Aggressive')
	dys += np.sum(category=='Dysphoric')
	pos += np.sum(category=='Positive')
	other += np.sum(category=='Other')
	'''
	assert event_start.shape == start_time_in_seconds.shape

	return [sorted_category,event_start,event_end,start_time_in_seconds,end_time_in_seconds]

def convert_construct_to_digits(construct_list):

	res_list=np.zeros((len(construct_list)))
	for idx,name in enumerate(construct_list):

		if name == 'Aggressive' or name =='aggressive':
			res_list[idx] = 0
		elif name =='Dysphoric' or name == 'dysphoric':
			res_list[idx] = 1
		elif name == 'Positive' or name == 'positive':
			res_list[idx] = 2
		elif name == 'other' or name == 'Other':
			res_list[idx] = 3
		else:
			print ("Something is wrong with the constructs")
			
	return res_list

def combine_codes (parent, child):

	p = '2'
	c = '1'
	p_construct_labels,p_event_start,p_event_end,p_start_time_in_seconds,p_end_time_in_seconds= parent
	c_construct_labels,c_event_start,c_event_end,c_start_time_in_seconds,c_end_time_in_seconds= child
	
	p_sub= np.repeat(p, len(p_construct_labels))
	c_sub= np.repeat(c, len(c_construct_labels))

	construct_labels= np.concatenate((p_construct_labels,c_construct_labels))
	sub= np.concatenate((p_sub,c_sub))
	onset_time = np.concatenate((p_start_time_in_seconds,c_start_time_in_seconds))
	event_start= np.concatenate((p_event_start,c_event_start))

	indices = np.argsort(onset_time)
	onset_time= onset_time[indices]
	event_start= event_start[indices]
	sub= sub[indices]
	construct_labels= construct_labels[indices]
	start_time, end_time = onset_time[:-1], onset_time[1:]
	start_frame, end_frame= event_start[:-1], event_start[1:]


	return [construct_labels[:-1], sub[:-1], start_frame, end_frame, start_time, end_time]  


def handle_covarep_df (covarep_df, start, end,skip=25):
	res_vector= np.empty(0)

	data = covarep_df 
	#print ("Pass",start,end)
	covarep_vowelSpace= np.array(data['covarep_vowelSpace'][start:end:skip]).reshape(-1,1)
	covarep_MCEP_0= np.array(data['covarep_MCEP_0'][start:end:skip]).reshape(-1,1)
	covarep_MCEP_1= np.array(data['covarep_MCEP_1'][start:end:skip]).reshape(-1,1)
	covarep_VAD= np.array(data['covarep_VAD'][start:end:skip]).reshape(-1,1)
	covarep_f0 = np.array(data['covarep_f0'][start:end:skip]).reshape(-1,1)
	covarep_NAQ = np.array(data['covarep_NAQ'][start:end:skip]).reshape(-1,1)
	covarep_QOQ= np.array(data['covarep_QOQ'][start:end:skip]).reshape(-1,1)
	covarep_MDQ= np.array(data['covarep_MDQ'][start:end:skip]).reshape(-1,1)
	covarep_peakSlope= np.array(data['covarep_peakSlope'][start:end:skip]).reshape(-1,1)
	covarep_F1= np.array(data['covarep_F1'][start:end:skip]).reshape(-1,1)
	covarep_F2= np.array(data['covarep_F2'][start:end:skip]).reshape(-1,1)

	
	 
	res_vector= np.concatenate((covarep_vowelSpace,covarep_MCEP_0,covarep_MCEP_1,covarep_VAD,covarep_f0,covarep_NAQ,covarep_QOQ,covarep_MDQ,covarep_peakSlope,covarep_F1,covarep_F2),axis=1)

	
	if res_vector.size :
		med_cov= np.median(res_vector,axis=0)
		iqr_cov= iqr(res_vector[:,4:],axis=0)
	else:
		med_cov= np.zeros(res_vector.shape[1])
		iqr_cov= np.zeros(res_vector[:,4:].shape[1])

	
	res_vector=np.concatenate((med_cov,iqr_cov))
	return res_vector
def handle_opensmile_df (opensmile_dfs, start,end,skip=25):

	res_vector= np.empty(0)
	opensmile_lld_df,opensmile_csv_df,prosody_df,vad_df = opensmile_dfs

	
	csv_vector= opensmile_csv_df[start:end:skip]
	lld_vector= opensmile_lld_df[start:end:skip]
	prosody_vector= prosody_df [start:end:skip]
	vad_vector    = vad_df [start:end:skip]

	if csv_vector.size :#and lld_vector.size and prosody_vector.size and vad_vector.size:
	
		med_csv= np.median (csv_vector,axis=0)
		med_lld= np.median (lld_vector,axis=0)
		iqr_lld= iqr(lld_vector,axis=0)
		med_prosody=np.median (prosody_vector,axis=0)
		med_vad= np.median(vad_vector,axis=0)
	else:
		med_csv=np.zeros(csv_vector.shape[1])
		med_lld=np.zeros(lld_vector.shape[1])
		iqr_lld=np.zeros(lld_vector.shape[1])
		med_prosody=np.zeros(prosody_vector.shape[1])
		med_vad=np.zeros(vad_vector.shape[1])
	
	res_vector= np.concatenate((med_csv,med_lld,iqr_lld,med_prosody,med_vad))
	return res_vector

def read_zface_features (zface_filename):

	res_vector=np.empty(0)

	mat=loadmat(os.path.join(zface_dir,zface_filename))
	zface_data = mat['fit']

	no_frames=zface_data.shape[1]
	isTracked_m  = zface_data[0]['isTracked']
	headPose_m   = zface_data[0]['headPose']
	#pts_3d_m= zface_data[0]['pts_3d']
	#pts_2d_m= zface_data[0]['pts_2d']
	pdmPars_m = zface_data[0]['pdmPars']
	no_pdm_parameters = 30
	
	isTracked = np.zeros(no_frames)
	#pts_3d= np.zeros((no_frames,512*3))
	#pts_2d= np.zeros((no_frames,49*2))
	headPose = np.zeros((no_frames,3) )
	pdmPars = np.zeros((no_frames,no_pdm_parameters) )
	

	for ii in range (no_frames):
		isTracked[ii] = isTracked_m[ii][0]
		if isTracked[ii] != 0:
			headPose[ii]  = headPose_m[ii].reshape(1,3)[0]
			pdmPars[ii]   = pdmPars_m[ii].reshape(1,no_pdm_parameters)[0]
			#pts_3d[ii]	  = pts_3d_m[ii].ravel()
			#pts_2d[ii]	  = pts_2d_m[ii].ravel()

	#print (zface_filename,no_frames,start_list,end_list)
	pdmPars = pdmPars[:,:15]
	
	if no_frames < 10000:
		return res_vector
	
	vector=np.concatenate((pdmPars,headPose),axis=1)   #Use this line to add as many zface_features as you want for thee raw zface vector

	
	
	'''
	for idx, (start, end) in enumerate(zip(start_list,end_list)):

		amp_vector= vector[start:end,:]
		vel_vector= amp_vector[1:,:] - amp_vector[:-1,:]
		acc_vector= vel_vector[1:,:] - vel_vector[:-1,:]

		amp_vector= amp_vector [2:,:]
		vel_vector= vel_vector [1:,:]

		
		amp_stats = compute_statistics (amp_vector)
		vel_stats = compute_statistics (vel_vector)
		acc_stats = compute_statistics (acc_vector)

		feature_vector= np.hstack ([amp_stats,vel_stats,acc_stats])

		res_vector= np.vstack ([res_vector,feature_vector]) if res_vector.size else feature_vector
		
	'''	
	'''
	#for items in event_start:
		
	print (headPose.shape,pts_3d.shape,pts_2d.shape,vector.shape)
	input ('Here')
	'''
	
	return vector

def read_AUs (AU_filename):

	res_vector=np.empty(0)

	mat=loadmat(os.path.join(au_dir,AU_filename))
	au_data = mat['occurrence']

	no_frames=len(au_data)
	#print  (AU_filename,no_frames,start_list,end_list)

	if no_frames <10000:
		return res_vector
	
	vector = au_data.copy()

	'''
	for idx, (start,end) in enumerate(zip(start_list,end_list)):
		amp_vector= vector[start:end,:]
		vel_vector= amp_vector[1:,:] - amp_vector[:-1,:]
		#acc_vector= vel_vector[1:,:] - vel_vector[:-1,:]

		amp_vector= amp_vector[1:,:]
		vel_vector= vel_vector[:,:]

		amp_stats = compute_statistics (amp_vector)
		vel_stats = compute_statistics (vel_vector)
		#acc_stats = compute_statistics (acc_vector)

		feature_vector= np.hstack ([amp_stats,vel_stats])

		res_vector= np.vstack ([res_vector,feature_vector]) if res_vector.size else feature_vector
	'''
	return vector

def read_AU_intensity(AU_filename):

	res_vector=np.empty(0)
	mat= loadmat(os.path.join(au_intensity_dir,AU_filename))
	#au_data= mat['intensity']

	au6= mat['AU6'][0][0][0][0].reshape(-1,1)
	au12=mat['AU12'][0][0][0][0].reshape(-1,1)

	au10=mat['AU10'][0][0][0][0].reshape(-1,1)
	au14=mat['AU14'][0][0][0][0].reshape(-1,1)

	#assert len(au6) == len(au12) == len(au10) == len(au14)
	
	no_frames=len(au6)
	if no_frames < 10000:
		return res_vector

	vector= np.concatenate((au6,au10,au12,au14),axis=1)


	'''
	for idx, (start,end) in enumerate(zip(start_list,end_list)):
		amp_vector= vector [start : end , :]

		feature_vector= compute_statistics (amp_vector)

		res_vector= np.vstack ([res_vector,feature_vector]) if res_vector.size else feature_vector
	'''
	return vector

def read_features1(feature_set):
	res_vector = []

	total_audio=[]
	total_video=[]
	total_label= []
	total_speaker=[]
	total_time=[]
	total_frame=[]
	total_filename=[]

	
	for i, items in enumerate(feature_set):
		parent_life, child_life , turn_item = items[0], items[1], items[4]

			
		[p_zface_filename,p_au_filename,p_au_intensity_filename,p_covarep_filename,p_opensmile_filename_lld,p_opensmile_filename_csv,p_opensmile_prosody_file,\
								p_opensmile_vad_file, p_volume_file, p_upsampled_file]= items[2]

		[c_zface_filename,c_au_filename,c_au_intensity_filename,c_covarep_filename,c_opensmile_filename_lld,c_opensmile_filename_csv,c_opensmile_prosody_file,\
								c_opensmile_vad_file,c_volume_file,c_upsampled_file]  = items[3]
		
		[turn_start_time, turn_end_time, turn_duration, turn_subject,turn_text] = turn_item		

		
		parent_tuple=read_codes(parent_life)
		child_tuple =read_codes(child_life)

		[construct_labels, sub, start_frame, end_frame, start_time, end_time]  = combine_codes (parent_tuple,child_tuple)
		

		p_data= pd.read_hdf (os.path.join(upsampled_dir, p_upsampled_file), 'df')
		c_data= pd.read_hdf (os.path.join(upsampled_dir, c_upsampled_file), 'df')


		p_zface_mat,c_zface_mat = read_zface_features(p_zface_filename), read_zface_features(c_zface_filename)
		
		p_end_time, c_end_time = p_data.index[-1], c_data.index[-1]
		

		if p_end_time <540 or c_end_time <540 or p_zface_mat.size==0 or c_zface_mat.size==0:
			continue
		p_au_occ_mat, 	c_au_occ_mat= read_AUs(p_au_filename), read_AUs(c_au_filename)
		
		p_au_int_mat , c_au_int_mat=read_AU_intensity(p_au_intensity_filename), read_AU_intensity(c_au_intensity_filename)
		
		

		#p_zface_df   = p_data [ [col for col in p_data if col.startswith('zface')]]
		#p_au_occ_df  = p_data[[col for col in p_data if col.startswith('au_occ')]]
		#p_au_int_df  = p_data [[col for col in p_data if col.startswith('au_int')]]
		p_covarep_df = p_data [ [col for col in p_data if col.startswith('covarep')] ] #81
		p_opensmile_lld_df= p_data[ [col for col in p_data if col.startswith('GeMAPS_lld')] ] #111
		p_opensmile_csv_df= p_data[ [col for col in p_data if col.startswith('GeMAPS') and not col.startswith('GeMAPS_lld')]]
		p_prosody_df = p_data [ [col for col in p_data if col.startswith('prosody')]]
		p_vad_df    = p_data  [ [col for col in p_data if col.startswith('vad')]]
		p_volume_df  = p_data[  [col for col in p_data if col.startswith('volume')]]

		#c_zface_df   = p_data [ [col for col in p_data if col.startswith('zface')]]
		#c_au_occ_df  = p_data[[col for col in p_data if col.startswith('au_occ')]]
		#c_au_int_df  = p_data [[col for col in p_data if col.startswith('au_int')]]
		c_covarep_df = c_data [ [col for col in c_data if col.startswith('covarep')] ] #81
		c_opensmile_lld_df= c_data[ [col for col in c_data if col.startswith('GeMAPS_lld')] ] #111
		c_opensmile_csv_df= c_data[ [col for col in c_data if col.startswith('GeMAPS') and not col.startswith('GeMAPS_lld')]]
		c_prosody_df = c_data [ [col for col in c_data if col.startswith('prosody')]]
		c_vad_df    = c_data  [ [col for col in c_data if col.startswith('vad')]]
		c_volume_df  = c_data[  [col for col in c_data if col.startswith('volume')]]

		audio_feature_stack=None
		video_feature_stack=None
		for jj , (start_idx, end_idx, st, end ) in enumerate(zip(start_time,end_time,start_frame,end_frame)):
			#start_idx= start 
			#end_idx = end   # Chanege if you need some other sort of starts

			if sub [jj] == '2':
				covarep_vec= handle_covarep_df(p_covarep_df,start_idx,end_idx,skip=1)
				opensmile_vec= handle_opensmile_df([p_opensmile_lld_df, p_opensmile_csv_df, p_prosody_df, p_vad_df],start_idx,end_idx,skip=1)
				zface_vec=compute_statistics(p_zface_mat[st:end])
				au_occ_vec=compute_statistics(p_au_occ_mat[st:end])
				au_int_vec=compute_statistics(p_au_int_mat[st:end])
	
				
			else:
				covarep_vec= handle_covarep_df(c_covarep_df,start_idx,end_idx,skip=1)
				opensmile_vec= handle_opensmile_df([c_opensmile_lld_df, c_opensmile_csv_df, c_prosody_df, c_vad_df],start_idx,end_idx,skip=1)
				zface_vec=compute_statistics(c_zface_mat[st:end])
				au_occ_vec=compute_statistics(c_au_occ_mat[st:end])
				au_int_vec=compute_statistics(c_au_int_mat[st:end])
				
			audio_vec= np.concatenate((covarep_vec,opensmile_vec))
			video_vec=np.concatenate((zface_vec,au_occ_vec,au_int_vec))
			#feature_vec=np.concatenate((video_vec,audio_vec))
			audio_feature_stack= np.vstack([audio_feature_stack, audio_vec]) if audio_feature_stack is not None else audio_vec
			video_feature_stack= np.vstack([video_feature_stack, video_vec]) if video_feature_stack is not None else video_vec
			
				
		digit_labels= convert_construct_to_digits(construct_labels)
		
	
		total_audio.append(audio_feature_stack)
		total_video.append(video_feature_stack)
		total_label.append(digit_labels)
		total_speaker.append(sub)
		total_time.append([start_time,end_time])
		total_frame.append([start_frame,end_frame])
		total_filename.append([parent_life,child_life])
		
		print (i,parent_life, len(total_label),len(total_time),len(total_speaker),  audio_feature_stack.shape, video_feature_stack.shape, len(digit_labels), len(sub))

	
	res={}
	res['audio']= total_audio
	res['video']= total_video
	res['label']= total_label
	res['speaker']= total_speaker
	res['time'] = total_time
	res['frame']= total_frame
	res['filenames']= total_filename

	return res

def read_features(feature_set):
	res_vector = []

	total_audio=[]
	total_video=[]
	total_label= []
	total_speaker=[]
	total_time=[]
	total_frame=[]
	total_filename=[]
	
	total_turn_filename=[]
	total_turn_speaker=[]
	total_turn_time=[]
	total_turn_frame=[]
	total_turn_duration=[]

	
	for i, items in enumerate(feature_set):
		parent_life, child_life , turn_item = items[0], items[1], items[4]


			
		[p_zface_filename,p_au_filename,p_au_intensity_filename,p_covarep_filename,p_opensmile_filename_lld,p_opensmile_filename_csv,p_opensmile_prosody_file,\
								p_opensmile_vad_file, p_volume_file, p_upsampled_file]= items[2]

		[c_zface_filename,c_au_filename,c_au_intensity_filename,c_covarep_filename,c_opensmile_filename_lld,c_opensmile_filename_csv,c_opensmile_prosody_file,\
								c_opensmile_vad_file,c_volume_file,c_upsampled_file]  = items[3]
		
		[turn_family, turn_start_time, turn_end_time, turn_duration, turn_subject,turn_text] = turn_item		

		turn_start_frame, turn_end_frame = (turn_start_time *30).astype('int'), (turn_end_time *30).astype('int')
		
		parent_tuple=read_codes(parent_life)
		child_tuple =read_codes(child_life)

		[construct_labels, sub, start_frame, end_frame, start_time, end_time]  = combine_codes (parent_tuple,child_tuple)
		
		p_data= pd.read_hdf (os.path.join(upsampled_dir, p_upsampled_file), 'df')
		c_data= pd.read_hdf (os.path.join(upsampled_dir, c_upsampled_file), 'df')


		p_zface_mat,c_zface_mat = read_zface_features(p_zface_filename), read_zface_features(c_zface_filename)
		
		p_end_time, c_end_time = p_data.index[-1], c_data.index[-1]
		

		if p_end_time <540 or c_end_time <540 or p_zface_mat.size==0 or c_zface_mat.size==0:
			continue
		p_au_occ_mat, 	c_au_occ_mat= read_AUs(p_au_filename), read_AUs(c_au_filename)
		
		p_au_int_mat , c_au_int_mat=read_AU_intensity(p_au_intensity_filename), read_AU_intensity(c_au_intensity_filename)
		
		

		#p_zface_df   = p_data [ [col for col in p_data if col.startswith('zface')]]
		#p_au_occ_df  = p_data[[col for col in p_data if col.startswith('au_occ')]]
		#p_au_int_df  = p_data [[col for col in p_data if col.startswith('au_int')]]
		p_covarep_df = p_data [ [col for col in p_data if col.startswith('covarep')] ] #81
		p_opensmile_lld_df= p_data[ [col for col in p_data if col.startswith('GeMAPS_lld')] ] #111
		p_opensmile_csv_df= p_data[ [col for col in p_data if col.startswith('GeMAPS') and not col.startswith('GeMAPS_lld')]]
		p_prosody_df = p_data [ [col for col in p_data if col.startswith('prosody')]]
		p_vad_df    = p_data  [ [col for col in p_data if col.startswith('vad')]]
		p_volume_df  = p_data[  [col for col in p_data if col.startswith('volume')]]

		#c_zface_df   = p_data [ [col for col in p_data if col.startswith('zface')]]
		#c_au_occ_df  = p_data[[col for col in p_data if col.startswith('au_occ')]]
		#c_au_int_df  = p_data [[col for col in p_data if col.startswith('au_int')]]
		c_covarep_df = c_data [ [col for col in c_data if col.startswith('covarep')] ] #81
		c_opensmile_lld_df= c_data[ [col for col in c_data if col.startswith('GeMAPS_lld')] ] #111
		c_opensmile_csv_df= c_data[ [col for col in c_data if col.startswith('GeMAPS') and not col.startswith('GeMAPS_lld')]]
		c_prosody_df = c_data [ [col for col in c_data if col.startswith('prosody')]]
		c_vad_df    = c_data  [ [col for col in c_data if col.startswith('vad')]]
		c_volume_df  = c_data[  [col for col in c_data if col.startswith('volume')]]

		audio_feature_stack=None
		video_feature_stack=None
		for jj , (start_idx, end_idx, st, end ) in enumerate(zip(turn_start_time,turn_end_time,turn_start_frame,turn_end_frame)):
			#start_idx= start 
			#end_idx = end   # Chanege if you need some other sort of starts

			if turn_subject [jj] == '2':
				covarep_vec= handle_covarep_df(p_covarep_df,start_idx,end_idx,skip=1)
				opensmile_vec= handle_opensmile_df([p_opensmile_lld_df, p_opensmile_csv_df, p_prosody_df, p_vad_df],start_idx,end_idx,skip=1)
				zface_vec=compute_statistics(p_zface_mat[st:end])
				au_occ_vec=compute_statistics(p_au_occ_mat[st:end])
				au_int_vec=compute_statistics(p_au_int_mat[st:end])
	
				
			else:
				covarep_vec= handle_covarep_df(c_covarep_df,start_idx,end_idx,skip=1)
				opensmile_vec= handle_opensmile_df([c_opensmile_lld_df, c_opensmile_csv_df, c_prosody_df, c_vad_df],start_idx,end_idx,skip=1)
				zface_vec=compute_statistics(c_zface_mat[st:end])
				au_occ_vec=compute_statistics(c_au_occ_mat[st:end])
				au_int_vec=compute_statistics(c_au_int_mat[st:end])
				
			audio_vec= np.concatenate((covarep_vec,opensmile_vec))
			video_vec=np.concatenate((zface_vec,au_occ_vec,au_int_vec))
			#feature_vec=np.concatenate((video_vec,audio_vec))
			audio_feature_stack= np.vstack([audio_feature_stack, audio_vec]) if audio_feature_stack is not None else audio_vec
			video_feature_stack= np.vstack([video_feature_stack, video_vec]) if video_feature_stack is not None else video_vec
			
	
		digit_labels= convert_construct_to_digits(construct_labels)
		
	
		total_audio.append(audio_feature_stack)
		total_video.append(video_feature_stack)
		total_label.append(digit_labels)
		total_speaker.append(sub)
		total_time.append([start_time,end_time])
		total_frame.append([start_frame,end_frame])
		total_filename.append([parent_life,child_life])
		
		total_turn_speaker.append(turn_subject)
		total_turn_time.append([turn_start_time, turn_end_time])
		total_turn_frame.append([turn_start_frame, turn_end_frame])
		total_turn_filename.append(turn_family)
		total_turn_duration.append(turn_duration)

		
		print (i,parent_life, len(total_label),len(total_time),len(total_speaker),  audio_feature_stack.shape, video_feature_stack.shape, len(digit_labels), len(sub))

	
	res={}
	res['audio']= total_audio
	res['video']= total_video
	res['label']= total_label
	res['speaker']= total_speaker
	res['time'] = total_time
	res['frame']= total_frame
	res['filenames']= total_filename

	res['turn_filename']= total_turn_filename
	res['turn_speaker']= total_turn_speaker
	res['turn_duration']= total_turn_duration
	res['turn_time'] = total_turn_time
	res['turn_frame']= total_turn_frame

	return res

def compute_statistics(vector):
	
	if vector.size > 0 :
		max_vector=np.max(vector,axis=0)
		mean_vector=np.mean(vector,axis=0)
		std_vector=np.std(vector,axis=0)
		iqr_vector=iqr (vector,axis=0)
		median_vector= np.median(vector,axis=0)
	else:
		max_vector=np.zeros (vector.shape[1])
		mean_vector=np.zeros_like(max_vector)
		std_vector=np.zeros_like(max_vector)
		iqr_vector=np.zeros_like(max_vector)
		median_vector=np.zeros_like(max_vector)

	stats_vec= np.hstack([max_vector,mean_vector,std_vector,iqr_vector,median_vector])
	return stats_vec


txt_files = [ x for x in os.listdir (text_dir) ]

source_dir ='../../LIFE_Codes/kfold_data/'
save_dir = '/run/user/1435515045/gvfs/smb-share:server=terracotta.psychology.pitt.edu,share=projects/Sanchayan/tpot_turn_data/'

folds=sorted([name for name in os.listdir(source_dir) if name.startswith('split')])

unique_list=get_unique_filenames(os.listdir(life_code_dir))

def par_fold(idx,files):

	#for idx, files in enumerate(folds):

	print ('Fold ', idx)

	data= np.load(source_dir+files, allow_pickle=True).item()
	x_train,x_valid,x_test=data['x_train'],data['x_valid'],data['x_test']
	y_train,y_valid,y_test=data['y_train'],data['y_valid'],data['y_test']
	

	result_train=expand_filenames(x_train,y_train,unique_list)
	result_valid=expand_filenames(x_valid,y_valid,unique_list)
	result_test =expand_filenames(x_test,y_test,unique_list)

	valid=read_features(result_valid)
	np.save (save_dir + 'valid_' +str(idx) +'.npy', valid)
	
	test= read_features(result_test)
	np.save(save_dir +'test_' + str(idx) +'.npy',test)

	train =read_features(result_train)
	np.save(save_dir+'train_' + str(idx) +'.npy', train)

d = Parallel(n_jobs=10)(delayed(par_fold)(idx,files) for idx, files in enumerate(folds))
print ('Done')