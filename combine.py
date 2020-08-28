
import numpy as np 
import os 
import pdb 
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
plt.rcParams.update({'font.size': 12})

import seaborn as sns 

class_dict = {0: 'Aggressive' , 1:'Dysphoric' , 2:'Positive', 3:'other'}
root_dir = '/run/user/1435515045/gvfs/smb-share:server=terracotta.psychology.pitt.edu,share=projects/Sanchayan/tpot_data/'
folds=sorted([name for name in os.listdir(root_dir) if name.startswith('train')])

def load_text_data():
	data = np.load ('text_data1.npy',allow_pickle=True).item()
	
	family= data ['family'] 
	start_time=data ['start_time']
	end_time=data ['end_time'] 
	duration=data ['duration'] 
	subject=data ['subject'] 
	text=data ['text'] 

	return [family, start_time, end_time, duration, subject, text]


def turn_per_construct(t_start,t_end, c_start,c_end, t_speaker, c_speaker, t_duration):


	arr1= np.zeros_like(c_start) #------Number of overlaps + inside per construct 

	arr2= np.zeros_like(c_start) #--------Number of turns strictly in the window --

	arr3= np.zeros_like(c_start) #------Number of tutrns that wraps the construct
	
	arr4= np.zeros_like(c_start) #---------Number of turns whose majority fall in the construct interval
	for idx , (start_c, end_c, speaker_c) in enumerate(zip(c_start,c_end,c_speaker)):

		for jj, (start_t , end_t , speaker_t, duration ) in enumerate(zip(t_start, t_end, t_speaker, t_duration)):


			if speaker_c == speaker_t :
				# Case 1: All turns which fall in the construct interva even if it is partial +plus those that wrap
				if (start_t< start_c and end_t > start_c and end_c > end_t)\
					or (start_t < end_c and end_t > end_c and start_c < start_t) \
					or (start_t >= start_c and end_t <= end_c)\
					or (start_t < start_c and end_t > end_c) :
						arr1[idx] = arr1[idx] +1 
					
					
				# Case 2: Only those turns that is strictly between construct interval
				if start_t >= start_c and end_t <= end_c:
					arr2[idx] = arr2[idx] +1


				# Case 3 : Only those turns  which cover the entire construct intterval + plus extra
				if (start_t < start_c and end_t > end_c):
					arr3[idx]= arr3[idx] + 1


				#Case 4 : If the majority of turn falls in the constuct interval
				if ((end_t > start_c and start_t < start_c and end_c> end_t) and  (end_t - start_c > duration/2) ) \
					or ((start_t < end_c and end_t > end_c and start_c < start_t) and (end_c-start_t > duration/2)) \
					or (start_t >= start_c and end_t <= end_c) \
					or ((start_t < start_c and end_t> end_c )and (end_c - start_c > duration/2)) :
						arr4[idx] = arr4[idx] + 1
				

	return  arr1,arr2,arr3,arr4

def new_turn_per_construct(t_start,t_end, c_start,c_end, t_speaker, c_speaker, t_duration):


	arr1= np.zeros_like(c_start) #------Number of overlaps + inside per construct 

	arr2= np.zeros_like(c_start) #--------Number of turns strictly in the window --

	arr3= np.zeros_like(c_start) #------Number of tutrns that wraps the construct
	
	arr4= np.zeros_like(c_start) #---------Number of turns whose majority fall in the construct interval
	

	neg1 = np.zeros_like(c_start)
	neg2 = np.zeros_like(c_start)
	for idx , (start_c, end_c, speaker_c) in enumerate(zip(c_start,c_end,c_speaker)):

		for jj, (start_t , end_t , speaker_t, duration ) in enumerate(zip(t_start, t_end, t_speaker, t_duration)):

			if speaker_c == speaker_t :
				# Case 1: All turns which fall in the construct interva even if it is partial +plus those that wrap
				if (start_t< start_c and end_t > start_c and end_c > end_t)\
					or (start_t < end_c and end_t > end_c and start_c < start_t) \
					or (start_t >= start_c and end_t <= end_c)\
					or (start_t < start_c and end_t > end_c) :
						arr1[idx] = arr1[idx] +1 
					
					
				# Case 2: Only those turns that is strictly between construct interval
				if start_t >= start_c and end_t <= end_c:
					arr2[idx] = arr2[idx] +1


				# Case 3 : Only those turns  which cover the entire construct intterval + plus extra
				if (start_t < start_c and end_t > end_c):
					arr3[idx]= arr3[idx] + 1


				#Case 4 : If the majority of turn falls in the constuct interval
				if ((end_t > start_c and start_t < start_c and end_c> end_t) and  (end_t - start_c > duration/2) ) \
					or ((start_t < end_c and end_t > end_c and start_c < start_t) and (end_c-start_t > duration/2)) \
					or (start_t >= start_c and end_t <= end_c) \
					or ((start_t < start_c and end_t> end_c )and (end_c - start_c > duration/2)) :
						arr4[idx] = arr4[idx] + 1
			
			

			elif speaker_c != speaker_t :
				if (start_t< start_c and end_t > start_c and end_c > end_t)\
					or (start_t < end_c and end_t > end_c and start_c < start_t) \
					or (start_t >= start_c and end_t <=  end_c)\
					or (start_t < start_c and end_t > end_c) :
						neg1[idx] = neg1[idx] + 1

				if ((end_t > start_c and start_t < start_c and end_c> end_t) and  (end_t - start_c > duration/2) ) \
					or ((start_t < end_c and end_t > end_c and start_c < start_t) and (end_c-start_t > duration/2)) \
					or (start_t >= start_c and end_t <= end_c) \
					or ((start_t < start_c and end_t> end_c )and (end_c - start_c > duration/2)) :
						neg2[idx] = neg2[idx] + 1
			
	
	return  arr1,arr2,arr3,arr4, neg1, neg2


def get_proportion(t_start,t_end, c_start,c_end, t_speaker, c_speaker, t_duration, c_duration):


	arr1= np.zeros_like(c_start) #------Number of overlaps + inside per construct 

	arr2= np.zeros_like(c_start) #--------Number of turns strictly in the window --

	arr3= np.zeros_like(c_start) #------Number of tutrns that wraps the construct
	

	

	neg1 = np.zeros_like(c_start)
	neg2 = np.zeros_like(c_start)
	for idx , (start_c, end_c, speaker_c) in enumerate(zip(c_start,c_end,c_speaker)):
		for jj, (start_t , end_t , speaker_t, duration ) in enumerate(zip(t_start, t_end, t_speaker, t_duration)):

			if speaker_c == speaker_t :
				# Case 1: All turns which fall in the construct interva even if it is partial +plus those that wrap
				if (start_t< start_c and end_t > start_c and end_c > end_t) : 
					arr1[idx]  +=  end_t - start_c 

				elif (start_t < end_c and end_t > end_c and start_c < start_t):
					arr1[idx] +=  end_c - start_t 

				elif (start_t >= start_c and end_t <= end_c) :
					arr1[idx] += end_t - start_t 

				elif (start_t < start_c and end_t > end_c) :
					arr1[idx] += end_c- start_c 
					
					
				
			
			

			elif speaker_c != speaker_t :
				if (start_t< start_c and end_t > start_c and end_c > end_t) : 
					arr2[idx]  +=  end_t - start_c 

				elif (start_t < end_c and end_t > end_c and start_c < start_t):
					arr2[idx] +=  end_c - start_t 

				elif (start_t >= start_c and end_t <= end_c) :
					arr2[idx] += end_t - start_t 

				elif (start_t < start_c and end_t > end_c) :
					arr2[idx] += end_c - start_c 
				
	

	arr3 = c_duration - ( arr1 +  arr2 )
	
	arr4 = arr1 / c_duration
	arr5 = arr2 / c_duration


	indices = ~np.isnan(arr4)  & ~np.isnan(arr5)

	arr1= arr1[indices]
	arr2= arr2[indices]
	arr3= arr3[indices]
	arr4= arr4[indices]
	arr5= arr5[indices]
	

	return  arr1,arr2,arr3 , arr4, arr5 , indices



def get_overlap (start_t, end_t, start_c, end_c, speaker_t, speaker_c, duration_t, duration_c, mode='case2'):
	duration = 0


	if mode == 'case2':
		if start_c < start_t and end_c > start_t and end_t> end_c:
			duration = end_c - start_t 
		elif start_c<end_t and end_c > end_t and start_t < start_c:
			duration = end_t - start_c
		elif start_c >= start_t and end_c <= end_t:
			duration = end_c - start_c 
		elif start_c < start_t and end_c> end_t:
			duration = end_t - start_t 

	return duration

def get_majority_construct (t_start,t_end, c_start,c_end, t_speaker, c_speaker, t_duration,c_duration, label):

	arr2= np.zeros_like(t_start)
	label2= np.full(arr2.shape, -1)


	majority = np.zeros_like(t_start)
	for idx, (start_t , end_t , speaker_t, duration ) in enumerate(zip(t_start, t_end, t_speaker, t_duration)):

		for jj , (start_c, end_c, speaker_c, duration_c, construct) in enumerate(zip(c_start,c_end,c_speaker, c_duration, label)):

			if speaker_c== speaker_t:

				d = get_overlap (start_t, end_t , start_c , end_c , speaker_t, speaker_c, duration, duration_c)

				if d > arr2[idx]:
					arr2[idx] = d 
					label2[idx] = construct
					
	return arr2, label2

 	
def get_all_utterance (t_start, t_end, c_start, c_end, t_speaker, c_speaker, t_duration, c_duration, t_text):
	arr1= np.zeros_like(c_start)
	arr= []

	mark= np.full(arr1.shape, -1)
	
	for idx , (start_c, end_c, speaker_c) in enumerate(zip(c_start,c_end,c_speaker)):

		utterance = ""
		for jj, (start_t , end_t , speaker_t, duration , text) in enumerate(zip(t_start, t_end, t_speaker, t_duration, t_text)):
			if speaker_c == speaker_t :
				# Case 1: All turns which fall in the construct interva even if it is partial +plus those that wrap
				if (start_t< start_c and end_t > start_c and end_c > end_t)\
					or (start_t < end_c and end_t > end_c and start_c < start_t) \
					or (start_t >= start_c and end_t <= end_c)\
					or (start_t < start_c and end_t > end_c) :
						arr1[idx] = arr1[idx] +1 
						try:
							utterance += text + " "	
						except:
							pdb.set_trace()


		
		#arr= np.vstack [arr, utterance] if arr is not None else utterance
		if utterance != "":
			mark[idx]=len(utterance)
		arr.append(utterance.rstrip())
	arr= np.array(arr)
	
	return mark, arr  



def plot_histogram (x_label, y_label, title, name,   data):

	#bins = np.arange(0,np.max(arr)+1)
	#bins = np.arange(0,np.max(data)+1).astype(int)
	#bins = np.linspace (0, np.max(data) + 1, np.max(data) + 1)
	x= np.arange( 0, np.max(data)+1).astype(int)

	y= np.zeros_like(x).astype('float')
	for idx in x:
		y[idx]=np.sum (data == idx) / len(data)
	
	#bins = len(np.unique(data))
	#plt.hist(data, bins=bins)
	plt.bar (x=x, height=y, tick_label=list(x))
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.ylim(0.0,0.6)
	#plt.gcf().text(0.9, 0.1, "Totat="+str(len(data)), fontsize=10)
	plt.title(title)
	plt.savefig (name+'.jpg')
	plt.close()
	return 


def plot_percentage1 (x_label, y_label, title, name,   data):

	#bins = np.arange(0,np.max(arr)+1)
	#bins = np.arange(0,np.max(data)+1).astype(int)
	#bins = np.linspace (0, np.max(data) + 1, np.max(data) + 1)
	x= np.arange( 0, np.max(data)+1).astype(int)

	y= np.zeros_like(x)
	for idx in x:
		y[idx]=np.sum (data == idx)


	#pdb.set_trace()
	#ax= sns.violinplot (x=, y=y_label, data= data )
	
	#bins = len(np.unique(data))
	plt.hist(data, bins=5)
	#plt.bar (x=x, height=y, tick_label=list(x))
	#plt.xlabel(x_label)
	#plt.ylabel(y_label)
	#plt.gcf().text(0.9, 0.1, "Totat="+str(len(data)), fontsize=10)
	plt.title(title)
	plt.savefig (name+'.jpg')
	plt.close()
	return 

def plot_percentage (x_label, y_label, title, name,   data):

	y , x = data	

	ax= sns.violinplot (x=x, y=y)
	
	#bins = len(np.unique(data))
	#plt.hist(data, bins=5)
	#plt.bar (x=x, height=y, tick_label=list(x))
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	#plt.gcf().text(0.9, 0.1, "Totat="+str(len(data)), fontsize=10)
	plt.title(title)
	plt.savefig (name+'.jpg')
	plt.close()
	return 

def differential_analysis(x1_list, construct_list, title=None, case='case_1'):

	elements = []


	for i in np.unique (construct_list):

		#elements = x1_list [x1_list == 0]
	
		indices = (construct_list == i)
		new_list = x1_list [indices]
		

		if title is None :
			plot_histogram(y_label='Number of constructs (Total='+str(len(new_list))+')', x_label='Speaker-Turns/construct', title='Histogram for number of Speaker-Turns for ' + class_dict[int(i)], name='utterance_'+case + class_dict[int(i)], data=new_list)
		else:
			plot_histogram(y_label='Number of constructs (Total='+str(len(new_list))+')', x_label='Speaker-Turns/construct', title=title +' for '+ class_dict[int(i)], name='utterance_'+case + class_dict[int(i)], data=new_list)

		#elements.append(sum(indices))
	#pdb.set_trace()
	return elements

def subject_analysis (x1_list,construct_list, subject_list,version='same', case='case_1'):


	d = {'1': 'Child', '2': 'Mother'}

	for jj in np.unique (construct_list):

		indexes= construct_list == jj 
		x_list =  x1_list[indexes]
		sub_list= subject_list[indexes]

		for i in np.unique (subject_list):
			

			indices = sub_list == i
			new_list = x_list [indices]

			if version == 'same':
				plot_histogram (y_label='Number of constructs (Total='+str(len(new_list))+')', x_label='Speaker-Turns/construct', title='Histogram for number of Speaker-Turns for ' + d[i] +'_'+class_dict[int(jj)], name='utterance_'+ case + d[i] + class_dict[int(jj)], data=new_list)
			else:
				plot_histogram (y_label='Number of constructs (Total='+str(len(new_list))+')', x_label='Speaker-Turns/construct', title='Histogram for number of Other Speaker turns for ' + d[i] +'_'+ class_dict[int(jj)], name='utterance_'+ case + d[i]+ class_dict[int(jj)], data=new_list)
	
	return 

def subject_percentage1 (x1_list,construct_list, subject_list,version='same', case='case_1'):


	d = {'1': 'Child', '2': 'Mother'}

	for jj in np.unique (construct_list):

		indexes= construct_list == jj 
		x_list =  x1_list[indexes]
		sub_list= subject_list[indexes]

		for i in np.unique (subject_list):
			

			indices = sub_list == i
			new_list = x_list [indices]

			if version == 'same':
				plot_percentage (y_label='Number of constructs (Total='+str(len(new_list))+')', x_label='Proportion of Turn time/construct', title='Histogram for number of Speaker Time  for ' + d[i] +'_'+class_dict[int(jj)], name='utterance_'+ case + d[i] + class_dict[int(jj)], data=new_list)
			else:
				plot_percentage (y_label='Number of constructs (Total='+str(len(new_list))+')', x_label='Proportion of Turn time/construct', title='Histogram for number of Other Speaker Time for ' + d[i] +'_'+ class_dict[int(jj)], name='utterance_'+ case + d[i]+ class_dict[int(jj)], data=new_list)
	
	return 


def violin_plot (x1_list,construct_list, subject_list,version='same', case='case_1'):


	d = {'1': 'Child', '2': 'Mother'}

	
	for i in np.unique (subject_list):
			

			indices = subject_list == i
			new_list = x1_list [indices]
			new_construct_list= construct_list[indices]

			sorted_indices = np.argsort(new_construct_list)
			new_list = new_list [sorted_indices]
			new_construct_list= new_construct_list[sorted_indices]
			love_list = np.array([class_dict[int(j)] for j in new_construct_list])
			print (min(new_list))
			if version == 'same':
				plot_percentage (y_label='Proportion of Turn Time / Construct', x_label='Constructs', title='Violin Plot of Same Speaker in '+ d[i]+' constructs', name='utterance_'+ case + d[i] , data=[new_list, love_list])
			else:
				plot_percentage (y_label='Proportion of Turn Time / Construct', x_label='Constructs', title='Violin Plot of Other Speaker in' + d[i]+' constructs' , name='utterance_'+ case + d[i], data=[new_list,love_list])
	
	return 


def construct_per_turn (t_start,t_end, c_start,c_end, t_speaker, c_speaker, t_duration,c_duration, label=None, text=None):
	arr1= np.zeros_like(t_start)

	arr2= np.zeros_like(t_start)

	arr3= np.zeros_like(t_start)
	
	annot_arr1 = []
	annot_arr2 = []
	annot_arr3 = []

	annot_neg_arr1=[]
	annot_neg_arr2=[]

	neg_arr1= np.zeros_like(t_start)
	neg_arr2= np.zeros_like(t_start)
	neg_arr3= np.zeros_like(t_start)


	#---Onsets----
	onset_arr2=[]


	for idx, (start_t , end_t , speaker_t, duration , turn) in enumerate(zip(t_start, t_end, t_speaker, t_duration, text)):

		temp_arr1 =[]
		temp_arr2 =[]
		temp_arr3 =[]

		temp_neg_arr1=[]
		temp_neg_arr2=[]

		temp_onset_1=[]
		temp_onset_2=[]

		

		for jj , (start_c, end_c, speaker_c, duration_c, lab) in enumerate(zip(c_start,c_end,c_speaker, c_duration, label)):




			"Conditions for same person behavior constructs in current turn  "
			if speaker_t == speaker_c:

				#Case 1: If tthe onset falls anwhhere within the turn
				if start_c >= start_t  and start_c < end_t:
					arr1[idx] = arr1[idx] +1 
					temp_arr1.append (int(lab))
					temp_onset_1.append( [start_c,end_c] )
				


				#Case 2: Where all constructs in the speaker turn are considered as labels 
				if (start_c < start_t  and  end_c > start_t and end_t> end_c) or (start_c < end_t  and end_c > end_t and start_t <start_c) or (start_c >= start_t and end_c <= end_t) or (start_c < start_t and end_c>end_t):
					arr2[idx] = arr2[idx] +1 
					temp_arr2.append(int(lab))
					temp_onset_2.append([start_c,end_c])
			

				#Case 3: Only majority of construct
				if ((start_c < start_t  and  end_c > start_t and end_t> end_c)  and (end_c - start_t > duration_c /2) ) \
					or ((start_c < end_t and end_c > end_t and start_t < start_c) and (end_t - start_c > duration_c/2)) \
					or (start_c >= start_t and end_c <= end_t) \
					or ((start_c < start_t and end_c>end_t) and (end_t - start_t > duration_c/2)):
						arr3[idx]= arr3[idx] +1
						temp_arr3.append(int(lab)) 
	

			#" The below conditions are for other person's behavior constructs in current turn"	
			elif speaker_t != speaker_c:
				if start_c >= start_t  and start_c < end_t:
					#arr1[idx]= arr1[idx]+1 
					neg_arr1[idx] = neg_arr1[idx] +1 
					temp_neg_arr1.append(int(lab))

				#Case 2: Where all constructs in the speaker turn are considered as labels 
				if (start_c < start_t  and  end_c > start_t and end_t> end_c) or\
				   (start_c < end_t  and end_c > end_t and start_t <start_c) or \
				   (start_c >= start_t and end_c <= end_t) or \
				   (start_c < start_t and end_c>end_t):
					#arr2[idx]= arr2[idx]+1 
					neg_arr2[idx] = neg_arr2[idx] +1
					temp_neg_arr2.append(int(lab)) 
			

				#Case 3: Only majority of construct
				if ((start_c < start_t  and  end_c > start_t and end_t> end_c)  and (end_c - start_t > duration_c /2) ) \
					or ((start_c < end_t and end_c > end_t and start_t < start_c) and (end_t - start_c > duration_c/2)) \
					or (start_c >= start_t and end_c <= end_t) \
					or ((start_c < start_t and end_c>end_t) and (end_t - start_t > duration_c/2)):
						neg_arr3[idx]= neg_arr3[idx] +1 
		
		#print (turn, speaker_t, start_t , end_t , temp_arr2, temp_onset_2)

		#pdb.set_trace()
		annot_arr1.append (np.array(temp_arr1).astype('int'))
		annot_arr2.append (np.array(temp_arr2).astype('int'))
		annot_arr3.append (np.array(temp_arr3).astype('int'))

		annot_neg_arr1.append(np.array(temp_neg_arr1).astype('int'))
		annot_neg_arr2.append(np.array(temp_neg_arr2).astype('int'))


		onset_arr2.append (np.array(temp_onset_2).astype('float'))



	
	return arr1, arr2, arr3, neg_arr1, neg_arr2, [annot_arr2, annot_neg_arr2], [onset_arr2]

def turn_pauses_overlaps (t_start,t_end, c_start,c_end, t_speaker, c_speaker, t_duration,c_duration, label=None, text=None):
	
	overlap_count = np.zeros_like(t_start)
	overlap_duration= np.zeros_like(t_start).astype('float')

	pause_duration = np.zeros_like (t_start).astype('float')

	within_duration = np.zeros_like (t_start).astype('float')
	between_duration= np.zeros_like (t_start).astype('float')

	prev_end = 0
	prev_speaker= None

	
	for idx, (start_t , end_t , speaker_t, duration , turn) in enumerate(zip(t_start, t_end, t_speaker, t_duration, text)):


		# for overlaps
		if start_t < prev_end and idx > 0:
			overlap_count [idx] += 1
			overlap_duration[idx] = prev_end - start_t
		
		# for pauses
		if start_t > prev_end and idx > 0:
			pause_duration[idx] = start_t - prev_end


		# For within speaker pauses 
		if start_t > prev_end and speaker_t == prev_speaker and idx > 0:
			within_duration[idx] = start_t - prev_end

		# For between speaker pauses
		if start_t > prev_end and speaker_t != prev_speaker and idx > 0:
			between_duration[idx]= start_t - prev_end


		prev_end = end_t 
		prev_speaker= speaker_t
		

	
	return [overlap_count, overlap_duration], [pause_duration, within_duration, between_duration]

def turn_statistics (data_lists):

	turn_duration, overlap_count, overlap_duration, pause_duration, within_duration, between_duration, speaker= data_lists


	total_turns = speaker.size
	parent_turns = np.sum (speaker =='2')
	child_turns = np.sum (speaker =='1')
	
	child_duration = turn_duration[ speaker=='1']
	parent_duration = turn_duration [ speaker =='2']

	total_overlap = np.sum (overlap_count) 
	parent_overlap = np.sum(overlap_count [speaker == '2']) 
	child_overlap = np.sum (overlap_count [speaker == '1']) 

	
	total_ov_duration=  overlap_duration[overlap_count==1]
	child_ov_duration = overlap_duration [(speaker == '1') & (overlap_count==1)]
	parent_ov_duration= overlap_duration [(speaker == '2') & (overlap_count==1)]


	total_pause_duration = pause_duration [overlap_count==0]
	child_pause_duration = pause_duration [(speaker=='1') & (overlap_count==0)]
	parent_pause_duration = pause_duration [(speaker=='2') & (overlap_count==0)]


	total_within_duration = within_duration[within_duration > 0]
	child_within_duration = within_duration [ (speaker=='1') & (within_duration>0)]
	parent_within_duration= within_duration [ (speaker=='2') & (within_duration>0)]
	
	
	total_between_duration = between_duration[between_duration> 0]
	child_between_duration = between_duration [ (speaker=='1') & (between_duration>0)]
	parent_between_duration= between_duration [ (speaker=='2') & (between_duration>0)]
	

	print ("Total turns=", total_turns, "parent_turns=", parent_turns, float(parent_turns)/total_turns ,"child_turns=", child_turns, float(child_turns)/total_turns)

	
	print ("Total Overlap=", total_overlap, "parent_overlap=", parent_overlap, float(parent_overlap)/child_overlap, "child_overlap=", child_overlap,float(child_overlap)/total_overlap)

	print ("Turn duration=",np.mean (turn_duration), np.std (turn_duration))
	print ("child_duration=", np.mean(child_duration), np.std (child_duration))
	print ("parent_duration=", np.mean(parent_duration), np.std(parent_duration))


	print ("Mean overlap duration", np.mean(total_ov_duration), "Standard Deviation", np.std(total_ov_duration))
	print ("Child Overlap Duration", np.mean(child_ov_duration), np.std(child_ov_duration))
	print ("Parent Overlap Duration", np.mean(parent_ov_duration), np.std(parent_ov_duration))


	print ("Mean pause duration", np.mean(total_pause_duration), np.std(total_pause_duration), len(total_pause_duration))
	print ("Child pause_duration", np.mean(child_pause_duration), np.std(child_pause_duration), len(child_pause_duration))
	print ("Mother pause_duration", np.mean(parent_pause_duration), np.std(parent_pause_duration), len(parent_pause_duration))


	print ("Mean within duration", np.mean(total_within_duration) , np.std (total_within_duration), len(total_within_duration))
	print ("Child within duration", np.mean(child_within_duration), np.std (child_within_duration), len(child_within_duration))
	print ("Parentt within duration", np.mean(parent_within_duration), np.std(parent_within_duration), len(parent_within_duration))

	print ("Mean between durationn", np.mean(total_between_duration), np.std (total_between_duration), len(total_between_duration))
	print ("Child between duration", np.mean(child_between_duration), np.std (child_between_duration), len(child_between_duration))
	print ("Parent between_duration", np.mean(parent_between_duration), np.std (parent_between_duration), len(parent_between_duration))

	pdb.set_trace()
	return


def plot_turns (x_list, subject_list, version='same', case='case1'):
	def plot_histogram (x_label, y_label, title, name,   data):
		#bins = np.arange(0,np.max(arr)+1)
		#bins = np.arange(0,np.max(data)+1).astype(int)
		#bins = np.linspace (0, np.max(data) + 1, np.max(data) + 1)
		x= np.arange( 0, np.max(data)+1).astype(int)

		y= np.zeros_like(x).astype('float')
		for idx in x:
			y[idx]=np.sum (data == idx)  / len(data)

	
		#bins = len(np.unique(data))
		#plt.hist(data, bins=bins)
		plt.bar (x=x, height=y, tick_label=list(x))
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.ylim(0.0,1.0)
	
		#plt.gcf().text(0.9, 0.1, "Totat="+str(len(data)), fontsize=10)
		plt.title(title)
		plt.savefig (name+'.jpg')
		plt.close()
		return 


	dict_speaker = {'1': 'Child', '2': 'Mother'}

    #------------General---------------------#
	if version == 'same':
		plot_histogram ("Number OF Construct/ Turn", " Number of Turns (Proportion) ", title="Histogram of Number of  all speaker constructs", name='turn_'+version +'_'+case, data=x_list )
	else:
		plot_histogram ("Number OF Construct/ Turn", " Number of Turns (Proportion) ", title="Histogram of Number of Other speaker constructs", name='turn_'+version +'_'+case, data=x_list )


	#-------Differential---------------#
	for i in np.unique(subject_list):
		
		sub_indices = subject_list == i 
		sub_list   = x_list [ sub_indices]
		
	
		if version == 'same':
			
			plot_histogram ("Number OF Construct/ Turn", " Number of Turns (Proportion) ", title="Histogram of Number of all speaker  constructs for "+dict_speaker[i], name='turn_'+version + '_'+ dict_speaker[i] + '_'+case, data=sub_list )
		else:
			plot_histogram ("Number OF Construct/ Turn", " Number of Turns (Proportion) ", title="Histogram of Number of Other speaker constructs for "+dict_speaker[i], name='turn_'+version + '_'+ dict_speaker[i] +'_'+ case, data=sub_list )

	return 


def plot_turn_length (x_list, subject_list, case='case1'):
	def plot_histogram (x_label, y_label, title, name,   data):
		#bins = np.arange(0,np.max(arr)+1)
		#bins = np.arange(0,np.max(data)+1).astype(int)
		#bins = np.linspace (0, np.max(data) + 1, np.max(data) + 1)
		x= np.arange( 0, np.max(data)+1).astype(int)

		y= np.zeros_like(x).astype('float')
		for idx in x:
			y[idx]=np.sum (data == idx)  / len(data)

		"Ffor perceentage"
		weights=np.ones(len(data)) / len(data)	
		#bins = len(np.unique(data))
		#plt.hist(data, bins=bins)
		#plt.bar (x=x, height=y, tick_label=list(x))
		plt.hist(data, weights= weights,bins=10)
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0,decimals=0))
		#plt.gcf().text(0.9, 0.1, "Totat="+str(len(data)), fontsize=10)
		
		plt.title(title)
		plt.savefig (name+'.jpg')
		plt.close()
		return 


	dict_speaker = {'1': 'Child', '2': 'Mother'}


	#-------Differential---------------#
	for i in np.unique(subject_list):
		
		sub_indices = subject_list == i 
		sub_list   = x_list [ sub_indices]
		
	
		plot_histogram ("Turn duration (in sec)",  " Number of Turns (in percentage)", title="Distribution of "+dict_speaker[i]+" turn duration time", name='duration_'+ dict_speaker[i] + '_'+case, data=sub_list )
		
	plot_histogram ("Turn duration (in sec)", "Number of Turns (in percentage) ", title="Distribution of turn duration", name='duration_'+case, data=x_list )
	
	return 

def length_analysis (x1_list,construct_list, duration_list,  subject_list,version='same', case='case_1'):

	def plot_length (x_label, y_label, title, name,   data):

		y , x = data	


		plt.bar (x=x, height=y)
		#bins = len(np.unique(data))
		#plt.hist(data, bins=5)
		#plt.bar (x=x, height=y, tick_label=list(x))
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		#plt.gcf().text(0.9, 0.1, "Totat="+str(len(data)), fontsize=10)
		plt.title(title)
		plt.savefig (name+'.jpg')
		plt.close()
		return 


	d = {'1': 'Child', '2': 'Mother'}

	
	for i in np.unique (subject_list):
			

			indices = subject_list == i
			new_list = x1_list [indices]
			new_construct_list= construct_list[indices]
			new_duration_list = duration_list[indices]

			sorted_indices = np.argsort(new_duration_list)
			new_list = new_list [sorted_indices] + 1
			new_construct_list= new_construct_list[sorted_indices]
			new_duration_list= new_duration_list[sorted_indices]
			#love_list = np.array([class_dict[int(j)] for j in new_construct_list])
			print (max(new_duration_list))
			if version == 'same':
				plot_length (y_label='Number of Speaker-Turns', x_label='Construct duration (in sec)', title='Turn count  of Same Speaker in '+ d[i]+' constructs', name='utterance_'+ case + d[i] , data=[new_list, new_duration_list])
			else:
				plot_length (y_label='Number of Speaker-Turns', x_label='Construct duration (in sec)', title='Turn count  of Other Speaker in' + d[i]+' constructs' , name='utterance_'+ case + d[i], data=[new_list,new_duration_list])
	
	return 





def analyze(res):
	[family, start_time, end_time, duration, subject, text]= load_text_data()



	audio=res['audio']
	video=res['video']
	label=res['label']
	speaker=res['speaker']
	time=res['time']
	frame=res['frame']
	filename=res['filenames']

	
	x1_list=[]
	x2_list=[]
	x3_list=[]
	x4_list=[]
	x5_list=[]
	construct_list=[]
	speaker_list = []

	neg1_list=[]
	neg2_list=[]

	y1_list=[]
	y2_list=[]
	y3_list=[]
	ny1_list=[]
	ny2_list=[]
	speakert_list=[]
	turn_start_list=[]
	turn_end_list=[]
	turn_text_list=[]
	turn_duration_list=[]
	turn_label_list=[]
	turn_onset_list=[]




	#----------Declarattion for analysyis of gaps and pauses

	overlap_count_list = []
	overlap_duration_list=[]
	pause_duration_list=[]
	within_duration_list=[]
	between_duration_list=[]
	#--------------------



	onset = [x[0] for x in time]
	offset = [x[1] for x in time]

	start_frame =[x[0] for x in frame]
	end_frame = [x[1] for x in frame]

	file_list = [x[0] for x in filename]
	file_list = [x.split('_')[0][:-1][-4:] for x in file_list]

	dur_c=[] 
	dur_t=[]
	for idx, fam in enumerate(family):

		check_fam =np.array( [x.find(fam) for x in file_list]) 	
		check_fam = np.where(check_fam>-1)[0]

		if len (check_fam) > 0:
			idx_fam = check_fam[0]

			value_1= onset [idx_fam]
			value_2= offset[idx_fam]

			
			#x1,x2,x3,x4 = turn_per_construct (start_time[idx], end_time[idx], onset[check_fam[0]], offset[check_fam[0]], subject[idx], speaker[check_fam[0]], duration[idx])
			y1,y2,y3, ny1, ny2, [pos_annot, neg_annot], [onset_info]= construct_per_turn (start_time[idx], end_time[idx], onset[check_fam[0]], offset[check_fam[0]], subject[idx], speaker[check_fam[0]], duration[idx], value_2-value_1,
								label= label[idx_fam],\
								text= text[idx])
			
			[overlap_count,overlap_duration],[pause_duration, within_duration, between_duration]= turn_pauses_overlaps (start_time[idx], end_time[idx], onset[check_fam[0]], offset[check_fam[0]], subject[idx], speaker[check_fam[0]], duration[idx], value_2-value_1,
								label= label[idx_fam],\
								text= text[idx])
			

			x1,x2,x3,x4 ,neg1,neg2= new_turn_per_construct (start_time[idx], end_time[idx], onset[check_fam[0]], offset[check_fam[0]], subject[idx], speaker[check_fam[0]], duration[idx])
			
			#----------This gets majority constructs label per turns ------------#
			new_duration,new_label= get_majority_construct (start_time[idx], end_time[idx], onset[check_fam[0]], offset[check_fam[0]], subject[idx], speaker[check_fam[0]], duration[idx], value_2-value_1, label[check_fam[0]])
			

			_, _, _, _, x5 , indices= get_proportion (start_time[idx], end_time[idx], onset[check_fam[0]], offset[check_fam[0]], subject[idx], speaker[check_fam[0]], duration[idx], value_2-value_1)
			
			dur_c.extend ((value_2-value_1)[indices])
			dur_t.extend (duration[idx])
			#new_mark, new_utterance=  get_all_utterance (start_time[idx], end_time[idx], onset[check_fam[0]], offset[check_fam[0]], subject[idx], speaker[check_fam[0]], duration[idx], value_2-value_1, text[idx])
			
			x1_list.append(x1[indices])
			x2_list.append(x2[indices])
			x3_list.append(x3[indices])
			x4_list.append(x4[indices])
			x5_list.append(x5)
			construct_list.append(label[check_fam[0]][indices])

			speaker_list.append (speaker[check_fam[0]][indices])
			neg1_list.append(neg1[indices])
			neg2_list.append(neg2)


			"Speaker information . Preparing for turns"
			y1_list.append(y1)
			y2_list.append(y2)
			y3_list.append(y3)
			ny1_list.append(ny1)
			ny2_list.append(ny2)
			
			speakert_list.append(subject[idx])
			turn_duration_list.append (duration[idx])
			turn_start_list.append( start_time[idx])
			turn_end_list.append( end_time[idx])
			turn_text_list.append( text[idx])
			turn_label_list.append ( pos_annot )
			turn_onset_list.append (onset_info)


			"Turn analsis"
			overlap_count_list.append(overlap_count)
			overlap_duration_list.append(overlap_duration)
			pause_duration_list.append(pause_duration)
			within_duration_list.append(within_duration)
			between_duration_list.append(between_duration)




	x1_list=np.concatenate(np.array(x1_list))
	x2_list=np.concatenate(np.array(x2_list))
	x3_list=np.concatenate(np.array(x3_list))
	x4_list=np.concatenate(np.array(x4_list))
	x5_list=np.concatenate(np.array(x5_list))

	construct_list= np.concatenate(np.array(construct_list))
	speaker_list= np.concatenate(np.array(speaker_list))

	y1_list= np.concatenate(np.array(y1_list))
	y2_list= np.concatenate(np.array(y2_list))
	y3_list= np.concatenate(np.array(y3_list))
	ny1_list= np.concatenate(np.array(ny1_list))
	ny2_list= np.concatenate(np.array(ny2_list))
	turn_speaker_list=np.concatenate(np.array(speakert_list))

	pdb.set_trace()
	'''
	"Analyzing turn statistics"
	overlap_count_list= np.concatenate(np.array(overlap_count_list))
	overlap_duration_list= np.concatenate(np.array(overlap_duration_list))
	pause_duration_list= np.concatenate(np.array(pause_duration_list))
	within_duration_list=np.concatenate(np.array(within_duration_list))
	between_duration_list=np.concatenate(np.array(between_duration_list))
	turn_duration_list= np.concatenate (np.array(turn_duration_list))

	turn_statistics ( [turn_duration_list,overlap_count_list, overlap_duration_list, pause_duration_list, within_duration_list, between_duration_list, turn_speaker_list] )
	dur_c = np.array (dur_c)
	
	neg1_list=	np.concatenate( np.array(neg1_list) )
	neg2_list= 	np.concatenate( np.array(neg2_list) )
	'''

	#length_analysis (x1_list, construct_list, dur_c, speaker_list, version='same', case='case_1')
	#length_analysis (neg1_list, construct_list, dur_c, speaker_list, version='other', case='case_neg_1')

	
	#----------------------Violin Plots -----------------------------------#
	#violin_plot (x4_list, construct_list, speaker_list, version='same', case='case_1')
	#violin_plot (x5_list, construct_list, speaker_list, version='other', case='case_neg_1')
	
	#plot_percentage(y_label="Number of constructs", x_label="Proportion of turn time/construct duration", title="Histogram OF Speaker Time", name="x4",data=x4_list)
	#plot_percentage(y_label="Number of constructs", x_label="Proportion of turn time/ construct duration", title="Histogram of Other Speaker Time", name="x5", data=x5_list)	
	

	#print (differential_analysis(x1_list, construct_list))
	#print (differential_analysis(x4_list, construct_list))
	#pdb.set_trace()
	#differential_analysis ( x1_list, construct_list, case='case_1')
	#differential_analysis ( x4_list, construct_list, case='case_4') 
	
	#----------------Plottting for constructs -------------------#
	#subject_analysis  (x1_list, construct_list, speaker_list, version='same', case='case_1')
	#subject_analysis  (x4_list, construct_list,speaker_list, version='same',case='case_4')
	#subject_analysis  (neg1_list,construct_list, speaker_list, version='other',case='case_neg1')
	#subject_analysis  (neg2_list,construct_list ,speaker_list, version='other',case='case_neg2')

	#plot_histogram(y_label='Number of constructs (Total='+str(len(x1_list))+')', x_label='Number of Speaker-Turns/construct', title='Histogram for number of Speaker-Turns', name='uttereance_case1', data=x1_list)
	#plot_histogram(y_label='Number of utterances', x_label='Number of Utterances/Event', title='Histogram for number of utterances', name='uttereance_case2', data=x2_list)
	#plot_histogram(y_label='Number of utterances', x_label='Number of Utterances/Event', title='Histogram for number of utterances', name='uttereance_case3', data=x3_list)
	#plot_histogram(y_label='Number of constructs'+str(len(x4_list))+')', x_label='Number of Speaker-Turns/Construct', title='Histogram for number of Speaker-Turns', name='uttereance_case4', data=x4_list)
	'''
	#-----------------------------------------------------------------#
	plot_histogram(y_label='Number of constructs', x_label='Number of constructs/utterance', title='Histogram for number of constructs', name='construct_case1', data=y1_list)
	plot_histogram(y_label='Number of constructs', x_label='Number of constructs/utterance', title='Histogram for number of constructs', name='construct_case2', data=y2_list)
	plot_histogram(y_label='Number of constructs', x_label='Number of constructs/utterance', title='Histogram for number of constructs', name='construct_case3', data=y3_list)
	'''
	#differential_analysis ( neg1_list, construct_list, title='Nunber of Speaker-Turns for other speaker',case='case_neg1')
	#differential_analysis ( neg2_list, construct_list, title='Number of Speaker-Turns for other speaker',case='case_neg2')

	#plot_histogram(y_label='Number of constructs (Total='+str(len(neg1_list))+')', x_label="Number of Speaker-Turns/Event", title='Histogram for number of Speaker-Turns of the other speaker', name='uttereance_case_neg1', data=neg1_list)
	#plot_histogram(y_label='Number of constructs (Total='+str(len(neg2_list))+')', x_label='Number of Speaker-Turns/Event', title='Histogram for number of Speaker-Turns of the other speaker', name='uttereance_case_neg2', data=neg2_list)


	#---------------Plotting with respect to turns -------------- #
	
	#plot_turns (y1_list, turn_speaker_list  , version='same', case='case1')
	#plot_turns (ny1_list, turn_speaker_list, version='other', case='case1')
	#plot_turns (y2_list, turn_speaker_list  , version='same', case='case2')
	#plot_turns (ny2_list, turn_speaker_list, version='other', case='case2')
	#turn_duration_list= np.concatenate (np.array(turn_duration_list))
	#plot_turn_length(turn_duration_list, turn_speaker_list, case='case1')

	return 

for idx, items in enumerate(folds):

	res = np.load (os.path.join(root_dir,items),allow_pickle=True).item()

	
	analyze(res)

	if idx == 0 :
		break 	