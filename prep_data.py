import cv2
print (cv2.__version__)

import numpy as np
print (np.__version__)

import os

from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.


IMG_SIZE = 28


#######################################################################
# Préparation des images : resize/enregistrement
#Mise en commun de images de training avec les labels correspondants
#######################################################################
def label_img(nom_classes):
	word_label = nom_classes
  
	if word_label == 'Classe_1': return [1,0,0,0,0]
	elif word_label == 'Classe_2': return [0,1,0,0,0]
	elif word_label == 'Classe_3': return [0,0,1,0,0]
	elif word_label == 'Classe_4': return [0,0,0,1,0]
	elif word_label == 'Classe_5': return [0,0,0,0,1]

# appelle de la fonction ::::: label = label_img(nom)



training_data = []

file_path = open ('../CNN_tensorflow/file_path.txt','w')

repertoire = "../CNN_tensorflow/"
classes_defaut = ['Classe_1','Classe_2','Classe_3','Classe_4','Classe_5']
for nom in classes_defaut:
	print(repertoire+nom)
	pic_num=1
	for nom_img in os.listdir(repertoire+nom):
		print(repertoire+nom+'/'+nom_img)
		img=cv2.imread(repertoire+nom+'/'+nom_img, cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
		
		img=np.float32(img)
		if not os.path.exists(repertoire+'/prep_'+nom):
			os.makedirs(repertoire+'/prep_'+nom)

		cv2.imwrite(repertoire+'/prep_'+nom+'/'+str(pic_num)+'.jpg',img)
		file_path.write (repertoire+'/prep_'+nom+'/'+str(pic_num)+'.jpg'+'\n')
		print (pic_num)
		pic_num+=1
		label=label_img(nom)
		training_data.append([np.array(img),np.array(label)])

shuffle(training_data) # mélange aléatoirement les élémets de training data
np.save('train_data.npy', training_data) # fichier .npy = fichier binaire
file_path.closed	





