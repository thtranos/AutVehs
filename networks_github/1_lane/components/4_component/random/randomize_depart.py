import xml.etree.ElementTree as ET
import random
import numpy as np

filename = '4.rou.xml'
xmlTree = ET.parse(filename)
rootElement = xmlTree.getroot()

time = 0

for type_tag in rootElement.findall('vehicle'):
	random_number_speed = float(random.choice(np.arange(10,20)))
	random_number_depart = float(random.choice([1,2,3,4,5,6,7]))
	#dice = random.uniform(0,1)
	#if dice < 0.25:
		#random_number_depart = 1
	#else:
		#random_number_depart = 4
	random_number_pos = float(random.choice(np.arange(0,20)))


	type_tag.set('departSpeed', str(random_number_speed))
	type_tag.set('departPos', str(random_number_pos))
	type_tag.set('depart', str(time))
	time = time + random_number_depart

xmlTree.write(filename,encoding='UTF-8',xml_declaration=True)