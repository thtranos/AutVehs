import os
import sys
import datetime
import timeit
import xml.etree.ElementTree as ET
from simRunner2test import SimRunner
import sumolib
import argparse
import sys
import numpy as np
import cProfile, pstats
import importlib
import io
import time
from pympler.tracker import SummaryTracker
from sklearn import preprocessing
import random
import shutil

np.set_printoptions(threshold=sys.maxsize)
if 'SUMO_HOME' in os.environ:
	tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
	sys.path.append(tools)
else:
	sys.exit("please declare environment variable 'SUMO_HOME'")

import traci

def find_all_edges(scenario): #TT
	edges_file = scenario + params.net_path
	root_edges = ET.parse(edges_file).getroot()
	net = sumolib.net.readNet(scenario + params.net_path)

	all_edges = []

	lane_lengths = {}

	for type_tag in root_edges.findall('edge'):
		for tag in type_tag.findall('lane'):
			lane = tag.get('id')
			length = tag.get('length')
			lane_lengths[lane] = length
		edge = type_tag.get('id')
		isInternal = type_tag.get("function")
		if(edge not in all_edges and isInternal != "internal" and isInternal != "traffic_light"):
			all_edges.append(edge)

	return all_edges, lane_lengths

def produce_random_trips(name, sim_runner, episode):

	os.system(f'cd {params.path} && python3 /Users/theodoretranos/sumo/tools/randomTrips.py -n {params.save_name.split("_")[1]}.net.xml --random -b 0 -e 900 -p 4 --trip-attributes="departLane=\\"best\\" departSpeed=\\"10.0\\" departPos=\\"base\\"" --route-file {params.save_name.split("_")[1]}.rou.xml --fringe-factor 20000 -o {params.save_name.split("_")[1]}.trips.xml')
	#os.system(f'cd {params.path} && python3 /Users/theodoretranos/sumo/tools/randomTrips.py -n randomSmall.net.xml --random -b 0 -e 900 -p 4 --trip-attributes="departLane=\\"best\\" departSpeed=\\"10.0\\" departPos=\\"base\\"" --route-file small.rou.xml --fringe-factor 20000 -o small.trips.xml')
	os.system(f'cd {params.path} && python3 randomize_depart.py')
	intersections, centers, inc_edges, int_lanes, int_to_junc, junction_inc_len, lane_to_agent = find_junctions(name)	

	final_edges, num_of_vehs, veh_to_final = find_final_edges(name)


	all_edges, lane_lengths = find_all_edges(name)

	nfedges = find_non_final_edges(all_edges,final_edges)
	print('nfedges', nfedges)

	veh_edge_number = findMode(name)

	junctionMeanLength = findJunctionMeans(int_lanes, lane_lengths)

	sumoCmd = [sumoBinary, "-c", name + f"/osm.small.sumocfg", "--collision.action", "remove",
	   "--collision.check-junctions", "true", "--no-step-log", "true", "--disable-textures",
	   "--waiting-time-memory", str(max_steps), "--no-warnings", "true", "--collision.mingap-factor", "0"]

	if episode != 0:
		sim_runner.int_to_junc = int_to_junc
		sim_runner.num_of_vehs = num_of_vehs
		sim_runner.centers = centers
		sim_runner.final_edges = final_edges
		sim_runner.nfedges = nfedges
		sim_runner.junctions = intersections
		sim_runner.sumoCmd = sumoCmd
		sim_runner.intersections = intersections
		sim_runner.int_lanes = int_lanes
		sim_runner.junctionMeanLength = junctionMeanLength
		sim_runner.veh_to_final = veh_to_final
		sim_runner.junction_inc_len = junction_inc_len

		return sim_runner

	else:
		sim_runner = SimRunner(str(sys.argv[1]), veh_to_final, num_of_vehs, int_to_junc, junctionMeanLength, intersections, centers, final_edges, nfedges, int_lanes, total_episodes, max_steps, sumoCmd, veh_edge_number, junction_inc_len, lane_to_agent)

	return sim_runner

def find_non_final_edges(all_edges, final_edges): #TT
	nfe = []
	for edge in all_edges:
		if edge not in final_edges:
			nfe.append(edge)

	print("Number of agents: " + str(len(nfe)))  
	return nfe
'''
def find_final_edges(scenario): #TT
	passengers = scenario + params.trips_path
	root_passengers = ET.parse(passengers).getroot()

	final_edges = []
	veh_counter = 0
	veh_to_final = {}

	for type_tag in root_passengers.findall('trip'):
		edge = type_tag.get('to')

		if edge not in final_edges:
			final_edges.append(edge)

	for type_tag in root_passengers.findall('trip'):
		veh = type_tag.get('id')
		destination = type_tag.get('to')
		veh_to_final[veh] = destination
		veh_counter = veh_counter + 1

	print("Number of vehicles: ", veh_counter)
	return final_edges, veh_counter, veh_to_final
'''

def find_final_edges(scenario): #TT
	tree = scenario + params.route_path
	root = ET.parse(tree).getroot()

	final_edges = []
	veh_counter = 0
	veh_to_final = {}

	for vehicle in root.findall('vehicle'):
	    # Get the vehicle ID and route edges
	    vehicle_id = vehicle.get('id')
	    route_edges = vehicle.find('route').get('edges')

	    # Split the route edges string into a list of individual edges
	    edge_list = route_edges.split()

	    # Get the last edge in the list
	    last_edge = edge_list[-1]

	    # Add the last edge to the dictionary for this vehicle ID
	    veh_to_final[vehicle_id] = last_edge

	    veh_counter += 1

	    if last_edge not in final_edges:
	    	final_edges.append(last_edge)

	print("Number of vehicles: ", veh_counter)
	print("Final Edges:",len(final_edges))
	return final_edges, veh_counter, veh_to_final

def findMode(scenario):
	routes = scenario + params.route_path
	root_routes = ET.parse(routes).getroot()

	edge_to_edge = {}
	edge_to_number = {}
	veh_edge_number = {}

	for vehicle in root_routes.findall('vehicle'):
		for route in vehicle.findall('route'):
			route_edges = route.get('edges')
			route_edges = route_edges.split()
			for i in range(len(route_edges) - 1):
				if route_edges[i] in edge_to_edge:
					if route_edges[i+1] not in edge_to_edge[route_edges[i]]:
						edge_to_edge[route_edges[i]].append(route_edges[i+1])
				else:
					edge_to_edge[route_edges[i]] = [route_edges[i+1]]

	for i in edge_to_edge:
		c = 0
		for j in edge_to_edge[i]:
			edge_to_number[str(i)+str(j)] = c
			c += 1

	#for i in edge_to_edge:
		#for j in range(len(edge_to_number[i])):
			#edge_to_number[i][j] = edge_to_number[i][j]/len(edge_to_number[i])

	for vehicle in root_routes.findall('vehicle'):
		veh = vehicle.get('id')
		for route in vehicle.findall('route'):
			route_edges = route.get('edges')
			route_edges = route_edges.split()
			for i in range(len(route_edges) - 1):
				tag = veh+route_edges[i]
				veh_edge_number[tag] = edge_to_number[str(route_edges[i])+str(route_edges[i+1])]


	for i in veh_edge_number:
		if veh_edge_number[i] == 1:
			veh_edge_number[i] = 0.5
		if veh_edge_number[i] == 2:
			veh_edge_number[i] = 1

	return veh_edge_number




def findJunctionMeans(int_lanes, lane_lengths):
	junctionMeanLength = {}
	for junction in int_lanes:
		s = 0
		c = 0
		ts = 0
		tc = 0


		if int_lanes[junction] == -1:
			junctionMeanLength[junction] = 8.0

		else:
			for lane in int_lanes[junction]:

				if lane in int_lanes:
					for sub_lane in int_lanes[lane]:

						ts = ts + float(lane_lengths[sub_lane])
						tc = tc +1

					if(tc != 0):
						junctionMeanLength[lane] = ts/tc

					s = s + ts
					c = c + 1

					ts = 0
					tc = 0

				else:

					s = s + float(lane_lengths[lane])
					c = c + 1

			if(c != 0):
				junctionMeanLength[junction] = s/c

	return junctionMeanLength



def find_junctions(scenario):
	junctions_file = scenario + params.net_path
	root_junctions = ET.parse(junctions_file).getroot()
	net = sumolib.net.readNet(scenario + params.net_path)

	junctions = []
	intersections = {}
	inc_edges = {}
	centers = {}
	int_lanes = {}
	int_to_junc = {}
	junction_inc_len = {}
	lane_to_agent = {}
	three_junction_counter = 0
	four_junction_counter = 0

	for type_tag in root_junctions.findall('junction'):
		if type_tag.get('type') != 'priority':
			continue

		value = type_tag.get('id')
		value2 = type_tag.get('incLanes')
		value3 = type_tag.get('intLanes')

		junctions.append(value)
		inc_lanes = value2.split()
		junction_inc_len[value] = len(inc_lanes)
		intLanes = value3.split()

		if len(inc_lanes) == 3:
			three_junction_counter += 1
		if len(inc_lanes) == 4:
			four_junction_counter += 1

		for i in range(len(inc_lanes)):
			junction_type = len(inc_lanes)

			key = inc_lanes[i][:-2]
			lane_to_agent[key] = str(junction_type)+"_"+str(i)

		for inc_lane in inc_lanes:
			if inc_lane not in intersections:
				intersections[inc_lane] = value

			inc_edges[inc_lane] = value


		if intLanes == []:
			int_lanes[value] = -1
		else:
			int_lanes[value] = intLanes

		for lane in intLanes:
			int_to_junc[lane] = value

		value_x = type_tag.get('x')
		value_y = type_tag.get('y')
		centers[value] = {'x':value_x, 'y':value_y}

	for type_tag in root_junctions.findall('junction'):
		if type_tag.get('type') == 'internal':
			inclanes = type_tag.get('incLanes').split()
			intlanes = type_tag.get('intLanes').split()
			for lane in inclanes:
				if lane[0] == ':':
					lane_junction = lane[1:].split("_")[0]
				int_to_junc[lane] = lane_junction

			for lane in intlanes:
				if lane[0] == ':':
					lane_junction = lane[1:].split("_")[0]
				int_to_junc[lane] = lane_junction


	print("number of junctions:", len(junctions))
	print("4-way junctions:", four_junction_counter)
	print("T-junctions:", three_junction_counter)

	return intersections, centers, inc_edges, int_lanes, int_to_junc, junction_inc_len, lane_to_agent



if __name__ == "__main__":
	params = importlib.import_module(str(sys.argv[1]))
	print(str(sys.argv[1]))
	name = params.path
	gui = params.gui
	total_episodes = params.total_episodes
	max_steps = params.max_steps
	sim_runner = "_"

	if not os.path.exists("Experiments"):
		os.makedirs("Experiments")

	if params.BUILD == True:
		os.makedirs("Experiments/Experiment_"+params.save_name)
		os.makedirs("Experiments/Experiment_"+params.save_name+"/agents")
		os.makedirs("Experiments/Experiment_"+params.save_name+"/net")
		os.makedirs("Experiments/Experiment_"+params.save_name+"/results")
		os.makedirs("Experiments/Experiment_"+params.save_name+"/graphs")
		os.makedirs("Experiments/Experiment_"+params.save_name+"/parameters")
		os.makedirs("Experiments/Experiment_"+params.save_name+"/scripts")

		shutil.copy('PPO.py', "Experiments/Experiment_"+params.save_name+"/scripts"+'/PPO.py')
		shutil.copy('smallRandom.py', "Experiments/Experiment_"+params.save_name+"/scripts"+'/smallRandom.py')
		shutil.copy('simRunner2test.py', "Experiments/Experiment_"+params.save_name+"/scripts"+'/simRunner2test.py')
		shutil.copy(params.path+params.net_path, "Experiments/Experiment_"+params.save_name+"/net"+params.net_path)
		shutil.copy(params.path+params.sumocfg, "Experiments/Experiment_"+params.save_name+"/net"+params.sumocfg)
		if params.random == False:
				shutil.copy(params.path+params.trips_path, "Experiments/Experiment_"+params.save_name+"/net"+params.trips_path)
				shutil.copy(params.path+params.route_path, "Experiments/Experiment_"+params.save_name+"/net"+params.route_path)
		shutil.copy(f'{str(sys.argv[1])}.py', "Experiments/Experiment_"+params.save_name+"/parameters/"+str(sys.argv[1])+".py")

	if gui == False:
		sumoBinary = params.sumoBinary
	else:
		sumoBinary = params.sumoBinaryGui

	if params.random == False:
		intersections, centers, inc_edges, int_lanes, int_to_junc, junction_inc_len, lane_to_agent = find_junctions(name)

		#print('intersections',intersections)
		#print('centers',centers)
		#print('inc_edges',inc_edges)
		#print('int_lanes',int_lanes)
		#print('int_to_junc',int_to_junc)		
		final_edges, num_of_vehs, veh_to_final = find_final_edges(name)

		all_edges, lane_lengths = find_all_edges(name)

		nfedges = find_non_final_edges(all_edges,final_edges)

		veh_edge_number = findMode(name)
		print('all edges', len(all_edges))
		print('all nfedges', len(nfedges))
		print('nfedges', len(nfedges))
		sumoCmd = [sumoBinary, "-c", name + params.sumocfg, "--collision.action", "remove", "--collision.mingap-factor", "0",
			   	"--collision.check-junctions", "true", "--no-step-log", "true", "--disable-textures",
			   	"--waiting-time-memory", str(max_steps), "--no-warnings", "true"]

		junctionMeanLength = findJunctionMeans(int_lanes, lane_lengths)


		sim_runner = SimRunner(str(sys.argv[1]), veh_to_final, num_of_vehs, int_to_junc, junctionMeanLength, intersections, centers, final_edges, nfedges, int_lanes, total_episodes, max_steps, sumoCmd, veh_edge_number, junction_inc_len, lane_to_agent) #TT

	print("----- Start time:", datetime.datetime.now())
	for episode in range(params.total_episodes):

		print('----- Episode {} of {}'.format(episode + 1, total_episodes))

		start = timeit.default_timer()
		if params.random == True:
			sim_runner = produce_random_trips(name,sim_runner,episode)
			shutil.copy(params.path+params.trips_path, "Experiments/Experiment_"+params.save_name+"/net"+params.trips_path)
			shutil.copy(params.path+params.route_path, "Experiments/Experiment_"+params.save_name+"/net"+params.route_path)
		sim_runner.run(episode)
		stop = timeit.default_timer()
		print('Time: ', round(stop - start, 1))
		print("+++++++++++++")

	print("----- End time:", datetime.datetime.now())
