import torch as T
import numpy as np
import sys
import os
import random
import timeit
import pandas as pd
import importlib
from PPO import PPO
from PPO2 import PPO2
from PPO3 import PPO3
from PPO4 import PPO4
from PPO5 import PPO5
import torch
from torch import sqrt
import math

np.set_printoptions(threshold=sys.maxsize)
if 'SUMO_HOME' in os.environ:
	tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
	sys.path.append(tools)
else:
	sys.exit("please declare environment variable 'SUMO_HOME'")

import traci


class SimRunner:
	def __init__(self, params, veh_to_final, num_of_vehs, int_to_junc, junctionMeanLength, intersections, centers, final_edges, nfedges, int_lanes, total_episodes, max_steps, sumoCmd, veh_edge_number, junction_inc_len, lane_to_agent):
		self.params_str = params
		self.params = importlib.import_module(self.params_str)
		self.int_to_junc = int_to_junc
		self.num_of_vehs = num_of_vehs
		self.centers = centers
		self.final_edges = final_edges
		self.nfedges = nfedges
		#print(self.nfedges)
		self.junctions = intersections
		#print(self.junctions)
		self.total_episodes = total_episodes
		self.sumoCmd = sumoCmd
		self.max_steps = max_steps
		self.intersections = intersections
		self.max_speed = self.params.max_speed
		self.min_speed = self.params.min_speed
		self.num_features = self.params.STATE_DIM
		self.agents = {}
		self.agents2 = {}
		self.agents3 = {}
		self.agents4 = {}
		self.agents5 = {}
		self.replay_buffers = {}
		self.int_lanes = int_lanes
		self.junctionMeanLength = junctionMeanLength
		self.flag = False
		self.veh_to_final = veh_to_final
		self.epsilon = 0.9
		self.veh_edge_number = veh_edge_number
		self.junction_inc_len = junction_inc_len
		self.reward_counter = {}
		self.state_counter = {}
		self.step_checker = {}
		self.agents_experiences = {}
		self.lane_to_agent = lane_to_agent

	def net_init(self):

		for edge in self.nfedges:
			if edge not in self.agents:
				if self.params.SCENARIO_AGENT[1] == 0:
					self.agents[edge] = PPO(str(edge), self.params_str)
				if self.params.SCENARIO_AGENT[1] == 1:
					self.agents[edge] = PPO2(str(edge), self.params_str)
				if self.params.SCENARIO_AGENT[1] == 2:
					self.agents[edge] = PPO3(str(edge), self.params_str)
				if self.params.SCENARIO_AGENT[1] == 3:
					self.agents[edge] = PPO4(str(edge), self.params_str)
				if self.params.SCENARIO_AGENT[1] == 4:
					self.agents[edge] = PPO5(str(edge), self.params_str)

				if self.params.ENSEMBLE == True:
					self.agents2[edge] = PPO2(str(edge), self.params_str)
					self.agents3[edge] = PPO3(str(edge), self.params_str)
					self.agents4[edge] = PPO4(str(edge), self.params_str)
					self.agents5[edge] = PPO5(str(edge), self.params_str)

				if self.params.TEST == True and self.params.SYN == False:
					self.agents[edge].load()
				elif self.params.TEST == True and self.params.SYN == True and self.params.ENSEMBLE == False:
					
					edge_info = self.lane_to_agent[edge]
					junction_type = edge_info.split("_")[0]
					agent_id = edge_info.split("_")[1]

					print(f'loaded agent [{agent_id} {junction_type}] for edge {edge}')
					self.agents[edge].load_component(junction_type,agent_id)

				elif self.params.TEST == True and self.params.SYN == True and self.params.ENSEMBLE == True:
					edge_info = self.lane_to_agent[edge]
					junction_type = edge_info.split("_")[0]
					agent_id = edge_info.split("_")[1]

					print(f'loaded agent [{agent_id} {junction_type}] for edge {edge}')
					self.agents[edge].load_component(junction_type,agent_id)					
					self.agents2[edge].load_component(junction_type,agent_id)
					self.agents3[edge].load_component(junction_type,agent_id)
					self.agents4[edge].load_component(junction_type,agent_id)
					self.agents5[edge].load_component(junction_type,agent_id)

	def find_leaders(self, vehicles, lanes):

		leaders = {}
		dists = {}
		junction_vehs = {}
		inside_vehs = {}
		uncontested_junctions = []

		for veh_id in vehicles:
			if veh_id not in lanes:
				continue
			lane = lanes[veh_id]
			if lane[:-2] not in self.nfedges:
				if lane[0] == ":":
					junc = lane.split("_")[0][1:]
				else:
					junc = lane

				if junc in inside_vehs:
					inside_vehs[junc].append(veh_id)
				else:
					inside_vehs[junc] = [veh_id]
			else:
				#print(veh_id, "is outside junction")
				junction = self.junctions[lane]
				if junction in junction_vehs:
					junction_vehs[junction].append(veh_id)
				else:
					junction_vehs[junction] = [veh_id]
		'''
		print("VEHS INSIDE JUNCTIONS")
		for i in inside_vehs:
			print(f"junction {i} has")
			print(inside_vehs[i])

		print("VEHS GOING TO JUNCTIONS")
		for i in junction_vehs:
			print(f"junction {i} waits")
			print(junction_vehs[i])		
		'''
		for veh in vehicles:
			veh_lane = traci.vehicle.getLaneID(veh)
			veh_x, veh_y = traci.vehicle.getPosition(veh)
			veh_leader = -1
			veh_dist = 10000

			if veh_lane[:-2] in self.nfedges:
				for other_veh in vehicles:
					if veh != other_veh and traci.vehicle.getLaneID(other_veh)[:-2] == traci.vehicle.getLaneID(veh)[:-2]:
						other_veh_lane = traci.vehicle.getLaneID(other_veh)
						other_veh_x, other_veh_y = traci.vehicle.getPosition(other_veh)					

						a = np.array((veh_x, veh_y))
						b = np.array((other_veh_x, other_veh_y))

						dist = np.linalg.norm(a-b)

						veh_pos = traci.vehicle.getLanePosition(veh)
						other_veh_pos = traci.vehicle.getLanePosition(other_veh)

						if dist < veh_dist and veh_pos < other_veh_pos:
							veh_leader = other_veh
							veh_dist = dist

				#if veh_leader == -1 and self.junctions[traci.vehicle.getLaneID(veh)] in inside_vehs:
					#dist = traci.lane.getLength(traci.vehicle.getLaneID(veh)) - traci.vehicle.getLanePosition(veh)
					#print(f'{veh} has leader inside a junction at dist {dist}')
					#if dist <= 100:
						#veh_dist = traci.lane.getLength(traci.vehicle.getLaneID(veh)) - traci.vehicle.getLanePosition(veh)
						#veh_leader = -2


			leaders[veh] = veh_leader
			dists[veh] = veh_dist

			#print(leaders, dists)

		return leaders, dists


	def choose_action(self, edge, state, veh_id, evalu):
		if(evalu == True or self.params.TEST == True):

			if self.params.ENSEMBLE == True:
				
				simulation_action1, action_tensor1, action_logprob_tensor1, state_tensor1 = self.agents[edge].select_action(state, evalu = True)
				simulation_action2, action_tensor2, action_logprob_tensor2, state_tensor2 = self.agents2[edge].select_action(state, evalu = True)
				simulation_action3, action_tensor3, action_logprob_tensor3, state_tensor3 = self.agents3[edge].select_action(state, evalu = True)
				simulation_action4, action_tensor4, action_logprob_tensor4, state_tensor4 = self.agents4[edge].select_action(state, evalu = True)
				simulation_action5, action_tensor5, action_logprob_tensor5, state_tensor5 = self.agents5[edge].select_action(state, evalu = True)
				
				simulation_action = np.mean(np.array([simulation_action1, simulation_action2,simulation_action3,simulation_action4,simulation_action5]))
				action_tensor = -1
				action_logprob_tensor = -1
				state_tensor = -1
			else:

				simulation_action, action_tensor, action_logprob_tensor, state_tensor = self.agents[edge].select_action(state, evalu = True)

		else:

			simulation_action, action_tensor, action_logprob_tensor, state_tensor = self.agents[edge].select_action(state)
			#print(simulation_action)
		return simulation_action, action_tensor, action_logprob_tensor, state_tensor


	def fuel_consumption(self, fuel):
		total_fuel = 0
		for key in fuel.keys():
			total_fuel += int(fuel[key])

		return total_fuel / self.num_of_vehs

	def calculate_time(self, duration):
		total_duration = 0
		for key in duration.keys():
			total_duration += int(duration[key])

		return total_duration / self.num_of_vehs

	def get_reward(self, veh, dist, speed, next_lane, colliding_vehicles, prev_veh_to_agent, next_veh_to_agent, state, prev_vip, next_vip):
		done = False

		if veh in colliding_vehicles:
			if prev_vip[veh] == 0:
				reward = -5
			elif prev_vip[veh] == 1:
				reward = 0
			else:
				reward = 0
			done = True
			return reward, done

		norm_speed = speed[veh]/self.max_speed
		if next_lane[veh][:-2] == self.veh_to_final[veh] or prev_veh_to_agent[veh] != next_veh_to_agent[veh]:
			reward = 5
			done = True
			return reward, done


		safe_distance = self.params.safe_distance

		distance = safe_distance / (min(dist[veh], safe_distance))

		dist_to_junc = (traci.lane.getLength(traci.vehicle.getLaneID(veh)) - traci.vehicle.getLanePosition(veh))
		if dist_to_junc <= 5:
			dist_to_junc = -1
		else:
			dist_to_junc -= 5

		if dist_to_junc > 95:
			dist_to_junc = -2

		if (next_lane[veh][:-2] not in self.nfedges) or (next_lane[veh][:-2] in self.nfedges and  dist_to_junc == -1):
			if prev_vip[veh] == 0:
				reward = -2 - norm_speed
			if prev_vip[veh] == 1:			
				reward = norm_speed
			if prev_vip[veh] == -1:
				reward = norm_speed
			return reward, done

		if next_vip[veh] == 1:
			reward = norm_speed

		else:

			if distance == 1.0:

				reward = norm_speed

			else:

				reward = -distance*norm_speed


		return reward, done

	def apply_acceleration(self, actions):
		new_speeds = {}
		for veh_id in actions.keys():
			current_speed = traci.vehicle.getSpeed(veh_id)
			
			action = float (actions[veh_id])

			if action >= 0:
				acceleration = action * 3
			else:
				acceleration = action * 5

			#print(acceleration)

			#print(veh_id, acceleration)
			next_speed = min(max([current_speed + acceleration, self.min_speed]), self.max_speed)
			new_speeds[veh_id] = next_speed
			traci.vehicle.setSpeed(veh_id, next_speed)

		#print("----")
		return new_speeds


	def get_state(self, veh_id, leader, leader_distance, veh_to_agent, vip):
		nn_state = []
		lane_id = traci.vehicle.getLaneID(veh_id)

		# Eco-vehicle's velocity
		nn_state.append(traci.vehicle.getSpeed(veh_id) / self.max_speed)
		# Eco-vehicle's lane position

		leader_lane_id = traci.vehicle.getLaneID(veh_id)

		lane_position = traci.vehicle.getLanePosition(veh_id)

		lane_length = traci.lane.getLength(lane_id)
		leader_lane_length = traci.lane.getLength(leader_lane_id)
		
		prev_lane_id = veh_to_agent[veh_id]

		prev_lane_length = traci.lane.getLength(prev_lane_id)

		dist_to_junc = (traci.lane.getLength(traci.vehicle.getLaneID(veh_id)) - traci.vehicle.getLanePosition(veh_id))
		if dist_to_junc <= 5:
			dist_to_junc = -1
		else:
			dist_to_junc -= 5

		if dist_to_junc > 95:
			dist_to_junc = -2

		if lane_id[:-2] not in self.nfedges or (lane_id[:-2] in self.nfedges and dist_to_junc == -1):
			nn_state = []
			nn_state.append(traci.vehicle.getSpeed(veh_id) / self.max_speed)
			nn_state.append(-1.0)
			nn_state.append(-1.0)
			nn_state.append(-1.0)
			nn_state.append(-1.0)
			#return np.full(self.num_features, -1)
			return np.array(nn_state)

		if dist_to_junc == -2:
			nn_state.append(0.0)
		else:
			nn_state.append(dist_to_junc/95)

		if leader == -1:
			nn_state.append(-1.0)
			nn_state.append(-1.0)
		else:			
			nn_state.append(traci.vehicle.getSpeed(str(leader)) / self.max_speed)
			if leader_distance > 100:
				nn_state.append(-1.0)
			else:
				nn_state.append(leader_distance/100)

		nn_state.append(vip[veh_id])

		return np.array(nn_state)


	def get_vip(self, vehicles, speeds, lanes, leaders):


		vip = {}
		in_vehs = {}
		out_vehs = {}
		offline_vehs = {}

		for lane in self.intersections:
			junc = self.intersections[lane]
			if junc not in in_vehs:
				in_vehs[junc] = []
				out_vehs[junc] = []
				offline_vehs[junc] = []

		for veh_id in vehicles:
			if veh_id not in lanes:
				continue
			lane = lanes[veh_id]

			dist_to_junc = (traci.lane.getLength(traci.vehicle.getLaneID(veh_id)) - traci.vehicle.getLanePosition(veh_id))
			if dist_to_junc <= 5:
				dist_to_junc = -1
			else:
				dist_to_junc -= 5

			if dist_to_junc > 95:
				dist_to_junc = -2

			pos = traci.vehicle.getLanePosition(veh_id)
			if lane[:-2] in self.nfedges:

				junction = self.intersections[lane]

				if dist_to_junc == -1:

					in_vehs[junction].append(veh_id)
				
				elif dist_to_junc == -2:

					offline_vehs[junction].append(veh_id)

				else:

					out_vehs[junction].append(veh_id)

			else:

				if lane[:-2] != self.veh_to_final[veh_id]:
					junction = self.int_to_junc[lane]

					in_vehs[junction].append(veh_id)



		for junction in in_vehs:

			if len(in_vehs[junction]) > 0:
				for veh in in_vehs[junction]:
					vip[veh] = -1
					traci.vehicle.setColor(veh,[255,255,0,255])
				if len(out_vehs[junction]) > 0:
					for veh in out_vehs[junction]:
						vip[veh] = 0
						traci.vehicle.setColor(veh,[255,255,0,255])

			else:
				vip_veh = -1
				vip_time = 100000
				for veh_id in out_vehs[junction]:
					speed = speeds[veh_id]
					if speed == 0 : speed = 0.1				
					dist = traci.lane.getLength(traci.vehicle.getLaneID(veh_id)) - traci.vehicle.getLanePosition(veh_id) - 5
					exp_time = dist #/ speed
					if exp_time < vip_time and leaders[veh_id] == -1:
						vip_veh = veh_id
						vip_time = exp_time

				for veh in out_vehs[junction]:
					if veh == vip_veh:
						vip[veh] = 1
						traci.vehicle.setColor(veh,[0,255,0,255])
					else:
						vip[veh] = 0
						traci.vehicle.setColor(veh,[255,255,0,255])


		for veh in vehicles:
			if veh not in vip:
				vip[veh] = 0
				traci.vehicle.setColor(veh,[255,255,0,255])


		for list_veh in list(offline_vehs.values()):
			for veh in list_veh:
				vip[veh] = 0
				traci.vehicle.setColor(veh,[255,0,0,255])


		return vip

	def eucl_dist(self, veh_x, veh_y, center_x, center_y):
		a = np.array((veh_x, veh_y))
		b = np.array((center_x, center_y))

		return np.linalg.norm(a-b)

	def run(self, episode):
			
		#random_seed = 74
		#T.manual_seed(random_seed)
		#np.random.seed(random_seed)
		episode = episode + 1

		if(episode % self.params.EVALUATION_EVERY == 0 and episode != 1):
			print("~ EVALUATION EPISODE START ~")

		self.net_init()
		traci.start(self.sumoCmd)

		#Stats
		total_collisions = 0
		duration = {}
		fuel = {}
		avg_speed = []
		avg_reward = {}
		stats = []
		stats_eval = []
		spawns = {}

		new_experiences = {}
		running_vehs = []
		fins = []

		#T+1 Variables
		next_veh_to_agent = {}


		#T Variables
		prev_veh_to_agent = {}

		step = 0

		while step < self.params.max_steps:

			if(step == 0):
				traci.simulationStep()

			if(step == int(25/100*self.max_steps)):
				print("--- 25% COMPLETE ---")
			elif(step == int(50/100*self.max_steps)):
				print("--- 50% COMPLETE ---")
			elif(step == int(75/100*self.max_steps)):
				print("--- 75% COMPLETE ---")

			next_velocities = {}
			next_leaders = {}
			next_leader_dists = {}
			next_lanes = {}
			next_fc_distances = {}
			next_state = {}
			next_vehicles = []
			next_leaders = {}
			dist_to_junc = {}
			prev_vip = []

			#T Variables
			prev_velocities = {}
			prev_leaders = {}
			prev_leader_dists = {}
			prev_lanes = {}
			prev_fc_distances = {}
			prev_state = {}
			prev_vehicles = []
			prev_leaders = {}


			#Get next veh list
			prev_vehicles = list(traci.vehicle.getIDList())

			for veh_id in prev_vehicles:

				prev_lanes[veh_id] = traci.vehicle.getLaneID(veh_id)

				if veh_id in duration:
					duration[veh_id] += 1
					fuel[veh_id] += traci.vehicle.getFuelConsumption(veh_id)
				else:
					duration[veh_id] = 1
					fuel[veh_id] = traci.vehicle.getFuelConsumption(veh_id)

				if veh_id not in running_vehs:
					#print(veh_id)
					lane = traci.vehicle.getLaneID(veh_id)[:-2]
					if lane not in spawns:
						spawns[lane] = 1
					else:
						spawns[lane] = spawns[lane] + 1
					traci.vehicle.setSpeedMode(veh_id, 32)
					#traci.vehicle.setSpeed(veh_id, 10.0)
					traci.vehicle.setLaneChangeMode(veh_id, 0b001000000000)
					running_vehs.append(veh_id)

				prev_velocities[veh_id] = traci.vehicle.getSpeed(veh_id)

				if prev_lanes[veh_id][:-2] in self.nfedges:
					prev_veh_to_agent[veh_id] = prev_lanes[veh_id]

			'''
			for veh_id in prev_vehicles:
				leader_info = traci.vehicle.getLeader(veh_id)
				prev_leaders[veh_id] = -1
				prev_leader_dists[veh_id] = 1000
				if leader_info is not None and traci.vehicle.getLaneID(leader_info[0])[:-2] == traci.vehicle.getLaneID(veh_id)[:-2] and leader_info[1]:
					leader_id = leader_info[0]
					leader_lane = traci.vehicle.getLaneID(leader_id)
					veh_x, veh_y = traci.vehicle.getPosition(veh_id)
					lead_x, lead_y = traci.vehicle.getPosition(leader_id)

					prev_leaders[veh_id] = leader_id
					prev_leader_dists[veh_id] = self.eucl_dist(veh_x, veh_y, lead_x, lead_y)
			'''

			prev_leaders, prev_leader_dists = self.find_leaders(prev_vehicles, prev_lanes)

			prev_vip = self.get_vip(prev_vehicles, prev_velocities, prev_lanes, prev_leaders)

			for veh_id in prev_vehicles:
				prev_state[veh_id] = self.get_state(veh_id, prev_leaders[veh_id], prev_leader_dists[veh_id], prev_veh_to_agent, prev_vip)

			actions = {}
			real_actions = {}
			speeds = {}

			for veh_id in prev_vehicles:
				if prev_lanes[veh_id][:-2] == self.veh_to_final[veh_id]:
					action = 1
				else:
					action, action_tensor, action_logprob_tensor, state_tensor = self.choose_action(prev_veh_to_agent[veh_id][:-2], prev_state[veh_id], veh_id, (episode % self.params.EVALUATION_EVERY == 0 and episode != 1))
					#if veh_id == '8':
					#if action < 0:
						#print(f'state {prev_state[veh_id]}')
						#print(f'action {action}')
						#print("------")
					if(episode % self.params.EVALUATION_EVERY != 0 or episode == 1 and self.params.TEST == False):
						tag = (str(prev_veh_to_agent[veh_id][:-2]) +","+ str(veh_id))
						if tag in self.agents_experiences:
							self.agents_experiences[tag].append([state_tensor, action_tensor, action_logprob_tensor])
						else:
							self.agents_experiences[tag] = [[state_tensor, action_tensor, action_logprob_tensor]]

				actions[veh_id] = action

			speeds = self.apply_acceleration(actions)

			v_speed = 0
			cars = 0
			for v in speeds:
				v_speed = v_speed + float(speeds[v])
				cars = cars + 1


			if cars != 0:
				avg_speed.append(v_speed / cars)

			traci.simulationStep()

			step += 1

			collisions = traci.simulation.getCollidingVehiclesNumber()
			col_vehicles = list(traci.simulation.getCollidingVehiclesIDList())
			if col_vehicles != []:
				print(step,col_vehicles)
				for vehicle in col_vehicles:
					if vehicle in duration:
						del duration[vehicle]

			#if col_vehicles != []:
				#print(col_vehicles)
			total_collisions += collisions
			next_vehicles = list(traci.vehicle.getIDList())

			for veh_id in next_vehicles:

				if veh_id in prev_vehicles:

					next_lanes[veh_id] = traci.vehicle.getLaneID(veh_id)

					if next_lanes[veh_id][:-2] in self.nfedges:
						next_veh_to_agent[veh_id] = next_lanes[veh_id]
					else:
						next_veh_to_agent[veh_id] = prev_veh_to_agent[veh_id]

			'''
			for veh_id in next_vehicles:
				if veh_id in prev_vehicles:
					leader_info = traci.vehicle.getLeader(veh_id)
					next_leaders[veh_id] = -1
					next_leader_dists[veh_id] = 1000
					if leader_info is not None and traci.vehicle.getLaneID(leader_info[0])[:-2] == traci.vehicle.getLaneID(veh_id)[:-2]:
						leader_id = leader_info[0]
						leader_lane = traci.vehicle.getLaneID(leader_id)
						veh_x, veh_y = traci.vehicle.getPosition(veh_id)
						lead_x, lead_y = traci.vehicle.getPosition(leader_id)
						if next_veh_to_agent[leader_id][:-2] == next_veh_to_agent[veh_id][:-2]:
							next_leaders[veh_id] = leader_id
							next_leader_dists[veh_id] = self.eucl_dist(veh_x, veh_y, lead_x, lead_y)
						else:
							next_leaders[veh_id] = -1
							next_leader_dists[veh_id] = 1000
			'''

			next_leaders, next_leader_dists = self.find_leaders(next_vehicles, next_lanes)

			for veh_id in next_vehicles:
				if veh_id in prev_vehicles:
					next_velocities[veh_id] = traci.vehicle.getSpeed(veh_id)

			next_vip = self.get_vip(next_vehicles, next_velocities, next_lanes, next_leaders)


			for veh_id in prev_vehicles:
				if(prev_lanes[veh_id][:-2] not in self.final_edges):
					if veh_id in col_vehicles or prev_veh_to_agent[veh_id] != next_veh_to_agent[veh_id] or next_lanes[veh_id][:-2] == self.veh_to_final[veh_id]:
						next_state[veh_id] = np.full(self.num_features, -2)

					else: 

						next_state[veh_id] = self.get_state(veh_id, next_leaders[veh_id], next_leader_dists[veh_id], next_veh_to_agent, next_vip)
					reward = self.get_reward(veh_id, next_leader_dists, next_velocities,
											next_lanes, col_vehicles, prev_veh_to_agent, next_veh_to_agent, prev_state[veh_id], prev_vip, next_vip)
					tag = (str(prev_veh_to_agent[veh_id][:-2]) +","+ str(veh_id))

					if(step == self.max_steps):
						if(episode % self.params.EVALUATION_EVERY != 0 or episode == 1 and self.params.TEST == False):
							self.agents_experiences[tag][-1].extend([reward[0], True])
						if veh_id not in avg_reward:
							avg_reward[veh_id] = [reward[0]]
						else:
							avg_reward[veh_id].append(reward[0])

					else:
						if(episode % self.params.EVALUATION_EVERY != 0 or episode == 1 and self.params.TEST == False):
							self.agents_experiences[tag][-1].extend([reward[0], reward[1]])
						if veh_id not in avg_reward:
							avg_reward[veh_id] = [reward[0]]
						else:
							avg_reward[veh_id].append(reward[0])
						#if veh_id in ['13','14']:
							#print(veh_id, reward[0])
						#if(veh_id == '0'):
							#print(reward[0])
						#if veh_id == '178':
							#print(reward[0], reward[1])

		traci.close()

		if self.params.train_steps <= episode:
			for agent in self.agents:
				self.agents[agent].adjust_lr()

		if (episode % self.params.ACTION_STD_UPDATE == 0):
			for agent in self.agents:
				self.agents[agent].decay_action_std(self.params.ACTION_STD_DECAY_RATE, self.params.MIN_ACTION_STD)

		if(self.params.TEST == False and episode % self.params.TRAIN_EVERY == 0 and episode % self.params.EVALUATION_EVERY != 0):
			for pair in self.agents_experiences:
				agent, veh_id = pair.split(",")
				#print(agent, veh_id)

				flag = False
				for experience in self.agents_experiences[pair]:
					if len(experience) != 5:
						print(f'{veh_id} at agent {agent} had an illegal experience')
						flag = True

				if flag == True:
					continue

				for experience in self.agents_experiences[pair]:
					self.agents[agent].buffer.states.append(experience[0])
					self.agents[agent].buffer.actions.append(experience[1])
					self.agents[agent].buffer.logprobs.append(experience[2])
					self.agents[agent].buffer.rewards.append(experience[3])
					self.agents[agent].buffer.is_terminals.append(experience[4])

				self.agents[agent].buffer.capacity += 1

			print("--- UPDATING AGENTS! ---")
			for agent in self.agents:
				if self.agents[agent].buffer.capacity >= 1:
					self.agents[agent].update()

			for agent in self.agents:
				self.agents[agent].buffer.clear()
				self.agents[agent].buffer.capacity = 0	

			self.agents_experiences = {}

		if(episode % self.params.SAVE_EVERY == 0 and self.params.TEST == False and self.params.SAVE_AGENTS == True):
			print("!--- Saving Agents! ---!")
			for agent in self.agents:
				self.agents[agent].save()

		avg_speed = sum(avg_speed) / len(avg_speed)
		#avg_reward = sum(avg_reward) / len(avg_reward)

		avg_rewards = []
		for veh in avg_reward:
			avg_rewards.append(sum(avg_reward[veh]) / len(avg_reward[veh]))

		avg_rewards = sum(avg_rewards) / len(avg_rewards)

		mean_durations = []

		for i in duration:
			mean_durations.append(duration[i])

		duration = np.mean(mean_durations)


		print("!--- Episode Results ---!")
		print("SPAWNS:",spawns)
		print(f"Action STD: {self.agents[random.choice(list(self.agents))].action_std}")
		print("Average Speed: {}".format(avg_speed))
		print("Collisions: {}".format(total_collisions / 2))
		print("Duration: {}".format(duration))
		print("Average Fuel Consumption: {}".format(self.fuel_consumption(fuel)))
		print("Average Reward: {}".format(avg_rewards))

		if(episode % self.params.EVALUATION_EVERY == 0 and episode != 1):
			print("~ EVALUATION EPISODE END ~")

		if(self.params.PRINT_STATS == True):
			if(self.params.TEST == False):
				if(episode % self.params.EVALUATION_EVERY != 0):
					cur_data = [episode, avg_speed, total_collisions/2, duration, self.fuel_consumption(fuel), avg_rewards]
					stats.append(cur_data)
					stats = pd.DataFrame(stats, index = [str(episode)], columns = ["episode","average speed","collisions","duration","fuel consumption","average reward"])
					stats.to_csv("Experiments/Experiment_"+self.params.save_name+"/results/results:"+str(self.params.save_name)+'.csv', mode='a', index=False, header=False)
					os.system("python3 graphMaker.py Experiments/Experiment_"+self.params.save_name+"/results/results:"+str(self.params.save_name) + " Experiments/Experiment_"+self.params.save_name+"/graphs/")
				else:
					cur_data = [episode, avg_speed, total_collisions/2, duration, self.fuel_consumption(fuel), avg_rewards]
					stats.append(cur_data)
					stats = pd.DataFrame(stats, index = [str(episode)], columns = ["episode","average speed","collisions","duration","fuel consumption","average reward"])
					stats.to_csv("Experiments/Experiment_"+self.params.save_name+"/results/results:"+str(self.params.save_name)+'_eval.csv', mode='a', index=False, header=False)
					os.system("python3 graphMaker.py Experiments/Experiment_"+self.params.save_name+"/results/results:"+str(self.params.save_name)+'_eval'+ " Experiments/Experiment_"+self.params.save_name+"/graphs/" + " eval")
			else:

				filename = 'experiments.csv'

				# Check if file exists
				file_exists = os.path.isfile(filename)

				# If file doesn't exist, create it and add header row
				if not file_exists:
				    with open(filename, mode='w', newline='') as f:
				        writer = csv.writer(f)
				        writer.writerow(['Network', 'Scenario', 'Agent', 'Average Speed', 'Collisions', 'Average Duration'])

				# Append data row
				with open(filename, mode='a', newline='') as f:
				    writer = csv.writer(f)
				    writer.writerow([self.params.NETWORK, self.params.SCENARIO_AGENT[0], self.params.SCENARIO_AGENT[1], avg_speed, total_collisions/2],duration)
				    