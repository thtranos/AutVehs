import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import importlib
import numpy as np
from random import sample
from random import choice

################################## set device ##################################
# set device to cpu or cuda

device = torch.device('cpu')
if(torch.cuda.is_available()): 
	device = torch.device('cuda:0') 
	torch.cuda.empty_cache()
	print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
	print("Device set to : cpu")


################################## PPO Policy ##################################
class RolloutBuffer:
	def __init__(self):
		self.actions = []
		self.states = []
		self.logprobs = []
		self.rewards = []
		self.is_terminals = []
		self.capacity = 0

	def clear(self):
		del self.actions[:]
		del self.states[:]
		del self.logprobs[:]
		del self.rewards[:]
		del self.is_terminals[:]
		self.capacity = 0

	def remove_first(self):
		index = self.is_terminals.index(True)

		del self.actions[0:index+1]
		del self.states[0:index+1]
		del self.logprobs[0:index+1]
		del self.rewards[0:index+1]
		del self.is_terminals[0:index+1]
		self.capacity += 1

	def trim(self, maxi):
		if capacity > maxi and maxi != 0:
			for _ in range(capacity-maxi):
				self.remove_first


class ActorCritic(nn.Module):
	def __init__(self, params):
		super(ActorCritic, self).__init__()
		self.params_str = params
		self.params = importlib.import_module(self.params_str)
		non = self.params.NUMBER_OF_NEURONS
		non2 = self.params.NUMBER_OF_NEURONS2
		
		self.action_dim = self.params.ACTION_DIM
		self.std = self.params.ACTION_STD_INIT
		self.action_var = torch.full((self.action_dim,), self.params.ACTION_STD_INIT * self.params.ACTION_STD_INIT).to(device)
		# actor
		self.actor = nn.Sequential(
						nn.Linear(self.params.STATE_DIM, non),
						nn.ReLU(),
						nn.Linear(non, non2),
						nn.ReLU(),
						nn.Linear(non2, 1),
						nn.Tanh()
					)
		# critic
		self.critic = nn.Sequential(
						nn.Linear(self.params.STATE_DIM, non),
						nn.ReLU(),
						nn.Linear(non, non2),
						nn.ReLU(),
						nn.Linear(non2, self.action_dim)
					)


	def set_action_std(self, new_action_std):
		#self.std = new_action_std
		self.action_var = torch.full((self.params.ACTION_DIM,), new_action_std * new_action_std).to(device)

	def forward(self):
		raise NotImplementedError
	
	def act(self, state, evalu=False):
		action_mean = self.actor(state)
		cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
		
		dist = MultivariateNormal(action_mean, cov_mat)

		#dist = Normal(action_mean, self.std)

		if(evalu == False):
			action = dist.sample().clip(-1,1)

		else:

			action = action_mean

		action_logprob = dist.log_prob(action)
		
		return action.detach(), action_logprob.detach()
	
	def evaluate(self, state, action):

		action_mean = self.actor(state).clip(-1,1)

		action_var = self.action_var.expand_as(action_mean)
		cov_mat = torch.diag_embed(action_var).to(device)
		dist = MultivariateNormal(action_mean, cov_mat)

		#dist = Normal(action_mean, self.std)
		
		# For Single Action Environments.
		if self.action_dim == 1:
			action = action.reshape(-1, self.action_dim)

		action_logprobs = dist.log_prob(action)
		dist_entropy = dist.entropy()
		state_values = self.critic(state)

		return action_logprobs, state_values, dist_entropy


class PPO3:
	def __init__(self, name, params):

		self.params_str = params
		self.params = importlib.import_module(self.params_str)
		self.action_std = self.params.ACTION_STD_INIT
		self.gamma = self.params.GAMMA
		self.eps_clip = self.params.EPS_CLIP
		self.K_epochs = self.params.LEARN_NUM
		self.action_dim = self.params.ACTION_DIM
		self.state_dim = self.params.STATE_DIM
		self.buffer = RolloutBuffer()
		self.name = name
		self.trajectory_counter = 0
		self.entropy = self.params.ENTROPY

		self.policy = ActorCritic(self.params_str).to(device)
		self.optimizer = torch.optim.Adam([
						{'params': self.policy.actor.parameters(), 'lr': self.params.LR_ACTOR, 'weight_decay': self.params.WD_ACTOR},
						{'params': self.policy.critic.parameters(), 'lr': self.params.LR_CRITIC, 'weight_decay': self.params.WD_CRITIC}
					])

		self.policy_old = ActorCritic(self.params_str).to(device)
		self.policy_old.load_state_dict(self.policy.state_dict())
		
		self.MseLoss = nn.MSELoss()

	def set_action_std(self, new_action_std):
		self.action_std = new_action_std
		self.policy.set_action_std(new_action_std)
		self.policy_old.set_action_std(new_action_std)
		if new_action_std == self.params.MIN_ACTION_STD:
			self.entropy = 0

	def get_batch_indeces(self, epochs, terminals):
		batches = []
		batch = []
		b_per_e = []
		c = 0
		for t in terminals:
			if t:
				batch.append(c)
				batches.append(batch)
				batch = []
				c += 1
			else:
				batch.append(c)
				c += 1

		batch_size = len(batches)//20
		#batch_size = self.params.BATCH_SIZE
		for i in range(epochs):
			sample_i = sample(batches, batch_size)
			#sample_i = choice(batches, size = batch_size, replace = False)

			epoch_i_indeces = []
			for s in sample_i:
				epoch_i_indeces.extend(s)

			b_per_e.append(epoch_i_indeces)

		return b_per_e

	def decay_action_std(self, action_std_decay_rate, min_action_std):
		self.action_std = self.action_std - action_std_decay_rate
		self.action_std = round(self.action_std, 4)
		if (self.action_std <= min_action_std):
			self.action_std = min_action_std
		else:
			self.set_action_std(self.action_std)

	def select_action(self, state, evalu = False):

		with torch.no_grad():
			state = torch.FloatTensor(state).to(device)
			action, action_logprob = self.policy_old.act(state, evalu)

		simulation_action = action.detach().cpu().numpy().flatten()
		#self.buffer.states.append(state)
		#self.buffer.actions.append(action)
		#self.buffer.logprobs.append(action_logprob)

		return simulation_action, action, action_logprob, state

	def update(self):
		# Monte Carlo estimate of returns
		returns = []
		discounted_reward = 0
		for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
			if is_terminal:
				discounted_reward = 0
			discounted_reward = reward + (self.gamma * discounted_reward)
			returns.insert(0, discounted_reward)
			
		returns = torch.tensor(returns, dtype=torch.float32).to(device)

		# Normalizing the rewards
		#returns = (returns - returns.mean()) / (returns.std() + 1e-8)

		# convert list to tensor
		old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
		old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
		old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)


		value_loss_epoch = 0 
		action_loss_epoch = 0

		# Optimize policy for K epochs
		for i in range(self.K_epochs):


			logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

			# match state_values tensor dimensions with rewards tensor
			state_values = torch.squeeze(state_values)

			"""
			advantages = []
			advantage = 0
			next_value = 0

			for r, t, v in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals), reversed(state_values.detach())):
				if t:
					discounted_reward = 0
					next_value = 0
				td_error = r + next_value * self.gamma - v
				advantage = td_error + (advantage * self.gamma * 0.95)
				next_value = v
				advantages.insert(0, advantage)

			advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
			advantages = (advantages - advantages.mean()) / advantages.std()
			"""

			# Finding the ratio (pi_theta / pi_theta__old)
			ratios = torch.exp(logprobs - old_logprobs.detach())

			# Finding Surrogate Loss
			advantages = returns - state_values.detach()   
			surr1 = ratios * advantages
			surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

			# final loss of clipped objective PPO

			#loss_1 = -torch.min(surr1, surr2)
			#loss_2 = self.MseLoss(state_values, returns)


			#print('actor loss mean', torch.mean(loss_1), 'critic loss mean',torch.mean(loss_2))

			loss = -torch.min(surr1, surr2) + 0.1 * self.MseLoss(state_values, returns) #- self.entropy*dist_entropy

			#print(f"epoch {_}: loss {torch.min(surr1, surr2).mean()}")
			# take gradient step
			self.optimizer.zero_grad()
			#torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
			loss.mean().backward()
			self.optimizer.step()


			value_loss_epoch += self.MseLoss(state_values, returns).item()
			action_loss_epoch += -torch.min(surr1, surr2).mean().item() 
		#print("-----------------------------------")
			
		# Copy new weights into old policy
		self.policy_old.load_state_dict(self.policy.state_dict())

		# clear buffer
		#self.buffer.clear()

		return(value_loss_epoch/self.K_epochs,action_loss_epoch/self.K_epochs)

	def adjust_lr(self):
		for g in self.optimizer.param_groups:
			g['lr'] = g['lr'] * 0.98



	def save(self):
		torch.save(self.policy_old.state_dict(), "Experiments/Experiment_"+self.params.save_name+"/agents/" + self.name)
   
	def load(self):
		self.policy_old.load_state_dict(torch.load("Experiments/Experiment_"+self.params.save_name+"/agents/" + self.name, map_location=lambda storage, loc: storage))
		self.policy.load_state_dict(torch.load("Experiments/Experiment_"+self.params.save_name+"/agents/" + self.name, map_location=lambda storage, loc: storage))


	def load_component(self, component, agent):
		#self.policy_old.load_state_dict(torch.load(f"Experiments/Experiment_component_{component}_random_22/agents/{agent}", map_location=lambda storage, loc: storage))
		#self.policy.load_state_dict(torch.load(f"Experiments/Experiment_component_{component}_random_22/agents/{agent}", map_location=lambda storage, loc: storage))
		self.policy_old.load_state_dict(torch.load(f"Experiments/Experiment_component_{component}_random_402/agents/{agent}", map_location=lambda storage, loc: storage))
		self.policy.load_state_dict(torch.load(f"Experiments/Experiment_component_{component}_random_402/agents/{agent}", map_location=lambda storage, loc: storage))
