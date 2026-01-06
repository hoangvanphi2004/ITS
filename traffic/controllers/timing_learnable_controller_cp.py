import traci
import torch
from traffic.config import FIXED_TIME_PLANS
from traffic.controllers.signal_controller import SignalController
import numpy as np

class TimingLearnableController(SignalController):
	def __init__(self, agent, buffer, metrics, model_paths = None, device='cuda', min_green_time=5, max_green_time=80, save_path=None):
		import os
		self.min_green_time = min_green_time
		self.max_green_time = max_green_time
		self.device = device
		self.agent = agent
		self.buffer = buffer
		self.metrics = metrics
		self.save_path = save_path
		if self.save_path is not None and not os.path.exists(self.save_path):
			os.makedirs(self.save_path, exist_ok=True)
		self.current_phase = {tls_id: len(FIXED_TIME_PLANS[tls_id]) - 2 for tls_id in FIXED_TIME_PLANS}
		self.turn_time = {tls_id: 0 for tls_id in FIXED_TIME_PLANS}
		self.duration = {tls_id: 0 for tls_id in FIXED_TIME_PLANS}
		self.is_yellow = {tls_id: False for tls_id in FIXED_TIME_PLANS}
		self.total_reward = {tls_id: 0.0 for tls_id in FIXED_TIME_PLANS}
		# For delta-based reward: store last waiting time per TLS
		self.last_waiting_time = {tls_id: 0.0 for tls_id in FIXED_TIME_PLANS}
		self.previous_sample = {tls_id: {
			'state': None,
			'action': 0,
			'logprob': 0.0,
			'reward': 0.0,
			'value': 0.0,
			'done': False
		} for tls_id in FIXED_TIME_PLANS}

	def reset(self, current_time):
		self.previous_sample = {tls_id: {
			'state': None,
			'action': 0,
			'logprob': 0.0,
			'reward': 0.0,
			'value': 0.0,
			'done': False
		} for tls_id in FIXED_TIME_PLANS}
		self.current_phase = {tls_id: len(FIXED_TIME_PLANS[tls_id]) - 2 for tls_id in FIXED_TIME_PLANS}
		self.turn_time = {tls_id: 0 for tls_id in FIXED_TIME_PLANS}
		self.duration = {tls_id: 0 for tls_id in FIXED_TIME_PLANS}
		self.total_reward = {tls_id: 0.0 for tls_id in FIXED_TIME_PLANS}
		self.last_waiting_time = {tls_id: 0.0 for tls_id in FIXED_TIME_PLANS}
	
	# Implement later
	def get_reward(self, tls_id):
		#print(f"Getting reward for TLS {tls_id}")
		#print(f"  Current waiting time: {self.metrics.get_current_waiting_time_at_tls(tls_id) / 100.0:.4f}")
		return -self.metrics.get_current_waiting_time_at_tls(tls_id) / 100.0
	
	def add_sample_to_buffer(self, tls_id):
		buffer = self.buffer
		sample = self.previous_sample[tls_id]
		buffer.store(sample['state'], sample['action'], sample['logprob'], sample['reward'], sample['done'], sample['value'])

	def add_last_sample_to_buffer(self, tls_id):
		buffer = self.buffer
		sample = self.previous_sample[tls_id]
		sample['reward'] = self.get_reward(tls_id)
		sample['done'] = True
		buffer.store(sample['state'], sample['action'], sample['logprob'], sample['reward'], sample['done'], sample['value'])

	def fishish_path_and_get_batch(self, tls_id, current_time):
		buffer = self.buffer
		last_value = self.agent.net.forward(torch.tensor(self.build_obs(tls_id, current_time), dtype=torch.float32, device=self.device).unsqueeze(0))[2].item()
		batch = self.agent.finish_path(last_value, buffer=buffer)
		return batch
	
	def update_agent(self, batch):
		self.agent.update(batch)

	def build_obs(self, tls_id, current_time):
		# 1. Queue length của tất cả các intersection (theo thứ tự tls_id)
		queue_lengths = []
		for tid in FIXED_TIME_PLANS:
			lanes = traci.trafficlight.getControlledLanes(tid)
			for lane in lanes:
				queue_lengths.append(len(traci.lane.getLastStepVehicleIDs(lane)))
		# 2. One-hot encode tls_id
		tls_ids = list(FIXED_TIME_PLANS.keys())
		one_hot = [1 if tid == tls_id else 0 for tid in tls_ids]
		# 3. remain_time của tất cả các tls_id
		turn_times = [self.turn_time[tid] for tid in tls_ids]
		# 4. phase hiện tại của tất cả các tls_id
		phases = [self.current_phase[tid] for tid in tls_ids]
		duration = [self.duration[tid] for tid in tls_ids]
		# Gộp lại thành 1 vector

		# print(f"DEBUG: Building obs for TLS {tls_id} at time {current_time:.1f}s")
		# print(f"  One-hot: {one_hot}")
		# print(f"  Queue lengths: {queue_lengths}")
		# print(f"  Turn times: {turn_times}")
		# print(f"  Phases: {phases}")
		# print(f"  Durations: {duration}")
		obs = np.array(one_hot + [current_time] + queue_lengths + turn_times + phases + duration, dtype=np.float32)
		#print(obs)
		return obs

	def next_green_phase(self, tls_id):
		current_phase = self.current_phase[tls_id]
		for i in range(1, len(FIXED_TIME_PLANS[tls_id]) + 1):
			next_phase = (current_phase + i) % len(FIXED_TIME_PLANS[tls_id])
			state = FIXED_TIME_PLANS[tls_id][next_phase][1]
			if 'G' in state:
				return next_phase
		return current_phase

	def apply_control(self, tls_id, current_time, **kwargs):
		turn_time = self.turn_time[tls_id]
		duration = self.duration[tls_id]
		if self.is_yellow[tls_id]:
			if current_time - turn_time >= 3:
				"""print(f"[{current_time:.1f}s] TLS {tls_id} within yellow phase transition.")
				print(f"  Previous phase: {self.current_phase[tls_id]}")
				print(f"  Next phase: {self.next_green_phase(tls_id)}")"""
				# if(self.next_green_phase(tls_id) - (self.current_phase[tls_id] + 1) % len(FIXED_TIME_PLANS[tls_id])) != 0:
				# 	print(f"WARNING: TLS {tls_id} yellow phase transition error at time {current_time:.1f}s")
				# 	print(f"  Current phase: {self.current_phase[tls_id]}")
				# 	print(f"  Next green phase: {self.next_green_phase(tls_id)}")
				# 	print(f"  tls_id phases: {FIXED_TIME_PLANS[tls_id]}	")
				# 	print(f"  Current time: {current_time}, turn_time: {turn_time}, duration: {duration}")
				self.current_phase[tls_id] = self.next_green_phase(tls_id)
				state = FIXED_TIME_PLANS[tls_id][self.current_phase[tls_id]]
				traci.trafficlight.setRedYellowGreenState(tls_id, state[1])
				self.is_yellow[tls_id] = False
			else:
				pass
		if current_time - turn_time >= duration + 3:
			agent = self.agent
			self.duration[tls_id] = 0  # Reset duration to avoid multiple triggers
			obs = self.build_obs(tls_id, current_time)

			self.previous_sample[tls_id]['reward'] = self.get_reward(tls_id)
			self.total_reward[tls_id] += self.previous_sample[tls_id]['reward']

			self.previous_sample[tls_id]['done'] = False
			if(self.previous_sample[tls_id]['state'] is not None):
				self.add_sample_to_buffer(tls_id)

			action, logprob, value = agent.select_action(obs)
			next_timing = action
			
			self.turn_time[tls_id] = current_time
			self.duration[tls_id] = next_timing.item()
			self.current_phase[tls_id] = (self.current_phase[tls_id] + 1) % len(FIXED_TIME_PLANS[tls_id])
			self.is_yellow[tls_id] = True
			state = FIXED_TIME_PLANS[tls_id][self.current_phase[tls_id]]

			self.previous_sample[tls_id]['state'] = obs
			self.previous_sample[tls_id]['action'] = action.item()
			self.previous_sample[tls_id]['value'] = value.item()
			self.previous_sample[tls_id]['logprob'] = logprob.item()
			self.previous_sample[tls_id]['done'] = True
			
			#print(f"[{current_time:.1f}s] TLS {tls_id} set to phase {self.current_phase[tls_id]} for {self.duration[tls_id]}s (action: {action.item():.4f}, logprob: {logprob.item():.4f}, value: {value.item():.4f})")
			try:
				traci.trafficlight.setRedYellowGreenState(tls_id, state[1])
				#print(f"DEBUG: TLS {tls_id} with current phase {self.current_phase[tls_id]} state set to {state[1]}")
			except Exception as e:
				print(f"Error setting state for {tls_id}: {e}")
		else:
			pass
	def save_models(self):
		for tls_id in FIXED_TIME_PLANS:
			model_file = f"{self.save_path}/agent_{tls_id}.pt"
			self.agent.save_model(model_file)

