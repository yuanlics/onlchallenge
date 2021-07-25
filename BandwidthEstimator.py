#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from cacb.cacb import ContinuousActionContextualBanditModel
import pickle

import torch
import numpy as np
from utils.packet_info import PacketInfo
from utils.packet_record import PacketRecord
from deep_rl.actor_critic import ActorCritic
from collections import deque


UNIT_K = 1000
UNIT_M = 1000000
MAX_BANDWIDTH_MBPS = 8
MIN_BANDWIDTH_MBPS = 0.01
LOG_MAX_BANDWIDTH_MBPS = np.log(MAX_BANDWIDTH_MBPS)
LOG_MIN_BANDWIDTH_MBPS = np.log(MIN_BANDWIDTH_MBPS)


def liner_to_log(value):
    # from 10kbps~8Mbps to 0~1
    value = np.clip(value / UNIT_M, MIN_BANDWIDTH_MBPS, MAX_BANDWIDTH_MBPS)
    log_value = np.log(value)
    return (log_value - LOG_MIN_BANDWIDTH_MBPS) / (LOG_MAX_BANDWIDTH_MBPS - LOG_MIN_BANDWIDTH_MBPS)


def log_to_linear(value):
    # from 0~1 to 10kbps to 8Mbps
    value = np.clip(value, 0, 1)
    log_bwe = value * (LOG_MAX_BANDWIDTH_MBPS - LOG_MIN_BANDWIDTH_MBPS) + LOG_MIN_BANDWIDTH_MBPS
    return np.exp(log_bwe) * UNIT_M


def append_log(s):
    LOG_FILE = 'webrtc.log'
    with open(LOG_FILE, 'a') as f:
        f.write(s)
        f.write('\n')


class Estimator(object):
    def __init__(self, step_time=200):  # from rtc_env.py -> GymEnv -> init
        self.packet_record = PacketRecord()
        self.packet_record.reset()
        self.step_time = step_time
        
        model_path = "cacb.pkl"
        state = np.array([0.0, 0.0, 0.0, 0.0])
        with open(model_path, 'rb') as f:
            self.cacb = pickle.load(f)
        action, _ = self.cacb.predict(state, epsilon=0, exploration_width=0, exploration_strategy=None)
        self.bandwidth_prediction = round(log_to_linear(action))
        
#         state_dim = 4
#         action_dim = 1
#         exploration_param = 0.05
#         model_path = "./model/ppo_2021_07_25_04_32_42.pth"
        
# #         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.device = torch.device("cpu")
#         self.model = ActorCritic(state_dim, action_dim, exploration_param, self.device).to(self.device)
#         self.model.load_state_dict(torch.load(model_path))
#         self.model.random_action = False

# #         self.history = deque(maxlen=buffer_size)

#         states = [0.0, 0.0, 0.0, 0.0]
#         torch_tensor_states = torch.FloatTensor(torch.Tensor(states).reshape(1, -1)).to(self.device)
#         action, _, _ = self.model.forward(torch_tensor_states)
#         self.bandwidth_prediction = round(log_to_linear(action.item()))
        
        append_log(f'MY INIT BWE: {self.bandwidth_prediction/UNIT_M:.6f} mbps')
        self.last_call = "init"
        

    def report_states(self, stats: dict):
        '''
        stats is a dict with the following items
        {
            "send_time_ms": uint,
            "arrival_time_ms": uint,
            "payload_type": int,
            "sequence_number": uint,
            "ssrc": int,
            "padding_length": uint,
            "header_length": uint,
            "payload_size": uint
        }
        '''
        self.last_call = "report_states"
        # collect data  # from rtc_env.py -> GymEnv -> step
        packet_info = PacketInfo()
        packet_info.payload_type = stats["payload_type"]
        packet_info.ssrc = stats["ssrc"]
        packet_info.sequence_number = stats["sequence_number"]
        packet_info.send_timestamp = stats["send_time_ms"]
        packet_info.receive_timestamp = stats["arrival_time_ms"]
        packet_info.padding_length = stats["padding_length"]
        packet_info.header_length = stats["header_length"]
        packet_info.payload_size = stats["payload_size"]
        packet_info.bandwidth_prediction = self.bandwidth_prediction
        self.packet_record.on_receive(packet_info)
        

    def get_estimated_bandwidth(self)->int:
        from sklearn.ensemble import GradientBoostingRegressor
        
        if self.last_call and self.last_call == "report_states":
            self.last_call = "get_estimated_bandwidth"
            
            # from rtc_env.py -> GymEnv -> step
            state = []
            receiving_rate = self.packet_record.calculate_receiving_rate(interval=self.step_time)  # bps
            state.append(liner_to_log(receiving_rate))  # 01norm
            delay = self.packet_record.calculate_average_delay(interval=self.step_time)  # ms
            state.append(min(delay/1000, 1))  # enforce delay(s) <= 1s
            loss_ratio = self.packet_record.calculate_loss_ratio(interval=self.step_time)  # ratio
            state.append(loss_ratio)
            latest_prediction = self.packet_record.calculate_latest_prediction()  # bps
            state.append(liner_to_log(receiving_rate) - liner_to_log(latest_prediction))  # 01norm over_estimation
            append_log(f'MY STATE: [{receiving_rate/UNIT_M:.6f} mbps, {state[1]:.3f} s, {state[2]:.3f}, {state[3]:.3f}]')
            
            # from main.py
#             state = torch.FloatTensor(state).reshape(1, -1).to(self.device)
            
            # from ppo_agent.py -> PPO -> select_action
#             action, _, _ = self.model.forward(state)
#             self.bandwidth_prediction = round(log_to_linear(action.item()))

            action, _ = self.cacb.predict(state, epsilon=0, exploration_width=0, exploration_strategy=None)
            self.bandwidth_prediction = round(log_to_linear(action))
        
            append_log(f'MY BWE: {self.bandwidth_prediction/UNIT_M:.6f} mbps')
        
        return self.bandwidth_prediction
