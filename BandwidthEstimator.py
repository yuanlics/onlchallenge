#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

LOG_FILE = 'webrtc.log'
STATIC_BWE = 20 * UNIT_M


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
    with open(LOG_FILE, 'a') as f:
        f.write(s)
        f.write('\n')


class Estimator(object):
    def __init__(self, model_path="./model/ppo_2021_07_20_16_07_41.pth", step_time=200):  # from rtc_env.py -> GymEnv -> init
        # model parameters
        state_dim = 4
        action_dim = 1
        # the std var of action distribution
        exploration_param = 0.05
        buffer_size = 5  # history packet records
        
        # load model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(state_dim, action_dim, exploration_param, self.device).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.random_action = False
        # the model to get the input of model
        self.packet_record = PacketRecord()
        self.packet_record.reset()
        self.history = deque(maxlen=buffer_size)
        self.step_time = step_time
#         # init
#         states = [0.0, 0.0, 0.0, 0.0]
#         torch_tensor_states = torch.FloatTensor(torch.Tensor(states).reshape(1, -1)).to(self.device)
#         action, _, _ = self.model.forward(torch_tensor_states)
# #         self.bandwidth_prediction = log_to_linear(action)
        action = STATIC_BWE  # DEBUG

        self.bandwidth_prediction = action
        append_log(f'M INIT BWE: {self.bandwidth_prediction} bps')
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
        # clear data  # from rtc_env.py -> GymEnv -> step
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
        if self.last_call and self.last_call == "report_states":
            self.last_call = "get_estimated_bandwidth"

            # DEBUG  # from rtc_env.py -> GymEnv -> step
            states = []
            receiving_rate = self.packet_record.calculate_receiving_rate(interval=self.step_time) / UNIT_M  # mbps
            states.append(receiving_rate)
            delay = self.packet_record.calculate_average_delay(interval=self.step_time) / UNIT_K  # s
            states.append(max(min(delay, 1), 0))  # enforce 0 <= delay <= 1s
            loss_ratio = self.packet_record.calculate_loss_ratio(interval=self.step_time)
            states.append(loss_ratio)
            latest_prediction = self.packet_record.calculate_latest_prediction() / UNIT_M  # mbps
            states.append(latest_prediction - receiving_rate)  # over_estimation
            
            append_log(f'M STATE: [{states[0]:.3f}, {states[1]:.3f}, {states[2]:.3f}, {states[3]:.3f}]')
            torch_tensor_states = torch.FloatTensor(torch.Tensor(states).reshape(1, -1)).to(self.device)
            
#             # from ppo_agent.py -> PPO -> select_action
#             action, _, _ = self.model.forward(torch_tensor_states)
# #             self.bandwidth_prediction = log_to_linear(action)
            action = STATIC_BWE  # DEBUG

            # from rtc_env.py -> GymEnv -> step
            self.bandwidth_prediction = action
            append_log(f'M BWE: {self.bandwidth_prediction} bps')
        

        return self.bandwidth_prediction
