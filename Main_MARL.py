from ENV_chengdu import sim_ENV
from dispatch_chengdu import OrderAllocator
import traci
import torch
import numpy as np
import random
import os
import pandas as pd
import xml.etree.ElementTree as ET
from lxml import etree

def train(num_frames = 90000):
    xml_file = 'loss_log_4.12.xml'
    reward_file = 'reward_log_4.12.xml'
    folder_name = "pth_store_4.12"
    os.makedirs(folder_name, exist_ok=True)
    update_cnt = 0
    env1 = sim_ENV()
    allocator = OrderAllocator()
    # LOAD_pth(allocator, "pth_store_4.4", 24150)
    for frame_idx in range(1, num_frames + 1):
        select_and_execute_mode(allocator, env1, frame_idx)
        score = store_memory(allocator)
        print(f"{frame_idx}此时")
        traci.close()
        update_cnt = update_model(frame_idx, update_cnt, allocator, xml_file)
        store_pth( allocator, folder_name, frame_idx)
        Test_now(env1, allocator, reward_file, frame_idx)


def Test_now(env1, allocator, reward_file, frame_idx=50):
    if frame_idx % 30 == 0 and frame_idx != 0:
        env1.reset()
        allocator.reset()
        done = False
        update_dict = {}
        # 执行选中的模式
        while not done:
            env1.sim_step(update_dict)
            allocator.update_order_info(env1.reservations_dict)
            allocator.update_vehicle_info(env1.vehicle_info)
            update_dict = allocator.iterate(min(frame_idx * 5, 25000))  # 调用选中的模式函数
            current_time = traci.simulation.getTime()
            # print(current_time)
            if current_time >= 10000 or allocator.simulation_end():
                done = True
        score = store_memory(allocator)
        print(f"此时{frame_idx}的分数{score}")
        store_reward(frame_idx, score, reward_file)
        traci.close()

def store_reward(frame_idx, reward, reward_file):
    # 加载现有的 XML 文件，如果不存在则创建一个新的根节点
    try:
        tree = ET.parse(reward_file)
        root = tree.getroot()
    except FileNotFoundError:
        root = ET.Element("rewards")
        tree = ET.ElementTree(root)

    # 创建一个新的子节点来存储当前 frame 的 reward
    frame_reward = ET.SubElement(root, "frame_reward", frame=str(frame_idx), reward=str(reward))

    # 将 XML 树写入文件
    tree.write(reward_file, encoding='utf-8', xml_declaration=True)


def LOAD_pth(allocator, folder_name,frame_idx = 2000):
    folder_name = folder_name
    frame_idx_to_load = frame_idx # 假设你想加载第90000帧时的模型权重
    for agent_name in allocator.order_agents.keys():
        agent = allocator.order_agents[agent_name]
        model_path = os.path.join(folder_name, f"{agent_name}order_agent_{frame_idx_to_load}.pth")
        agent.ActorNetwork.load_state_dict(torch.load(model_path))

    for agent_name in allocator.order_critic.keys():
        agent = allocator.order_critic[agent_name]
        model_path = os.path.join(folder_name, f"{agent_name}order_critic_{frame_idx_to_load}.pth")
        agent.CriticNetwork.load_state_dict(torch.load(model_path))


def Test1(frame_idx = 8350, dense=50):
    scores = []
    env1 = sim_ENV(False, dense)
    allocator = OrderAllocator()
    # 加载训练好的权重
    folder_name = "pth_store_4.12"
    frame_idx_to_load = frame_idx
    for agent_name in allocator.order_agents.keys():
        agent = allocator.order_agents[agent_name]
        model_path = os.path.join(folder_name, f"{agent_name}order_agent_{frame_idx_to_load}.pth")
        agent.ActorNetwork.load_state_dict(torch.load(model_path))
        agent.ActorNetwork.eval()  # 设置为评估模式

    for agent_name in allocator.order_critic.keys():
        agent = allocator.order_critic[agent_name]
        model_path = os.path.join(folder_name, f"{agent_name}order_critic_{frame_idx_to_load}.pth")
        agent.CriticNetwork.load_state_dict(torch.load(model_path))
        agent.CriticNetwork.eval()  # 设置为评估模式
    env1.reset()
    allocator.reset()
    done = False
    update_dict = {}
    # 执行选中的模式
    while not done:
        env1.sim_step(update_dict)
        allocator.update_order_info(env1.reservations_dict)
        allocator.update_vehicle_info(env1.vehicle_info)
        update_dict = allocator.iterate()  # 调用选中的模式函数
        current_time = traci.simulation.getTime()
        # print(current_time)
        if current_time >= 10000 or allocator.simulation_end():
            done = True
    score = store_memory(allocator)
    scores.append(score)
    print(f"此时的分数{score}")
    traci.close()


def Test_km(dense=50):
    scores = []
    env1 = sim_ENV(False,dense)
    allocator = OrderAllocator()
    env1.reset()
    allocator.reset()
    done = False
    update_dict = {}
    # 执行选中的模式
    while not done:
        env1.sim_step(update_dict)
        allocator.update_order_info(env1.reservations_dict)
        allocator.update_vehicle_info(env1.vehicle_info)
        update_dict = allocator.iterate(0)  # 调用选中的模式函数
        current_time = traci.simulation.getTime()
        # print(current_time)
        if current_time >= 10000 or allocator.simulation_end():
            done = True
    score = store_memory(allocator)
    scores.append(score)
    print(f"此时的分数{score}")
    traci.close()


def select_and_execute_mode(allocator, env1, frame_idx):
    # 重置环境和分配器
    env1.reset()
    allocator.reset()
    done = False
    update_dict = {}

    # 执行选中的模式
    while not done:
        env1.sim_step(update_dict)
        allocator.update_order_info(env1.reservations_dict)
        allocator.update_vehicle_info(env1.vehicle_info)
        update_dict = allocator.iterate(min(frame_idx, 12000))  # 调用选中的模式函数
        current_time = traci.simulation.getTime()
        # print(current_time)
        if current_time >= 200:
            done = True


def store_memory(allocator):
    score = 0
    for interaction_dict in allocator.storage.values():
        # 获取每个 interaction_dict 中的数据
        obs_sequence = interaction_dict['状态空间序列']
        action_sequence = interaction_dict['动作空间序列']
        reward_sequence = interaction_dict['reward']
        Q_sequence = interaction_dict['Q']
        next_obs = interaction_dict['结尾状态']

        score += sum(reward_sequence)

        if next_obs == []:
            done = True
            next_obs = obs_sequence[0]
        else:
            done = False

        for obs, action, reward, Q, (agent_name, agent) in zip(obs_sequence, action_sequence,reward_sequence, Q_sequence, allocator.order_critic.items()):
            agent.memory.store(obs, action, reward, Q, next_obs, done)

        for obs, action, reward, Q, (agent_name, agent) in zip(obs_sequence, action_sequence,reward_sequence,  Q_sequence, allocator.order_agents.items()):
            agent.memory.store(obs, action, reward, Q, next_obs, done)
    return score


def update_model1(frame_idx, update_cnt, allocator):
    for agent_name, agent in allocator.order_agents.items():
        loss = agent.update_model()
    if frame_idx >= 20:
        for agent_name, agent in allocator.order_critic.items():
            loss = agent.update_model()
        update_cnt += 1
    return update_cnt


def update_model(frame_idx, update_cnt, allocator, xml_file):
    # 如果存在现有的 XML 文件，则先读取现有数据
    root = None
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except FileNotFoundError:
        # 如果文件不存在，则创建新的根节点
        root = ET.Element("agents")
        tree = ET.ElementTree(root)

        # 创建或更新代理的损失值
    for agent_name, agent in allocator.order_agents.items():
        loss = agent.update_model()

        # 查找是否存在具有相同name的agent元素
        agent_element = root.find(".//agent[@name='{}']".format(agent_name))
        if agent_element is None:
            # 如果不存在，则创建一个新的agent元素
            agent_element = ET.SubElement(root, "agent", name=agent_name)

            # 向agent元素中添加新的frame子元素
        frame_element = ET.SubElement(agent_element, "frame", id=str(frame_idx), Loss=str(loss))

        # 如果满足条件，更新评论家代理的模型，并记录损失值

    for agent_name, agent in allocator.order_critic.items():
        if agent.memory.size >= 500:
            loss = agent.update_model()

            # 同样查找或创建agent元素
            agent_element = root.find(".//agent[@name='{}']".format(agent_name))
            if agent_element is None:
                agent_element = ET.SubElement(root, "agent", name=agent_name)

                # 添加frame子元素
            frame_element = ET.SubElement(agent_element, "frame", id=str(frame_idx), Loss=str(loss))

        update_cnt += 1

        # 将 XML 数据写入文件
    tree.write(xml_file, encoding='utf-8', xml_declaration=True)

    return update_cnt


def store_pth(allocator, folder_name, frame_idx):
    if frame_idx % 500 == 0 and frame_idx != 0:
        for agent_name, agent in allocator.order_critic.items():
            model_path = os.path.join(folder_name, f"{agent_name}order_critic_{frame_idx}.pth")
            torch.save(agent.CriticNetwork.state_dict(), model_path)

        for agent_name, agent in allocator.order_agents.items():
            model_path = os.path.join(folder_name, f"{agent_name}order_agent_{frame_idx}.pth")
            torch.save(agent.ActorNetwork.state_dict(), model_path)

def calculate_average_waiting_time(tripinfo_file):
    tree = ET.parse(tripinfo_file)
    root = tree.getroot()

    waiting_times = []
    for personinfo in root.findall('personinfo'):
        ride = personinfo.find('ride')
        waiting_time = float(ride.attrib['waitingTime'])
        waiting_times.append(waiting_time)

    average_waiting_time = sum(waiting_times) / len(waiting_times)
    return average_waiting_time

def generate_average_waiting_time_xml():
    root = ET.Element('dense_values')

    for dense_value in range(20, 101, 5):
        Test1(20000, dense_value)
        average_waiting_time = calculate_average_waiting_time(f'tripinfo.xml')
        dense_element = ET.SubElement(root, 'dense_value')
        dense_element.set('value', str(dense_value))
        dense_element.set('average_waitingTime', f'{average_waiting_time:.3f}')

    tree = ET.ElementTree(root)
    tree.write('average_waiting_times_RL.xml')




if __name__ == "__main__":
    generate_average_waiting_time_xml()
        # 这里可以对每个环境进行操作，例如运行仿真等
