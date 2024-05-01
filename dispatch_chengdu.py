import traci
import os
import time
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import linear_sum_assignment
from AC_chengdu import *


class OrderAllocator:
    """
    需要生成一个订单分配器类，在每一秒可以读取车辆类的信息，也可以读取订单类的信息（提前从ＸＭＬ文件知道）
    因此订单分配器拥有一个迭代函数，车辆信息传递给这个函数，更新车辆信息和订单信息
    同时订单类拥有调用不同算法的能力，可以在每一步根据是否有新的订单信息，决定要不要调用分配算法
    因此其拥有一个函数用来判断是否需要调用分配算法
    如果需要分配，订单函数会调用其他函数获得分配结果，例如调用ＫＭ算法，分配结果是车辆和目前订单的匹配关系，调用函数的输入和输出方式仍然为字典
    因此其拥有一个函数用来选择指定的分配算法，传入分配的参数字典，并且获得输出结果的字典
    收到分配算法给的信息后，其会将分配结果做成ｓｕｍｏ能听懂的形式，并组成一个行程字典，供给sumo调用
    因此其拥有一个处理信息的函数，接收匹配结果，处理成ｓｕｍｏ能接受的字典形式数据
    """
    def __init__(self):

        self.vehicle_info = {}  # 新增属性，用于保存车辆信息字典
        self.reservations_dict = {}
        self.veh_num_in_state = 20
        self.category = ['E20', 'E28', 'E36', 'E9', 'E16', '-E18', '-E39']
        self.category = ['844894006#9', '409988943#1', '-100292708#2', '100305008#1', '-115625963#0', '430062181#1',
                         '589463632#1', '-100306491', '499418539#1']

        self.via_dict = {}
        self.dispatch_dict = {}
        self.vehicle_id = []
        self.storage = {}
        self.current_time = 0
        self.previous_time = None

        self.order_critic = {}
        self.order_agents = {}
        for category_name in self.category:
            critic_name = f"critic_agent_{category_name}"
            actor_name = f"Actor_agent_{category_name}"
            agent_C = Critic_agent(3000, 64)
            self.order_critic[critic_name] = agent_C
            agent = Actor_agent(self.order_critic[critic_name], 2000, 1)  # 修改 Actor_agent 的初始化以接收 Critic_agent 实例
            self.order_agents[actor_name] = agent

    def reset(self):
        folder_path = os.path.dirname(os.path.abspath(__file__))
        self.vehicle_info = {}  # 新增属性，用于保存车辆信息字典
        self.reservations_dict = {}
        self.veh_num_in_state = 20
        self.category = ['E20', 'E28', 'E36', 'E9', 'E16', '-E18', '-E39']
        self.category = ['844894006#9', '409988943#1', '-100292708#2', '100305008#1', '-115625963#0', '430062181#1',
                         '589463632#1', '-100306491', '499418539#1']
        self.load_vehicles_from_xml(f"{folder_path}/result.rou.xml")
        self.load_orders_from_xml(f"{folder_path}/result.rou.xml")

        self.via_dict = {}
        self.dispatch_dict = {}
        self.vehicle_id = []
        self.storage = {}
        self.current_time = 0
        self.previous_time = None

    def load_vehicles_from_xml(self, vehicles_xml):
        # 解析XML文件
        tree = ET.parse(vehicles_xml)
        root = tree.getroot()
        # 遍历所有的vehicle元素
        for vehicle_elem in root.findall('vehicle'):
            veh_id = vehicle_elem.get('id')
            # 初始化车辆信息字典（如果它还不存在）
            if veh_id not in self.vehicle_info:
                self.vehicle_info[veh_id] = {
                    'id': veh_id,
                    'x': None,
                    'y': None,
                    'lane': None,
                    'lane_pos': None,
                    'current_time': None,
                    'costs_to_orders': None,
                    'average_speed': None,
                    'env_state': None,
                    'state': None,  # 车辆目前的状态，需要处理
                    'remaining_battery': None,
                    'total_orders': [],  # 总共的订单对象，需要处理
                    'resides_orders': [],  # 剩余的订单对象，需要处理
                    'current_order': None,  # 目前的订单对象，需要处理
                    'predict_idle_time': 0,  # 车辆的预估空闲时间，需要处理
                    'predict_idle_edge': None,  # 车辆的预估空闲位置，需要处理
                    'predict_idle_x': None,
                    'predict_idle_y': None,
                }

    def load_orders_from_xml(self, orders_xml):
        # 解析XML字符串
        tree = ET.parse(orders_xml)
        root = tree.getroot()
        # 初始化一个空字典来存储解析后的数据

        # 遍历XML中的person元素
        for person in root.findall('person'):
            # 获取person元素的id属性
            person_id = person.get('id')
            # 初始化一个空字典来存储当前person的信息
            person_info = {}
            # 遍历person元素下的ride元素
            for ride in person.findall('ride'):
                # 提取ride元素的信息并存储到person_info字典中
                person_info['fromEdge'] = ride.get('from')
                person_info['fromPos_x'] = traci.simulation.convert2D(ride.get('from'),0)[0]
                person_info['fromPos_y'] = traci.simulation.convert2D(ride.get('from'),0)[1]
                person_info['started'] = ride.get('started')
                person_info['toEdge'] = ride.get('to')
                person_info['ended'] = ride.get('ended')
                person_info['depart'] = ride.get('depart')
                person_info['departPos'] = 'free'  # 似乎这个值在XML中是固定的，或者需要根据实际情况修改
                person_info['persons'] = None
                person_info['group'] = None
                person_info['arrivalPos'] = None
                person_info['reservationTime'] = None
                person_info['state'] = None
                person_info['dispatch_state'] = None  # 分配状态，需要处理
                person_info['dispatch_time'] = None  # 分配的时间，需要处理
                person_info['finish_time'] = None  # 完成的时间，需要处理
                person_info['current_vehicle'] = None  # 分配的车辆，需要处理
                min_distance = float('inf')  # 初始化最小距离为无穷大
                nearest_edge = None  # 初始化最近的边为None
                for edge in self.category:
                    distance = traci.simulation.getDistanceRoad(person_info['fromEdge'], 0, edge, 0, isDriving=True)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_edge = edge
            person_info['belong_to'] = nearest_edge
            self.reservations_dict[person_id] = person_info

    def update_reward(self, reward_type, order = None, veh = None):
        if reward_type == 'order':
            belong_to_index = self.category.index(order['belong_to'])
            reward = 1000 - (order['finish_time'] - order['reservationTime'])
            self.storage[order['dispatch_time']]['reward'][belong_to_index] += reward

        elif reward_type == 'order_choice':
            """
            用于在训练时，判断是否需要进行订单分配调度。
            :return: 如果需要订单分配调度则返回True，否则返回False。
            """
            # 检查是否有未分配的订单
            unassigned_orders = [order_id for order_id, info in self.reservations_dict.items()
                                 if info.get('dispatch_state') == '未分配' and info.get('state') == 2]

            # 如果有未分配的订单，则需要进行订单分配调度
            return len(unassigned_orders) > 0
        else:
            # 处理其他类型的订单或者未知类型的逻辑
            print("未找到该类型：处理其他或未知类型的订单分配...")
            # ...（具体的实现代码）

    def update_vehicle_info(self, vehicle_data):
        """
        更新车辆信息字典。

        :param vehicle_data: 包含车辆信息的字典，格式为 {veh_id: {...}}
        """
        for veh_id, info in vehicle_data.items():
            self.current_time = info['current_time']
            # 确保提供的车辆ID在当前的vehicle_info中（如果不存在，则添加它）
            if veh_id not in self.vehicle_info:
                self.vehicle_info[veh_id] = {}

            """需要优先单独更新派遣系统里车辆的状态"""
            last_state = self.vehicle_info.get(veh_id, {}).get('state', '')
            last_env_state = self.vehicle_info.get(veh_id, {}).get('env_state', -1)
            current_env_state = info.get('env_state', -1)
            if last_state is None and last_env_state == 0 and current_env_state == 0:
                self.vehicle_info[veh_id]['state'] = '空闲'  # 直接在这里更新state字段
            if last_state == '载客' and last_env_state == 2 and current_env_state == 0:
                self.vehicle_info[veh_id]['state'] = '空闲'  # 直接在这里更新state字段
                order_id = self.vehicle_info[veh_id]['current_order']
                order = self.reservations_dict.get(order_id)
                order['dispatch_state'] = '已完成'
                order['finish_time'] = self.current_time
                self.update_reward('order', order, info)
                self.vehicle_info[veh_id]['current_order'] = None
                if self.vehicle_info[veh_id]['resides_orders']:
                    self.vehicle_info[veh_id]['resides_orders'].pop(0)


                # 更新或添加车辆的具体信息
            self.vehicle_info[veh_id].update({
                'x': info.get('x', self.vehicle_info[veh_id].get('x')),
                'y': info.get('y', self.vehicle_info[veh_id].get('y')),
                'current_time': info.get('current_time', self.vehicle_info[veh_id].get('current_time')),
                'lane': info.get('lane', self.vehicle_info[veh_id].get('lane')),
                'lane_pos': info.get('lane_pos', self.vehicle_info[veh_id].get('lane')),
                'costs_to_orders': info.get('costs_to_orders', self.vehicle_info[veh_id].get('costs_to_orders')),
                'average_speed': info.get('average_speed', self.vehicle_info[veh_id].get('average_speed')),
                'env_state': info.get('env_state', self.vehicle_info[veh_id].get('env_state'))
            })
            self.predict_train(veh_id)

    def update_order_info(self, new_orders_info):
        for order_id, info in new_orders_info.items():
            if order_id not in self.reservations_dict:
                self.reservations_dict[order_id] = {}

            """需要优先单独更新派遣系统里dispatch_state订单的状态"""
            last_state = self.reservations_dict.get(order_id, {}).get('state', '')
            last_dispatch_state = self.reservations_dict.get(order_id, {}).get('dispatch_state', -1)
            current_state = info.get('state', -1)
            # 如果上一步车辆状态为“载客”且env_state为2，当前env_state为0，则更新状态为“空闲”
            if current_state == 1:
                self.reservations_dict[order_id]['dispatch_state'] = '未分配'  # 直接在这里更新state字段
            if last_dispatch_state == '已分配' and current_state == 8:
                self.reservations_dict[order_id]['dispatch_state'] = '已接取'  # 直接在这里更新state字段

                # 保留原始字典中的信息，如果新信息中没有提供该字段，则使用原始信息
            updated_info = {
                'persons': info.get('persons', self.reservations_dict[order_id].get('persons')),
                'group': info.get('group', self.reservations_dict[order_id].get('group')),
                'fromEdge': info.get('fromEdge', self.reservations_dict[order_id].get('fromEdge')),
                'toEdge': info.get('toEdge', self.reservations_dict[order_id].get('toEdge')),
                'departPos': info.get('departPos', self.reservations_dict[order_id].get('departPos')),
                'arrivalPos': info.get('arrivalPos', self.reservations_dict[order_id].get('arrivalPos')),
                'depart': info.get('depart', self.reservations_dict[order_id].get('depart')),
                'reservationTime': info.get('reservationTime', self.reservations_dict[order_id].get('reservationTime')),
                'state': info.get('state', self.reservations_dict[order_id].get('state'))
            }

            self.reservations_dict[order_id].update(updated_info)

    def predict_train(self, vehicle_id):
        # 获取车辆信息
        vehicle = self.vehicle_info.get(vehicle_id)
        if not vehicle:
            raise ValueError(f"Vehicle {vehicle_id} not found.")
            # 获取当前订单以及后续的订单序列

        # 初始化变量
        current_edge = vehicle['lane']  # 假设lane的格式为'edge_laneindex'
        current_pos = vehicle['lane_pos']  # 获取车辆的当前位置（沿边的距离）,需要改成订阅
        arrival_edge = current_edge
        arrival_time = vehicle['current_time']  # 获取当前仿真时间
        xingcheng  = vehicle['resides_orders']


        # 遍历订单序列进行预测
        for order_id in xingcheng:
            order_info = self.reservations_dict.get(order_id)
            if not order_info:
                raise ValueError(f"Order {order_id} not found.")
            from_edge = order_info['fromEdge']
            to_edge = order_info['toEdge']

            # 预测到达订单出发地的时间（如果不在同一个边，则需要计算距离并除以速度）
            if from_edge != current_edge:
                try:
                    distance = traci.simulation.getDistanceRoad(current_edge, current_pos, from_edge, 0, isDriving=True)
                except Exception as e:
                    distance = traci.simulation.getDistanceRoad(current_edge, 0, from_edge, 0, isDriving=True)
                travel_time = distance / vehicle['average_speed']  # 假设平均速度是恒定的
                arrival_time += travel_time
                current_edge = from_edge  # 更新当前边
                current_pos = 0  # 假设在边的起点接到乘客

            # 预测到达订单目的地的时间（同样需要计算距离并除以速度）
            distance = traci.simulation.getDistanceRoad(from_edge, current_pos, to_edge, 0, isDriving=True)
            travel_time = distance / vehicle['average_speed']  # 假设平均速度是恒定的
            arrival_time += travel_time
            arrival_edge = to_edge  # 更新到达边

            # 在这里可以添加额外的逻辑来处理乘客上车、下车等时间，以及可能的等待时间
            # 假设车辆在完成一个订单后立刻前往下一个订单的出发地（如果有的话）
            # 这里需要获取下一个订单的信息，并更新current_edge和current_pos等变量

        vehicle['predict_idle_time'] = arrival_time
        vehicle['predict_idle_edge'] = arrival_edge
        if not xingcheng:
            vehicle['predict_idle_x'] = vehicle['x']
            vehicle['predict_idle_y'] = vehicle['y']
            return
        pos = traci.simulation.convert2D(arrival_edge, 0)
        vehicle['predict_idle_x'] = pos[0]
        vehicle['predict_idle_y'] = pos[1]

    def need_dispatch(self, order_type):
        if order_type == 'order':
            """
            判断是否需要进行订单分配调度。
            :return: 如果需要订单分配调度则返回True，否则返回False。
            """
            # 检查是否有空闲车辆
            idle_vehicles = [veh_id for veh_id, info in self.vehicle_info.items() if info.get('state') == '空闲']

            # 检查是否有未分配的订单
            unassigned_orders = [order_id for order_id, info in self.reservations_dict.items()
                                 if info.get('dispatch_state') == '未分配' and info.get('state') == 2]

            # 如果既有空闲车辆又有未分配的订单，则需要进行订单分配调度
            return len(idle_vehicles) > 0 and len(unassigned_orders) > 0

        elif order_type == 'order_train':
            """
            用于在训练时，判断是否需要进行订单分配调度。
            :return: 如果需要订单分配调度则返回True，否则返回False。
            """
            # 检查是否有未分配的订单
            unassigned_orders = [order_id for order_id, info in self.reservations_dict.items()
                                 if info.get('dispatch_state') == '未分配' and info.get('state') == 2]

            # 如果有未分配的订单，则需要进行订单分配调度
            return len(unassigned_orders) > 0
        else:
            # 处理其他类型的订单或者未知类型的逻辑
            print("未找到该类型：处理其他或未知类型的订单分配...")
            # ...（具体的实现代码）

    def need_order_dispatch(self):
        """
        判断是否需要进行订单分配调度。
        :return: 如果需要订单分配调度则返回True，否则返回False。
        """
        # 检查是否有空闲车辆
        idle_vehicles = [veh_id for veh_id, info in self.vehicle_info.items() if info.get('state') == '空闲']

        # 检查是否有未分配的订单
        unassigned_orders = [order_id for order_id, info in self.reservations_dict.items()
                             if info.get('dispatch_state') == '未分配' and info.get('state') == 2]

        # 如果既有空闲车辆又有未分配的订单，则需要进行订单分配调度
        return len(idle_vehicles) > 0 and len(unassigned_orders) > 0

    def need_order_dispatch_train(self):
        """
        用于在训练时，判断是否需要进行订单分配调度。
        :return: 如果需要订单分配调度则返回True，否则返回False。
        """
        # 检查是否有未分配的订单
        unassigned_orders = [order_id for order_id, info in self.reservations_dict.items()
                             if info.get('dispatch_state') == '未分配' and info.get('state') == 2]

        # 如果有未分配的订单，则需要进行订单分配调度
        return len(unassigned_orders) > 0 and self.current_time > 101

    def perform_one_round_interaction(self, frame_idx = 30000):
        """
        Conduct a round of multi-agent decision-making
        """
        interaction_dict = {
            '订单序列': [],
            '车辆序列': [],
            '状态空间序列': [],
            '动作空间序列': [],
            'reward': [0] * len(self.category),
            '结尾状态': [],
            'Q': []
        }

        # 进行 num_agents 次决策
        for idx, agent_index in enumerate(self.category):
            # 筛选订单
            orders = self.filter_orders()
            interaction_dict['订单序列'].append(orders)

            # 筛选车辆
            vehicle_ids = self.filter_vehicles(self.veh_num_in_state)
            interaction_dict['车辆序列'].append(vehicle_ids)

            # 构建状态空间
            state_space = self.construct_state_space(orders, vehicle_ids)
            interaction_dict['状态空间序列'].append(state_space)

            # 获取智能体决策动作
            action = self.choose_next_action(state_space, agent_index, orders, vehicle_ids, idx, frame_idx )
            interaction_dict['动作空间序列'].append(action)
            print(action)

            reward_Q, vehicle_id = self.Evaluate_action_value(agent_index, orders, action, vehicle_ids)

            # 处理订单分配
            self.handle_order_assignment(orders, vehicle_ids, action, agent_index)
            # total_value = self.Evaluate_value(vehicle_id)
            # interaction_dict['Q'].append((reward_Q + total_value)*0.01)
            interaction_dict['Q'].append((reward_Q) * 0.01)

        if self.previous_time is not None and self.storage.get(self.previous_time) is not None:
            previous_interaction_dict = self.storage[self.previous_time]
            previous_interaction_dict['结尾状态'] = interaction_dict['状态空间序列'][0]

        self.previous_time = self.current_time
        return interaction_dict

    def Evaluate_action_value(self, agent_index, orders, action, vehicle_ids):
        index = self.category.index(agent_index)
        # 条件1: orders[index] is None 且 action = self.veh_num_in_state
        if orders[index] is None and action == self.veh_num_in_state:
            reward_Q = 1000  # 给予正反馈，数值可根据实际情况调整
            vehicle_id = '20'
        # 条件2: orders[index] is None 且 action != self.veh_num_in_state
        elif orders[index] is None and action != self.veh_num_in_state:
            reward_Q = 0  # 无反馈
            vehicle_id = vehicle_ids[action]
        # 条件3: orders[index] is not None 且 action = self.veh_num_in_state
        elif orders[index] is not None and action == self.veh_num_in_state:
            reward_Q = -4000  # 给予负反馈，数值可根据实际情况调整
            vehicle_id = '20'
        # 条件4: orders[index] is not None 且 action != self.veh_num_in_state
        elif orders[index] is not None and action < self.veh_num_in_state:
            vehicle_id = vehicle_ids[action]
            order_id = orders[index][0]
            current_edge = self.vehicle_info[vehicle_id]['predict_idle_edge']
            from_edge = self.reservations_dict[order_id[0]]['fromEdge']
            distance = traci.simulation.getDistanceRoad(current_edge, 0, from_edge, 0, isDriving=True)
            predict_idle_time = self.vehicle_info[vehicle_id]['predict_idle_time']
            cost = predict_idle_time + distance / self.vehicle_info[vehicle_id]['average_speed']
            reward_Q = 1000 - cost  # 正常计算reward_Q
        else:
            reward_Q = 0  # 对于其他未明确指定的情况，可以给予无反馈或适当的默认值
        return reward_Q, vehicle_id

    def Evaluate_value(self, vehicle_id):
        # 筛选出所有在self.vehicle_info中状态为空闲或前往的车辆
        idle_or_going_vehicles = [
            vehicle_id for vehicle_id in self.vehicle_info.keys()
        ]  # 这里我们是选择了所有的车辆，如果后期需要增加最空闲的车辆，需要增加一个筛选工作
        if vehicle_id in idle_or_going_vehicles:
            # 如果存在，则移除它
            idle_or_going_vehicles.remove(vehicle_id)

        # 筛选出self.reservations_dict中所有dispatch_state为已分配或未分配状态的预约
        allocated_or_unallocated_reservations = [
            reservation_id for reservation_id, reservation_data in self.reservations_dict.items()
            if reservation_data.get('dispatch_state') in ['未分配'] and reservation_data.get('state') == 2
        ]

        cost_matrix = np.zeros((len(idle_or_going_vehicles), len(allocated_or_unallocated_reservations)))
        for i, vehicle_id in enumerate(idle_or_going_vehicles):
            for j, order_id in enumerate(allocated_or_unallocated_reservations):
                current_edge = self.vehicle_info[vehicle_id]['predict_idle_edge']
                from_edge = self.reservations_dict[order_id]['fromEdge']
                distance = traci.simulation.getDistanceRoad(current_edge, 0, from_edge, 0, isDriving=True)
                predict_idle_time = self.vehicle_info[vehicle_id]['predict_idle_time']
                cost = predict_idle_time + distance / self.vehicle_info[vehicle_id]['average_speed']

                cost_matrix[i][j] = cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total_value = 0
        for r, c in zip(row_ind, col_ind):
            total_value += 1000 - cost_matrix[r][c]
        return total_value

    def perform_one_round_interaction_rd(self):
        """
        Make a round of random decisions
        """
        interaction_dict = {
            '订单序列': [],
            '车辆序列': [],
            '状态空间序列': [],
            '动作空间序列': [],
            'reward': 0,
            '结尾状态': []
        }

        # 进行 num_agents 次决策
        for idx, agent_index in enumerate(self.category):
            # 筛选订单
            orders = self.filter_orders()
            interaction_dict['订单序列'].append(orders)

            # 筛选车辆
            vehicle_ids = self.filter_vehicles(self.veh_num_in_state)
            interaction_dict['车辆序列'].append(vehicle_ids)

            # 构建状态空间
            state_space = self.construct_state_space(orders, vehicle_ids)
            interaction_dict['状态空间序列'].append(state_space)

            # 获取智能体决策动作
            action = self.choose_next_action_random(state_space, agent_index)
            interaction_dict['动作空间序列'].append(action)

            # 处理订单分配
            reward = self.handle_order_assignment(orders, vehicle_ids, action, agent_index)
            interaction_dict['reward'][idx] += reward

        if self.previous_time is not None and self.storage.get(self.previous_time) is not None:
            previous_interaction_dict = self.storage[self.previous_time]
            previous_interaction_dict['结尾状态'] = interaction_dict['状态空间序列'][0]

        self.previous_time = self.current_time
        return interaction_dict

    def perform_one_round_interaction_km(self):
        """
        Using KM algorithm for one round of decision-making
        """
        interaction_dict = {
            '订单序列': [],
            '车辆序列': [],
            '状态空间序列': [],
            '动作空间序列': [],
            'reward': [0] * len(self.category),
            '结尾状态': []
        }

        # 筛选订单
        orders = self.filter_orders()

        # 筛选车辆
        vehicle_ids = self.filter_vehicles(self.veh_num_in_state)

        # 构建状态空间
        state_space = self.construct_state_space(orders, vehicle_ids)

        # 获取智能体决策动作
        action = self.choose_next_action_km(orders, vehicle_ids)
        interaction_dict['动作空间序列'] = action

        for i, agent_index in enumerate(self.category):
            interaction_dict['订单序列'].append(orders)
            interaction_dict['车辆序列'].append(vehicle_ids)
            interaction_dict['状态空间序列'].append(state_space)
            # 处理订单分配
            self.handle_order_assignment(orders, vehicle_ids, action[i], agent_index)

        if self.previous_time is not None and self.storage.get(self.previous_time) is not None:
            previous_interaction_dict = self.storage[self.previous_time]
            previous_interaction_dict['结尾状态'] = interaction_dict['状态空间序列'][0]

        self.previous_time = self.current_time

        return interaction_dict

    def handle_order_assignment(self, orders, vehicle_ids, action, agent_index):

        if action < self.veh_num_in_state:  # 分配车辆
            vehicle_id = vehicle_ids[action]
            vehicle = self.vehicle_info[vehicle_id]
            index = self.category.index(agent_index)
            if orders[index] is not None:
                order_id = orders[index][0]
                order_info = self.reservations_dict[order_id]
                order_info['dispatch_state'] = '已分配'
                order_info['current_vehicle'] = vehicle_id
                order_info['dispatch_time'] = self.current_time

                vehicle['total_orders'].append(order_id)
                vehicle['resides_orders'].append(order_id)
            self.predict_train(vehicle_id)  # 更新车辆状态


    def choose_next_action_random(self, state, agent_index):
        # 这里应该有对应的智能体网络的实现，返回一个动作
        # 示例: action = self.actor_networks[agent_index].forward(state)
        action = np.random.randint(0, self.veh_num_in_state + 1)  # 随机选择一个动作，仅作为示例
        return action

    def choose_next_action(self, state, agent_index, orders, vehicle_ids, idx, frame_idx=30000):
        threshold1 = np.random.rand()
        threshold2 = np.random.rand()
        if frame_idx == 0:
            threshold2 = 1  #　使用ｋｍ算法时

        if frame_idx / 30000 < threshold1:
            if 0.5 < threshold2:
                actions = self.choose_next_action_km(orders, vehicle_ids)
                action = actions[idx]
            else:
                action = np.random.randint(0, self.veh_num_in_state + 1)
        else:
            agent_name = f"critic_agent_{agent_index}"
            agent = self.order_critic.get(agent_name)
            action = agent.select_action(state)
        #else:
            #agent_name = f"Actor_agent_{agent_index}"
            #agent = self.order_agents.get(agent_name)
            #action = agent.select_action(state)
        return action


    def choose_next_action_km(self, orders, vehicle_ids):
        cost_matrix = np.zeros((len(vehicle_ids), len(orders)))
        for i, vehicle_id in enumerate(vehicle_ids):
            for j, order_id in enumerate(orders):
                if order_id is None:
                    cost_matrix[i][j] = 99999
                else:
                    current_edge = self.vehicle_info[vehicle_id]['predict_idle_edge']
                    from_edge = self.reservations_dict[order_id[0]]['fromEdge']
                    distance = traci.simulation.getDistanceRoad(current_edge, 0, from_edge, 0, isDriving=True)
                    predict_idle_time = self.vehicle_info[vehicle_id]['predict_idle_time']
                    cost = predict_idle_time + distance/self.vehicle_info[vehicle_id]['average_speed']
                    cost_matrix[i][j] = cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # 使用argsort获取col_ind的排序索引
        sorted_indices = np.argsort(col_ind)

        # 使用这些索引来对col_ind和row_ind进行排序
        sorted_col_ind = col_ind[sorted_indices]
        sorted_row_ind = row_ind[sorted_indices]
        none_indices = [idx for idx, order in enumerate(orders) if order is None]
        for none_idx in none_indices:
            sorted_row_ind[none_idx] = self.veh_num_in_state
        return sorted_row_ind

    def filter_vehicles(self, num_vehicles=20):
        # 获取所有车辆的ID和对应的predict_idle_time
        vehicles = [(veh_id, vehicle['predict_idle_time']) for veh_id, vehicle in self.vehicle_info.items()]
        # 根据predict_idle_time和ID对车辆进行排序
        sorted_vehicles = sorted(vehicles, key=lambda x: (x[1], x[0]))
        # 提取前num_vehicles个车辆的ID
        vehicle_ids = [veh_id for veh_id, _ in sorted_vehicles[:num_vehicles]]
        # vehicle_ids = sorted(vehicle_ids, key=lambda x: int(x))
        # 如果车辆数量不足num_vehicles，则用None填充列表
        vehicle_ids.extend([None] * (num_vehicles - len(vehicle_ids)))
        return vehicle_ids

    def filter_orders(self):
        """筛选订单的类，适用于筛选出一轮强化学习需要的订单，其订单按照出现时间排序，筛选出每个区域中最需要当前决策的订单"""
        filtered_orders = [None] * len(self.category)
        selected_orders = set()

        # 按照类别顺序逐个筛选订单
        for belong_to in self.category:
            min_started = float('inf')  # 初始化最小的started时间
            min_order_id = None  # 初始化最小started时间对应的订单ID
            min_order_info = None  # 初始化最小started时间对应的订单信息

            # 遍历订单信息字典，筛选符合条件的订单
            for order_id, order_info in self.reservations_dict.items():
                if (order_info['dispatch_state'] == '未分配' and
                        order_info['started'] is not None and
                        order_info['belong_to'] == belong_to):
                    # 更新最小started时间和对应的订单信息
                    if float(order_info['started']) < min_started:
                        min_started = float(order_info['started'])
                        min_order_id = order_id
                        min_order_info = order_info

            # 如果存在符合条件的订单，将其加入到对应类别的位置
            if min_order_id is not None:
                filtered_orders[self.category.index(belong_to)] = [min_order_id, min_order_info['fromPos_x'],
                                                                 min_order_info['fromPos_y']]
                selected_orders.add(belong_to)  # 将该类别加入到已选择的类别集合中

        return filtered_orders

    def construct_state_space(self, orders, vehicle_ids):
        # 这里是基于给定的订单信息和车辆信息来构建状态空间的完整实现
        cost_matrix = np.zeros((len(orders), len(vehicle_ids)))  # 创建一个五行十列的全零矩阵
        for i, order_id in enumerate(orders):
            for j, vehicle_id in enumerate(vehicle_ids):
                if order_id is None:
                    cost_matrix[i][j] = 99999
                else:
                    current_edge = self.vehicle_info[vehicle_id]['predict_idle_edge']
                    from_edge = self.reservations_dict[order_id[0]]['fromEdge']
                    distance = traci.simulation.getDistanceRoad(current_edge, 0, from_edge, 0, isDriving=True)
                    predict_idle_time = self.vehicle_info[vehicle_id]['predict_idle_time']
                    cost = predict_idle_time + distance / self.vehicle_info[vehicle_id][
                        'average_speed'] - self.current_time
                    cost_matrix[i][j] = cost
                cost_array = cost_matrix.reshape(-1)
                state_space = cost_array

        return state_space

    def get_order_state(self, person_info):
        """每个区域返回当前的订单信息和区域所剩余的订单信息量"""
        return [person_info['fromPos_x'], person_info['fromPos_y'],
                self.get_remaining_orders_count(person_info['belong_to'])]

    def get_remaining_orders_count(self, belong_to):
        count = 0
        for order_id, order_info in self.reservations_dict.items():
            if order_info['belong_to'] == belong_to and order_info['dispatch_state'] == '未分配':
                count += 1
        return count

    def generate_via_dict(self):
        """
        每个车辆在字典里有如上的三个属性：总订单，剩余订单以及目前订单，
        我们需要根据这些信息决定车辆在每一步进行的ｖｉａ和ｄｉｓｐａｔｃｈ操作，（这两个为字典形式传给ｓｕｍｏ）
        在每一步中我们需要检查车辆目前的状态，如果['state'] = '空闲'，则将车辆'resides_orders'的第一个赋予给'current_order'
        如果'resides_orders'为空，则不赋予
        生成的对应ｖia字典如   via[vehicle_id] = 'current_order'
            else:
                via[vehicle_id] = None
        除了以上我们要同步以上的order信息和车辆信息
         self.vehicle_info[vehicle_id]['state'] = '前往'
         self.vehicle_info[vehicle_id]['current_order'] = order_id
         self.reservations_dict[order_id]['dispatch_state'] = '已分配'
        self.reservations_dict[order_id]['current_vehicle'] = vehicle_id
        """
        via = {}  # 创建via字典，用于存储当前车辆对应的当前订单
        for vehicle_id, vehicle_data in self.vehicle_info.items():
            if vehicle_data['state'] == '空闲' and vehicle_data['resides_orders']:
                # 如果车辆状态为空闲，并且有剩余订单
                current_order = vehicle_data['resides_orders'][0]  # 获取剩余订单列表的第一个订单
                reservation_data = self.reservations_dict.get(current_order)
                order_edge = reservation_data.get('fromEdge')
                vehicle_edge = vehicle_data['lane']
                if vehicle_edge != order_edge:
                    via[vehicle_id] = order_edge  # 将当前订单设置为via字典的值
                    # 更新车辆信息和订单信息
                    self.vehicle_info[vehicle_id]['state'] = '前往'
                    self.vehicle_info[vehicle_id]['current_order'] = current_order
                    order_data = self.reservations_dict.get(current_order)
                    if order_data:
                        order_data['dispatch_state'] = '已分配'
                        order_data['current_vehicle'] = vehicle_id
        return via

    def generate_dispatch_dict(self):
        self.dispatch_dict = {}
        # 遍历所有车辆，寻找处于“前往”状态的车辆
        for vehicle_id, vehicle_data in self.vehicle_info.items():
            if vehicle_data['state'] == '前往':
                lane = vehicle_data.get('lane')  # 读取车辆的'lane'信息
                lane_Pos = vehicle_data.get('lane_pos')  # 读取车辆的'lane'信息
                current_order = vehicle_data.get('current_order')  # 读取车辆的'current_order'信息


                # 检查'current_order'是否存在于'reservations_dict'中
                if current_order in self.reservations_dict:
                    reservation_data = self.reservations_dict[current_order]
                    from_edge = reservation_data.get('fromEdge')  # 读取订单的'fromEdge'信息
                    try:
                        costs = traci.simulation.getDistanceRoad(lane, lane_Pos, from_edge, 0,
                                                                    isDriving=True)
                    except Exception as e:
                        costs = traci.simulation.getDistanceRoad(lane, 0, from_edge, 0, isDriving=True)
                    # 比较'lane'和'fromEdge'是否相等
                    if costs < 15000:
                        # 更新车辆状态为“载客”
                        self.vehicle_info[vehicle_id]['state'] = '载客'
                        # 更新订单状态为“已接取”
                        self.reservations_dict[current_order]['dispatch_state'] = '已接取'
                        # 将派遣结果添加到'dispatch_dict'中
                        self.dispatch_dict[vehicle_id] = current_order
                        print(f"Vehicle {vehicle_id} has picked up order {current_order}.")
                    else:
                        print(f"Vehicle {vehicle_id} is going to the wrong lane. Expected: {from_edge}, Actual: {lane}")
                else:
                    print(f"Current order {current_order} not found in reservations_dict.")
        return self.dispatch_dict  # 返回派遣结果字典

    def iterate(self, frame_idx = 30000):
        if self.need_order_dispatch_train() :
        # if self.need_order_dispatch():
            self.storage[self.current_time] = self.perform_one_round_interaction(frame_idx)

        give_env_dict = {'via': self.generate_via_dict(), 'dispatch': self.generate_dispatch_dict()}
        return give_env_dict

    def iterate_rd(self):
        if self.need_order_dispatch_train() :
            self.storage[self.current_time] = self.perform_one_round_interaction_rd()

        give_env_dict = {'via': self.generate_via_dict(), 'dispatch': self.generate_dispatch_dict()}
        return give_env_dict

    def iterate_km(self):
        if self.need_order_dispatch_train() :
            self.storage[self.current_time] = self.perform_one_round_interaction_km()

        give_env_dict = {'via': self.generate_via_dict(), 'dispatch': self.generate_dispatch_dict()}
        return give_env_dict

    def simulation_end(self):
        """
        判断仿真是否结束。
        :return: 如果所有订单都已完成则返回True，否则返回False。
        """
        # 检查是否所有订单都已完成
        all_orders_completed = all(info.get('dispatch_state') == '已完成' for info in self.reservations_dict.values())
        if all_orders_completed:
            print()
        # 如果所有订单都已完成，则返回True，否则返回False
        return all_orders_completed

# allocator = OrderAllocator('vehicles.xml', 'orders.xml')
# allocator.iterate()  # 在实际应用中，这个调用可能在一个循环中持续进行

