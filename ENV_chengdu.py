import traci
# 这是一个指向sumo的命令库
import traci.constants as tc
from chargingnet import *
import subprocess
import time
import logging
import os
import xml.etree.ElementTree as ET
from dispatch_chengdu import OrderAllocator
"""这是一个从头搭建的成都的仿真环境"""


class sim_ENV:
    def __init__(self, is_test=False, dense=50):
        self.folder_path = os.path.dirname(os.path.abspath(__file__))

        self.attributes = [traci.constants.VAR_ROUTE_ID, traci.constants.VAR_ROAD_ID, traci.constants.VAR_LANEPOSITION,
                           traci.constants.VAR_DISTANCE, traci.constants.VAR_DEPARTURE, traci.constants.VAR_TIME, traci.constants.VAR_POSITION]
        # 这里添加订阅，所有想要从sumo仿真中获得的数据都可以先提前设置好订阅,订阅分为本来车辆就有的订阅和设备订阅，可以在traci,constant里查看
        self.attributes2 = ["device.battery.energyConsumed",
                            "device.battery.energyCharged",
                            "device.battery.chargeLevel",
                            "device.battery.capacity",
                            "device.battery.chargingStationId",
                            "device.battery.totalEnergyConsumed"]

        self.net_file = os.path.join(self.folder_path, "chengdu.net.xml")  # 地图网络文件名net_file.net.xml  simmap.net.xml
        self.route_file = "result.rou.xml"  # 用于config的路由文件名
        self.begin_time = "0"  # 仿真开始时间
        self.end_time = "9000"  # 仿真结束时间
        self.step_length = "1"  # 仿真的时间步长
        self.seed = 722  # 用来生成随机旅程的随机数种子,int(time.time())
        self.num_trips = 20  # 用来生成一共要生成多少辆车子
        self.num_trips1 = dense  # 用来生成一共要生成多少订单
        self.start_time = 0  # 用来生成生成车子的开始时间
        self.end_time = 1  # 根据车流密度用来生成生成车子的结束时间
        self.start_time1 = 98  # 用来生成生成订单的开始时间
        self.end_time1 = 100  # 根据车流密度用来生成生成订单的结束时间
        self.intermediate: int = 1  # 中间的途径点数量
        self.vehicle_add = os.path.join(self.folder_path, "vehicle_add.add.xml")  # 添加车子类型的xml文件
        if is_test:
            self.sumoBinary = "/usr/bin/sumo-gui"  # sumo仿真的地址，可以是sumo-gui
        else:
            self.sumoBinary = "/usr/bin/sumo"
        self.Ele_loss = 10  # 馈电状态每秒钟损失
        self.chargingnet = None
        self.vehicle_info = {}  # 用来存储每辆车的信息字典
        self.reservations_dict = {}


    def run_randomTrips(self):

        perid0: float = (self.end_time - self.start_time) / self.num_trips  # 每辆车出现的平均间隔时间
        perid1: float = (self.end_time1 - self.start_time1) / self.num_trips1  # 每辆车出现的平均间隔时间

        """构建命令,这一段代码是生成随机的trip"""
        command = f'/home/huapeng/anaconda3/bin/python "/usr/share/sumo/tools/randomTrips.py" -s {self.seed} -n "{self.net_file}" -b {self.start_time} -e {self.end_time} -p {perid0}  ' \
                  f'--edge-param SS --intermediate {self.intermediate}  --trip-attributes="type= \\"typedist1\\"" --additional-file "{self.vehicle_add}" '

        """这一段代码是调用上面随机生成的trip然后开始分析得出想要的数据,https://sumo.dlr.de/docs/duarouter.html"""
        command2 = f'duarouter --route-files "merged.xml" --additional-files "{self.vehicle_add}" --net-file "{self.net_file}" ' \
                   f'--output-file result.rou.xml --exit-times True --route-length True '

        """构建命令,这一段代码是生成随机的ride，这些命令可以在randomtrip.py里查看"""
        command3 = f'/home/huapeng/anaconda3/bin/python "/usr/share/sumo/tools/randomTrips.py" -n "{self.net_file}" --min-distance "200"  --length  ' \
                   f'--pedestrians --personrides "taxi"  --max-dist "2000" -b {self.start_time1} -e {self.end_time1} -p {perid1} -o "trips.ride.xml" --random-departpos '


        try:
            # 运行命令并等待它完成
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"命令执行失败：{e}")
        except FileNotFoundError as e:
            print("未找到 randomTrips.py，请检查路径是否正确。")

        try:
            # 运行命令并等待它完成
            subprocess.run(command3, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"命令执行失败：{e}")
        except FileNotFoundError as e:
            print("未找到 randomTrips.py，请检查路径是否正确。")

        tree1 = ET.parse("trips.trips.xml")
        root1 = tree1.getroot()

        tree2 = ET.parse("trips.ride.xml")
        root2 = tree2.getroot()

        # 将第二个XML文件的内容添加到第一个XML文件的根节点下
        for child in root2:
            root1.append(child)

        # 创建新的合并后的XML文件
        merged_tree = ET.ElementTree(root1)
        merged_tree.write("merged.xml")

        try:
            # 运行命令并等待它完成
            subprocess.run(command2, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"命令执行失败：{e}")
        except FileNotFoundError as e:
            print("未找到 ，请检查路径是否正确。")


    #<additional-files value="CS_add.add.xml"/>
    def create_sumo_config_file(self):
        config_file_path = f"{self.folder_path}/config3.sumocfg"

        with open(config_file_path, "w") as file:
            file.write(f'''<?xml version="1.0" encoding="UTF-8"?>
    <configuration>
        <input>
            <net-file value="{self.net_file}"/>
            <route-files value="{self.route_file}"/>
            <gui-settings-file value="fig.xml"/>


        </input>
         <output>
            <battery-output value="Battery.out.xml"/>
            <battery-output.precision value="4"/>
            <device.battery.probability value="1"/>
            <summary-output value="summary_100.xml"/>
            <tripinfo-output value="tripinfo.xml"/>
        </output>
        <time>
            <begin value="{self.begin_time}"/>
            <end value="9999"/>
            <step-length value="{self.step_length}"/>
        </time>
        <taxidevice>
            <device.taxi.dispatch-algorithm value ="traci"/>
            <device.taxi.dispatch-algorithm.output value ="tripinfo2.xml"/>
        </taxidevice>
    </configuration>''')


    def reset(self):
        self.run_randomTrips()
        self.create_sumo_config_file()

        traci.start([self.sumoBinary, "-c", f"{self.folder_path}/config3.sumocfg", "--time-to-teleport", "999"])
        #  获得当前的时间
        current_time = traci.simulation.getTime()

    def reset_train(self):
        self.run_randomTrips()
        self.create_sumo_config_file()
        traci.start([self.sumoBinary, "-c", f"{self.folder_path}/config3.sumocfg", "--time-to-teleport", "999"])
        #  获得当前的时间
        current_time = traci.simulation.getTime()
        while current_time <= self.end_time1:
            env1.sim_step()
            current_time = traci.simulation.getTime()

    def sim_step(self, xingcheng = None):
        """
        这是一个单次更新的函数，
        更新的时候需要首先读取一个调度字典，里面包含了每辆车应该前往哪里，
        再读取一个订单字典，更新乘客应该出现在哪里，
        在此基础上运行一次迭代步
        并且将运行结果后的信息，传递出来，分别给车辆管理，订单系统，调度系统
        """
        # 读取调度字典 xingcheng，如果字典不为空
        if xingcheng:
            for vehicle_id, lanes in xingcheng['via'].items():
                # 设置车辆的路径（假设lanes是一个有效的车道列表）
                traci.vehicle.setVia(vehicle_id, lanes)
                traci.vehicle.rerouteTraveltime(vehicle_id)

            for vehicle_id, order_id in xingcheng['dispatch'].items():
                traci.vehicle.dispatchTaxi(vehicle_id, [order_id])

        """１　先进行订阅"""
        for vehicle_id in traci.vehicle.getIDList():  # 这里对每一辆出现的车进行订阅，可以优化
            traci.vehicle.subscribe(vehicle_id, self.attributes)
            traci.vehicle.subscribeParameterWithKey(vehicle_id, "device.battery.actualBatteryCapacity")
        current_time = traci.simulation.getTime()
        """2 根据调度信息运行一步迭代步"""
        traci.simulationStep()

        """3 获得订阅更新订单信息和更新车辆信息"""

        ride = traci.person.getTaxiReservations(0)
        for reservation in ride:
            self.reservations_dict[reservation.id] = {
                'persons': reservation.persons,
                'group': reservation.group,
                'fromEdge': reservation.fromEdge,
                'toEdge': reservation.toEdge,
                'departPos': reservation.departPos,
                'arrivalPos': reservation.arrivalPos,
                'depart': reservation.depart,
                'reservationTime': reservation.reservationTime,
                'state': reservation.state
            }
        # 现在 reservations_dict 包含了所有以id为键的预订数据
        # 你可以打印它或者进行其他处理
        trip_0 = traci.vehicle.getTaxiFleet(0)
        self.update_vehicle_info(trip_0)


        """
        if current_time == 502:
            traci.vehicle.setVia("0", ["-E58","-E113","-E114","E71"])
            traci.vehicle.dispatchTaxi("0", [2])
        if current_time == 510:
            traci.vehicle.setVia("0", ["-E113", "-E114", "E71"])
            traci.vehicle.dispatchTaxi("0", [0, 0, 1, 1])

        if current_time <= 500:
            traci.vehicle.setVia("0", "-E57")  # 给该车辆增加上需要的途经点
            traci.vehicle.rerouteTraveltime("0")
            traci.vehicle.setChargingStationStop("0","cS_-E57",duration=800)

        # 这里车辆可以完成停靠充电等任务

        """

    def update_vehicle_info(self,trip_0:tuple):
        # 遍历所有车辆，更新它们的信息
        current_time = traci.simulation.getTime()
        for veh_id in traci.vehicle.getIDList():
            sub = traci.vehicle.getSubscriptionResults(veh_id)  # 获取该车辆的订阅
            # stops = traci.vehicle.getStops(veh_id)
            # route = traci.vehicle.getRoute(veh_id)
            # print(f"{current_time}时，更新后车{veh_id}的路程", route, stops)
            # 获取车辆的基本信息
            if sub != {}:
                # 计算平均车速（这里简化为当前速度，实际中可能需要记录历史速度来计算平均值）
                average_speed = sub[132]/(current_time-sub[58]+1)
                if veh_id in trip_0:
                    env_state = 0
                else:
                    env_state = 2

                # 计算距离每个订单的距离（这里需要知道订单的位置和车辆的位置）
                # 假设有一个方法get_order_positions()返回所有订单的位置字典{order_id: (x, y)}
                order_positions = self.filter_reservations_by_state(2)   # 实际中需要替换为真实的订单位置获取逻辑
                distances_to_orders = None
                    #{order_id: self.calculate_distance(sub[80], order_pos) for order_id, order_pos in order_positions.items()}

                # 更新车辆信息字典
                self.vehicle_info[veh_id] = {
                    'x': sub[66][0],
                    'y': sub[66][1],
                    'lane': sub[80],
                    'lane_pos': max(sub[86],0),
                    'costs_to_orders': distances_to_orders,
                    'average_speed': average_speed,
                    'env_state': env_state,
                    'current_time': current_time
                }
    def filter_reservations_by_state(self, state):
        """过滤预订字典中特定状态的预订并返回新的字典。"""
        filtered_reservations = {
            reservation_id: reservation['fromEdge']
            for reservation_id, reservation in self.reservations_dict.items()
            if reservation['state'] == state
        }
        return filtered_reservations


    @staticmethod
    def calculate_distance(start_edge_id, end_edge_id):
        shortest_route = traci.simulation.findRoute(start_edge_id, end_edge_id)
        if shortest_route:
            # 获取最短路径的长度
            shortest_distance = shortest_route.cost
            return shortest_distance
        else:
            return None





if __name__ == "__main__":
    # 配置日志记录器

    env1 = sim_ENV(True)
    env1.reset()

    allocator = OrderAllocator()
    done = False
    update_dict = {}
    while not done:
        env1.sim_step(update_dict)
        allocator.update_order_info(env1.reservations_dict)
        allocator.update_vehicle_info(env1.vehicle_info)
        update_dict = allocator.iterate()
        print(allocator.storage)

        current_time = traci.simulation.getTime()
        if current_time >= 5000:
            done = True
    traci.close()

