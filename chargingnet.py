import traci
class ChargingPile:
    def __init__(self, id, location):
        self.id = id
        self.location = location
        self.jieshutime = 0

class ChargingStation:
    def __init__(self, id, location):
        self.id = id
        self.location = location
        self.charging_piles = {}
        self.reserved_vehicles = []

    @property
    def max_time(self):  # 用来返回几个充电桩中需要等待的最长
        max_t = 0
        for i in self.charging_piles.values():
           if i.jieshutime > max_t:
               max_t = i.jieshutime
        return max_t

    def add_charging_pile(self, id, location):
        self.charging_piles[id] = ChargingPile(id, location)

    def reserve(self, vehicle_id):
        self.reserved_vehicles.append(vehicle_id)

    def depart(self,vehicle_id):
        self.reserved_vehicles.remove(vehicle_id)
        # 车辆充满电的时候会将车辆从该充电站的列表里删除
class ChargingNetwork:
    def __init__(self):
        self.stations = {}
        self.time = 0

    @property
    def stations_reserved(self):
        nums = []
        for i in self.stations.values():
            nums.append(len(i.reserved_vehicles))
        return nums



    def add_station(self, id, station):
        self.stations[id] = station  # 这是字典的形式将充电站保存起来 字典名是充电站的id，值是充电站本身

    def add_pile_to_station(self, station_id, pile_id, location):
        self.stations[station_id].add_charging_pile(pile_id, location)

    def reserve(self, station_id, vehicle_id):
        self.stations[station_id].reserve(vehicle_id)

    def increment_time(self):
        self.time += 1

    def find_distance(self, edge):  # 获取某个位置到当前所有充电桩的路程距离
        distance = []
        for value in self.stations.values():
            distance.append(round(traci.simulation.getDistanceRoad(edge, 0, value.location, 0, isDriving=True),2))
        return distance

    def initialize(self):
        chargingstation1 = ChargingStation(1, "E54")
        chargingstation1.add_charging_pile("-E59", "-E59")
        chargingstation1.add_charging_pile("-E57", "-E57")
        chargingstation1.add_charging_pile("-E55", "-E55")
        chargingstation1.add_charging_pile("-E61", "-E61")
        chargingstation1.add_charging_pile("-E63", "-E63")
        chargingstation2 = ChargingStation(2, "E66")
        chargingstation2.add_charging_pile("-E75", "-E75")
        chargingstation2.add_charging_pile("-E73", "-E73")
        chargingstation2.add_charging_pile("-E67", "-E67")
        chargingstation2.add_charging_pile("-E69", "-E69")
        chargingstation2.add_charging_pile("-E71", "-E71")
        chargingstation3 = ChargingStation(3, "E78")
        chargingstation3.add_charging_pile("-E86", "-E86")
        chargingstation3.add_charging_pile("-E84", "-E84")
        chargingstation3.add_charging_pile("-E82", "-E82")
        chargingstation3.add_charging_pile("-E80", "-E80")
        chargingstation3.add_charging_pile("-E87", "-E87")
        chargingstation4 = ChargingStation(4, "E91")
        chargingstation4.add_charging_pile("-E96", "-E96")
        chargingstation4.add_charging_pile("-E97", "-E97")
        chargingstation4.add_charging_pile("-E98", "-E98")
        chargingstation4.add_charging_pile("-E99", "-E99")
        chargingstation4.add_charging_pile("-E100", "-E100")
        chargingstation5 = ChargingStation(5, "E104")
        chargingstation5.add_charging_pile("-E114", "-E114")
        chargingstation5.add_charging_pile("-E113", "-E113")
        chargingstation5.add_charging_pile("-E112", "-E112")
        chargingstation5.add_charging_pile("-E111", "-E111")
        chargingstation5.add_charging_pile("-E110", "-E110")

        self.add_station(chargingstation1.id, chargingstation1)
        self.add_station(chargingstation2.id, chargingstation2)
        self.add_station(chargingstation3.id, chargingstation3)
        self.add_station(chargingstation4.id, chargingstation4)
        self.add_station(chargingstation5.id, chargingstation5)


if __name__ == "__main__":

    chargingstation1 = ChargingStation(1, "E54")
    chargingstation1.add_charging_pile("-E59", "-E59")
    chargingstation1.add_charging_pile("-E57", "-E57")
    chargingstation1.add_charging_pile("-E55", "-E55")
    chargingstation1.add_charging_pile("-E61", "-E61")
    chargingstation1.add_charging_pile("-E63", "-E63")
    chargingstation2 = ChargingStation(2, "E66")
    chargingstation2.add_charging_pile("-E75", "-E75")
    chargingstation2.add_charging_pile("-E73", "-E73")
    chargingstation2.add_charging_pile("-E67", "-E67")
    chargingstation2.add_charging_pile("-E69", "-E69")
    chargingstation2.add_charging_pile("-E71", "-E71")
    chargingstation3 = ChargingStation(3, "E78")
    chargingstation3.add_charging_pile("-E75", "-E75")
    chargingstation3.add_charging_pile("-E73", "-E73")
    chargingstation3.add_charging_pile("-E67", "-E67")
    chargingstation3.add_charging_pile("-E69", "-E69")
    chargingstation3.add_charging_pile("-E71", "-E71")
    chargingstation4 = ChargingStation(4, "E91")
    chargingstation4.add_charging_pile("-E96", "-E96")
    chargingstation4.add_charging_pile("-E97", "-E97")
    chargingstation4.add_charging_pile("-E98", "-E98")
    chargingstation4.add_charging_pile("-E99", "-E99")
    chargingstation4.add_charging_pile("-E100", "-E100")
    chargingstation5 = ChargingStation(5, "E104")
    chargingstation5.add_charging_pile("-E114", "-E114")
    chargingstation5.add_charging_pile("-E113", "-E113")
    chargingstation5.add_charging_pile("-E112", "-E112")
    chargingstation5.add_charging_pile("-E111", "-E111")
    chargingstation5.add_charging_pile("-E110", "-E110")
    chargingnet = ChargingNetwork()
    chargingnet.add_station(chargingstation1.id, chargingstation1)
    chargingnet.add_station(chargingstation2.id, chargingstation2)
    chargingnet.add_station(chargingstation3.id, chargingstation3)
    chargingnet.add_station(chargingstation4.id, chargingstation4)
    chargingnet.add_station(chargingstation5.id, chargingstation5)
    print(chargingnet.stations[1].location)

