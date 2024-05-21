from src.common.optimization_problem import OptimizationProblem

import math
from matplotlib import pyplot as plt
from collections import namedtuple

TIME_FACTOR = 10


class VRP:
    VisitData = namedtuple("CustomerData", "demand service_time earliest, latest")

    def __init__(self, pb):
        # Sizes
        self._num_veh = pb.get_nb_trucks()
        self._num_cust = pb.get_num_nodes() - 1
        self._n = self._num_cust + self._num_veh * 2

        # First, last, customer groups
        self._first = tuple(self._num_cust + i for i in range(self._num_veh))
        self._last = tuple(self._num_cust + self._num_veh + i for i in range(self._num_veh))
        self._cust = tuple(range(self._num_cust))

        # Time and load limits
        self._max_horizon = pb.get_max_horizon()
        self._capacity = pb.get_capacity()

        # Node mapping
        pnode = [i + 1 for i in range(self._num_cust)] + [0] * (2 * self._num_veh)

        # Visit data
        self._visit_data = \
            tuple(VRP.VisitData(pb.get_demand(pnode[c]), pb.get_service_time(pnode[c]), pb.get_earliest_start(pnode[c]),
                                pb.get_latest_start(pnode[c])) for c in self._cust) + \
            tuple(VRP.VisitData(0, 0, 0, self._max_horizon) for _ in self._first + self._last)

        # Distance
        self.distance_matrix = [
            [pb.get_distance(pnode[i], pnode[j]) for j in range(self._n)]
            for i in range(self._n)
        ]

    def first(self): return self._first

    def last(self): return self._last

    def vehicles(self): return zip(range(self._num_veh), self._first, self._last)

    def customers(self): return self._cust

    def all(self): return range(self._n)

    def get_num_customers(self): return self._num_cust

    def get_num_visits(self): return self._n

    def get_num_vehicles(self): return self._num_veh

    def get_first(self, veh): return self._first[veh]

    def get_last(self, veh): return self._last[veh]

    def get_capacity(self): return self._capacity

    def get_max_horizon(self): return self._max_horizon

    def get_demand(self, i): return self._visit_data[i].demand

    def get_service_time(self, i): return self._visit_data[i].service_time

    def get_earliest_start(self, i): return self._visit_data[i].earliest

    def get_latest_start(self, i): return self._visit_data[i].latest

    def get_distance(self, i, j): return self.distance_matrix[i][j]


class CVRPTW(OptimizationProblem):
    def __init__(self, benchmark_name, instance_name, data, solution, run_history) -> None:
        super().__init__(benchmark_name, instance_name, "CVRPTW", data, solution, run_history)
        self.nb_trucks = self._data['nb_trucks']
        self.truck_capacity = self._data['truck_capacity']
        self.max_horizon = self._data['max_horizon']
        self.nb_customers = self._data['nb_customers']
        self.depot_xy = self._data['depot_xy']
        self.customers_xy = self._data['customers_xy']
        self.demands = self._data['demands']
        self.earliest_start = self._data['earliest_start']
        self.latest_start = self._data['latest_start']
        self.service_time = self._data['service_time']
        self._xy = self._data['_xy']

    def read_elem(self, filename):
        with open(filename) as f:
            return [str(elem) for elem in f.read().split()]

    # The input files follow the "Solomon" format.
    def read(self, filename):

        def skip_elems(n):
            for _ in range(n):
                next(file_it)

        file_it = iter(self.read_elem(filename))

        skip_elems(4)

        self.nb_trucks = int(next(file_it))
        self.truck_capacity = int(next(file_it))

        skip_elems(13)

        self.depot_xy = (int(next(file_it)), int(next(file_it)))

        skip_elems(2)

        self.max_horizon = int(next(file_it))

        skip_elems(1)

        idx = 0
        while True:
            val = next(file_it, None)
            if val is None: break
            idx = int(val) - 1
            self.customers_xy.append((int(next(file_it)), int(next(file_it))))
            self.demands.append(int(next(file_it)))
            ready = int(next(file_it))
            due = int(next(file_it))
            stime = int(next(file_it))
            self.earliest_start.append(ready)
            self.latest_start.append(due)
            self.service_time.append(stime)

        self.nb_customers = idx + 1
        self._xy = [self.depot_xy] + self.customers_xy

    def get_num_nodes(self):
        return self.nb_customers + 1

    def get_nb_trucks(self):
        return self.nb_trucks

    def get_capacity(self):
        return self.truck_capacity

    def get_max_horizon(self):
        return TIME_FACTOR * self.max_horizon

    def get_demand(self, i):
        assert i >= 0
        assert i < self.get_num_nodes()
        if i == 0:
            return 0
        return self.demands[i - 1]

    def get_service_time(self, i):
        assert i >= 0
        assert i < self.get_num_nodes()
        if i == 0:
            return 0
        return TIME_FACTOR * self.service_time[i - 1]

    def get_earliest_start(self, i):
        assert i >= 0
        assert i < self.get_num_nodes()
        if i == 0:
            return 0
        return TIME_FACTOR * self.earliest_start[i - 1]

    def vehicles(self):
        return zip(range(self.nb_trucks), [0] * self.nb_trucks, [0] * self.nb_trucks)

    def get_latest_start(self, i):
        assert i >= 0
        assert i < self.get_num_nodes()
        if i == 0:
            return 0
        return TIME_FACTOR * self.latest_start[i - 1]

    def _get_distance(self, from_, to_):
        c1, c2 = self._xy[from_], self._xy[to_]
        dx, dy, d = c2[0] - c1[0], c2[1] - c1[1], 0.0
        d = math.sqrt(dx * dx + dy * dy)
        return int(math.floor(d * TIME_FACTOR))

    def get_distance(self, from_, to_):
        assert from_ >= 0, f"from_ = {from_}, to_ = {to_}"
        assert from_ < self.get_num_nodes(), f"from_ = {from_}, to_ = {to_}"
        assert to_ >= 0, f"from_ = {from_}, to_ = {to_}"
        assert to_ < self.get_num_nodes(), f"from_ = {from_}, to_ = {to_}"
        return self._get_distance(from_, to_)

    def to_dict(self):
        data = {
            'nb_trucks': self.nb_trucks,
            'truck_capacity': self.truck_capacity,
            'max_horizon': self.max_horizon,
            'nb_customers': self.nb_customers,
            'depot_xy': self.depot_xy,
            'customers_xy': self.customers_xy,
            'demands': self.demands,
            'earliest_start': self.earliest_start,
            'latest_start': self.latest_start,
            'service_time': self.service_time,
            '_xy': self._xy
        }
        return data

    def from_dict(self, data):
        self.nb_trucks = data['nb_trucks']
        self.truck_capacity = data['truck_capacity']
        self.max_horizon = data['max_horizon']
        self.nb_customers = data['nb_customers']
        self.depot_xy = data['depot_xy']
        self.customers_xy = data['customers_xy']
        self.demands = data['demands']
        self.earliest_start = data['earliest_start']
        self.latest_start = data['latest_start']
        self.service_time = data['service_time']
        self._xy = data['_xy']

    @property
    def xy(self):
        return self._xy


def visualize_path(path, data, save=None):
    loc_x, loc_y = list(zip(*data.customers_xy))

    n = range(1, len(loc_x)+1)

    plt.scatter(loc_x, loc_y, c='b')
    plt.plot(data.depot_xy[0], data.depot_xy[1], c='r', marker='s')
    for i, txt in enumerate(n):
        plt.annotate(txt, (loc_x[i], loc_y[i]))

    loc_x = [data.depot_xy[0]] + list(loc_x)
    loc_y = [data.depot_xy[1]] + list(loc_y)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    c = 0

    v = 0
    for route in path['paths']:
        if len(route) > 2:
            for i in range(len(route) - 1):
                plt.plot([loc_x[route[i]], loc_x[route[i + 1]]], [loc_y[route[i]], loc_y[route[i + 1]]], c=colors[c],
                         alpha=0.3)
            c += 1
            if c >= len(colors):
                c = 0
        v += 1

    if 'total_distance' in path:
        plt.title(f"Total distance: {path['total_distance']}")
    else:
        plt.title("Paths")
    if save:
        plt.savefig(save, format='pdf')
    plt.show()



def validate_path(path, data):
    vrp = data

    total_distance = 0
    v = 0
    visited = [False for i in range(vrp.nb_customers)]
    for route in path['paths']:
        # route = [-1 if i >= vrp.get_num_customers() else i for i in route]
        # print("Route", route)
        assert route[0] == 0 and route[-1] == 0, f"Vehicle {v} does not start or end at the depot"
        if len(route) > 2:
            arrive = 0
            total_load = 0
            for idx, nd in enumerate(route[:-1]):
                if nd != 0:  # Depo doesn't need this check
                    assert visited[nd-1] == False, f"Customer {nd} has already been been visited"
                    visited[nd-1] = True
                early = vrp.get_earliest_start(nd)
                late = vrp.get_latest_start(nd)
                start = max(arrive, early)

                # print(f"Node {nd} arrive {arrive} early {early} late {late} start {start}")

                if nd != 0:  # Depo doesn't need this check
                    assert start <= late, f"Too late for node {nd}"

                nxt = route[idx + 1]
                locald = vrp.get_distance(nd, nxt)

                # print(f"Distance from {nd} to {nxt} is {locald}")

                if nd != 0:
                    serv = vrp.get_service_time(nd)
                    arrive = start + serv + locald
                    total_distance += locald
                    total_load += vrp.get_demand(nd)
                if nd == 0:
                    arrive = start + locald
                    total_distance += locald
            assert total_load <= vrp.get_capacity(), f"Vehicle {v} exceeds its capacity"
            # print("Route len", arrive)
        v += 1

    assert all(visited), "Not all customers have been visited"

    total_distance /= TIME_FACTOR
    if 'total_distance' in path:
        target = path['total_distance']
        diff = abs(total_distance - target)
        assert diff < 0.1, f"Total distance {total_distance} does not match objective value {target}"
    else:
        # print("No distance found in solution, adding it", total_distance)
        path['total_distance'] = total_distance
    # print("Valid solution, total_distance =", total_distance)
    return True
