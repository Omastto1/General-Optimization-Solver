# - *- coding: utf- 8 - *-

import math

from matplotlib import pyplot as plt

TIME_FACTOR = 10

class CVRPTWProblem:
    def __init__(self):
        self.nb_trucks = -1
        self.truck_capacity = -1
        self.max_horizon = -1
        self.nb_customers = -1
        self.depot_xy = None
        self.customers_xy = []
        self.demands = []
        self.earliest_start = []
        self.latest_start = []
        self.service_time = []
        self._xy = None

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
        assert from_ >= 0
        assert from_ < self.get_num_nodes()
        assert to_ >= 0
        assert to_ < self.get_num_nodes()
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


def visualize_path(path, data, cvrptw_prob):
    loc_x, loc_y = list(zip(*cvrptw_prob.customers_xy))

    n = range(1, len(loc_x)+1)

    plt.scatter(loc_x, loc_y, c='b')
    plt.plot(cvrptw_prob.depot_xy[0], cvrptw_prob.depot_xy[1], c='r', marker='s')
    for i, txt in enumerate(n):
        plt.annotate(txt, (loc_x[i], loc_y[i]))

    loc_x = [cvrptw_prob.depot_xy[0]] + list(loc_x)
    loc_y = [cvrptw_prob.depot_xy[1]] + list(loc_y)

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

    plt.show()


def validate_path(path, data):
    vrp = data

    total_distance = 0
    v = 0
    visited = [False for i in range(vrp.nb_customers)]
    for route in path['paths']:
        # route = [-1 if i >= vrp.get_num_customers() else i for i in route]
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

                print(f"Node {nd} arrive {arrive} early {early} late {late} start {start}")

                if nd != 0:  # Depo doesn't need this check
                    assert start <= late, f"Too late for node {nd}"

                nxt = route[idx + 1]
                locald = vrp.get_distance(nd, nxt)

                print(f"Distance from {nd} to {nxt} is {locald}")

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
        print("No distance found in solution, adding it", total_distance)
        path['total_distance'] = total_distance
    print("Valid solution, total_distance =", total_distance)
