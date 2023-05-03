import localsolver
import sys
import math


def read_elem(filename):
    with open(filename) as f:
        return [str(elem) for elem in f.read().split()]


def main(instance_file, str_time_limit, output_file):
    #
    # Read instance data
    #
    nb_customers, nb_trucks, truck_capacity, dist_matrix_data, dist_depot_data, \
        demands_data, service_time_data, earliest_start_data, latest_end_data, \
        max_horizon = read_input_cvrptw(instance_file)

    with localsolver.LocalSolver() as ls:
        #
        # Declare the optimization model
        #
        model = ls.model

        # Sequence of customers visited by each truck
        customers_sequences = [model.list(nb_customers) for k in range(nb_trucks)]

        # All customers must be visited by the trucks
        model.constraint(model.partition(customers_sequences))

        # Create LocalSolver arrays to be able to access them with an "at" operator
        demands = model.array(demands_data)
        earliest = model.array(earliest_start_data)
        latest = model.array(latest_end_data)
        service_time = model.array(service_time_data)
        dist_matrix = model.array()
        for n in range(nb_customers):
            dist_matrix.add_operand(model.array(dist_matrix_data[n]))
        dist_depot = model.array(dist_depot_data)

        dist_routes = [None] * nb_trucks
        end_time = [None] * nb_trucks
        home_lateness = [None] * nb_trucks
        lateness = [None] * nb_trucks

        # A truck is used if it visits at least one customer
        trucks_used = [(model.count(customers_sequences[k]) > 0) for k in range(nb_trucks)]
        nb_trucks_used = model.sum(trucks_used)

        for k in range(nb_trucks):
            sequence = customers_sequences[k]
            c = model.count(sequence)

            # The quantity needed in each route must not exceed the truck capacity
            demand_lambda = model.lambda_function(lambda i: demands[sequence[i]])
            route_quantity = model.sum(model.range(0, c), demand_lambda)
            model.constraint(route_quantity <= truck_capacity)

            # Distance traveled by each truck
            dist_lambda = model.lambda_function(
                lambda i: model.at(dist_matrix, sequence[i - 1], sequence[i]))
            dist_routes[k] = model.sum(model.range(1, c), dist_lambda) \
                + model.iif(c > 0, dist_depot[sequence[0]] + dist_depot[sequence[c - 1]], 0)

            # End of each visit
            end_time_lambda = model.lambda_function(
                lambda i, prev:
                    model.max(
                        earliest[sequence[i]],
                        model.iif(
                            i == 0, dist_depot[sequence[0]],
                            prev + model.at(dist_matrix, sequence[i - 1], sequence[i])))
                    + service_time[sequence[i]])

            end_time[k] = model.array(model.range(0, c), end_time_lambda)

            # Arriving home after max horizon
            home_lateness[k] = model.iif(
                trucks_used[k],
                model.max(
                    0,
                    end_time[k][c - 1] + dist_depot[sequence[c - 1]] - max_horizon),
                0)

            # Completing visit after latest end
            late_lambda = model.lambda_function(
                lambda i: model.max(0, end_time[k][i] - latest[sequence[i]]))
            lateness[k] = home_lateness[k] + model.sum(model.range(0, c), late_lambda)

        # Total lateness
        total_lateness = model.sum(lateness)

        # Total distance traveled
        total_distance = model.div(model.round(100 * model.sum(dist_routes)), 100)

        # Objective: minimize the number of trucks used, then minimize the distance traveled
        model.minimize(total_lateness)
        model.minimize(nb_trucks_used)
        model.minimize(total_distance)

        model.close()

        # Parameterize the solver
        ls.param.time_limit = int(str_time_limit)

        ls.solve()

        #
        # Write the solution in a file with the following format:
        #  - number of trucks used and total distance
        #  - for each truck the customers visited (omitting the start/end at the depot)
        #
        if output_file is not None:
            with open(output_file, 'w') as f:
                f.write("%d %d\n" % (nb_trucks_used.value, total_distance.value))
                for k in range(nb_trucks):
                    if trucks_used[k].value != 1:
                        continue
                    # Values in sequence are in [0..nbCustomers-1]. +1 is to put it back in
                    # [1..nbCustomers] as in the data files (0 being the depot)
                    for customer in customers_sequences[k].value:
                        f.write("%d " % (customer + 1))
                    f.write("\n")


# The input files follow the "Solomon" format
def read_input_cvrptw(filename):
    file_it = iter(read_elem(filename))

    for i in range(4):
        next(file_it)

    nb_trucks = int(next(file_it))
    truck_capacity = int(next(file_it))

    for i in range(13):
        next(file_it)

    depot_x = int(next(file_it))
    depot_y = int(next(file_it))

    for i in range(2):
        next(file_it)

    max_horizon = int(next(file_it))

    next(file_it)

    customers_x = []
    customers_y = []
    demands = []
    earliest_start = []
    latest_end = []
    service_time = []

    while True:
        val = next(file_it, None)
        if val is None:
            break
        i = int(val) - 1
        customers_x.append(int(next(file_it)))
        customers_y.append(int(next(file_it)))
        demands.append(int(next(file_it)))
        ready = int(next(file_it))
        due = int(next(file_it))
        stime = int(next(file_it))
        earliest_start.append(ready)
        # in input files due date is meant as latest start time
        latest_end.append(due + stime)
        service_time.append(stime)

    nb_customers = i + 1

    # Compute distance matrix
    distance_matrix = compute_distance_matrix(customers_x, customers_y)
    distance_depots = compute_distance_depots(depot_x, depot_y, customers_x, customers_y)

    return nb_customers, nb_trucks, truck_capacity, distance_matrix, distance_depots, \
        demands, service_time, earliest_start, latest_end, max_horizon


# Computes the distance matrix
def compute_distance_matrix(customers_x, customers_y):
    nb_customers = len(customers_x)
    distance_matrix = [[None for i in range(nb_customers)] for j in range(nb_customers)]
    for i in range(nb_customers):
        distance_matrix[i][i] = 0
        for j in range(nb_customers):
            dist = compute_dist(customers_x[i], customers_x[j],
                                customers_y[i], customers_y[j])
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist
    return distance_matrix


# Computes the distances to depot
def compute_distance_depots(depot_x, depot_y, customers_x, customers_y):
    nb_customers = len(customers_x)
    distance_depots = [None] * nb_customers
    for i in range(nb_customers):
        dist = compute_dist(depot_x, customers_x[i], depot_y, customers_y[i])
        distance_depots[i] = dist
    return distance_depots


def compute_dist(xi, xj, yi, yj):
    return math.sqrt(math.pow(xi - xj, 2) + math.pow(yi - yj, 2))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python cvrptw.py input_file [output_file] [time_limit]")
        sys.exit(1)

    instance_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    str_time_limit = sys.argv[3] if len(sys.argv) > 3 else "20"

    main(instance_file, str_time_limit, output_file)