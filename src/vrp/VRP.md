This folder focuses on Vehicle Routing Problem (VRP) and its variants. VRP is a combinatorial optimization and integer programming problem seeking to service a number of customers with a fleet of vehicles. The goal is to minimize the total route cost. The VRP is a generalization of the Traveling Salesman Problem (TSP).

Three solvers are implemented in this folder, all of them for the CVRPTW (Vehicle Routing Problem with Time Windows) variant. The CVRPTW instances are taken from the Solomon benchmark.
Best known results are taken from http://web.cba.neu.edu/~msolomon/problems.htm and https://www.sintef.no/projectweb/top/vrptw/100-customers/

The solvers are:

Integer model using Docplex

Integer model using OR-Tools

Interval model using Docplex

These solvers solve an instance of CVRPTW (defined in problem.py) and return the best found list of paths, one path for each vehicle. 0 marks the depot, and the rest of the numbers are the customers to be visited. The solvers also return the total cost of the solution.
