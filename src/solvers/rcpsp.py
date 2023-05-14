from docplex.cp.model import CpoModel


def solve_rcpsp(no_jobs, no_resources, durations, successors, capacities, requests, validate=False):
    mdl = CpoModel()

    x = [ mdl.interval_var(size = duration, name=f"{i}") for i, duration in enumerate(durations) ] # (4)

    mdl.add( [ mdl.minimize ( mdl.max( [mdl.end_of(x[i]) for i in range(no_jobs)] ) ) ] )# (1)

    # mdl.add( [ mdl.sum( mdl.pulse(x[i],parsed_input["job_specifications"][i]["request_duration"][f"R{k+1}"]) for i in range(no_jobs)) <= parsed_input["resources"]["renewable_resources"]["renewable_availabilities"][k] for k in range(parsed_input["resources"]["renewable_resources"]["number_of_resources"]) ] )# (2)
    mdl.add( [ mdl.sum( mdl.pulse(x[i], requests[k][i]) for i in range(no_jobs)) <= capacities[k] for k in range(no_resources) ] )# (2)

    mdl.add( [mdl.end_before_start( x[i], x[successor - 1] ) for (i, job_successors) in enumerate(successors) for successor in job_successors] ) # (3)

    # TODO: IS THIS NEEDED?
    # Define the initial conditions
    # mdl.add(x[0].get_start() == 0)

    sol = mdl.solve(LogVerbosity='Terse')\
    # sol = mdl.solve(TimeLimit=10)

    if sol:
        if validate:
            try:
                print("Validating solution...")
                validate_rcpsp(sol, x, successors, sol.get_objective_value())
                print("Solution is valid.")
            except AssertionError as e:
                print("Solution is invalid.")
                print(e)
                return None, None
            
        print("Project completion time:", sol.get_objective_value())
        # for i in range(no_jobs):
        #     print(f"Activity {i}: start={sol[i].get_start()}, end={sol[i].get_end()}")
    else:
        print("No solution found.")

    return sol, x


def validate_rcpsp(sol, x, successors, horizon):
    assert sol.get_objective_value() <= horizon, "Project completion time exceeds horizon."

    for i, job_successors in enumerate(successors):
        for successor in job_successors:
            assert sol.get_var_solution(x[i]).get_end() <= sol.get_var_solution(x[successor - 1]).get_start(), f"Job {i} ends after job {successor} starts."

    return True