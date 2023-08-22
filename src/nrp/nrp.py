from docplex.mp.model import Model

# Data.
num_nurses = 4
num_shifts = 3
num_days = 3
all_nurses = range(num_nurses)
all_shifts = range(num_shifts)
all_days = range(num_days)

# Creates the model.
model = Model()
# Creates shift variables.
# shifts[(n, d, s)]: nurse 'n' works shift 's' on day 'd'.
shifts = {}

for n in all_nurses:
    for d in all_days:
        for s in all_shifts:
            shifts[(n, d, s)] = model.binary_var()

for d in all_days:
    for s in all_shifts:
        model.add_constraint(model.sum(shifts[(n, d, s)] for n in all_nurses) == 1)

    # Each nurse works at most one shift per day.
for n in all_nurses:
    for d in all_days:
        model.add_constraint(model.sum(shifts[(n, d, s)] for s in all_shifts) <= 1)

model.solve(log_output=True, )
