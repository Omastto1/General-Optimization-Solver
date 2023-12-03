from docplex.mp.model import Model

from utils import *

fname = "..\\..\\data\\NRP\\NRC\\Instance6.txt"
problem = NRCProblem()
problem.load_from_txt(fname)

mdl = Model(name="nurses")

# first global collections to iterate upon
all_nurses = problem.staff.keys()
all_shifts = problem.shifts.keys()
all_days = list(range(problem.horizon))

# the assignment variables.
assigned = mdl.binary_var_cube(keys1=all_nurses, keys2=all_days, keys3=all_shifts, name="assign_%s_%s_%s")
print("assigned: ", assigned)

# vacations

vacations = []

for nurse in problem.days_off.keys():
    for day in problem.days_off[nurse]:
        # print(nurse, day)
        for shift in problem.cover_requirements:
            if shift['Day'] == day:
                temp = assigned[nurse, shift['Day'], shift['ShiftID']]
                vacations.append(temp)
                mdl.add_constraint(temp == 0)
        # for day in days:
        #     vacations.append(assigned[nurse, day])

print("vacations: ", vacations)

# max 1 shift per day

for nurse in all_nurses:
    for day in all_days:
        mdl.add_constraint(mdl.sum(assigned[nurse, day, shift] for shift in all_shifts) <= 1)

print(problem.shifts['L']['CannotFollow'])
