try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
try:
    from urllib2 import urlopen
except ImportError:
    from urllib.request import urlopen
# This notebook requires pandas to work
import pandas as pd
from pandas import DataFrame

from utils import *

fname = "..\\..\\data\\NRP\\NRC\\Instance6.txt"
problem = NRCProblem()
problem.load_from_txt(fname)

# Use pandas to read the file, one tab for each table.
data_url = "original example/nurses_data.xls"
nurse_xls_file = pd.ExcelFile(data_url)

df_skills = nurse_xls_file.parse('Skills')
df_depts = nurse_xls_file.parse('Departments')
df_shifts = nurse_xls_file.parse('Shifts')
# Rename df_shifts index
df_shifts.index.name = 'shiftId'

# Index is column 0: name
df_nurses = nurse_xls_file.parse('Nurses', header=0, index_col=0)
df_nurse_skilles = nurse_xls_file.parse('NurseSkills')
df_vacations = nurse_xls_file.parse('NurseVacations')
df_associations = nurse_xls_file.parse('NurseAssociations')
df_incompatibilities = nurse_xls_file.parse('NurseIncompatibilities')

# Display the nurses dataframe
print("#nurses = {}".format(len(df_nurses)))
print("#shifts = {}".format(len(df_shifts)))
print("#vacations = {}".format(len(df_vacations)))

# maximum work time (in hours)
max_work_time = 40

# maximum number of shifts worked in a week.
max_nb_shifts = 5

days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
day_of_weeks = dict(zip(days, range(7)))


# utility to convert a day string e.g. "Monday" to an integer in 0..6
def day_to_day_of_week(day):
    return day_of_weeks[day.strip().lower()]


# for each day name, we normalize it by stripping whitespace and converting it to lowercase
# " Monday" -> "monday"
df_shifts["dow"] = df_shifts.day.apply(day_to_day_of_week)

df_shifts["wstart"] = df_shifts.start_time + 24 * df_shifts.dow


# an auxiliary function to calculate absolute end time of a shift
def calculate_absolute_endtime(start, end, dow):
    return 24 * dow + end + (24 if start >= end else 0)


# store the results in a new column
df_shifts["wend"] = df_shifts.apply(lambda row: calculate_absolute_endtime(
    row.start_time, row.end_time, row.dow), axis=1)

df_shifts["duration"] = df_shifts.wend - df_shifts.wstart

# also compute minimum demand in nurse-hours
df_shifts["min_demand"] = df_shifts.min_req * df_shifts.duration

from docplex.mp.model import Model

mdl = Model(name="nurses")

# first global collections to iterate upon
# all_nurses = df_nurses.index.values
# all_shifts = df_shifts.index.values
all_nurses = problem.staff.keys()
all_shifts = problem.shifts.keys()

# the assignment variables.
assigned = mdl.binary_var_matrix(keys1=all_nurses, keys2=all_shifts, name="assign_%s_%s")

# Organize decision variables in a DataFrame
df_assigned = DataFrame({'assigned': assigned})
df_assigned.index.names = ['all_nurses', 'all_shifts']

# Re-organize the Data Frame as a pivot table with nurses as row index and shifts as columns:
df_assigned_pivot = df_assigned.unstack(level='all_shifts')

# Create a pivot using nurses and shifts index as dimensions
# df_assigned_pivot = df_assigned.reset_index().pivot(index='all_nurses', columns='all_shifts', values='assigned')

# Create a Data Frame representing a list of shifts sorted by wstart and duration.
# One keeps only the three relevant columns: 'shiftId', 'wstart' and 'wend' in the resulting Data Frame
df_sorted_shifts = df_shifts.sort_values(['wstart', 'duration']).reset_index()[['shiftId', 'wstart', 'wend']]


# Add 'day of week' column to vacations Data Frame
df_vacations['dow'] = df_vacations.day.apply(day_to_day_of_week)

# Join 'df_vacations', 'df_shifts' and 'df_assigned' Data Frames to create the list of 'forbidden' assigments.
# The 'reset_index()' function is invoked to move 'shiftId' index as a column in 'df_shifts' Data Frame, and
# to move the index pair ('all_nurses', 'all_shifts') as columns in 'df_assigned' Data Frame.
# 'reset_index()' is invoked so that a join can be performed between Data Frame, based on column names.
df_assigned_reindexed = df_assigned.reset_index()
df_vacation_forbidden_assignments = df_vacations.merge(df_shifts.reset_index()[['dow', 'shiftId']]).merge(
    df_assigned_reindexed, left_on=['nurse', 'shiftId'], right_on=['all_nurses', 'all_shifts'])

for forbidden_assignment in df_vacation_forbidden_assignments.itertuples():
    # to forbid an assignment just set the variable to zero.
    mdl.add_constraint(forbidden_assignment.assigned == 0)
print("# vacation forbids: {} assignments".format(len(df_vacation_forbidden_assignments)))

# Join 'df_assignment' Data Frame twice, based on associations to get corresponding decision variables pairs for all shifts
# The 'suffixes' parameter in the second merge indicates our preference for updating the name of columns that occur both
# in the first and second argument Data Frames (in our case, these columns are 'all_nurses' and 'assigned').
df_preferred_assign = df_associations.merge(
    df_assigned_reindexed, left_on='nurse1', right_on='all_nurses').merge(
    df_assigned_reindexed, left_on=['nurse2', 'all_shifts'], right_on=['all_nurses', 'all_shifts'],
    suffixes=('_1', '_2'))

for preferred_assign in df_preferred_assign.itertuples():
    mdl.add_constraint(preferred_assign.assigned_1 == preferred_assign.assigned_2)

# Join assignment Data Frame twice, based on incompatibilities Data Frame to get corresponding decision variables pairs
#  for all shifts
df_incompatible_assign = df_incompatibilities.merge(
    df_assigned_reindexed, left_on='nurse1', right_on='all_nurses').merge(
    df_assigned_reindexed, left_on=['nurse2', 'all_shifts'], right_on=['all_nurses', 'all_shifts'],
    suffixes=('_1', '_2'))

for incompatible_assign in df_incompatible_assign.itertuples():
    mdl.add_constraint(incompatible_assign.assigned_1 + incompatible_assign.assigned_2 <= 1)


# auxiliary function to create worktime variable from a row
def make_var(row, varname_fmt):
    return mdl.continuous_var(name=varname_fmt % row.name, lb=0)


# apply the function over nurse rows and store result in a new column
df_nurses["worktime"] = df_nurses.apply(lambda r: make_var(r, "worktime_%s"), axis=1)

# Use pandas' groupby operation to enforce constraint calculating worktime for each nurse as the sum of all assigned
#  shifts times the duration of each shift
for nurse, nurse_assignments in df_assigned.groupby(level='all_nurses'):
    mdl.add_constraint(df_nurses.worktime[nurse] == mdl.dot(nurse_assignments.assigned, df_shifts.duration))


# we use pandas' apply() method to set an upper bound on all worktime variables.
def set_max_work_time(v):
    v.ub = max_work_time
    # Optionally: return a string for fancy display of the constraint in the Output cell
    return str(v) + ' <= ' + str(v.ub)


df_nurses["worktime"].apply(convert_dtype=False, func=set_max_work_time)

# Use pandas' groupby operation to enforce minimum requirement constraint for each shift
for shift, shift_nurses in df_assigned.groupby(level='all_shifts'):
    mdl.add_constraint(mdl.sum(shift_nurses.assigned) >= df_shifts.min_req[shift])

# again leverage pandas to create a series of expressions: costs of each nurse
total_salary_series = df_nurses.worktime * df_nurses.pay_rate

# compute global salary cost using pandas sum()
# Note that the result is a DOcplex expression: DOcplex if fully compatible with pandas
total_salary_cost = total_salary_series.sum()
mdl.add_kpi(total_salary_cost, "Total salary cost")

# Set Cplex mipgap to 1e-5 to enforce precision to be of the order of a unit (objective value magnitude is ~1e+5).
mdl.parameters.mip.tolerances.mipgap = 1e-5

s = mdl.solve(log_output=True)
assert s, "solve failed"
mdl.report()

# Create a pandas Series containing actual shift assignment decision variables value
s_assigned = df_assigned.assigned.apply(lambda v: v.solution_value)

# Create a pivot table by (nurses, shifts), using pandas' "unstack" method to transform the 'all_shifts' row index
#  into columns
df_res = s_assigned.unstack(level='all_shifts')

# Display the first few rows of the resulting pivot table
df_res.head()
