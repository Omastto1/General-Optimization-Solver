import numpy as np
from matplotlib import pyplot as plt

from src.common.optimization_problem import OptimizationProblem


class NRC(OptimizationProblem):
    def __init__(self, benchmark_name, instance_name, data, solution, run_history) -> None:
        super().__init__(benchmark_name, instance_name, "NRC", data, solution, run_history)
        self.horizon = self._data["horizon"]
        self.shifts = self._data["shifts"]
        self.staff = self._data["staff"]
        self.days_off = self._data["days_off"]
        self.shift_on_requests = self._data["shift_on_requests"]
        self.shift_off_requests = self._data["shift_off_requests"]
        self.cover_requirements = self._data["cover_requirements"]

    def load_from_txt(self, path):
        sections = {}
        current_section = None
        with open(path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('SECTION_'):
                    current_section = line
                    sections[current_section] = []
                elif current_section:
                    sections[current_section].append(line)

        self.horizon = int(sections['SECTION_HORIZON'][2])

        self.shifts = parse_shift_section(sections['SECTION_SHIFTS'])
        self.staff = parse_staff(sections['SECTION_STAFF'])
        self.days_off = parse_daysoff_section(sections['SECTION_DAYS_OFF'])
        self.shift_on_requests = parse_requests_section(sections['SECTION_SHIFT_ON_REQUESTS'])
        self.shift_off_requests = parse_requests_section(sections['SECTION_SHIFT_OFF_REQUESTS'])
        self.cover_requirements = parse_cover_section(sections['SECTION_COVER'])

    def to_dict(self):
        return self.__dict__

    def from_dict(self, data):
        self.__dict__ = data

def parse_staff(lines):
    # ID, MaxShifts, MaxTotalMinutes, MinTotalMinutes, MaxConsecutiveShifts, MinConsecutiveShifts, MinConsecutiveDaysOff, MaxWeekends
    section_data = {}
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            split = line.split(',')
            shifts = split[1].strip().split('|')
            max_of_shift = {key: int(value) for (key, value) in [shift.split('=') for shift in shifts]}
            section_data[split[0].strip()] = {
                'max_of_shift': max_of_shift,
                'max_total_minutes': int(split[2]),
                'min_total_minutes': int(split[3]),
                'max_consecutive_shifts': int(split[4]),
                'min_consecutive_shifts': int(split[5]),
                'min_consecutive_days_off': int(split[6]),
                'max_weekends': int(split[7])
            }
    return section_data


def parse_shift_section(lines):
    shifts = {}
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            parts = line.split(',')
            shift_info = {
                'Length': int(parts[1]),
                'CannotFollow': [] if len(parts) < 3 or parts[2] == '' else parts[2].strip().split('|')
            }
            shifts[parts[0].strip()] = shift_info
    return shifts


def parse_daysoff_section(lines):
    days_off = {}
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            day_indexes = line.split(',')
            days_off[day_indexes[0].strip()] = list(map(int, day_indexes[1:]))
    return days_off


def parse_requests_section(lines):
    requests = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            parts = line.split(',')
            request_info = {
                'EmployeeID': parts[0].strip(),
                'Day': int(parts[1]),
                'ShiftID': parts[2].strip(),
                'Weight': int(parts[3])
            }
            requests.append(request_info)
    return requests


def parse_cover_section(lines):
    cover_requirements = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            parts = line.split(',')
            cover_info = {
                'Day': int(parts[0]),
                'ShiftID': parts[1].strip(),
                'Requirement': int(parts[2]),
                'WeightForUnder': int(parts[3]),
                'WeightForOver': int(parts[4])
            }
            cover_requirements.append(cover_info)
    return cover_requirements


def visualize_nrp(schedule_data, instance, obj_value, instance_name=None):
    # Extract unique shifts
    unique_shifts = sorted(set(shift for row in schedule_data for shift in row))

    # Define colors for each shift
    shift_colors = {shift: (1, 1, 1, 1.0) if shift == ' ' else plt.cm.tab10(i) for i, shift in enumerate(unique_shifts)}
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.imshow([[shift_colors[shift] for shift in row] for row in schedule_data], aspect='auto')

    # Annotate cells with shift labels
    for i in range(len(schedule_data)):
        for j in range(len(schedule_data[0])):
            plt.text(j, i, schedule_data[i][j], ha='center', va='center', color='black')

    # Customize ticks and labels
    plt.yticks(np.arange(len(instance.staff.keys())), instance.staff.keys())
    plt.xticks(np.arange(instance.horizon), np.arange(instance.horizon) + 1)
    plt.xlabel('Day')
    plt.ylabel('Nurse')

    plt.tick_params(axis='both', bottom=True, top=True, left=True, right=True, labelbottom=True, labeltop=True,
                    labelleft=True, labelright=True)

    plt.title('Nurse Rostering Schedule ' + (instance_name if instance_name else ''))
    plt.text(len(schedule_data[0]) + 0.5, len(schedule_data)*1.01 + 0.45, f"Penalty = {obj_value}", ha='right', va='center', color='gray', fontsize=8)

    # plt.show()


def validate_nrp(instance, sol, result):
    # print(result)

    # Check hard constraints

    staff_keys = list(instance.staff.keys())
    for nurse in instance.staff.keys():
        max_of_shift = {key: 0 for key in instance.staff[nurse]['max_of_shift'].keys()}
        nurse_int = staff_keys.index(nurse)
        consecutive_shifts = 0
        min_consecutive_shifts = instance.staff[nurse]['min_consecutive_shifts']
        consecutive_days_off = 0
        total_minutes = 0
        consecutive_shifts_max = 0
        consecutive_days_off_max = 0
        weekends = [False for _ in range(instance.horizon // 7)]
        for day in range(instance.horizon):
            shift = result[nurse_int, day]
            if shift in max_of_shift.keys():
                max_of_shift[shift] += 1
            if shift != ' ':
                if day in instance.days_off[nurse]:
                    assert False, f"Day off constraint violated for nurse {nurse} on day {day}"
                consecutive_shifts += 1
                min_consecutive_shifts += 1
                consecutive_days_off = 0
                total_minutes += instance.shifts[shift]['Length']
                if consecutive_shifts > consecutive_shifts_max:
                    consecutive_shifts_max = consecutive_shifts
                if day != instance.horizon - 1 and result[nurse_int, day + 1] in instance.shifts[shift]['CannotFollow']:
                    assert False, f"Shift {shift} cannot follow {result[nurse_int, day + 1]}"
            else:
                if 0 < min_consecutive_shifts < instance.staff[nurse]['min_consecutive_shifts']:
                    assert False, f"Min consecutive shifts constraint violated for nurse {nurse} {consecutive_shifts} < {instance.staff[nurse]['min_consecutive_shifts']}"
                consecutive_shifts = 0
                min_consecutive_shifts = 0
                consecutive_days_off += 1
                if consecutive_days_off > consecutive_days_off_max:
                    consecutive_days_off_max = consecutive_days_off

            if (day % 7 == 5 or day % 7 == 6) and shift != ' ':
                weekends[day // 7] = True

        assert consecutive_shifts_max <= instance.staff[nurse]['max_consecutive_shifts'], f"Max consecutive shifts constraint violated for nurse {nurse} {consecutive_shifts_max} > {instance.staff[nurse]['max_consecutive_shifts']}"
        assert consecutive_days_off_max >= instance.staff[nurse]['min_consecutive_days_off'], f"Min consecutive days off constraint violated for nurse {nurse} {consecutive_days_off_max} < {instance.staff[nurse]['min_consecutive_days_off']}"
        assert total_minutes <= instance.staff[nurse]['max_total_minutes'], f"Total minutes constraint violated for nurse {nurse}, {total_minutes} > {instance.staff[nurse]['max_total_minutes']}"
        assert total_minutes >= instance.staff[nurse]['min_total_minutes'], f"Total minutes constraint violated for nurse {nurse}, {total_minutes} < {instance.staff[nurse]['min_total_minutes']}"
        assert sum(weekends) <= instance.staff[nurse]['max_weekends'], f"Weekends constraint violated for nurse {nurse}"

        for shift in instance.staff[nurse]['max_of_shift'].keys():
            assert max_of_shift[shift] <= instance.staff[nurse]['max_of_shift'][shift], f"Shift {shift} constraint violated for nurse {nurse}"

    penalty = 0

    # Shift on requests

    for request in instance.shift_on_requests:
        nurse, day, shift, weight = request.get('EmployeeID'), request.get('Day'), request.get(
            'ShiftID'), request.get('Weight')
        nurse_int = staff_keys.index(nurse)
        if result[nurse_int, day] != shift:
            penalty += weight

    # Shift off requests

    for request in instance.shift_off_requests:
        nurse, day, shift, weight = request.get('EmployeeID'), request.get('Day'), request.get(
            'ShiftID'), request.get('Weight')
        nurse_int = staff_keys.index(nurse)
        if result[nurse_int, day] == shift:
            penalty += weight

    # Cover

    for cover in instance.cover_requirements:
        day, shift, preferred, weight_under, weight_over = cover.get('Day'), cover.get('ShiftID'), cover.get(
            'Requirement'), cover.get('WeightForUnder'), cover.get('WeightForOver')
        assigned = np.sum(result[:, day] == shift)
        if assigned < preferred:
            penalty += (preferred - assigned) * weight_under
        elif assigned > preferred:
            penalty += (assigned - preferred) * weight_over

    diff = sol.get_objective_values()[0] - penalty
    assert diff < 1e-6, f"Penalty {penalty} does not match objective value {sol.get_objective_values()[0]}"

    return True
