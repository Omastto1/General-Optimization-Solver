import json

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


class NRCProblem:
    def __init__(self):
        self.horizon = -1
        self.shifts = []
        self.staff = {}
        self.days_off = {}
        self.shift_on_requests = []
        self.shift_off_requests = []
        self.cover_requirements = []

    def read_elem(self, filename):
        with open(filename) as f:
            return [str(elem) for elem in f.read().split()]

    # The input files follow the NRC format.
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

    def save_to_json(self, path):
        with open(path, 'w') as outfile:
            json.dump(self.__dict__, outfile)

    def load_from_json(self, path):
        with open(path, 'r') as infile:
            self.__dict__ = json.load(infile)


if __name__ == '__main__':
    sections = {}
    current_section = None
    fname = "..\\..\\data\\NRP\\NRC\\Instance6.txt"
    with open(fname, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('SECTION_'):
                current_section = line
                sections[current_section] = []
            elif current_section:
                sections[current_section].append(line)

    parsed_sections = {}
    parsed_sections['HORIZON'] = int(sections['SECTION_HORIZON'][2])

    parsed_sections['SHIFTS'] = parse_shift_section(sections['SECTION_SHIFTS'])
    parsed_sections['STAFF'] = parse_staff(sections['SECTION_STAFF'])
    parsed_sections['DAYS_OFF'] = parse_daysoff_section(sections['SECTION_DAYS_OFF'])
    parsed_sections['SHIFT_ON_REQUESTS'] = parse_requests_section(sections['SECTION_SHIFT_ON_REQUESTS'])
    parsed_sections['SHIFT_OFF_REQUESTS'] = parse_requests_section(sections['SECTION_SHIFT_OFF_REQUESTS'])
    parsed_sections['COVER'] = parse_cover_section(sections['SECTION_COVER'])

    # Print or manipulate the parsed data as needed
    print(parsed_sections)
