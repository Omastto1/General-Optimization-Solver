def convert_time_to_seconds(results):
    """accepts list of triples, the third one being string with a time spent on a task when the better solution was found
    converts the time to seconds and replaces the string with the number of seconds
    """

    for i in range(len(results)):
        unit = results[i][2][-1]  # Get the last character of the time
        if unit not in ['s', 'm', 'h']:  # If the unit is not minutes or hours
            print("Error: Unknown unit", unit)
        time = float(results[i][2][:-1])  # Get the time without the last character
        if unit == 'm':  # If the unit is minutes
            results[i][2] = time * 60  # Convert minutes to seconds
        elif unit == 'h':  # If the unit is hours
            results[i][2] = time * 3600  # Convert hours to seconds
        else:
            results[i][2] = time  # Otherwise, the unit is seconds
        
    return results