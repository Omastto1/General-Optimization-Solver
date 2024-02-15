{
    'no_projects' (int) : 
    'number_of_jobs' (int) : number of jobs including supersouce and sink
    'resources' (dict) : 
    {
        "renewable_resources" | "non_renewable_resources" | "doubly_constrained_resources" (dict) : 
        {
            "number_of_resources" (int) : 
            "availabilities" (dict) : {"resource name" (str): "resurce availability" (int)}
        }
    }
    'project_information' (dict) : project specifications
    {
        'pronr' (int): project number
        'no_jobs' (int) : number of jobs
        'rel_data' (int) : release date
        'duedate' (int) :
        'tardcost' (int) :
        'mpm_time' (int) :
    }
    'job_specification' (list[dict]):
    [
        {
            'jobnr' (int) : job number
            'no_modes' (int): number of modes
            'modes' (dict):
            {
                'mode' (int) : mode number
                'duration' (int) : duration
                'request_durations' (dict) : so far not divided to renewable / non-renewable / doubly constrained
            }
            'no_successors' (int) : number of successors
            'successors' (list[int]) : list of successor job numbers
        }
    ]
    'resource_availabilities' (dict) : 
    {
        # so far not divided to renewable / non-renewable / doubly constrained
        "renewable_resources" | "non_renewable_resources" | "doubly_constrained_resources" (dict) : 
        {
            "resource name" (str): "resurce availability" (int)
        }
    }
}