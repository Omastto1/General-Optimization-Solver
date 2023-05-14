************************************************************************
file with basedata            : mm44_.bas
initial value random generator: 600952212
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  86
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     10      0       14        7       14
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2          10  11
   3        3          1           7
   4        3          2           5   6
   5        3          1           9
   6        3          3           9  10  11
   7        3          2           8  10
   8        3          1           9
   9        3          1          12
  10        3          1          12
  11        3          1          12
  12        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     3       7    7   10    0
         2     4       6    6    0    7
         3     7       6    6    6    0
  3      1     3       7    8    0    8
         2     8       4    7    0    7
         3     9       3    5    0    5
  4      1     1       8    8    0    8
         2     9       7    8    9    0
         3    10       4    7    0    3
  5      1     5       6   10   10    0
         2     7       5    9    0    6
         3     9       2    9    0    5
  6      1     5       8    7    8    0
         2     6       8    5    5    0
         3     7       8    3    0    7
  7      1     7       5   10    9    0
         2     8       4    7    8    0
         3     9       4    3    8    0
  8      1     1       6    7    9    0
         2     6       6    6    5    0
         3    10       6    5    0    6
  9      1     3       9    9    5    0
         2     8       9    9    0    6
         3     8       9    8    4    0
 10      1     4       3    8    0    6
         2     8       2    7    0    5
         3    10       2    5    1    0
 11      1     2       5    9    0    9
         2     5       4    7    3    0
         3     7       1    7    0    7
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   19   21   64   63
************************************************************************
