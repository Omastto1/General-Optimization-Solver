************************************************************************
file with basedata            : c1530_.bas
initial value random generator: 484099375
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  18
horizon                       :  124
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     16      0       20        3       20
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           7  16
   3        3          2           5  10
   4        3          2           6  11
   5        3          3           8   9  11
   6        3          1          15
   7        3          1           9
   8        3          2          12  13
   9        3          2          12  13
  10        3          1          11
  11        3          1          15
  12        3          1          17
  13        3          2          14  17
  14        3          1          15
  15        3          1          18
  16        3          1          18
  17        3          1          18
  18        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     2       5    7    6    0
         2     6       4    7    5    0
         3     8       3    5    5    0
  3      1     1       4    9    0    7
         2     1       5    9    6    0
         3     7       4    7    3    0
  4      1     2      10    6    0    6
         2     4       9    4    8    0
         3     6       9    3    4    0
  5      1     4       8    9    4    0
         2     5       5    9    0    9
         3     9       4    6    4    0
  6      1     6       4    7    8    0
         2     9       2    5    7    0
         3     9       3    6    0    8
  7      1     6       7    7    5    0
         2     7       4    5    4    0
         3    10       3    5    2    0
  8      1     2       7    7    4    0
         2     5       7    7    0    6
         3     7       3    7    0    3
  9      1     4       5    3    0    9
         2     5       5    3    8    0
         3     8       5    2    8    0
 10      1     2       9    4    0    8
         2     2       9    4    7    0
         3     5       9    4    4    0
 11      1     6       6    9    4    0
         2     9       4    7    0    5
         3     9       5    7    4    0
 12      1     1       4    3    0    6
         2     5       3    3    6    0
         3     8       3    2    4    0
 13      1     6       7    8    0    7
         2     8       4    7    7    0
         3     9       4    6    4    0
 14      1     1      10    3    4    0
         2     3       4    3    3    0
         3     7       3    3    3    0
 15      1     1       7    6    8    0
         2     2       6    5    5    0
         3     6       4    3    0    8
 16      1     5       4   10    0   10
         2     5       6   10    4    0
         3     8       4   10    0    9
 17      1     1       5    5    9    0
         2     3       4    4    8    0
         3     8       2    3    6    0
 18      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   22   25   98   89
************************************************************************
