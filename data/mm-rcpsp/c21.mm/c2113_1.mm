************************************************************************
file with basedata            : c2113_.bas
initial value random generator: 29136
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  18
horizon                       :  135
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     16      0       34        5       34
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           8   9  12
   3        3          3           6   7  10
   4        3          2           5  12
   5        3          3           6   7  10
   6        3          2           8   9
   7        3          3           8  13  14
   8        3          2          11  17
   9        3          3          11  13  14
  10        3          3          11  13  14
  11        3          2          15  16
  12        3          2          15  17
  13        3          2          15  16
  14        3          2          16  17
  15        3          1          18
  16        3          1          18
  17        3          1          18
  18        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     3       2   10    7    0
         2     5       2   10    6    0
         3    10       1   10    6    0
  3      1     1       7    8    0   10
         2     3       3    6    0    7
         3     8       1    6    0    6
  4      1     3      10    5    0    1
         2     4       8    4    0    1
         3     7       4    2    0    1
  5      1     6       6   10    0    4
         2     9       1    8    4    0
         3     9       4    5    0    4
  6      1     2       7    8    1    0
         2     6       7    5    0    6
         3    10       7    4    0    5
  7      1     6       9    9    0    9
         2     6      10    8    3    0
         3     9       3    7    0   10
  8      1     5      10    6    9    0
         2     9       9    5    9    0
         3    10       9    3    0    6
  9      1     5       8    8    0    4
         2     6       8    7    5    0
         3     8       7    3    5    0
 10      1     6       7    1    4    0
         2     6       7    1    0    4
         3     8       5    1    4    0
 11      1     7       2   10    0    3
         2     8       2    3    0    3
         3     8       1    3    5    0
 12      1     3       6    8    0    9
         2     7       6    8    0    6
         3     9       5    7    0    6
 13      1     2       7    7    0    8
         2     7       5    7    6    0
         3     8       5    2    0    7
 14      1     2       8    6    7    0
         2     2       6    6   10    0
         3     4       2    5    5    0
 15      1     7       5    7    6    0
         2     8       3    7    6    0
         3    10       1    6    0    7
 16      1     4       3    9    5    0
         2     6       2    7    4    0
         3     9       2    3    0    8
 17      1     1       8    7    0    6
         2     6       8    6    0    4
         3     8       7    4    0    3
 18      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   13   13   38   51
************************************************************************
