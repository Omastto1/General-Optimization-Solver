jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	5		2 3 4 7 9 
2	3	6		16 13 12 11 8 6 
3	3	4		13 12 11 5 
4	3	4		14 12 11 6 
5	3	3		22 16 10 
6	3	5		22 19 18 17 15 
7	3	3		18 13 11 
8	3	4		22 20 18 14 
9	3	4		22 20 18 13 
10	3	3		18 17 15 
11	3	4		32 27 20 17 
12	3	3		27 19 17 
13	3	5		27 25 23 21 19 
14	3	2		32 17 
15	3	4		28 23 21 20 
16	3	4		27 25 24 19 
17	3	4		28 25 23 21 
18	3	5		32 29 27 26 24 
19	3	4		32 29 28 26 
20	3	3		29 25 24 
21	3	2		26 24 
22	3	2		29 24 
23	3	3		33 30 29 
24	3	3		34 33 30 
25	3	3		34 33 31 
26	3	2		33 30 
27	3	2		31 28 
28	3	6		45 41 40 38 37 36 
29	3	3		45 40 34 
30	3	2		40 31 
31	3	4		45 38 37 36 
32	3	4		45 40 37 36 
33	3	2		37 35 
34	3	3		38 37 36 
35	3	6		51 48 45 44 41 40 
36	3	6		51 49 48 44 43 39 
37	3	4		50 48 47 42 
38	3	3		49 48 47 
39	3	2		50 42 
40	3	2		49 42 
41	3	2		46 43 
42	3	1		46 
43	3	1		47 
44	3	1		46 
45	3	1		46 
46	3	1		52 
47	3	1		52 
48	3	1		52 
49	3	1		52 
50	3	1		52 
51	3	1		52 
52	1	0		
************************************************************************
REQUESTS/DURATIONS
jobnr.	mode	dur	R1	R2	N1	N2	
------------------------------------------------------------------------
1	1	0	0	0	0	0	
2	1	6	10	7	6	0	
	2	7	9	4	4	0	
	3	9	9	2	0	8	
3	1	1	9	5	6	0	
	2	1	7	3	0	6	
	3	7	7	2	4	0	
4	1	9	7	9	4	0	
	2	10	7	9	3	0	
	3	10	7	8	0	5	
5	1	2	3	5	7	0	
	2	6	2	5	0	2	
	3	7	2	2	0	2	
6	1	2	3	1	0	7	
	2	3	2	1	7	0	
	3	10	2	1	0	5	
7	1	1	8	7	5	0	
	2	3	7	5	3	0	
	3	8	6	5	3	0	
8	1	4	9	4	0	4	
	2	9	8	3	7	0	
	3	10	7	3	4	0	
9	1	5	9	6	9	0	
	2	6	9	2	7	0	
	3	7	9	1	7	0	
10	1	5	6	10	0	4	
	2	7	5	7	0	4	
	3	8	3	6	2	0	
11	1	2	1	7	0	8	
	2	9	1	7	8	0	
	3	10	1	5	0	3	
12	1	5	4	10	0	4	
	2	7	3	8	0	3	
	3	10	2	5	0	3	
13	1	3	7	9	0	8	
	2	7	6	8	0	6	
	3	8	4	8	0	4	
14	1	8	5	5	9	0	
	2	8	4	3	0	5	
	3	10	3	2	0	5	
15	1	1	5	5	9	0	
	2	4	4	3	7	0	
	3	9	2	2	5	0	
16	1	5	10	8	9	0	
	2	8	6	6	0	3	
	3	9	3	6	0	3	
17	1	4	7	7	5	0	
	2	5	5	7	4	0	
	3	10	4	7	3	0	
18	1	6	6	10	0	9	
	2	7	4	9	0	7	
	3	8	3	8	0	3	
19	1	2	10	6	8	0	
	2	4	6	6	0	8	
	3	8	5	5	3	0	
20	1	2	5	3	0	6	
	2	8	3	2	0	6	
	3	10	3	2	0	5	
21	1	3	2	4	0	2	
	2	6	1	4	0	2	
	3	8	1	1	0	2	
22	1	2	4	8	5	0	
	2	4	3	7	0	7	
	3	5	3	7	0	6	
23	1	2	6	8	7	0	
	2	3	5	7	6	0	
	3	4	5	7	0	4	
24	1	2	6	7	0	8	
	2	5	5	6	6	0	
	3	6	5	5	6	0	
25	1	1	9	6	10	0	
	2	6	7	6	10	0	
	3	8	6	1	10	0	
26	1	3	7	5	0	4	
	2	5	5	5	0	3	
	3	5	5	5	3	0	
27	1	3	5	5	7	0	
	2	7	4	4	0	6	
	3	7	4	4	5	0	
28	1	2	2	3	7	0	
	2	7	1	3	6	0	
	3	9	1	1	5	0	
29	1	3	6	6	0	6	
	2	4	3	5	6	0	
	3	8	1	5	0	2	
30	1	3	6	3	0	7	
	2	6	4	3	5	0	
	3	8	4	3	0	6	
31	1	2	4	2	0	7	
	2	6	3	2	5	0	
	3	10	2	2	3	0	
32	1	1	6	8	0	8	
	2	2	6	8	5	0	
	3	6	5	6	0	6	
33	1	2	5	6	0	8	
	2	3	4	5	0	8	
	3	6	4	2	0	8	
34	1	1	7	7	0	8	
	2	4	4	5	0	7	
	3	9	1	3	3	0	
35	1	1	4	9	5	0	
	2	4	4	6	2	0	
	3	6	3	6	0	5	
36	1	2	5	7	7	0	
	2	5	3	6	5	0	
	3	10	2	6	0	2	
37	1	6	6	6	10	0	
	2	7	6	4	7	0	
	3	10	6	3	7	0	
38	1	5	9	7	0	10	
	2	6	9	6	0	9	
	3	7	9	3	0	9	
39	1	4	10	10	0	4	
	2	7	4	9	0	3	
	3	7	1	7	4	0	
40	1	5	7	7	6	0	
	2	9	7	6	0	2	
	3	10	7	6	1	0	
41	1	3	7	7	0	5	
	2	6	5	7	5	0	
	3	8	3	7	5	0	
42	1	5	10	8	0	7	
	2	8	8	8	7	0	
	3	10	7	8	0	1	
43	1	3	6	3	8	0	
	2	7	6	3	0	8	
	3	10	5	3	3	0	
44	1	2	8	3	0	3	
	2	4	6	2	0	3	
	3	5	6	2	0	2	
45	1	2	6	10	0	7	
	2	6	6	6	0	4	
	3	9	6	5	0	3	
46	1	2	6	1	8	0	
	2	8	5	1	0	7	
	3	9	3	1	0	7	
47	1	1	10	9	3	0	
	2	2	5	7	0	3	
	3	4	5	5	3	0	
48	1	4	10	3	6	0	
	2	7	8	2	0	5	
	3	10	6	1	2	0	
49	1	4	7	8	10	0	
	2	4	5	6	0	4	
	3	10	4	5	0	4	
50	1	1	8	7	0	5	
	2	3	8	6	0	3	
	3	5	8	3	1	0	
51	1	2	8	10	0	10	
	2	3	3	7	7	0	
	3	6	3	7	6	0	
52	1	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2
	32	28	202	196

************************************************************************
