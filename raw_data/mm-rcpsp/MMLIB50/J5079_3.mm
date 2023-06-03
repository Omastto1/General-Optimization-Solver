jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	4		2 4 5 7 
2	3	3		9 8 3 
3	3	5		19 13 12 11 10 
4	3	3		13 12 6 
5	3	5		20 19 16 14 13 
6	3	4		20 19 16 10 
7	3	4		19 15 14 11 
8	3	4		19 18 16 14 
9	3	4		18 16 15 14 
10	3	3		18 15 14 
11	3	5		25 20 18 17 16 
12	3	6		25 23 22 21 18 17 
13	3	5		23 22 18 17 15 
14	3	5		25 23 22 21 17 
15	3	6		33 31 29 26 25 24 
16	3	4		29 24 23 21 
17	3	5		34 33 31 29 24 
18	3	5		34 33 31 26 24 
19	3	4		34 31 29 24 
20	3	6		33 32 31 30 28 26 
21	3	5		37 33 32 27 26 
22	3	5		38 34 33 30 27 
23	3	4		34 30 28 26 
24	3	4		32 30 28 27 
25	3	5		40 39 37 34 32 
26	3	4		40 38 36 35 
27	3	5		51 45 42 40 36 
28	3	4		42 40 37 35 
29	3	4		51 38 37 36 
30	3	5		51 45 42 39 37 
31	3	5		51 45 43 42 39 
32	3	5		51 47 45 43 41 
33	3	3		45 43 39 
34	3	1		35 
35	3	5		51 50 47 45 41 
36	3	2		43 39 
37	3	3		47 43 41 
38	3	2		43 42 
39	3	2		47 41 
40	3	2		50 44 
41	3	1		44 
42	3	1		44 
43	3	3		50 49 48 
44	3	1		46 
45	3	1		46 
46	3	2		49 48 
47	3	2		49 48 
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
2	1	1	4	9	10	0	
	2	2	2	3	0	5	
	3	3	2	2	1	0	
3	1	1	8	8	0	5	
	2	8	6	6	3	0	
	3	9	2	3	1	0	
4	1	3	8	7	9	0	
	2	4	7	5	0	4	
	3	8	7	2	9	0	
5	1	2	9	8	8	0	
	2	4	4	4	8	0	
	3	10	4	3	5	0	
6	1	2	6	8	0	7	
	2	3	4	7	6	0	
	3	7	3	7	0	3	
7	1	4	5	7	0	8	
	2	6	3	6	0	5	
	3	10	1	4	3	0	
8	1	1	10	10	0	10	
	2	3	9	9	0	9	
	3	6	9	8	0	9	
9	1	5	10	4	5	0	
	2	8	8	3	4	0	
	3	9	8	3	3	0	
10	1	8	5	7	0	9	
	2	9	4	5	3	0	
	3	9	3	3	0	4	
11	1	1	7	9	6	0	
	2	6	4	4	5	0	
	3	9	3	3	3	0	
12	1	2	10	9	10	0	
	2	3	9	8	0	3	
	3	7	8	8	0	3	
13	1	2	5	5	0	7	
	2	4	5	4	2	0	
	3	7	1	2	1	0	
14	1	2	4	3	10	0	
	2	3	3	3	0	2	
	3	9	3	3	10	0	
15	1	5	10	4	5	0	
	2	9	8	4	0	5	
	3	10	8	4	0	4	
16	1	2	7	5	0	2	
	2	3	7	4	0	2	
	3	6	7	3	3	0	
17	1	1	4	7	10	0	
	2	2	4	6	5	0	
	3	6	2	5	3	0	
18	1	2	5	8	10	0	
	2	6	5	5	8	0	
	3	8	4	4	0	2	
19	1	3	9	8	8	0	
	2	7	4	7	0	6	
	3	10	1	6	0	2	
20	1	2	7	9	0	7	
	2	7	7	9	0	6	
	3	8	6	9	3	0	
21	1	2	9	5	0	7	
	2	3	5	3	2	0	
	3	5	5	3	0	5	
22	1	4	7	9	8	0	
	2	7	5	8	6	0	
	3	9	3	8	5	0	
23	1	4	5	9	6	0	
	2	6	4	6	0	4	
	3	8	3	4	2	0	
24	1	1	9	9	8	0	
	2	4	7	7	8	0	
	3	9	6	6	8	0	
25	1	1	7	7	0	4	
	2	5	6	6	0	4	
	3	6	4	6	4	0	
26	1	3	9	6	0	9	
	2	4	7	3	0	8	
	3	9	7	2	0	6	
27	1	2	5	10	5	0	
	2	3	5	9	3	0	
	3	4	5	9	2	0	
28	1	5	10	6	0	8	
	2	6	9	4	0	8	
	3	7	9	1	0	7	
29	1	3	10	10	9	0	
	2	8	5	8	9	0	
	3	9	3	7	9	0	
30	1	5	7	8	9	0	
	2	7	6	5	0	4	
	3	9	5	3	9	0	
31	1	6	5	6	0	5	
	2	7	4	5	0	4	
	3	9	4	5	0	3	
32	1	1	6	3	8	0	
	2	2	6	3	7	0	
	3	7	4	3	0	2	
33	1	4	8	6	0	6	
	2	5	4	4	0	6	
	3	8	2	4	2	0	
34	1	1	6	7	0	8	
	2	8	4	6	0	5	
	3	9	3	3	0	4	
35	1	1	5	5	0	1	
	2	2	3	4	7	0	
	3	6	3	4	6	0	
36	1	7	8	2	7	0	
	2	8	5	1	0	3	
	3	9	3	1	0	1	
37	1	3	9	7	5	0	
	2	4	7	7	0	3	
	3	8	6	5	0	2	
38	1	1	4	8	9	0	
	2	2	4	5	0	2	
	3	6	3	4	0	2	
39	1	1	8	7	10	0	
	2	6	7	7	9	0	
	3	10	7	7	8	0	
40	1	8	7	7	5	0	
	2	8	3	7	0	6	
	3	10	3	7	5	0	
41	1	3	8	5	5	0	
	2	7	4	5	0	9	
	3	8	1	5	0	9	
42	1	1	10	8	8	0	
	2	6	9	7	5	0	
	3	8	8	7	4	0	
43	1	3	3	9	0	8	
	2	7	3	9	4	0	
	3	10	3	9	0	5	
44	1	4	8	8	0	7	
	2	8	7	7	0	6	
	3	10	6	6	0	3	
45	1	9	8	3	0	4	
	2	9	5	3	3	0	
	3	10	5	2	0	3	
46	1	2	2	5	1	0	
	2	4	2	4	1	0	
	3	4	2	3	0	5	
47	1	7	4	10	5	0	
	2	9	3	9	0	5	
	3	10	3	9	5	0	
48	1	1	6	2	6	0	
	2	4	5	1	3	0	
	3	10	3	1	0	2	
49	1	1	5	8	9	0	
	2	3	5	8	0	3	
	3	5	4	6	7	0	
50	1	1	6	4	2	0	
	2	7	5	4	0	9	
	3	9	5	4	2	0	
51	1	3	8	8	8	0	
	2	4	7	8	4	0	
	3	8	7	8	1	0	
52	1	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2
	24	23	105	74

************************************************************************
