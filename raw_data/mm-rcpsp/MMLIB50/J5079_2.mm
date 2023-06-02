jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	5		2 3 4 5 8 
2	3	5		12 11 10 9 6 
3	3	3		12 9 7 
4	3	5		27 23 14 13 12 
5	3	5		27 23 14 13 12 
6	3	7		27 23 20 16 15 14 13 
7	3	6		27 20 18 15 14 10 
8	3	5		27 20 18 15 10 
9	3	4		23 20 16 15 
10	3	5		23 21 19 17 16 
11	3	4		23 20 17 15 
12	3	3		20 16 15 
13	3	4		22 21 18 17 
14	3	4		28 26 22 19 
15	3	3		26 21 19 
16	3	5		37 28 25 24 22 
17	3	5		37 28 26 25 24 
18	3	5		37 33 31 25 24 
19	3	5		37 33 32 31 24 
20	3	4		33 31 25 24 
21	3	4		36 33 31 25 
22	3	6		36 33 32 31 30 29 
23	3	2		31 24 
24	3	3		36 30 29 
25	3	3		32 30 29 
26	3	3		36 32 30 
27	3	3		36 32 30 
28	3	3		36 34 31 
29	3	4		51 40 38 34 
30	3	2		35 34 
31	3	4		45 44 38 35 
32	3	3		51 38 34 
33	3	5		51 45 44 40 39 
34	3	5		45 44 43 41 39 
35	3	4		51 50 43 40 
36	3	4		51 49 48 41 
37	3	4		50 49 48 41 
38	3	3		48 43 42 
39	3	5		50 49 48 47 46 
40	3	2		48 41 
41	3	1		42 
42	3	2		47 46 
43	3	2		49 46 
44	3	1		50 
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
2	1	1	6	10	6	0	
	2	3	4	6	0	3	
	3	8	3	6	5	0	
3	1	4	9	5	0	9	
	2	6	6	3	2	0	
	3	10	4	2	2	0	
4	1	4	7	1	0	2	
	2	5	5	1	0	1	
	3	10	3	1	0	1	
5	1	1	10	8	0	4	
	2	2	7	7	8	0	
	3	4	6	6	0	4	
6	1	2	7	9	5	0	
	2	3	6	9	0	7	
	3	7	6	9	4	0	
7	1	2	7	8	0	3	
	2	5	7	7	0	2	
	3	9	7	7	7	0	
8	1	5	2	8	9	0	
	2	8	2	8	4	0	
	3	9	2	8	2	0	
9	1	3	10	10	0	6	
	2	6	6	7	4	0	
	3	8	2	7	4	0	
10	1	5	3	4	0	3	
	2	7	3	3	4	0	
	3	10	3	3	0	1	
11	1	6	9	3	7	0	
	2	7	9	3	0	4	
	3	9	9	3	1	0	
12	1	1	5	9	5	0	
	2	5	3	9	4	0	
	3	6	3	8	4	0	
13	1	1	3	5	5	0	
	2	7	2	3	3	0	
	3	10	2	3	2	0	
14	1	2	4	7	7	0	
	2	8	4	7	0	9	
	3	10	3	6	2	0	
15	1	1	9	7	2	0	
	2	4	7	6	0	7	
	3	6	6	6	1	0	
16	1	1	9	6	4	0	
	2	3	4	3	3	0	
	3	8	1	2	0	2	
17	1	5	6	6	0	7	
	2	7	5	4	4	0	
	3	10	3	3	1	0	
18	1	1	6	5	0	4	
	2	2	5	3	4	0	
	3	8	5	3	1	0	
19	1	3	8	6	0	6	
	2	4	8	4	0	5	
	3	9	6	4	0	4	
20	1	3	5	7	0	5	
	2	9	3	4	0	4	
	3	10	1	3	0	4	
21	1	5	9	6	0	6	
	2	6	7	4	6	0	
	3	9	5	2	0	4	
22	1	3	8	3	0	6	
	2	9	4	3	0	5	
	3	9	1	3	3	0	
23	1	6	7	6	8	0	
	2	7	5	6	0	3	
	3	9	2	5	1	0	
24	1	1	8	8	0	8	
	2	6	7	7	9	0	
	3	8	4	5	0	1	
25	1	2	8	5	0	9	
	2	6	6	5	5	0	
	3	10	6	5	2	0	
26	1	1	7	6	8	0	
	2	2	7	4	6	0	
	3	8	6	4	6	0	
27	1	1	3	5	3	0	
	2	2	2	3	2	0	
	3	10	2	3	0	1	
28	1	1	5	9	9	0	
	2	2	5	9	5	0	
	3	3	3	8	0	6	
29	1	5	6	3	5	0	
	2	8	4	3	0	7	
	3	10	4	3	0	6	
30	1	5	5	6	6	0	
	2	6	5	5	3	0	
	3	8	4	4	0	3	
31	1	2	7	6	0	10	
	2	9	6	4	1	0	
	3	9	3	4	0	4	
32	1	1	4	9	8	0	
	2	9	4	9	7	0	
	3	10	2	9	7	0	
33	1	1	3	5	0	8	
	2	9	2	4	0	6	
	3	10	2	2	0	6	
34	1	2	2	7	7	0	
	2	3	1	4	7	0	
	3	10	1	2	0	2	
35	1	2	9	2	0	8	
	2	5	7	2	4	0	
	3	6	5	2	3	0	
36	1	1	8	9	0	8	
	2	4	4	4	0	6	
	3	6	4	3	0	5	
37	1	3	8	5	4	0	
	2	6	4	5	4	0	
	3	10	4	3	4	0	
38	1	1	10	8	5	0	
	2	5	9	7	0	4	
	3	6	9	6	0	2	
39	1	2	8	8	0	1	
	2	2	6	6	7	0	
	3	9	6	4	6	0	
40	1	5	8	9	4	0	
	2	6	7	8	5	0	
	3	7	7	8	4	0	
41	1	3	8	9	0	8	
	2	4	8	6	0	8	
	3	6	8	3	3	0	
42	1	4	5	10	8	0	
	2	6	5	9	6	0	
	3	7	5	9	5	0	
43	1	5	3	2	0	8	
	2	6	2	2	0	5	
	3	7	2	2	0	2	
44	1	1	4	8	0	3	
	2	5	2	8	3	0	
	3	8	1	8	0	2	
45	1	5	5	3	4	0	
	2	10	3	2	0	2	
	3	10	2	1	2	0	
46	1	4	7	7	7	0	
	2	6	6	7	0	8	
	3	8	3	6	3	0	
47	1	2	5	2	0	7	
	2	5	5	2	0	4	
	3	8	5	2	2	0	
48	1	5	7	7	0	6	
	2	6	7	7	0	3	
	3	10	6	5	6	0	
49	1	2	4	3	0	1	
	2	5	3	3	0	1	
	3	10	2	3	4	0	
50	1	4	8	7	7	0	
	2	6	6	6	5	0	
	3	7	4	6	3	0	
51	1	4	6	4	0	4	
	2	4	4	4	6	0	
	3	8	2	4	0	3	
52	1	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2
	17	18	85	70

************************************************************************
