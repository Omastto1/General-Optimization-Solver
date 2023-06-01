jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	10		2 3 4 5 7 8 10 12 13 16 
2	3	7		27 25 24 22 19 11 9 
3	3	7		30 27 25 23 19 17 6 
4	3	8		27 24 23 22 19 18 15 14 
5	3	8		31 27 26 25 23 20 18 15 
6	3	6		26 24 22 20 18 15 
7	3	6		32 27 26 24 19 17 
8	3	7		35 33 31 30 26 22 21 
9	3	4		31 23 18 15 
10	3	6		32 31 26 24 19 18 
11	3	6		39 35 32 26 21 20 
12	3	7		39 38 35 34 33 29 22 
13	3	6		39 35 34 27 24 22 
14	3	6		51 39 34 32 31 20 
15	3	5		51 36 35 32 21 
16	3	5		38 35 34 26 22 
17	3	3		36 35 21 
18	3	11		49 48 47 40 39 38 37 36 35 34 33 
19	3	8		49 48 39 37 36 34 33 29 
20	3	8		50 49 48 38 37 36 29 28 
21	3	7		49 48 38 37 34 29 28 
22	3	6		51 48 37 36 32 28 
23	3	6		51 50 49 48 44 28 
24	3	7		49 48 46 40 38 37 36 
25	3	7		51 47 46 42 40 37 36 
26	3	8		51 48 46 45 44 43 42 40 
27	3	6		51 47 43 42 40 36 
28	3	6		47 46 45 43 42 40 
29	3	6		47 46 45 43 42 40 
30	3	5		46 44 42 40 39 
31	3	5		46 45 43 42 40 
32	3	5		49 45 44 42 41 
33	3	4		46 45 42 41 
34	3	3		50 44 43 
35	3	3		44 42 41 
36	3	2		44 41 
37	3	2		45 43 
38	3	2		44 42 
39	3	2		43 41 
40	3	1		41 
41	3	1		52 
42	3	1		52 
43	3	1		52 
44	3	1		52 
45	3	1		52 
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
2	1	4	5	0	6	0	
	2	6	4	0	0	6	
	3	10	3	0	3	0	
3	1	1	0	7	0	5	
	2	3	5	0	0	4	
	3	5	0	6	6	0	
4	1	6	6	0	0	6	
	2	7	3	0	0	6	
	3	10	3	0	0	3	
5	1	1	0	5	7	0	
	2	4	8	0	5	0	
	3	10	6	0	0	3	
6	1	1	8	0	6	0	
	2	3	7	0	6	0	
	3	9	7	0	0	5	
7	1	4	7	0	0	6	
	2	5	0	5	2	0	
	3	6	0	5	1	0	
8	1	2	0	7	1	0	
	2	7	2	0	1	0	
	3	10	1	0	0	5	
9	1	4	0	5	0	9	
	2	5	2	0	0	7	
	3	7	0	3	0	4	
10	1	2	0	6	0	10	
	2	2	0	5	3	0	
	3	5	0	1	2	0	
11	1	4	6	0	0	4	
	2	5	6	0	0	2	
	3	8	6	0	7	0	
12	1	4	8	0	0	9	
	2	6	7	0	2	0	
	3	8	0	2	0	8	
13	1	1	0	7	0	3	
	2	5	0	4	0	2	
	3	6	0	4	0	1	
14	1	4	5	0	0	6	
	2	7	5	0	0	5	
	3	7	0	1	5	0	
15	1	5	6	0	0	10	
	2	5	4	0	6	0	
	3	9	4	0	3	0	
16	1	2	0	8	0	9	
	2	6	0	6	0	7	
	3	7	0	5	0	5	
17	1	1	9	0	7	0	
	2	3	6	0	4	0	
	3	6	0	5	3	0	
18	1	2	0	8	0	10	
	2	3	0	5	0	4	
	3	3	9	0	0	2	
19	1	2	0	5	0	5	
	2	7	7	0	0	5	
	3	10	7	0	0	3	
20	1	4	0	8	6	0	
	2	6	5	0	0	7	
	3	6	0	6	2	0	
21	1	3	0	8	0	7	
	2	9	0	3	0	7	
	3	10	0	3	0	6	
22	1	6	0	6	5	0	
	2	6	6	0	0	2	
	3	7	5	0	0	1	
23	1	3	10	0	0	7	
	2	3	9	0	1	0	
	3	4	0	2	1	0	
24	1	1	0	8	0	6	
	2	5	8	0	0	3	
	3	7	0	7	0	2	
25	1	5	8	0	8	0	
	2	9	6	0	7	0	
	3	9	0	7	7	0	
26	1	7	0	4	0	7	
	2	8	0	3	4	0	
	3	8	0	1	0	6	
27	1	3	0	7	7	0	
	2	4	5	0	0	7	
	3	5	0	7	0	4	
28	1	2	2	0	7	0	
	2	6	2	0	4	0	
	3	7	0	7	2	0	
29	1	3	0	7	0	10	
	2	4	0	5	1	0	
	3	7	6	0	0	9	
30	1	5	7	0	0	7	
	2	6	6	0	7	0	
	3	6	0	3	0	3	
31	1	6	0	6	7	0	
	2	7	0	6	0	9	
	3	8	6	0	6	0	
32	1	3	0	6	8	0	
	2	3	0	4	0	6	
	3	5	4	0	0	4	
33	1	1	2	0	9	0	
	2	7	0	1	6	0	
	3	10	0	1	5	0	
34	1	3	0	5	0	4	
	2	5	0	4	4	0	
	3	6	5	0	1	0	
35	1	1	5	0	7	0	
	2	2	3	0	5	0	
	3	7	0	7	2	0	
36	1	2	0	3	7	0	
	2	5	5	0	0	5	
	3	8	0	2	3	0	
37	1	1	0	1	10	0	
	2	9	0	1	0	5	
	3	10	5	0	3	0	
38	1	2	0	6	6	0	
	2	8	5	0	0	4	
	3	9	4	0	2	0	
39	1	2	2	0	10	0	
	2	3	0	3	9	0	
	3	7	0	2	9	0	
40	1	8	0	7	7	0	
	2	10	6	0	0	6	
	3	10	5	0	7	0	
41	1	2	0	8	0	6	
	2	4	2	0	6	0	
	3	7	0	4	6	0	
42	1	1	0	5	0	8	
	2	9	8	0	8	0	
	3	9	0	3	0	5	
43	1	3	0	1	8	0	
	2	4	0	1	7	0	
	3	6	0	1	0	8	
44	1	1	9	0	0	7	
	2	4	7	0	0	6	
	3	7	0	7	0	6	
45	1	1	4	0	0	9	
	2	7	0	8	0	7	
	3	8	1	0	4	0	
46	1	1	0	7	8	0	
	2	3	3	0	0	6	
	3	10	0	5	7	0	
47	1	3	0	7	3	0	
	2	3	2	0	0	6	
	3	4	2	0	0	5	
48	1	1	8	0	0	2	
	2	1	7	0	9	0	
	3	3	6	0	9	0	
49	1	3	0	6	3	0	
	2	4	5	0	2	0	
	3	10	0	3	0	4	
50	1	4	0	5	9	0	
	2	8	2	0	0	4	
	3	9	0	4	0	4	
51	1	1	0	5	6	0	
	2	6	0	3	0	4	
	3	9	0	2	0	3	
52	1	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2
	23	18	134	153

************************************************************************
