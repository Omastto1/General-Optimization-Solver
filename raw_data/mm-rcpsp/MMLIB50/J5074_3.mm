jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	3		2 3 8 
2	3	3		6 5 4 
3	3	6		16 12 11 10 7 6 
4	3	8		19 16 15 14 13 11 10 9 
5	3	5		13 12 11 9 7 
6	3	3		19 13 9 
7	3	4		18 17 15 14 
8	3	3		23 19 13 
9	3	5		29 26 23 21 18 
10	3	5		29 27 26 21 17 
11	3	4		27 21 20 17 
12	3	5		30 29 24 23 19 
13	3	3		27 20 17 
14	3	5		33 29 27 26 22 
15	3	5		30 26 24 23 21 
16	3	4		30 25 24 21 
17	3	3		32 24 22 
18	3	3		33 27 22 
19	3	3		26 25 21 
20	3	4		33 32 26 25 
21	3	5		35 33 32 31 28 
22	3	3		30 28 25 
23	3	2		33 25 
24	3	5		42 41 35 33 31 
25	3	4		42 41 35 31 
26	3	4		42 41 35 31 
27	3	5		42 41 38 35 32 
28	3	3		42 41 34 
29	3	2		38 32 
30	3	4		39 38 37 36 
31	3	2		38 34 
32	3	3		39 37 36 
33	3	3		40 39 38 
34	3	2		39 36 
35	3	3		46 39 37 
36	3	4		51 46 44 40 
37	3	5		51 50 48 44 43 
38	3	3		51 46 44 
39	3	3		51 50 45 
40	3	2		48 43 
41	3	2		50 43 
42	3	2		50 47 
43	3	1		45 
44	3	1		45 
45	3	1		47 
46	3	1		48 
47	3	1		49 
48	3	1		49 
49	3	1		52 
50	3	1		52 
51	3	1		52 
52	1	0		
************************************************************************
REQUESTS/DURATIONS
jobnr.	mode	dur	R1	R2	N1	N2	
------------------------------------------------------------------------
1	1	0	0	0	0	0	
2	1	1	4	0	8	6	
	2	4	0	5	5	5	
	3	9	0	4	4	2	
3	1	5	0	7	9	5	
	2	7	0	5	8	4	
	3	9	6	0	7	3	
4	1	3	9	0	3	8	
	2	4	0	4	3	8	
	3	5	5	0	3	8	
5	1	2	0	7	7	10	
	2	2	9	0	5	9	
	3	7	0	4	5	9	
6	1	3	8	0	7	10	
	2	5	0	8	4	9	
	3	6	0	6	3	8	
7	1	3	9	0	8	6	
	2	5	7	0	5	5	
	3	9	1	0	4	2	
8	1	5	0	6	5	9	
	2	8	0	6	5	7	
	3	10	0	2	4	6	
9	1	2	6	0	8	6	
	2	5	0	5	8	5	
	3	9	3	0	7	5	
10	1	3	6	0	3	6	
	2	4	6	0	3	3	
	3	9	6	0	3	2	
11	1	4	0	7	7	3	
	2	7	0	5	6	3	
	3	9	2	0	6	1	
12	1	1	7	0	6	1	
	2	8	7	0	4	1	
	3	9	0	7	1	1	
13	1	2	1	0	6	9	
	2	6	1	0	5	8	
	3	10	0	8	5	8	
14	1	9	9	0	7	7	
	2	9	0	7	6	5	
	3	10	0	6	5	4	
15	1	5	9	0	7	4	
	2	9	0	2	6	4	
	3	10	0	2	5	2	
16	1	5	0	10	7	9	
	2	7	0	10	6	9	
	3	8	0	10	5	9	
17	1	4	8	0	9	7	
	2	6	8	0	8	7	
	3	10	8	0	7	7	
18	1	1	5	0	9	5	
	2	2	4	0	8	5	
	3	6	4	0	8	3	
19	1	7	0	6	5	2	
	2	10	0	5	4	2	
	3	10	4	0	2	2	
20	1	3	0	8	9	7	
	2	6	9	0	9	6	
	3	7	9	0	9	5	
21	1	1	8	0	5	7	
	2	1	0	6	5	6	
	3	7	3	0	4	4	
22	1	2	10	0	8	6	
	2	3	7	0	7	5	
	3	4	0	3	7	5	
23	1	3	0	9	3	7	
	2	4	0	5	1	6	
	3	6	3	0	1	5	
24	1	2	0	9	5	8	
	2	4	0	8	5	7	
	3	5	0	7	5	7	
25	1	1	10	0	7	3	
	2	2	5	0	7	3	
	3	5	0	4	3	2	
26	1	4	0	9	9	6	
	2	8	5	0	7	5	
	3	10	0	6	7	2	
27	1	6	6	0	5	10	
	2	6	0	9	3	10	
	3	9	4	0	3	10	
28	1	4	4	0	3	9	
	2	8	3	0	3	8	
	3	9	3	0	1	8	
29	1	5	1	0	3	4	
	2	7	1	0	2	4	
	3	8	0	4	1	2	
30	1	5	0	8	8	7	
	2	9	5	0	5	7	
	3	9	0	3	2	7	
31	1	1	0	8	4	5	
	2	1	5	0	4	3	
	3	9	0	3	4	2	
32	1	2	0	8	2	7	
	2	6	3	0	1	5	
	3	9	0	3	1	3	
33	1	2	0	2	7	5	
	2	3	0	2	4	5	
	3	10	2	0	4	2	
34	1	4	0	8	4	5	
	2	4	5	0	4	5	
	3	5	4	0	4	2	
35	1	2	9	0	4	8	
	2	7	5	0	2	6	
	3	8	0	6	2	5	
36	1	2	0	7	6	6	
	2	4	4	0	5	5	
	3	5	1	0	3	4	
37	1	2	7	0	6	6	
	2	6	0	7	4	6	
	3	7	0	6	2	6	
38	1	1	0	6	7	9	
	2	9	0	6	5	9	
	3	10	0	4	1	9	
39	1	3	0	2	6	6	
	2	8	0	2	5	4	
	3	10	5	0	5	3	
40	1	1	8	0	3	6	
	2	7	6	0	3	5	
	3	8	5	0	3	3	
41	1	2	10	0	7	3	
	2	5	0	5	6	3	
	3	10	4	0	6	2	
42	1	3	5	0	5	7	
	2	8	0	2	4	6	
	3	8	1	0	1	3	
43	1	2	10	0	10	9	
	2	5	0	4	7	7	
	3	9	0	3	4	6	
44	1	2	9	0	5	8	
	2	7	0	8	4	8	
	3	9	7	0	2	8	
45	1	4	10	0	8	7	
	2	5	0	8	8	6	
	3	10	0	8	5	4	
46	1	7	0	8	8	8	
	2	8	0	6	7	8	
	3	9	6	0	7	6	
47	1	7	2	0	10	8	
	2	9	1	0	5	7	
	3	10	1	0	1	7	
48	1	4	0	6	9	7	
	2	7	4	0	8	4	
	3	8	0	5	7	2	
49	1	1	4	0	10	10	
	2	2	3	0	10	9	
	3	6	0	3	10	8	
50	1	4	0	2	8	8	
	2	8	1	0	8	4	
	3	9	1	0	6	2	
51	1	3	0	7	7	7	
	2	7	0	4	5	7	
	3	8	0	4	4	5	
52	1	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2
	17	19	237	257

************************************************************************
