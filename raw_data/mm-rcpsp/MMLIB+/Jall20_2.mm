jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	4		2 3 4 5 
2	3	5		17 11 10 8 7 
3	3	4		16 10 8 6 
4	3	6		19 17 16 10 9 8 
5	3	6		25 19 16 14 13 10 
6	3	3		15 9 7 
7	3	5		25 19 18 14 12 
8	3	3		25 13 12 
9	3	3		25 20 14 
10	3	2		15 12 
11	3	3		24 21 16 
12	3	3		24 21 20 
13	3	6		37 33 28 24 23 22 
14	3	3		27 24 21 
15	3	3		37 24 21 
16	3	3		30 23 20 
17	3	3		37 24 21 
18	3	5		33 27 24 23 22 
19	3	2		24 21 
20	3	6		37 33 29 28 27 26 
21	3	4		33 29 23 22 
22	3	4		34 32 31 30 
23	3	2		34 26 
24	3	2		29 26 
25	3	5		42 38 37 35 33 
26	3	4		41 38 36 31 
27	3	4		42 40 34 32 
28	3	4		42 38 34 32 
29	3	3		41 38 31 
30	3	4		42 40 38 35 
31	3	4		42 40 39 35 
32	3	3		41 39 35 
33	3	3		41 40 36 
34	3	2		51 36 
35	3	4		50 47 44 43 
36	3	4		50 47 44 43 
37	3	2		50 41 
38	3	1		39 
39	3	5		51 50 47 46 45 
40	3	3		50 44 43 
41	3	3		51 45 43 
42	3	2		46 45 
43	3	1		46 
44	3	1		45 
45	3	1		48 
46	3	1		49 
47	3	1		48 
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
2	1	2	9	4	9	5	
	2	3	8	3	8	4	
	3	10	5	3	8	4	
3	1	3	7	8	6	9	
	2	5	7	5	6	8	
	3	9	7	2	6	5	
4	1	1	3	8	10	7	
	2	5	2	6	8	5	
	3	8	2	5	6	1	
5	1	2	6	6	7	4	
	2	4	4	5	6	3	
	3	6	4	2	5	2	
6	1	1	10	5	7	7	
	2	3	8	5	5	7	
	3	4	7	5	3	3	
7	1	2	3	7	8	7	
	2	6	2	4	7	3	
	3	8	1	2	6	2	
8	1	3	4	6	3	2	
	2	4	3	6	3	2	
	3	6	2	6	2	2	
9	1	3	6	2	8	9	
	2	4	6	2	8	7	
	3	7	5	2	8	6	
10	1	2	9	6	4	8	
	2	8	8	3	4	7	
	3	9	8	3	4	6	
11	1	3	6	9	4	6	
	2	8	4	8	4	5	
	3	10	3	8	1	5	
12	1	1	6	5	10	8	
	2	2	4	3	6	7	
	3	3	2	2	4	7	
13	1	3	2	8	4	7	
	2	4	2	7	4	7	
	3	8	2	7	2	6	
14	1	5	8	9	10	5	
	2	6	8	8	9	3	
	3	7	8	7	9	1	
15	1	2	8	4	1	10	
	2	7	7	3	1	7	
	3	8	6	2	1	7	
16	1	3	6	9	8	5	
	2	8	5	6	6	4	
	3	9	5	4	5	4	
17	1	4	2	7	7	5	
	2	5	2	6	5	5	
	3	6	2	5	3	4	
18	1	3	6	7	4	9	
	2	7	6	3	4	6	
	3	9	6	2	4	2	
19	1	4	7	2	3	4	
	2	5	5	1	3	4	
	3	6	5	1	3	3	
20	1	5	6	7	9	3	
	2	6	5	6	4	3	
	3	8	5	6	4	2	
21	1	7	8	6	6	9	
	2	8	8	6	4	7	
	3	10	8	6	3	6	
22	1	4	6	4	5	7	
	2	5	3	3	5	6	
	3	6	3	3	5	3	
23	1	1	10	10	9	10	
	2	2	9	6	7	7	
	3	8	8	3	7	6	
24	1	6	5	8	9	7	
	2	8	3	8	6	6	
	3	10	3	6	5	4	
25	1	3	6	6	9	9	
	2	7	2	5	7	7	
	3	8	1	2	5	6	
26	1	7	9	9	7	9	
	2	8	7	7	4	9	
	3	10	7	6	1	9	
27	1	8	10	9	1	5	
	2	9	5	9	1	4	
	3	10	4	9	1	3	
28	1	6	7	8	8	7	
	2	7	7	6	5	6	
	3	10	7	4	5	4	
29	1	5	6	5	6	3	
	2	9	4	5	5	3	
	3	10	4	5	5	2	
30	1	2	4	2	8	6	
	2	4	3	2	7	4	
	3	5	3	2	7	3	
31	1	3	9	6	8	7	
	2	7	8	5	6	6	
	3	10	7	3	2	6	
32	1	6	7	4	7	5	
	2	7	6	2	7	4	
	3	9	4	2	6	1	
33	1	1	4	6	4	3	
	2	7	3	6	4	2	
	3	10	3	5	2	2	
34	1	5	7	4	7	8	
	2	6	6	3	6	6	
	3	10	3	3	2	6	
35	1	1	9	7	8	6	
	2	4	5	7	8	3	
	3	5	4	6	8	2	
36	1	3	10	6	8	7	
	2	5	7	4	7	5	
	3	10	6	3	5	3	
37	1	4	4	8	6	5	
	2	6	3	5	5	2	
	3	8	1	4	2	1	
38	1	3	8	10	4	5	
	2	7	6	9	2	2	
	3	9	5	8	1	2	
39	1	1	8	7	6	6	
	2	6	7	6	6	5	
	3	8	6	3	5	4	
40	1	3	5	8	9	8	
	2	5	4	6	9	8	
	3	6	3	6	9	7	
41	1	1	7	9	2	8	
	2	9	6	9	1	7	
	3	10	4	9	1	6	
42	1	7	5	4	7	9	
	2	8	4	3	6	6	
	3	9	4	2	4	4	
43	1	1	9	7	9	2	
	2	4	6	7	8	2	
	3	10	3	5	6	1	
44	1	3	5	9	8	8	
	2	8	4	6	7	8	
	3	9	4	3	4	6	
45	1	3	8	9	1	5	
	2	4	7	8	1	3	
	3	7	5	6	1	2	
46	1	1	3	6	10	5	
	2	8	2	5	10	3	
	3	9	1	2	10	1	
47	1	3	9	8	9	7	
	2	4	5	7	7	3	
	3	5	2	5	7	1	
48	1	2	8	8	2	6	
	2	3	8	8	2	5	
	3	5	8	8	1	4	
49	1	4	5	5	5	3	
	2	5	5	4	3	3	
	3	6	5	4	2	3	
50	1	6	8	9	3	2	
	2	7	7	9	3	1	
	3	10	6	9	3	1	
51	1	1	7	6	8	7	
	2	2	6	5	6	7	
	3	8	4	4	6	7	
52	1	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2
	18	20	268	252

************************************************************************
