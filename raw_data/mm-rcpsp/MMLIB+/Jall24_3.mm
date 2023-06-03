jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	5		2 3 4 5 8 
2	3	4		16 9 7 6 
3	3	3		16 7 6 
4	3	2		9 6 
5	3	2		19 6 
6	3	3		17 14 11 
7	3	2		12 10 
8	3	4		19 17 14 13 
9	3	1		10 
10	3	5		20 19 18 17 15 
11	3	4		26 20 18 12 
12	3	1		13 
13	3	2		21 15 
14	3	2		20 15 
15	3	5		28 27 25 23 22 
16	3	5		28 26 25 24 20 
17	3	4		31 26 22 21 
18	3	4		35 33 28 25 
19	3	3		31 29 24 
20	3	6		38 33 32 31 30 27 
21	3	5		33 32 30 28 27 
22	3	2		29 24 
23	3	5		38 33 31 30 29 
24	3	5		38 35 33 32 30 
25	3	4		38 32 30 29 
26	3	3		38 36 30 
27	3	3		36 35 34 
28	3	5		47 45 38 37 34 
29	3	6		46 45 44 42 40 36 
30	3	3		45 37 34 
31	3	7		47 46 45 44 42 41 40 
32	3	5		46 45 41 39 37 
33	3	6		50 47 46 45 44 40 
34	3	5		46 44 42 41 40 
35	3	5		45 44 42 41 40 
36	3	4		51 47 41 39 
37	3	3		49 44 40 
38	3	3		51 44 43 
39	3	3		50 49 48 
40	3	2		51 43 
41	3	2		50 48 
42	3	2		50 49 
43	3	1		48 
44	3	1		48 
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
2	1	1	6	6	7	8	
	2	5	5	3	7	5	
	3	8	4	2	7	5	
3	1	7	5	3	10	4	
	2	8	4	3	5	3	
	3	10	3	2	3	2	
4	1	4	6	3	7	8	
	2	5	4	3	6	5	
	3	10	2	3	5	5	
5	1	1	5	6	4	5	
	2	4	4	6	4	5	
	3	7	3	5	3	4	
6	1	1	10	6	8	6	
	2	5	8	4	6	3	
	3	9	6	4	6	2	
7	1	2	7	5	8	7	
	2	7	7	3	8	5	
	3	8	7	3	8	3	
8	1	8	4	5	9	5	
	2	9	3	5	9	4	
	3	10	3	4	9	2	
9	1	1	5	7	5	9	
	2	7	5	4	4	7	
	3	10	5	4	4	6	
10	1	7	4	7	3	5	
	2	8	4	7	2	4	
	3	10	2	6	2	3	
11	1	5	4	4	7	7	
	2	6	3	3	7	6	
	3	9	2	3	7	5	
12	1	2	4	5	6	6	
	2	4	4	3	6	6	
	3	5	2	2	6	6	
13	1	7	5	2	5	6	
	2	9	4	2	4	5	
	3	10	3	2	4	5	
14	1	4	9	9	7	4	
	2	8	9	7	5	2	
	3	10	9	6	2	2	
15	1	2	10	10	7	4	
	2	4	6	10	6	4	
	3	6	4	10	4	4	
16	1	7	5	4	10	6	
	2	8	3	2	4	5	
	3	9	1	2	3	5	
17	1	7	5	6	9	5	
	2	8	3	5	7	5	
	3	9	2	5	6	5	
18	1	1	2	8	3	10	
	2	3	1	5	3	5	
	3	6	1	2	3	4	
19	1	4	2	3	3	6	
	2	5	2	2	3	6	
	3	9	2	2	3	5	
20	1	8	5	5	9	9	
	2	9	4	4	9	7	
	3	10	3	4	9	5	
21	1	3	5	4	6	4	
	2	7	3	4	6	3	
	3	9	1	2	4	3	
22	1	3	3	6	5	9	
	2	4	2	6	3	8	
	3	10	2	3	2	7	
23	1	6	8	10	3	6	
	2	7	6	6	3	5	
	3	10	5	4	1	4	
24	1	4	10	6	8	9	
	2	5	9	6	7	6	
	3	10	8	5	7	1	
25	1	3	8	10	5	6	
	2	6	6	9	5	6	
	3	7	6	9	5	5	
26	1	5	8	8	3	1	
	2	9	7	5	3	1	
	3	10	7	4	3	1	
27	1	1	5	6	6	7	
	2	5	5	6	4	4	
	3	8	4	6	3	2	
28	1	2	9	8	7	8	
	2	7	7	7	6	5	
	3	9	7	6	4	4	
29	1	2	9	7	10	7	
	2	5	7	4	8	4	
	3	8	7	3	4	3	
30	1	6	3	9	4	8	
	2	8	2	8	4	6	
	3	9	2	7	4	6	
31	1	4	9	2	10	9	
	2	5	8	2	9	6	
	3	10	8	2	9	5	
32	1	5	5	10	7	7	
	2	6	5	7	5	7	
	3	7	4	3	4	7	
33	1	2	9	8	7	3	
	2	5	7	6	5	3	
	3	8	6	6	3	2	
34	1	4	7	9	2	8	
	2	5	5	9	2	6	
	3	7	4	9	2	6	
35	1	2	10	8	4	5	
	2	8	9	8	4	4	
	3	9	9	6	3	3	
36	1	2	8	2	8	6	
	2	3	5	1	6	6	
	3	5	1	1	6	5	
37	1	5	7	4	8	10	
	2	6	6	4	5	7	
	3	10	5	3	2	1	
38	1	2	7	5	8	8	
	2	5	7	5	8	7	
	3	10	6	5	6	7	
39	1	1	7	1	8	1	
	2	3	5	1	7	1	
	3	10	4	1	5	1	
40	1	2	4	7	10	2	
	2	6	4	4	10	2	
	3	7	4	3	10	2	
41	1	1	6	6	4	8	
	2	2	5	5	4	8	
	3	5	4	4	2	7	
42	1	8	7	8	8	6	
	2	9	5	4	6	5	
	3	10	5	4	4	4	
43	1	3	6	3	6	8	
	2	4	5	2	5	7	
	3	9	4	1	4	6	
44	1	2	8	6	9	4	
	2	6	5	5	7	3	
	3	7	5	4	7	3	
45	1	1	7	5	4	2	
	2	2	5	5	4	1	
	3	6	4	5	4	1	
46	1	3	2	9	6	9	
	2	4	2	6	5	9	
	3	5	1	3	3	9	
47	1	1	10	6	9	9	
	2	4	6	4	9	5	
	3	5	5	3	8	3	
48	1	1	7	8	8	4	
	2	3	6	8	5	2	
	3	6	4	8	3	2	
49	1	1	9	7	9	5	
	2	4	6	3	7	5	
	3	5	3	2	7	5	
50	1	2	8	7	5	4	
	2	7	4	7	3	4	
	3	9	2	7	1	4	
51	1	2	4	4	9	7	
	2	3	3	3	7	7	
	3	5	2	2	5	6	
52	1	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2
	31	29	307	283

************************************************************************
