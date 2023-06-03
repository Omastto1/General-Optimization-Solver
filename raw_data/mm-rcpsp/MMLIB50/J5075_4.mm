jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	6		2 3 4 5 9 10 
2	3	5		21 17 12 8 6 
3	3	6		21 18 15 14 13 12 
4	3	5		21 20 18 14 11 
5	3	3		21 12 7 
6	3	4		20 18 14 11 
7	3	3		20 14 13 
8	3	3		20 15 13 
9	3	2		14 12 
10	3	2		15 11 
11	3	1		13 
12	3	5		24 23 22 20 19 
13	3	1		16 
14	3	1		16 
15	3	1		16 
16	3	4		24 23 22 19 
17	3	4		24 23 22 20 
18	3	4		28 25 24 23 
19	3	4		36 28 27 25 
20	3	3		36 26 25 
21	3	1		22 
22	3	7		40 35 34 32 29 28 27 
23	3	4		36 34 29 26 
24	3	7		42 41 40 35 34 32 31 
25	3	5		35 34 33 32 29 
26	3	2		40 27 
27	3	3		33 31 30 
28	3	3		37 33 30 
29	3	6		44 42 41 39 38 37 
30	3	6		51 45 44 42 39 38 
31	3	5		51 44 39 38 37 
32	3	5		51 45 39 38 37 
33	3	4		44 41 39 38 
34	3	5		51 47 46 44 43 
35	3	3		45 44 38 
36	3	2		41 40 
37	3	4		50 47 46 43 
38	3	4		50 49 47 46 
39	3	3		47 46 43 
40	3	3		51 50 45 
41	3	3		51 50 49 
42	3	2		49 46 
43	3	2		49 48 
44	3	2		50 49 
45	3	1		46 
46	3	1		48 
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
2	1	7	0	5	4	0	
	2	8	0	5	0	2	
	3	10	4	0	4	0	
3	1	3	1	0	2	0	
	2	7	1	0	1	0	
	3	8	0	3	0	5	
4	1	4	7	0	8	0	
	2	9	5	0	6	0	
	3	10	4	0	5	0	
5	1	1	5	0	0	8	
	2	4	0	6	4	0	
	3	10	2	0	4	0	
6	1	1	4	0	0	8	
	2	2	0	7	0	7	
	3	3	2	0	0	6	
7	1	5	6	0	8	0	
	2	7	0	4	0	7	
	3	10	0	3	2	0	
8	1	6	7	0	0	4	
	2	6	5	0	5	0	
	3	10	5	0	0	3	
9	1	1	0	9	0	9	
	2	3	3	0	0	7	
	3	4	0	8	0	7	
10	1	2	0	5	0	7	
	2	5	2	0	0	6	
	3	7	1	0	2	0	
11	1	1	8	0	0	6	
	2	2	7	0	0	5	
	3	5	7	0	0	2	
12	1	2	0	3	0	6	
	2	5	7	0	0	6	
	3	6	0	2	2	0	
13	1	6	8	0	7	0	
	2	8	7	0	0	3	
	3	9	0	5	7	0	
14	1	3	0	4	0	8	
	2	5	0	4	0	7	
	3	5	2	0	8	0	
15	1	3	0	4	5	0	
	2	5	0	4	3	0	
	3	10	4	0	0	9	
16	1	1	6	0	0	7	
	2	4	5	0	0	4	
	3	10	5	0	0	3	
17	1	1	6	0	6	0	
	2	1	6	0	0	2	
	3	8	6	0	0	1	
18	1	2	0	8	6	0	
	2	3	0	8	0	9	
	3	10	6	0	3	0	
19	1	1	4	0	0	7	
	2	4	4	0	0	4	
	3	5	4	0	0	1	
20	1	5	7	0	0	6	
	2	6	7	0	0	5	
	3	9	0	5	0	4	
21	1	1	0	8	6	0	
	2	3	8	0	4	0	
	3	4	8	0	0	7	
22	1	1	0	9	10	0	
	2	4	0	9	0	8	
	3	9	0	9	8	0	
23	1	3	9	0	0	8	
	2	6	0	6	0	7	
	3	7	0	6	0	6	
24	1	6	4	0	0	7	
	2	9	3	0	0	7	
	3	10	2	0	4	0	
25	1	4	0	10	0	6	
	2	5	2	0	3	0	
	3	6	0	9	2	0	
26	1	4	10	0	0	4	
	2	6	0	8	1	0	
	3	8	0	7	1	0	
27	1	7	6	0	4	0	
	2	8	0	6	3	0	
	3	10	0	4	0	4	
28	1	3	0	9	0	6	
	2	3	0	9	4	0	
	3	4	6	0	0	5	
29	1	8	9	0	6	0	
	2	8	0	7	0	6	
	3	9	0	6	4	0	
30	1	6	6	0	8	0	
	2	7	5	0	6	0	
	3	9	4	0	0	5	
31	1	4	0	8	8	0	
	2	5	0	8	6	0	
	3	7	0	8	0	7	
32	1	2	7	0	0	2	
	2	3	5	0	0	2	
	3	4	0	3	0	2	
33	1	3	5	0	4	0	
	2	6	4	0	2	0	
	3	7	0	3	2	0	
34	1	3	0	9	6	0	
	2	3	6	0	0	6	
	3	9	3	0	3	0	
35	1	1	3	0	8	0	
	2	5	2	0	0	9	
	3	8	2	0	6	0	
36	1	2	0	7	3	0	
	2	3	4	0	2	0	
	3	10	3	0	1	0	
37	1	2	8	0	7	0	
	2	3	0	5	4	0	
	3	5	4	0	0	1	
38	1	3	0	7	9	0	
	2	8	6	0	0	7	
	3	10	0	7	0	2	
39	1	1	10	0	2	0	
	2	1	0	3	2	0	
	3	2	9	0	2	0	
40	1	2	0	10	0	8	
	2	3	3	0	3	0	
	3	10	0	8	2	0	
41	1	1	0	5	0	6	
	2	6	5	0	0	6	
	3	10	0	3	6	0	
42	1	2	0	10	0	7	
	2	3	2	0	6	0	
	3	4	0	7	0	3	
43	1	1	0	2	0	9	
	2	5	0	2	5	0	
	3	10	0	1	3	0	
44	1	2	0	9	0	6	
	2	4	7	0	2	0	
	3	7	7	0	1	0	
45	1	3	0	8	0	6	
	2	4	0	7	0	4	
	3	10	5	0	0	2	
46	1	4	0	6	8	0	
	2	10	7	0	0	4	
	3	10	0	4	6	0	
47	1	4	0	6	9	0	
	2	6	0	6	0	8	
	3	7	0	6	0	7	
48	1	4	0	9	6	0	
	2	4	6	0	4	0	
	3	4	0	8	0	2	
49	1	1	7	0	5	0	
	2	7	6	0	0	7	
	3	9	0	5	4	0	
50	1	3	0	10	9	0	
	2	6	0	8	0	5	
	3	10	1	0	9	0	
51	1	5	0	7	0	8	
	2	7	0	5	1	0	
	3	9	2	0	1	0	
52	1	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2
	17	18	115	157

************************************************************************
