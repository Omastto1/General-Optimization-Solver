jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 4 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	5		2 3 4 5 6 
2	3	5		14 12 10 9 7 
3	3	4		14 12 8 7 
4	3	4		15 14 13 10 
5	3	3		15 14 10 
6	3	3		15 12 10 
7	3	4		21 15 13 11 
8	3	2		15 10 
9	3	3		21 15 11 
10	3	2		21 11 
11	3	8		26 25 23 20 19 18 17 16 
12	3	6		26 25 23 21 17 16 
13	3	6		26 24 23 20 19 18 
14	3	6		35 28 26 25 24 18 
15	3	1		16 
16	3	6		35 31 30 28 24 22 
17	3	6		38 31 30 29 27 24 
18	3	4		31 30 27 22 
19	3	3		30 28 22 
20	3	6		38 35 34 33 30 27 
21	3	6		38 34 33 31 30 29 
22	3	4		38 34 33 29 
23	3	4		37 35 30 29 
24	3	4		39 34 33 32 
25	3	3		38 37 29 
26	3	2		33 29 
27	3	6		44 43 40 39 37 36 
28	3	6		44 43 40 39 37 36 
29	3	2		39 32 
30	3	2		44 32 
31	3	5		44 41 40 39 36 
32	3	4		43 41 40 36 
33	3	4		43 42 40 37 
34	3	5		45 44 43 41 40 
35	3	3		45 43 40 
36	3	2		48 42 
37	3	2		47 41 
38	3	4		51 48 47 46 
39	3	1		42 
40	3	4		51 50 48 46 
41	3	4		51 50 48 46 
42	3	3		47 46 45 
43	3	2		49 47 
44	3	2		50 46 
45	3	2		50 49 
46	3	1		49 
47	3	1		50 
48	3	1		49 
49	3	1		52 
50	3	1		52 
51	3	1		52 
52	1	0		
************************************************************************
REQUESTS/DURATIONS
jobnr.	mode	dur	R1	R2	N1	N2	N3	N4	
------------------------------------------------------------------------
1	1	0	0	0	0	0	0	0	
2	1	6	5	6	2	10	7	1	
	2	7	5	5	2	5	7	1	
	3	8	5	4	2	1	5	1	
3	1	1	8	7	6	10	3	5	
	2	2	7	6	6	7	3	5	
	3	8	2	5	6	2	2	4	
4	1	3	9	7	9	1	8	4	
	2	9	6	4	5	1	7	3	
	3	10	6	2	1	1	7	1	
5	1	3	7	9	10	6	8	9	
	2	4	5	8	7	6	5	6	
	3	6	1	7	6	4	2	4	
6	1	1	6	7	9	7	5	2	
	2	3	5	7	7	3	4	1	
	3	5	5	7	6	3	4	1	
7	1	2	5	6	5	3	8	9	
	2	7	5	6	5	2	7	7	
	3	8	5	5	2	2	2	5	
8	1	2	7	10	6	5	1	4	
	2	3	6	7	5	3	1	4	
	3	9	4	5	5	3	1	4	
9	1	3	7	4	5	5	5	7	
	2	6	5	3	5	4	3	3	
	3	9	4	3	5	1	1	2	
10	1	3	6	8	3	8	8	2	
	2	4	4	7	2	6	7	2	
	3	10	2	7	2	5	2	2	
11	1	2	9	9	6	7	6	5	
	2	5	6	8	5	7	6	4	
	3	8	3	8	5	7	4	2	
12	1	5	10	4	6	7	10	7	
	2	7	8	2	4	7	9	5	
	3	8	7	1	4	6	9	5	
13	1	5	8	4	3	8	5	5	
	2	7	5	4	3	7	4	4	
	3	10	4	4	3	6	2	4	
14	1	1	6	9	5	8	5	7	
	2	2	4	4	5	7	4	7	
	3	7	3	4	4	6	2	7	
15	1	2	6	7	7	7	6	7	
	2	4	5	6	5	7	6	6	
	3	8	4	6	4	4	6	3	
16	1	1	6	10	9	9	2	3	
	2	2	6	7	9	6	1	3	
	3	9	6	6	8	3	1	2	
17	1	1	1	7	8	2	5	6	
	2	3	1	5	6	1	4	6	
	3	10	1	5	6	1	4	4	
18	1	3	7	7	9	4	10	9	
	2	4	5	6	7	2	9	9	
	3	5	5	6	2	1	8	7	
19	1	5	7	5	8	5	8	6	
	2	8	6	3	7	5	5	5	
	3	10	4	2	7	2	2	4	
20	1	1	7	10	2	5	7	5	
	2	2	7	5	2	5	6	4	
	3	9	5	3	1	5	5	3	
21	1	1	9	8	6	6	8	5	
	2	3	7	3	6	6	8	5	
	3	4	7	1	6	6	6	5	
22	1	1	9	7	8	7	2	3	
	2	8	3	5	7	6	1	2	
	3	10	1	2	7	6	1	2	
23	1	3	7	9	3	3	5	7	
	2	5	4	9	3	3	5	4	
	3	10	2	8	2	3	5	3	
24	1	1	10	4	7	2	6	8	
	2	5	6	3	5	2	3	8	
	3	7	5	2	4	1	1	7	
25	1	3	5	8	10	9	9	7	
	2	6	3	8	9	7	6	6	
	3	8	2	6	8	6	6	4	
26	1	3	10	8	8	1	5	7	
	2	5	9	6	5	1	4	4	
	3	7	8	6	5	1	4	3	
27	1	1	2	6	2	9	8	9	
	2	3	1	2	2	5	8	6	
	3	4	1	2	2	5	7	6	
28	1	2	5	4	9	8	1	5	
	2	4	5	4	9	8	1	4	
	3	8	3	4	9	8	1	3	
29	1	6	4	6	5	1	8	5	
	2	7	3	4	4	1	5	5	
	3	8	1	4	3	1	4	5	
30	1	3	8	5	8	6	3	1	
	2	4	6	4	5	5	3	1	
	3	6	6	4	3	4	3	1	
31	1	1	6	6	4	7	4	9	
	2	4	6	5	3	7	3	5	
	3	5	6	5	1	7	3	1	
32	1	2	9	4	7	6	2	4	
	2	6	8	4	7	5	2	3	
	3	9	8	3	7	4	2	2	
33	1	1	5	3	4	8	9	10	
	2	4	4	3	3	8	6	7	
	3	7	4	3	3	8	5	3	
34	1	1	8	6	10	7	1	10	
	2	2	7	6	8	6	1	8	
	3	9	7	5	5	4	1	8	
35	1	6	5	8	5	9	8	8	
	2	7	4	4	4	9	6	7	
	3	8	4	2	4	9	6	6	
36	1	5	8	5	2	10	6	3	
	2	9	8	4	2	10	5	2	
	3	10	8	3	2	10	5	1	
37	1	8	5	5	5	3	4	10	
	2	9	4	5	3	2	3	9	
	3	10	3	3	2	2	3	8	
38	1	2	6	8	9	8	2	8	
	2	3	6	8	9	5	2	7	
	3	9	6	7	9	4	2	5	
39	1	1	5	6	10	5	3	3	
	2	8	4	5	6	5	2	3	
	3	9	4	3	6	4	2	3	
40	1	1	2	9	7	4	7	9	
	2	5	2	6	6	4	7	3	
	3	7	1	6	6	4	3	2	
41	1	5	5	8	8	8	8	10	
	2	9	3	8	7	7	6	8	
	3	10	1	8	7	7	3	7	
42	1	4	4	6	8	7	10	6	
	2	9	2	5	7	4	10	4	
	3	10	2	5	6	3	10	2	
43	1	4	5	8	9	9	10	2	
	2	5	3	4	9	5	6	1	
	3	9	3	4	8	4	3	1	
44	1	3	7	8	7	8	5	9	
	2	6	6	7	6	7	5	7	
	3	10	5	6	5	6	3	5	
45	1	7	5	8	1	6	6	4	
	2	9	5	7	1	3	5	4	
	3	10	5	5	1	2	5	3	
46	1	1	6	7	8	7	7	4	
	2	5	6	5	7	6	4	2	
	3	7	5	4	4	3	1	2	
47	1	4	6	3	10	10	1	1	
	2	7	6	2	8	9	1	1	
	3	9	5	1	8	8	1	1	
48	1	3	9	4	10	9	3	6	
	2	6	6	2	6	9	3	5	
	3	7	4	1	6	9	3	2	
49	1	4	6	8	1	1	5	5	
	2	5	4	4	1	1	4	5	
	3	8	4	3	1	1	3	5	
50	1	2	5	7	6	5	7	7	
	2	4	5	5	6	2	7	7	
	3	6	5	5	6	1	7	7	
51	1	1	8	9	6	10	10	6	
	2	2	4	5	6	9	6	6	
	3	7	3	4	6	7	4	3	
52	1	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2	N 3	N 4
	20	20	253	237	209	209

************************************************************************
