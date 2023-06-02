jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	7		2 3 4 5 6 7 8 
2	3	5		14 13 12 10 9 
3	3	5		14 13 12 11 9 
4	3	4		14 13 10 9 
5	3	4		13 12 11 9 
6	3	6		25 17 15 14 13 12 
7	3	6		25 20 19 18 17 14 
8	3	7		25 23 20 19 18 17 15 
9	3	7		25 23 21 20 19 17 15 
10	3	1		11 
11	3	5		23 20 19 18 16 
12	3	4		23 20 19 18 
13	3	3		27 22 18 
14	3	3		23 22 16 
15	3	6		35 28 27 26 24 22 
16	3	5		35 28 27 26 24 
17	3	5		35 28 27 26 24 
18	3	3		26 24 21 
19	3	3		35 28 24 
20	3	2		27 22 
21	3	6		39 35 33 30 29 28 
22	3	5		37 32 31 30 29 
23	3	2		27 26 
24	3	4		37 32 31 29 
25	3	4		39 34 32 28 
26	3	5		39 38 37 32 30 
27	3	4		38 37 36 34 
28	3	4		42 38 37 36 
29	3	3		38 36 34 
30	3	5		51 43 42 40 36 
31	3	3		39 38 36 
32	3	2		42 33 
33	3	3		51 40 36 
34	3	5		51 47 44 43 42 
35	3	5		50 47 46 44 38 
36	3	5		50 47 46 44 41 
37	3	4		50 47 45 43 
38	3	2		51 40 
39	3	3		51 49 46 
40	3	2		49 45 
41	3	2		49 45 
42	3	2		50 46 
43	3	2		49 46 
44	3	1		45 
45	3	1		48 
46	3	1		48 
47	3	1		49 
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
2	1	2	8	4	5	0	
	2	3	8	4	0	3	
	3	9	6	4	0	3	
3	1	7	6	5	8	0	
	2	9	5	2	0	2	
	3	10	5	1	0	2	
4	1	3	7	5	5	0	
	2	5	3	4	5	0	
	3	9	1	4	0	9	
5	1	1	9	2	7	0	
	2	6	6	2	0	1	
	3	9	5	2	0	1	
6	1	5	7	8	9	0	
	2	7	7	5	9	0	
	3	8	7	3	8	0	
7	1	2	7	9	0	9	
	2	6	5	7	3	0	
	3	7	3	5	0	5	
8	1	7	3	10	0	9	
	2	8	3	8	0	9	
	3	9	2	5	9	0	
9	1	6	3	9	0	5	
	2	7	3	7	0	3	
	3	10	3	7	0	2	
10	1	1	9	3	2	0	
	2	7	8	1	1	0	
	3	8	8	1	0	6	
11	1	5	8	7	0	8	
	2	5	6	7	8	0	
	3	10	4	4	6	0	
12	1	5	8	8	0	5	
	2	9	5	5	0	5	
	3	9	5	3	8	0	
13	1	3	10	7	3	0	
	2	7	8	7	3	0	
	3	7	7	5	0	8	
14	1	2	5	9	0	6	
	2	3	4	8	4	0	
	3	6	3	7	0	3	
15	1	2	8	9	5	0	
	2	2	5	7	0	3	
	3	8	3	5	0	2	
16	1	4	10	4	4	0	
	2	5	8	4	0	3	
	3	9	8	2	0	3	
17	1	2	5	9	9	0	
	2	6	5	8	7	0	
	3	9	1	6	0	9	
18	1	2	8	10	0	7	
	2	5	6	10	0	5	
	3	8	4	10	5	0	
19	1	3	8	7	6	0	
	2	4	7	6	5	0	
	3	5	5	6	5	0	
20	1	2	8	8	9	0	
	2	3	6	7	8	0	
	3	7	4	6	7	0	
21	1	1	8	8	0	6	
	2	6	6	4	0	5	
	3	10	2	4	6	0	
22	1	1	5	8	0	7	
	2	2	4	7	0	7	
	3	3	4	5	0	7	
23	1	7	6	7	7	0	
	2	8	4	6	0	6	
	3	9	3	2	0	4	
24	1	3	8	2	7	0	
	2	5	7	2	0	4	
	3	5	5	1	5	0	
25	1	3	5	4	0	6	
	2	9	5	4	3	0	
	3	10	5	2	2	0	
26	1	4	5	5	0	3	
	2	8	4	3	5	0	
	3	9	3	2	0	2	
27	1	1	4	5	4	0	
	2	3	2	5	3	0	
	3	7	2	5	2	0	
28	1	2	9	9	7	0	
	2	3	6	8	5	0	
	3	8	3	8	0	7	
29	1	5	8	9	6	0	
	2	7	7	8	0	5	
	3	10	7	8	0	4	
30	1	7	3	7	0	9	
	2	8	3	6	6	0	
	3	9	2	6	6	0	
31	1	3	5	6	7	0	
	2	6	4	5	0	3	
	3	8	3	5	0	3	
32	1	1	3	7	0	6	
	2	3	3	6	0	6	
	3	10	3	6	4	0	
33	1	4	7	5	0	7	
	2	8	5	4	0	5	
	3	9	1	3	0	5	
34	1	1	6	6	6	0	
	2	2	4	4	5	0	
	3	3	3	3	0	8	
35	1	1	6	7	0	8	
	2	4	6	5	0	8	
	3	8	2	5	0	7	
36	1	5	7	9	5	0	
	2	7	7	8	0	8	
	3	9	7	6	0	6	
37	1	2	8	6	0	8	
	2	5	7	5	0	6	
	3	10	7	2	0	5	
38	1	3	5	7	0	2	
	2	8	3	7	0	2	
	3	9	3	6	0	1	
39	1	5	8	5	0	1	
	2	7	7	4	4	0	
	3	8	4	2	4	0	
40	1	3	8	8	0	6	
	2	8	7	6	0	6	
	3	9	7	4	0	5	
41	1	1	7	9	0	9	
	2	8	6	8	1	0	
	3	10	6	6	1	0	
42	1	1	8	9	0	9	
	2	6	6	9	4	0	
	3	10	4	9	3	0	
43	1	1	4	9	7	0	
	2	2	3	9	0	5	
	3	3	3	8	0	3	
44	1	1	8	10	0	1	
	2	3	6	5	2	0	
	3	8	6	2	0	1	
45	1	1	3	7	4	0	
	2	2	2	6	0	4	
	3	2	1	4	2	0	
46	1	1	8	10	6	0	
	2	3	4	9	4	0	
	3	4	3	9	0	9	
47	1	1	4	5	5	0	
	2	6	4	5	0	2	
	3	8	3	5	0	2	
48	1	1	7	9	1	0	
	2	10	5	9	0	5	
	3	10	5	9	1	0	
49	1	1	3	7	0	4	
	2	4	3	6	0	4	
	3	9	3	4	0	4	
50	1	4	9	6	8	0	
	2	5	6	5	6	0	
	3	6	3	3	5	0	
51	1	6	4	6	0	9	
	2	9	3	5	8	0	
	3	9	3	1	0	8	
52	1	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2
	46	46	129	148

************************************************************************
