jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 4 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	15		2 3 4 6 7 8 9 10 12 13 14 15 17 21 24 
2	3	5		35 30 25 18 5 
3	3	14		51 50 49 36 35 34 33 32 31 29 28 26 23 18 
4	3	9		50 49 48 38 30 25 20 18 16 
5	3	10		50 37 36 34 33 32 29 28 26 19 
6	3	8		51 37 33 32 27 26 22 11 
7	3	7		32 29 28 27 26 20 19 
8	3	8		48 47 32 30 29 28 26 25 
9	3	9		50 49 48 45 41 37 33 32 28 
10	3	6		43 41 33 31 29 26 
11	3	8		50 49 48 47 43 41 40 29 
12	3	6		49 47 43 41 36 26 
13	3	5		49 39 34 32 26 
14	3	9		51 49 48 47 46 45 44 41 40 
15	3	5		50 41 32 30 28 
16	3	6		45 44 43 41 32 28 
17	3	5		50 47 42 41 26 
18	3	7		47 46 45 44 42 41 40 
19	3	7		51 49 48 47 43 42 41 
20	3	4		44 35 34 33 
21	3	6		51 48 47 45 44 41 
22	3	4		46 44 41 28 
23	3	5		48 47 46 42 41 
24	3	5		49 46 45 41 40 
25	3	4		45 43 41 33 
26	3	3		45 44 40 
27	3	3		48 43 40 
28	3	2		42 40 
29	3	2		46 42 
30	3	2		46 43 
31	3	2		44 40 
32	3	1		40 
33	3	1		40 
34	3	1		43 
35	3	1		41 
36	3	1		42 
37	3	1		40 
38	3	1		42 
39	3	1		41 
40	3	1		52 
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
jobnr.	mode	dur	R1	R2	N1	N2	N3	N4	
------------------------------------------------------------------------
1	1	0	0	0	0	0	0	0	
2	1	2	9	6	5	9	5	5	
	2	3	8	6	3	8	4	4	
	3	7	8	6	3	6	3	4	
3	1	2	8	7	8	10	8	7	
	2	4	6	5	4	9	7	5	
	3	9	3	5	4	8	7	2	
4	1	2	5	8	6	8	7	4	
	2	8	5	4	5	6	6	3	
	3	10	4	2	5	4	6	3	
5	1	3	9	7	8	10	5	1	
	2	5	6	5	5	8	5	1	
	3	8	5	3	5	8	2	1	
6	1	1	9	10	3	10	4	9	
	2	5	5	8	2	5	4	9	
	3	6	4	7	2	5	4	9	
7	1	2	7	1	7	9	6	8	
	2	7	6	1	6	8	6	8	
	3	8	4	1	5	8	4	7	
8	1	2	9	9	5	7	7	7	
	2	4	9	5	4	3	6	5	
	3	5	9	4	3	2	5	5	
9	1	4	10	3	7	8	5	4	
	2	5	8	3	7	8	4	3	
	3	6	6	2	7	6	1	2	
10	1	7	1	9	8	9	9	10	
	2	8	1	8	5	6	7	8	
	3	9	1	8	4	4	7	6	
11	1	6	2	7	8	9	10	8	
	2	7	2	7	7	8	8	7	
	3	10	2	7	7	8	7	5	
12	1	4	7	2	3	6	8	8	
	2	5	3	1	2	5	6	3	
	3	8	3	1	1	5	5	1	
13	1	1	5	9	8	7	8	3	
	2	2	5	6	7	4	6	2	
	3	3	5	4	6	2	2	2	
14	1	8	5	4	4	7	10	5	
	2	9	5	4	2	5	8	4	
	3	10	2	3	2	3	8	3	
15	1	1	9	9	4	9	5	8	
	2	2	9	9	4	8	5	7	
	3	9	9	9	3	7	5	5	
16	1	2	4	8	5	4	4	3	
	2	4	3	7	5	4	3	2	
	3	8	2	7	5	4	2	1	
17	1	4	4	6	6	6	8	4	
	2	5	2	5	4	4	5	3	
	3	8	2	3	2	4	5	3	
18	1	1	7	4	4	7	3	6	
	2	3	7	3	4	4	3	5	
	3	5	6	3	2	2	2	5	
19	1	6	6	5	8	5	7	7	
	2	9	5	3	7	5	6	6	
	3	10	3	3	7	5	4	6	
20	1	5	9	8	6	8	10	6	
	2	6	8	5	6	4	7	3	
	3	10	8	5	3	2	2	1	
21	1	3	9	6	2	3	6	6	
	2	5	5	6	2	2	5	5	
	3	6	2	4	2	2	4	5	
22	1	4	5	7	7	8	3	8	
	2	5	5	6	3	7	1	6	
	3	6	5	6	2	3	1	6	
23	1	2	9	5	10	9	1	5	
	2	3	8	5	7	9	1	5	
	3	6	7	4	7	9	1	5	
24	1	2	5	4	8	4	3	3	
	2	7	4	3	6	4	3	3	
	3	9	4	3	4	4	3	2	
25	1	1	7	5	4	9	10	8	
	2	5	6	5	2	7	10	6	
	3	10	2	5	2	7	10	6	
26	1	8	4	2	6	8	5	7	
	2	9	3	2	4	8	2	5	
	3	10	3	2	4	7	2	5	
27	1	7	5	8	3	4	7	7	
	2	9	3	8	2	4	5	6	
	3	10	3	8	1	4	2	3	
28	1	1	6	6	10	5	6	7	
	2	3	4	5	9	5	5	6	
	3	5	3	4	9	5	5	4	
29	1	5	5	8	2	7	2	6	
	2	6	3	7	2	6	2	6	
	3	7	2	6	2	6	1	6	
30	1	2	3	8	10	3	6	2	
	2	3	3	6	9	3	6	2	
	3	10	3	4	8	3	6	2	
31	1	2	3	8	4	3	2	9	
	2	4	3	7	4	3	2	9	
	3	7	3	7	4	3	2	8	
32	1	1	8	7	3	3	4	9	
	2	4	8	6	3	2	2	9	
	3	9	8	4	3	1	2	9	
33	1	1	7	8	8	4	2	9	
	2	2	4	5	7	4	1	8	
	3	4	2	3	7	4	1	8	
34	1	2	10	4	2	6	6	5	
	2	4	4	4	2	5	5	4	
	3	10	3	2	1	4	4	4	
35	1	1	4	7	6	5	1	10	
	2	4	4	5	6	2	1	6	
	3	5	3	4	4	1	1	3	
36	1	1	9	5	3	5	9	7	
	2	4	7	5	3	5	7	7	
	3	8	6	4	2	5	4	6	
37	1	1	8	3	3	3	8	6	
	2	2	7	2	2	3	7	6	
	3	9	5	2	1	3	2	4	
38	1	3	7	5	6	4	5	9	
	2	5	7	5	4	4	5	7	
	3	6	7	5	4	2	3	5	
39	1	6	6	10	4	8	8	6	
	2	7	5	5	3	8	4	6	
	3	10	3	1	1	7	4	6	
40	1	1	4	9	10	1	5	3	
	2	3	3	9	8	1	5	2	
	3	5	3	8	7	1	4	2	
41	1	3	8	6	5	7	4	5	
	2	4	7	4	3	5	4	3	
	3	6	5	3	3	3	4	2	
42	1	1	8	8	5	7	6	6	
	2	7	8	8	4	6	4	4	
	3	10	8	8	4	4	1	2	
43	1	5	6	7	9	3	5	9	
	2	7	4	5	6	2	4	8	
	3	10	3	1	4	2	3	8	
44	1	3	9	2	10	4	4	6	
	2	4	7	1	9	3	3	6	
	3	6	7	1	7	3	2	4	
45	1	4	8	7	7	6	8	7	
	2	5	5	7	5	4	6	5	
	3	6	2	7	4	3	2	3	
46	1	3	7	6	7	9	2	9	
	2	6	6	5	5	4	1	9	
	3	9	2	5	5	4	1	9	
47	1	2	8	4	3	5	3	6	
	2	4	8	2	3	5	3	4	
	3	5	8	2	2	5	2	3	
48	1	6	5	4	8	6	6	5	
	2	8	4	3	6	5	5	4	
	3	9	4	2	4	4	2	3	
49	1	4	7	3	8	3	7	2	
	2	5	7	2	8	3	5	2	
	3	9	7	1	8	1	3	2	
50	1	8	7	2	10	6	9	4	
	2	9	6	2	7	5	8	4	
	3	10	5	2	4	3	7	3	
51	1	1	2	7	9	9	5	10	
	2	3	2	6	9	9	5	8	
	3	5	2	4	9	8	5	8	
52	1	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2	N 3	N 4
	59	61	280	289	259	290

************************************************************************
