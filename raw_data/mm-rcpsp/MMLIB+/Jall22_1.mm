jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	6		2 3 4 5 6 9 
2	3	7		18 15 13 12 11 8 7 
3	3	7		17 16 15 14 13 12 10 
4	3	7		18 17 16 15 14 12 10 
5	3	6		18 16 13 12 11 10 
6	3	6		25 18 17 12 10 8 
7	3	5		25 21 17 16 10 
8	3	4		20 19 16 14 
9	3	4		25 24 19 16 
10	3	4		24 22 20 19 
11	3	4		24 23 20 17 
12	3	3		24 22 21 
13	3	3		25 24 21 
14	3	2		24 21 
15	3	3		25 24 20 
16	3	2		23 22 
17	3	3		33 27 22 
18	3	4		33 32 27 24 
19	3	3		30 26 23 
20	3	5		33 32 30 29 26 
21	3	2		29 23 
22	3	4		32 30 29 26 
23	3	4		33 32 28 27 
24	3	2		30 26 
25	3	3		36 35 28 
26	3	2		36 28 
27	3	6		42 40 38 37 36 35 
28	3	3		40 34 31 
29	3	5		40 38 37 36 35 
30	3	4		42 39 38 36 
31	3	3		41 38 37 
32	3	3		41 38 37 
33	3	3		41 38 37 
34	3	5		51 43 42 41 39 
35	3	4		51 43 41 39 
36	3	5		51 49 44 43 41 
37	3	2		43 39 
38	3	5		51 49 48 47 46 
39	3	3		49 46 44 
40	3	3		48 46 43 
41	3	3		48 46 45 
42	3	3		47 46 45 
43	3	2		50 45 
44	3	2		47 45 
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
2	1	1	10	7	7	1	
	2	5	5	7	4	1	
	3	7	3	6	3	1	
3	1	2	7	8	6	1	
	2	4	6	5	6	1	
	3	9	4	2	6	1	
4	1	4	5	10	9	9	
	2	5	5	7	9	3	
	3	6	3	5	9	1	
5	1	1	8	6	3	7	
	2	2	4	3	3	5	
	3	4	4	3	3	4	
6	1	3	10	6	10	10	
	2	4	9	6	9	5	
	3	10	8	5	9	1	
7	1	5	9	3	7	6	
	2	6	4	3	5	6	
	3	9	2	1	5	5	
8	1	1	8	5	7	1	
	2	4	6	2	6	1	
	3	9	4	1	6	1	
9	1	6	8	3	3	9	
	2	7	6	3	2	9	
	3	8	6	3	2	8	
10	1	2	7	7	7	4	
	2	4	6	5	5	2	
	3	5	6	2	2	2	
11	1	3	7	6	4	7	
	2	5	5	3	4	7	
	3	7	5	2	2	7	
12	1	2	6	10	9	10	
	2	3	5	4	7	9	
	3	9	5	3	7	7	
13	1	4	9	9	8	9	
	2	6	8	8	8	8	
	3	7	4	6	6	7	
14	1	1	6	5	7	5	
	2	4	5	4	5	5	
	3	5	4	2	3	3	
15	1	5	8	9	8	9	
	2	6	7	9	7	8	
	3	7	6	8	7	8	
16	1	2	3	7	3	9	
	2	6	2	4	2	8	
	3	7	2	2	2	8	
17	1	3	10	6	2	8	
	2	4	9	6	2	8	
	3	9	9	3	2	8	
18	1	4	5	8	8	10	
	2	5	5	8	6	7	
	3	6	4	8	5	6	
19	1	4	7	8	9	10	
	2	8	6	6	7	8	
	3	9	6	4	5	7	
20	1	1	4	6	1	4	
	2	4	3	4	1	2	
	3	8	2	2	1	2	
21	1	1	4	9	3	7	
	2	9	4	8	3	6	
	3	10	3	7	3	5	
22	1	4	8	6	6	5	
	2	5	5	5	4	5	
	3	10	3	1	3	5	
23	1	7	9	8	8	3	
	2	9	7	8	6	3	
	3	10	7	8	5	2	
24	1	2	6	10	4	9	
	2	5	4	8	4	7	
	3	8	3	8	4	5	
25	1	4	6	7	3	4	
	2	6	6	7	1	4	
	3	8	6	7	1	3	
26	1	4	9	5	5	5	
	2	6	7	4	4	4	
	3	9	3	2	3	4	
27	1	3	6	6	8	2	
	2	5	5	5	7	1	
	3	6	4	5	7	1	
28	1	4	5	7	5	7	
	2	7	3	7	5	5	
	3	9	3	7	3	4	
29	1	6	5	6	5	2	
	2	7	4	5	4	2	
	3	8	3	4	3	2	
30	1	3	2	7	5	6	
	2	5	1	4	3	6	
	3	7	1	4	3	3	
31	1	2	5	10	7	7	
	2	3	4	6	5	6	
	3	5	4	6	4	4	
32	1	1	7	5	9	6	
	2	2	6	3	5	4	
	3	6	4	2	5	4	
33	1	5	6	4	5	8	
	2	8	5	4	2	8	
	3	9	2	4	1	6	
34	1	2	4	10	2	7	
	2	3	2	9	2	7	
	3	4	1	8	2	4	
35	1	2	4	10	2	1	
	2	3	3	10	2	1	
	3	4	1	10	2	1	
36	1	7	6	5	4	4	
	2	8	5	4	3	3	
	3	10	4	3	3	2	
37	1	4	8	7	4	3	
	2	5	7	7	3	2	
	3	9	5	4	3	1	
38	1	7	10	8	8	9	
	2	8	5	5	7	8	
	3	9	3	1	6	8	
39	1	1	6	3	8	8	
	2	7	4	2	7	7	
	3	10	3	1	7	6	
40	1	3	6	8	6	10	
	2	4	5	8	5	8	
	3	8	4	8	5	8	
41	1	1	9	10	3	8	
	2	4	9	8	2	7	
	3	7	9	8	2	6	
42	1	7	10	2	6	9	
	2	9	8	1	3	8	
	3	10	8	1	3	7	
43	1	2	7	9	8	9	
	2	8	6	7	7	8	
	3	9	2	6	6	8	
44	1	2	7	2	7	8	
	2	3	6	1	3	5	
	3	5	5	1	2	3	
45	1	1	3	8	4	5	
	2	8	3	4	2	4	
	3	9	3	2	2	3	
46	1	3	7	2	4	5	
	2	4	7	1	3	4	
	3	5	5	1	1	3	
47	1	2	4	3	3	6	
	2	4	3	3	2	4	
	3	8	2	2	1	3	
48	1	1	8	5	9	7	
	2	4	8	4	8	4	
	3	7	8	4	8	3	
49	1	2	3	8	8	10	
	2	6	3	7	7	9	
	3	10	3	4	2	8	
50	1	1	4	10	9	7	
	2	3	2	8	8	7	
	3	7	1	5	7	4	
51	1	2	7	9	2	9	
	2	6	6	8	1	8	
	3	7	4	8	1	8	
52	1	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2
	29	28	216	249

************************************************************************
