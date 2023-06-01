jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	4		2 3 4 9 
2	3	3		7 6 5 
3	3	3		8 7 5 
4	3	4		16 13 12 7 
5	3	6		19 16 14 13 12 11 
6	3	5		24 19 14 13 10 
7	3	4		19 15 14 10 
8	3	5		19 17 16 14 13 
9	3	5		24 19 14 13 12 
10	3	3		23 20 11 
11	3	4		22 21 18 17 
12	3	4		28 21 20 17 
13	3	5		27 25 23 22 18 
14	3	5		28 27 25 21 20 
15	3	5		31 28 27 24 20 
16	3	6		30 27 25 24 23 22 
17	3	6		32 31 30 27 26 25 
18	3	4		32 31 29 28 
19	3	4		32 31 30 25 
20	3	2		30 22 
21	3	5		34 33 32 30 29 
22	3	3		32 29 26 
23	3	5		41 40 36 35 34 
24	3	4		41 40 35 33 
25	3	2		41 29 
26	3	4		41 40 35 34 
27	3	3		40 35 33 
28	3	1		30 
29	3	3		40 36 35 
30	3	3		41 40 35 
31	3	2		45 33 
32	3	4		41 40 39 37 
33	3	4		51 39 37 36 
34	3	5		51 48 45 39 37 
35	3	4		51 39 38 37 
36	3	3		44 43 38 
37	3	5		50 49 47 44 42 
38	3	5		50 49 48 47 42 
39	3	4		49 47 43 42 
40	3	3		48 45 42 
41	3	3		51 48 46 
42	3	1		46 
43	3	1		46 
44	3	1		46 
45	3	1		47 
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
2	1	2	8	5	0	4	
	2	7	6	4	0	2	
	3	7	3	3	5	0	
3	1	2	4	9	0	10	
	2	7	4	8	0	10	
	3	9	4	6	0	10	
4	1	4	9	7	6	0	
	2	6	8	7	6	0	
	3	10	7	6	6	0	
5	1	2	6	9	0	5	
	2	8	3	8	0	4	
	3	9	1	7	0	3	
6	1	5	4	9	0	6	
	2	6	2	7	0	5	
	3	8	2	7	0	4	
7	1	6	9	3	10	0	
	2	7	7	3	10	0	
	3	10	6	2	0	3	
8	1	1	8	4	9	0	
	2	3	5	4	0	2	
	3	4	4	4	0	2	
9	1	1	7	10	0	1	
	2	5	7	8	4	0	
	3	8	5	8	0	1	
10	1	6	10	10	0	2	
	2	7	8	10	3	0	
	3	9	6	10	2	0	
11	1	4	6	5	0	7	
	2	5	5	5	3	0	
	3	7	2	3	0	5	
12	1	5	4	9	0	9	
	2	6	4	6	0	8	
	3	7	3	6	0	8	
13	1	5	2	8	4	0	
	2	9	2	7	3	0	
	3	9	1	7	0	1	
14	1	1	10	6	0	4	
	2	3	7	6	0	3	
	3	4	5	6	4	0	
15	1	1	6	8	8	0	
	2	4	6	5	5	0	
	3	7	6	5	4	0	
16	1	2	4	6	0	8	
	2	6	4	5	4	0	
	3	7	4	4	3	0	
17	1	3	9	8	7	0	
	2	7	7	6	6	0	
	3	9	4	6	6	0	
18	1	3	2	6	0	6	
	2	6	2	5	6	0	
	3	7	1	4	5	0	
19	1	1	4	6	6	0	
	2	3	3	5	5	0	
	3	5	3	4	3	0	
20	1	4	9	6	6	0	
	2	5	8	4	5	0	
	3	7	5	4	4	0	
21	1	4	6	6	0	8	
	2	5	6	6	0	7	
	3	8	6	3	0	6	
22	1	6	6	7	0	7	
	2	7	3	6	0	4	
	3	10	2	6	1	0	
23	1	7	5	9	0	8	
	2	7	5	7	9	0	
	3	10	4	4	0	6	
24	1	1	7	7	0	6	
	2	7	7	5	5	0	
	3	8	7	3	3	0	
25	1	2	8	6	9	0	
	2	3	7	6	7	0	
	3	5	5	6	7	0	
26	1	1	9	5	0	10	
	2	4	7	4	7	0	
	3	6	5	4	7	0	
27	1	3	4	9	0	9	
	2	4	2	9	3	0	
	3	6	2	9	2	0	
28	1	5	3	5	9	0	
	2	9	2	5	4	0	
	3	9	1	4	0	1	
29	1	1	6	2	0	6	
	2	2	3	2	0	6	
	3	2	2	2	2	0	
30	1	7	5	6	10	0	
	2	7	3	2	0	5	
	3	10	3	1	0	3	
31	1	5	10	5	9	0	
	2	6	9	4	0	4	
	3	8	8	4	0	4	
32	1	1	8	8	7	0	
	2	3	8	7	0	6	
	3	6	8	6	6	0	
33	1	2	6	4	0	5	
	2	3	6	2	7	0	
	3	10	6	2	0	5	
34	1	3	7	3	9	0	
	2	5	7	3	8	0	
	3	8	1	1	0	4	
35	1	1	2	9	4	0	
	2	3	1	8	0	2	
	3	7	1	8	3	0	
36	1	6	9	6	8	0	
	2	8	9	4	0	10	
	3	10	9	3	0	10	
37	1	1	9	9	0	3	
	2	7	6	4	0	4	
	3	8	6	4	0	3	
38	1	6	1	5	9	0	
	2	9	1	4	0	5	
	3	10	1	3	0	5	
39	1	3	5	9	0	8	
	2	5	4	8	0	3	
	3	9	4	8	4	0	
40	1	2	6	3	9	0	
	2	5	4	2	0	7	
	3	7	1	1	0	5	
41	1	1	6	8	7	0	
	2	3	5	5	0	6	
	3	5	2	1	5	0	
42	1	2	9	10	0	4	
	2	3	6	6	0	4	
	3	7	1	5	0	4	
43	1	5	6	7	7	0	
	2	6	6	7	6	0	
	3	9	6	5	0	2	
44	1	5	4	10	7	0	
	2	9	3	5	0	6	
	3	9	3	1	7	0	
45	1	2	10	7	0	8	
	2	9	9	3	2	0	
	3	10	9	2	2	0	
46	1	1	9	9	6	0	
	2	4	6	8	0	8	
	3	7	2	8	0	5	
47	1	1	10	6	0	5	
	2	7	10	6	7	0	
	3	8	10	6	6	0	
48	1	5	9	8	0	1	
	2	9	8	8	0	1	
	3	10	6	7	0	1	
49	1	5	10	2	0	1	
	2	9	9	2	7	0	
	3	10	9	1	0	1	
50	1	7	4	9	0	10	
	2	8	4	8	0	8	
	3	9	4	6	8	0	
51	1	1	7	9	0	9	
	2	6	5	8	9	0	
	3	6	2	7	0	1	
52	1	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2
	29	26	207	190

************************************************************************
