jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	6		2 3 4 5 6 8 
2	3	6		18 14 11 10 9 7 
3	3	5		16 15 14 11 9 
4	3	5		20 19 18 16 13 
5	3	4		20 16 13 12 
6	3	3		21 15 10 
7	3	4		20 19 16 13 
8	3	4		19 16 15 13 
9	3	6		28 22 21 20 19 17 
10	3	4		23 19 17 16 
11	3	3		28 19 13 
12	3	4		28 22 21 19 
13	3	6		27 26 24 23 22 21 
14	3	4		28 23 22 20 
15	3	4		28 27 26 23 
16	3	3		28 26 22 
17	3	4		36 29 27 24 
18	3	5		36 35 29 26 25 
19	3	4		37 36 27 24 
20	3	4		36 35 26 25 
21	3	4		37 36 35 25 
22	3	3		36 35 25 
23	3	3		36 35 25 
24	3	2		35 25 
25	3	3		32 31 30 
26	3	4		37 34 33 32 
27	3	2		35 30 
28	3	2		36 29 
29	3	4		39 38 37 34 
30	3	2		34 33 
31	3	2		40 33 
32	3	4		45 42 40 39 
33	3	3		45 39 38 
34	3	4		45 42 41 40 
35	3	4		45 42 41 40 
36	3	6		51 50 45 44 43 42 
37	3	6		51 50 45 44 43 42 
38	3	4		51 44 43 42 
39	3	3		44 43 41 
40	3	4		51 50 44 43 
41	3	4		50 49 48 46 
42	3	2		48 46 
43	3	2		49 47 
44	3	2		48 47 
45	3	2		48 47 
46	3	1		47 
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
2	1	4	3	8	9	0	
	2	6	2	6	9	0	
	3	7	1	6	9	0	
3	1	2	9	4	7	0	
	2	5	9	3	0	8	
	3	6	9	3	0	7	
4	1	3	7	8	0	7	
	2	4	6	6	0	6	
	3	10	6	5	0	2	
5	1	3	5	3	0	6	
	2	4	5	3	0	5	
	3	8	3	2	2	0	
6	1	3	5	7	8	0	
	2	5	5	4	3	0	
	3	8	3	3	0	8	
7	1	3	4	5	8	0	
	2	7	4	3	0	7	
	3	9	4	2	2	0	
8	1	1	7	7	9	0	
	2	4	6	7	5	0	
	3	8	4	7	3	0	
9	1	3	7	10	0	3	
	2	8	5	7	6	0	
	3	8	5	4	0	2	
10	1	5	6	2	7	0	
	2	6	5	1	6	0	
	3	7	4	1	6	0	
11	1	5	4	6	0	7	
	2	7	3	3	3	0	
	3	8	3	2	0	6	
12	1	2	8	5	0	2	
	2	8	4	4	0	2	
	3	10	3	3	0	2	
13	1	1	7	2	8	0	
	2	3	6	1	0	6	
	3	7	5	1	0	6	
14	1	1	6	9	8	0	
	2	2	6	7	8	0	
	3	7	2	7	0	3	
15	1	3	3	9	0	5	
	2	7	3	8	6	0	
	3	8	3	7	5	0	
16	1	4	9	10	6	0	
	2	7	9	10	5	0	
	3	9	8	10	3	0	
17	1	1	9	7	0	3	
	2	5	9	7	3	0	
	3	10	8	7	3	0	
18	1	3	8	9	5	0	
	2	6	5	6	0	1	
	3	8	2	2	4	0	
19	1	3	4	5	8	0	
	2	4	2	4	8	0	
	3	8	2	3	0	8	
20	1	8	3	5	0	9	
	2	8	3	4	6	0	
	3	10	3	3	0	3	
21	1	7	8	8	0	8	
	2	8	7	5	0	7	
	3	9	7	4	0	7	
22	1	2	6	5	9	0	
	2	5	5	5	8	0	
	3	9	4	5	5	0	
23	1	4	3	2	0	8	
	2	5	3	1	0	7	
	3	8	3	1	0	5	
24	1	2	8	10	0	8	
	2	3	3	7	0	6	
	3	4	2	3	3	0	
25	1	1	8	9	8	0	
	2	2	7	9	8	0	
	3	10	6	9	0	1	
26	1	4	3	6	4	0	
	2	5	2	4	2	0	
	3	7	2	3	0	4	
27	1	1	9	4	10	0	
	2	2	7	4	0	5	
	3	8	7	3	0	2	
28	1	5	7	9	0	10	
	2	10	6	8	0	9	
	3	10	5	6	5	0	
29	1	1	9	7	0	6	
	2	4	8	6	0	5	
	3	9	8	6	2	0	
30	1	4	4	8	4	0	
	2	6	2	6	3	0	
	3	8	2	5	3	0	
31	1	3	6	1	9	0	
	2	6	4	1	0	8	
	3	7	4	1	0	7	
32	1	2	5	2	8	0	
	2	4	5	1	0	4	
	3	5	3	1	4	0	
33	1	7	10	8	0	9	
	2	8	10	5	6	0	
	3	8	10	4	0	8	
34	1	1	7	3	3	0	
	2	3	3	2	2	0	
	3	8	2	2	2	0	
35	1	1	7	6	4	0	
	2	3	5	6	0	7	
	3	9	4	6	4	0	
36	1	4	4	7	7	0	
	2	7	2	7	4	0	
	3	7	2	6	0	5	
37	1	1	4	8	7	0	
	2	4	3	6	4	0	
	3	10	3	6	3	0	
38	1	2	6	5	5	0	
	2	3	6	5	4	0	
	3	7	3	5	3	0	
39	1	2	9	10	0	10	
	2	2	6	10	7	0	
	3	5	6	10	6	0	
40	1	4	9	5	0	4	
	2	8	6	3	0	3	
	3	10	4	1	0	3	
41	1	1	5	3	0	9	
	2	3	3	2	0	8	
	3	10	3	1	0	6	
42	1	1	7	9	0	9	
	2	2	4	7	2	0	
	3	9	3	6	0	8	
43	1	1	1	8	0	7	
	2	5	1	8	0	6	
	3	7	1	6	0	7	
44	1	2	8	9	0	7	
	2	4	7	7	0	6	
	3	10	7	6	4	0	
45	1	1	7	9	3	0	
	2	9	5	6	3	0	
	3	10	4	5	0	4	
46	1	7	8	5	5	0	
	2	8	5	5	3	0	
	3	9	3	4	0	8	
47	1	4	7	5	0	2	
	2	6	4	4	0	2	
	3	7	4	4	0	1	
48	1	2	4	4	0	10	
	2	4	4	2	7	0	
	3	6	4	2	0	9	
49	1	2	8	8	10	0	
	2	4	7	6	8	0	
	3	10	2	2	6	0	
50	1	1	10	5	8	0	
	2	4	8	2	0	2	
	3	9	6	1	0	2	
51	1	7	5	5	0	10	
	2	8	3	4	0	10	
	3	9	3	4	0	9	
52	1	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2
	19	21	145	146

************************************************************************
