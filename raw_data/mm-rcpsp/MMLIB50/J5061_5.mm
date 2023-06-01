jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	10		2 3 4 5 6 7 9 12 16 20 
2	3	7		33 21 19 18 17 11 8 
3	3	5		35 23 21 13 10 
4	3	3		17 14 13 
5	3	8		35 33 30 29 27 25 24 15 
6	3	4		35 24 23 10 
7	3	6		35 33 29 27 24 15 
8	3	5		30 27 24 23 15 
9	3	8		33 32 31 28 27 26 24 23 
10	3	8		41 34 33 31 29 28 27 22 
11	3	7		32 29 28 27 26 25 24 
12	3	6		32 31 30 29 27 21 
13	3	9		40 39 38 33 32 31 29 27 24 
14	3	6		40 39 32 31 30 23 
15	3	7		41 37 34 32 31 28 26 
16	3	5		40 39 31 27 23 
17	3	5		36 34 30 28 26 
18	3	8		50 41 40 39 38 36 34 27 
19	3	8		51 49 41 40 39 34 31 29 
20	3	5		46 41 40 29 28 
21	3	7		51 41 39 38 37 34 26 
22	3	8		51 48 46 40 39 38 32 30 
23	3	5		51 49 38 34 29 
24	3	6		51 49 45 41 37 34 
25	3	6		50 49 47 45 42 34 
26	3	8		50 49 48 46 45 44 43 40 
27	3	7		51 49 48 47 46 45 37 
28	3	7		51 49 45 44 43 39 38 
29	3	5		50 48 47 45 36 
30	3	5		49 47 45 44 37 
31	3	6		48 46 45 44 43 42 
32	3	5		50 47 45 43 42 
33	3	5		47 46 44 43 42 
34	3	4		48 46 44 43 
35	3	4		46 44 43 42 
36	3	2		44 37 
37	3	2		43 42 
38	3	2		47 42 
39	3	1		42 
40	3	1		42 
41	3	1		43 
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
jobnr.	mode	dur	R1	R2	N1	N2	
------------------------------------------------------------------------
1	1	0	0	0	0	0	
2	1	2	3	0	0	4	
	2	9	3	0	0	3	
	3	10	2	0	0	1	
3	1	3	0	7	5	0	
	2	7	0	5	0	7	
	3	10	0	5	3	0	
4	1	1	0	7	4	0	
	2	6	5	0	3	0	
	3	7	5	0	2	0	
5	1	1	6	0	5	0	
	2	5	0	4	0	4	
	3	5	0	2	1	0	
6	1	7	0	6	0	4	
	2	8	0	5	0	4	
	3	10	2	0	8	0	
7	1	5	0	7	0	9	
	2	7	0	6	8	0	
	3	9	6	0	0	4	
8	1	4	0	10	0	10	
	2	8	3	0	0	8	
	3	9	0	6	1	0	
9	1	7	6	0	0	7	
	2	9	0	6	6	0	
	3	10	4	0	0	5	
10	1	2	5	0	0	6	
	2	4	4	0	10	0	
	3	10	4	0	0	4	
11	1	3	8	0	0	5	
	2	10	0	6	0	2	
	3	10	6	0	0	1	
12	1	5	0	7	0	5	
	2	7	0	7	0	4	
	3	10	4	0	0	1	
13	1	2	8	0	0	9	
	2	3	7	0	2	0	
	3	10	4	0	0	5	
14	1	8	0	10	0	2	
	2	8	7	0	7	0	
	3	9	6	0	0	1	
15	1	1	0	5	0	5	
	2	6	0	3	0	4	
	3	9	0	3	0	3	
16	1	2	10	0	0	5	
	2	4	0	9	3	0	
	3	10	8	0	0	2	
17	1	3	0	6	0	4	
	2	5	0	5	0	2	
	3	9	0	2	0	2	
18	1	3	0	2	0	7	
	2	6	6	0	3	0	
	3	9	0	2	0	1	
19	1	2	10	0	6	0	
	2	6	10	0	5	0	
	3	10	10	0	4	0	
20	1	2	5	0	0	8	
	2	4	3	0	0	8	
	3	9	0	4	0	7	
21	1	3	0	7	0	10	
	2	4	1	0	1	0	
	3	8	1	0	0	9	
22	1	1	9	0	0	6	
	2	2	8	0	3	0	
	3	10	6	0	1	0	
23	1	4	1	0	0	10	
	2	7	0	4	0	7	
	3	10	1	0	6	0	
24	1	4	0	6	0	8	
	2	7	0	5	0	6	
	3	8	0	4	0	6	
25	1	2	6	0	0	5	
	2	2	0	4	0	4	
	3	6	0	3	0	3	
26	1	4	0	9	10	0	
	2	5	6	0	0	2	
	3	6	6	0	0	1	
27	1	3	7	0	0	7	
	2	8	5	0	0	3	
	3	10	0	3	4	0	
28	1	2	7	0	5	0	
	2	4	5	0	3	0	
	3	7	0	6	2	0	
29	1	5	0	4	0	9	
	2	5	7	0	0	7	
	3	7	7	0	0	6	
30	1	6	8	0	10	0	
	2	9	8	0	0	1	
	3	9	0	7	0	1	
31	1	4	0	3	3	0	
	2	6	0	2	0	6	
	3	7	0	1	0	6	
32	1	1	8	0	0	9	
	2	6	3	0	3	0	
	3	7	0	5	3	0	
33	1	4	0	8	5	0	
	2	9	7	0	4	0	
	3	10	5	0	3	0	
34	1	5	4	0	10	0	
	2	6	0	9	8	0	
	3	10	3	0	0	3	
35	1	1	0	3	0	4	
	2	2	0	3	9	0	
	3	7	0	3	7	0	
36	1	2	4	0	0	9	
	2	3	3	0	0	8	
	3	5	2	0	0	7	
37	1	1	0	6	0	5	
	2	3	0	5	0	4	
	3	4	6	0	5	0	
38	1	1	0	9	0	9	
	2	7	3	0	3	0	
	3	9	0	5	1	0	
39	1	1	10	0	1	0	
	2	5	0	7	1	0	
	3	9	8	0	0	9	
40	1	4	0	9	0	6	
	2	9	3	0	0	6	
	3	10	0	9	0	5	
41	1	1	10	0	0	6	
	2	3	0	5	0	5	
	3	7	4	0	2	0	
42	1	7	8	0	6	0	
	2	8	8	0	0	6	
	3	8	0	2	3	0	
43	1	3	10	0	0	1	
	2	4	8	0	3	0	
	3	9	5	0	0	1	
44	1	4	2	0	10	0	
	2	6	0	4	0	5	
	3	8	0	4	0	4	
45	1	2	0	6	10	0	
	2	3	9	0	8	0	
	3	10	0	4	0	1	
46	1	1	0	5	0	9	
	2	4	2	0	5	0	
	3	7	1	0	2	0	
47	1	3	0	5	0	6	
	2	4	0	3	0	3	
	3	9	0	2	0	1	
48	1	1	5	0	0	9	
	2	1	0	5	0	6	
	3	8	5	0	1	0	
49	1	6	0	5	6	0	
	2	6	6	0	5	0	
	3	10	0	4	4	0	
50	1	3	0	1	4	0	
	2	7	6	0	0	6	
	3	9	4	0	3	0	
51	1	5	6	0	0	9	
	2	6	6	0	0	4	
	3	9	4	0	0	3	
52	1	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2
	45	65	58	103

************************************************************************
