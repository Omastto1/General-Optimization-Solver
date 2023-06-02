jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 4 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	12		2 3 4 5 6 7 8 9 10 11 12 16 
2	3	8		33 31 27 21 19 18 14 13 
3	3	8		34 31 27 25 22 17 15 14 
4	3	9		33 31 25 24 23 22 21 18 17 
5	3	7		31 27 25 24 22 18 13 
6	3	6		27 25 24 22 18 13 
7	3	8		34 33 31 30 26 25 23 22 
8	3	6		39 31 30 27 23 18 
9	3	4		26 25 24 13 
10	3	8		39 37 34 31 28 26 25 24 
11	3	5		36 34 27 26 23 
12	3	8		51 38 35 34 30 29 28 27 
13	3	6		51 39 37 34 30 23 
14	3	3		37 26 20 
15	3	5		51 38 37 30 23 
16	3	9		51 50 42 38 37 34 33 32 30 
17	3	8		51 50 42 39 38 35 30 29 
18	3	7		42 37 36 35 34 32 26 
19	3	4		38 34 30 22 
20	3	4		51 30 28 24 
21	3	6		51 50 48 42 32 30 
22	3	7		51 50 46 41 37 35 32 
23	3	4		46 35 29 28 
24	3	6		50 42 41 36 35 32 
25	3	9		51 50 49 47 45 43 41 40 36 
26	3	8		51 50 47 46 45 41 40 38 
27	3	4		46 41 37 32 
28	3	7		50 49 48 47 44 43 42 
29	3	2		48 32 
30	3	4		47 43 41 36 
31	3	5		51 47 46 45 43 
32	3	4		49 45 43 40 
33	3	3		46 44 35 
34	3	4		48 45 44 43 
35	3	2		47 43 
36	3	2		46 44 
37	3	2		48 45 
38	3	2		48 43 
39	3	2		46 45 
40	3	1		44 
41	3	1		44 
42	3	1		45 
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
2	1	1	3	9	8	3	7	5	
	2	3	3	7	7	3	6	4	
	3	7	2	4	4	3	4	3	
3	1	1	6	10	6	8	3	10	
	2	2	5	10	4	6	3	10	
	3	10	5	10	4	3	3	10	
4	1	1	8	7	7	6	4	8	
	2	4	8	7	6	5	3	4	
	3	10	6	7	4	4	3	3	
5	1	5	8	8	10	9	8	6	
	2	6	4	5	10	7	4	5	
	3	9	3	3	10	5	2	5	
6	1	2	9	6	8	8	4	5	
	2	5	7	6	7	6	3	5	
	3	9	4	5	3	6	2	5	
7	1	1	8	5	8	10	9	9	
	2	6	7	5	8	6	9	4	
	3	8	2	4	7	5	9	2	
8	1	7	4	5	9	7	8	6	
	2	8	4	4	4	6	7	3	
	3	10	4	4	1	3	7	3	
9	1	7	8	5	10	6	10	5	
	2	9	7	2	10	6	9	5	
	3	10	7	1	10	6	9	5	
10	1	3	7	7	8	5	6	7	
	2	4	7	4	7	4	5	4	
	3	8	7	3	7	3	5	2	
11	1	4	6	6	7	5	6	6	
	2	8	6	5	5	3	2	6	
	3	9	5	4	5	2	2	5	
12	1	1	9	6	4	1	6	3	
	2	8	6	3	3	1	4	2	
	3	10	5	2	2	1	3	2	
13	1	5	7	10	2	10	5	10	
	2	7	5	7	1	9	5	5	
	3	8	4	6	1	9	5	3	
14	1	4	9	5	6	9	7	5	
	2	5	4	4	5	8	5	4	
	3	6	3	2	5	5	4	1	
15	1	5	8	7	7	8	6	8	
	2	9	7	4	7	8	5	6	
	3	10	7	3	7	6	1	6	
16	1	4	6	8	2	6	6	4	
	2	5	4	6	1	4	4	3	
	3	6	3	5	1	2	3	1	
17	1	8	7	9	9	3	9	4	
	2	9	7	8	7	1	7	2	
	3	10	5	8	7	1	5	2	
18	1	2	6	6	9	6	6	2	
	2	3	5	6	6	6	6	1	
	3	8	1	6	4	6	5	1	
19	1	1	6	5	6	5	3	4	
	2	6	4	4	5	5	1	4	
	3	7	4	3	5	4	1	2	
20	1	2	5	6	2	5	9	4	
	2	4	5	6	2	3	6	4	
	3	5	5	4	2	3	4	4	
21	1	4	6	6	6	7	5	9	
	2	8	4	4	3	7	2	8	
	3	9	1	4	2	7	2	8	
22	1	2	8	9	8	1	4	8	
	2	5	8	6	6	1	2	7	
	3	6	8	5	6	1	2	4	
23	1	2	1	4	9	3	3	3	
	2	5	1	3	6	2	3	2	
	3	7	1	3	5	2	2	2	
24	1	3	3	7	10	4	9	3	
	2	7	2	5	8	4	8	2	
	3	9	1	4	8	4	8	2	
25	1	3	8	4	9	6	8	8	
	2	4	8	3	6	4	8	7	
	3	9	5	3	4	4	8	6	
26	1	7	5	8	5	7	10	7	
	2	8	5	8	4	6	9	6	
	3	10	5	8	3	6	8	6	
27	1	1	5	8	8	7	9	6	
	2	2	4	6	8	5	3	3	
	3	9	1	5	8	3	3	2	
28	1	2	8	6	2	10	9	9	
	2	5	8	6	1	8	6	6	
	3	6	8	5	1	8	5	6	
29	1	2	2	2	9	2	4	2	
	2	3	1	2	8	2	4	2	
	3	6	1	2	6	1	4	2	
30	1	1	6	7	6	9	7	4	
	2	3	6	4	5	6	5	3	
	3	10	5	4	4	2	3	2	
31	1	2	2	9	10	3	7	7	
	2	3	2	6	9	2	7	5	
	3	6	2	6	9	2	4	2	
32	1	6	8	10	6	1	9	7	
	2	7	8	6	4	1	9	5	
	3	8	8	6	2	1	8	1	
33	1	3	3	8	9	7	5	7	
	2	4	1	6	8	4	5	6	
	3	10	1	6	7	3	4	5	
34	1	1	10	7	7	1	3	6	
	2	3	7	7	6	1	2	3	
	3	9	2	7	5	1	2	2	
35	1	5	4	5	6	3	2	10	
	2	7	4	5	6	3	1	9	
	3	8	2	5	6	3	1	7	
36	1	2	4	6	6	3	6	6	
	2	4	4	4	2	2	6	5	
	3	10	4	3	2	2	5	5	
37	1	1	5	6	8	6	7	7	
	2	4	4	6	7	5	6	5	
	3	10	4	4	2	4	4	4	
38	1	3	8	7	6	7	7	7	
	2	4	6	6	4	5	5	6	
	3	5	4	6	4	4	4	6	
39	1	4	5	9	7	4	6	5	
	2	8	4	8	6	3	6	4	
	3	10	3	6	5	3	6	4	
40	1	8	2	5	5	4	9	7	
	2	9	2	3	5	2	7	6	
	3	10	1	3	5	1	5	4	
41	1	4	5	4	9	7	7	10	
	2	8	5	4	8	3	7	6	
	3	9	5	4	8	3	5	2	
42	1	3	7	8	7	4	6	7	
	2	6	7	8	6	2	6	6	
	3	9	7	8	4	1	6	6	
43	1	3	4	5	7	4	6	7	
	2	4	4	3	3	3	6	4	
	3	10	4	2	3	3	3	4	
44	1	2	8	6	9	10	6	5	
	2	8	6	3	7	10	6	4	
	3	10	5	3	4	10	6	4	
45	1	2	5	6	8	6	9	4	
	2	4	5	4	8	6	8	2	
	3	5	5	2	8	6	6	2	
46	1	5	6	1	7	6	8	5	
	2	9	4	1	5	5	6	5	
	3	10	2	1	2	4	4	4	
47	1	1	6	3	10	6	8	3	
	2	2	5	3	9	5	7	3	
	3	5	5	2	9	2	6	3	
48	1	1	5	4	4	10	6	5	
	2	4	4	4	3	8	5	5	
	3	6	4	4	3	8	5	4	
49	1	2	8	9	10	7	6	4	
	2	4	5	4	9	6	5	4	
	3	8	3	2	8	5	4	4	
50	1	6	5	4	7	7	10	8	
	2	7	3	3	6	3	9	8	
	3	8	1	2	6	2	7	8	
51	1	3	5	9	4	9	8	4	
	2	5	3	8	3	6	7	4	
	3	9	1	5	3	5	7	4	
52	1	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2	N 3	N 4
	64	64	324	265	304	273

************************************************************************
