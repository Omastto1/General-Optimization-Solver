jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	6		2 3 5 6 9 10 
2	3	5		14 13 12 11 8 
3	3	3		11 7 4 
4	3	3		18 16 12 
5	3	3		16 13 11 
6	3	3		27 16 11 
7	3	5		27 25 22 17 16 
8	3	4		22 18 17 16 
9	3	3		19 18 15 
10	3	5		27 25 22 20 16 
11	3	5		30 22 21 19 18 
12	3	2		25 15 
13	3	5		30 27 22 19 18 
14	3	4		30 27 21 18 
15	3	5		30 27 22 21 20 
16	3	3		30 21 19 
17	3	3		26 21 20 
18	3	4		31 26 25 20 
19	3	4		31 29 28 26 
20	3	3		33 29 23 
21	3	3		31 24 23 
22	3	3		37 31 26 
23	3	3		39 34 28 
24	3	3		39 34 28 
25	3	3		39 34 28 
26	3	4		36 34 33 32 
27	3	4		36 34 33 32 
28	3	3		37 36 32 
29	3	3		37 34 32 
30	3	5		43 40 37 36 35 
31	3	5		44 43 40 38 36 
32	3	4		44 43 40 35 
33	3	3		40 39 35 
34	3	3		41 40 38 
35	3	3		51 41 38 
36	3	4		51 49 48 41 
37	3	4		50 47 45 42 
38	3	3		47 45 42 
39	3	5		50 49 48 47 46 
40	3	4		51 47 46 45 
41	3	3		50 47 46 
42	3	2		49 46 
43	3	2		48 46 
44	3	2		47 46 
45	3	1		48 
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
2	1	1	10	0	9	0	
	2	5	10	0	7	0	
	3	8	0	4	6	0	
3	1	2	0	3	7	0	
	2	7	5	0	6	0	
	3	8	2	0	0	2	
4	1	2	8	0	3	0	
	2	2	7	0	0	6	
	3	3	6	0	2	0	
5	1	1	7	0	0	8	
	2	4	6	0	0	6	
	3	7	5	0	0	6	
6	1	5	0	6	2	0	
	2	7	0	2	2	0	
	3	9	7	0	0	1	
7	1	2	0	6	0	8	
	2	2	6	0	1	0	
	3	7	4	0	0	8	
8	1	3	8	0	6	0	
	2	10	0	5	0	5	
	3	10	5	0	0	5	
9	1	4	0	6	6	0	
	2	6	0	4	5	0	
	3	6	5	0	4	0	
10	1	3	9	0	0	9	
	2	6	0	5	0	6	
	3	7	0	4	1	0	
11	1	2	7	0	7	0	
	2	4	7	0	6	0	
	3	5	0	8	6	0	
12	1	1	0	9	4	0	
	2	6	0	7	0	8	
	3	7	4	0	0	6	
13	1	1	9	0	8	0	
	2	6	0	4	7	0	
	3	10	0	4	6	0	
14	1	2	0	4	0	6	
	2	6	0	3	8	0	
	3	8	2	0	8	0	
15	1	4	0	9	6	0	
	2	6	0	8	0	4	
	3	7	0	7	0	3	
16	1	1	0	3	0	3	
	2	6	0	1	3	0	
	3	10	0	1	0	1	
17	1	3	0	4	10	0	
	2	4	3	0	0	8	
	3	8	0	4	3	0	
18	1	2	0	10	0	8	
	2	4	5	0	1	0	
	3	9	0	9	0	7	
19	1	4	4	0	0	3	
	2	5	0	5	0	2	
	3	6	0	5	4	0	
20	1	1	8	0	0	10	
	2	5	0	7	8	0	
	3	7	0	6	4	0	
21	1	1	0	6	7	0	
	2	2	0	3	3	0	
	3	3	0	2	1	0	
22	1	3	0	6	0	7	
	2	6	0	3	0	6	
	3	9	0	2	0	5	
23	1	1	10	0	7	0	
	2	3	6	0	7	0	
	3	4	4	0	4	0	
24	1	1	0	4	0	5	
	2	2	0	3	0	2	
	3	5	2	0	3	0	
25	1	4	0	7	0	9	
	2	6	5	0	1	0	
	3	9	0	7	0	6	
26	1	7	5	0	0	6	
	2	9	0	6	0	5	
	3	10	5	0	5	0	
27	1	5	0	7	8	0	
	2	8	3	0	7	0	
	3	9	0	3	0	1	
28	1	1	8	0	0	7	
	2	3	8	0	5	0	
	3	3	0	5	3	0	
29	1	6	10	0	3	0	
	2	6	0	5	0	8	
	3	9	0	2	1	0	
30	1	1	5	0	8	0	
	2	2	0	7	8	0	
	3	10	0	7	7	0	
31	1	4	5	0	0	2	
	2	5	5	0	0	1	
	3	9	0	6	4	0	
32	1	5	0	8	3	0	
	2	6	6	0	1	0	
	3	6	0	4	1	0	
33	1	3	9	0	7	0	
	2	4	0	3	0	5	
	3	9	6	0	0	1	
34	1	3	10	0	0	6	
	2	5	9	0	7	0	
	3	6	9	0	6	0	
35	1	4	4	0	10	0	
	2	6	4	0	0	4	
	3	8	0	3	5	0	
36	1	2	9	0	1	0	
	2	5	9	0	0	5	
	3	9	0	6	0	3	
37	1	2	0	8	0	5	
	2	2	0	6	1	0	
	3	9	0	5	0	1	
38	1	1	0	7	0	10	
	2	4	6	0	0	7	
	3	9	0	3	0	3	
39	1	2	0	6	8	0	
	2	8	0	6	7	0	
	3	8	0	6	0	2	
40	1	1	0	8	0	9	
	2	6	0	8	0	7	
	3	10	4	0	0	6	
41	1	4	0	9	10	0	
	2	4	7	0	9	0	
	3	10	6	0	0	1	
42	1	1	6	0	0	3	
	2	8	5	0	0	3	
	3	10	0	4	0	2	
43	1	2	0	9	2	0	
	2	6	0	8	0	8	
	3	8	0	7	2	0	
44	1	1	8	0	0	5	
	2	2	0	9	0	4	
	3	4	0	9	0	3	
45	1	1	0	3	8	0	
	2	5	0	1	7	0	
	3	9	2	0	7	0	
46	1	4	3	0	2	0	
	2	8	0	3	2	0	
	3	9	0	1	1	0	
47	1	5	8	0	8	0	
	2	9	8	0	0	3	
	3	10	8	0	0	2	
48	1	1	0	10	8	0	
	2	6	0	8	0	3	
	3	8	6	0	0	3	
49	1	1	8	0	3	0	
	2	2	0	8	0	3	
	3	6	3	0	2	0	
50	1	4	0	10	8	0	
	2	5	8	0	4	0	
	3	9	0	5	2	0	
51	1	6	6	0	0	10	
	2	6	4	0	5	0	
	3	6	0	3	3	0	
52	1	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2
	36	33	91	72

************************************************************************
