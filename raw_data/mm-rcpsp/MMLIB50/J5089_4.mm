jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	4		2 5 6 8 
2	3	2		7 3 
3	3	1		4 
4	3	5		18 13 12 10 9 
5	3	2		14 9 
6	3	4		23 20 18 12 
7	3	3		17 14 11 
8	3	3		18 17 13 
9	3	2		17 11 
10	3	4		20 17 15 14 
11	3	5		23 21 20 19 15 
12	3	3		19 17 15 
13	3	4		23 20 19 15 
14	3	3		27 22 19 
15	3	2		22 16 
16	3	6		33 30 29 27 26 24 
17	3	4		27 26 25 21 
18	3	4		34 31 25 22 
19	3	5		33 31 30 28 24 
20	3	4		33 30 29 26 
21	3	7		42 36 33 32 31 30 29 
22	3	3		33 29 26 
23	3	5		42 34 33 30 28 
24	3	7		42 40 39 38 35 34 32 
25	3	6		42 41 40 36 33 29 
26	3	2		42 28 
27	3	6		46 42 40 36 35 31 
28	3	5		40 39 38 36 32 
29	3	5		46 39 38 37 35 
30	3	3		41 40 38 
31	3	4		45 39 38 37 
32	3	3		43 41 37 
33	3	3		46 39 37 
34	3	2		45 36 
35	3	4		51 45 44 43 
36	3	1		37 
37	3	5		51 50 49 47 44 
38	3	5		51 50 49 47 44 
39	3	3		51 48 43 
40	3	2		44 43 
41	3	2		46 45 
42	3	1		43 
43	3	3		50 49 47 
44	3	1		48 
45	3	1		47 
46	3	1		48 
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
2	1	4	0	7	0	2	
	2	4	7	0	6	0	
	3	5	6	0	4	0	
3	1	1	0	6	0	5	
	2	2	0	5	8	0	
	3	8	8	0	6	0	
4	1	4	0	8	0	3	
	2	8	1	0	5	0	
	3	10	1	0	4	0	
5	1	6	9	0	0	8	
	2	8	0	4	0	5	
	3	9	0	3	0	4	
6	1	4	4	0	9	0	
	2	6	2	0	0	7	
	3	10	0	2	0	7	
7	1	1	5	0	0	7	
	2	2	5	0	0	6	
	3	3	5	0	0	5	
8	1	6	0	8	2	0	
	2	9	0	8	1	0	
	3	10	0	7	1	0	
9	1	4	8	0	9	0	
	2	4	0	3	5	0	
	3	5	0	3	4	0	
10	1	2	0	5	7	0	
	2	3	0	5	6	0	
	3	9	2	0	7	0	
11	1	1	0	5	3	0	
	2	2	0	5	2	0	
	3	6	8	0	0	3	
12	1	4	4	0	2	0	
	2	4	0	9	2	0	
	3	9	0	9	0	7	
13	1	3	7	0	10	0	
	2	4	0	6	9	0	
	3	5	0	6	8	0	
14	1	2	0	4	8	0	
	2	5	6	0	0	4	
	3	6	0	3	5	0	
15	1	3	0	10	8	0	
	2	5	0	7	5	0	
	3	8	5	0	4	0	
16	1	3	0	9	4	0	
	2	7	7	0	0	1	
	3	9	0	6	4	0	
17	1	1	0	10	0	5	
	2	6	4	0	3	0	
	3	8	0	9	2	0	
18	1	4	0	6	0	5	
	2	8	0	5	7	0	
	3	9	0	4	6	0	
19	1	1	0	3	0	6	
	2	4	0	3	0	5	
	3	4	4	0	8	0	
20	1	1	0	4	6	0	
	2	3	0	4	0	8	
	3	8	4	0	0	8	
21	1	1	0	6	0	5	
	2	3	6	0	0	4	
	3	9	0	4	0	4	
22	1	4	0	5	8	0	
	2	7	3	0	7	0	
	3	10	0	1	0	8	
23	1	5	0	4	0	7	
	2	6	5	0	6	0	
	3	8	5	0	5	0	
24	1	3	0	8	0	3	
	2	4	3	0	8	0	
	3	6	0	6	0	1	
25	1	7	10	0	8	0	
	2	7	10	0	0	4	
	3	8	10	0	0	2	
26	1	8	0	6	0	9	
	2	10	7	0	8	0	
	3	10	0	1	5	0	
27	1	6	0	7	7	0	
	2	9	5	0	5	0	
	3	9	0	6	4	0	
28	1	3	0	7	0	2	
	2	6	0	6	0	2	
	3	7	0	6	0	1	
29	1	1	5	0	10	0	
	2	2	3	0	0	6	
	3	6	0	5	8	0	
30	1	1	2	0	6	0	
	2	6	0	2	0	8	
	3	8	1	0	4	0	
31	1	2	0	8	10	0	
	2	4	4	0	0	3	
	3	8	0	8	0	2	
32	1	2	0	6	10	0	
	2	7	6	0	0	8	
	3	10	6	0	10	0	
33	1	5	3	0	7	0	
	2	10	0	7	6	0	
	3	10	0	6	0	4	
34	1	7	0	8	4	0	
	2	10	8	0	0	8	
	3	10	0	6	2	0	
35	1	1	5	0	0	9	
	2	6	4	0	0	9	
	3	7	4	0	8	0	
36	1	3	7	0	0	8	
	2	9	0	7	8	0	
	3	9	0	6	0	8	
37	1	2	4	0	9	0	
	2	8	3	0	9	0	
	3	9	2	0	9	0	
38	1	2	0	8	0	8	
	2	6	4	0	7	0	
	3	9	3	0	7	0	
39	1	4	0	8	4	0	
	2	5	9	0	0	3	
	3	10	0	5	2	0	
40	1	1	0	9	8	0	
	2	5	3	0	0	6	
	3	8	0	9	3	0	
41	1	3	0	9	0	5	
	2	4	0	6	0	4	
	3	8	0	6	0	3	
42	1	3	0	5	4	0	
	2	6	3	0	3	0	
	3	8	2	0	0	2	
43	1	1	4	0	0	9	
	2	4	4	0	7	0	
	3	9	2	0	7	0	
44	1	3	6	0	9	0	
	2	7	0	4	9	0	
	3	8	0	3	9	0	
45	1	4	4	0	7	0	
	2	6	0	3	7	0	
	3	9	3	0	0	3	
46	1	6	3	0	0	4	
	2	7	3	0	0	3	
	3	10	0	6	0	1	
47	1	2	8	0	3	0	
	2	8	8	0	2	0	
	3	9	6	0	0	6	
48	1	2	0	9	9	0	
	2	6	0	8	9	0	
	3	8	0	8	0	7	
49	1	3	8	0	7	0	
	2	5	0	8	4	0	
	3	8	0	5	2	0	
50	1	3	0	6	7	0	
	2	6	4	0	0	9	
	3	6	4	0	4	0	
51	1	2	0	3	5	0	
	2	9	0	3	0	3	
	3	9	3	0	3	0	
52	1	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2
	22	24	236	175

************************************************************************
