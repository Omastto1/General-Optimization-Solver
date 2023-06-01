jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	7		2 3 5 6 14 15 18 
2	6	3		10 7 4 
3	6	4		19 13 12 9 
4	6	3		13 12 8 
5	6	2		12 8 
6	6	2		12 8 
7	6	2		19 11 
8	6	4		29 24 17 16 
9	6	4		29 24 17 16 
10	6	4		29 24 17 16 
11	6	5		29 24 21 20 17 
12	6	3		24 17 16 
13	6	3		24 17 16 
14	6	5		29 24 22 21 20 
15	6	4		29 24 21 20 
16	6	3		22 21 20 
17	6	5		30 28 26 23 22 
18	6	5		30 28 26 23 22 
19	6	1		20 
20	6	6		35 30 28 27 26 23 
21	6	7		40 35 33 30 28 27 25 
22	6	7		40 37 36 35 33 31 27 
23	6	6		40 37 33 32 31 25 
24	6	6		36 35 34 33 32 30 
25	6	4		41 39 36 34 
26	6	4		39 38 37 32 
27	6	3		39 34 32 
28	6	3		36 32 31 
29	6	3		37 34 31 
30	6	4		42 39 38 37 
31	6	3		42 41 38 
32	6	4		46 43 42 41 
33	6	3		44 39 38 
34	6	2		42 38 
35	6	3		44 41 39 
36	6	2		44 38 
37	6	6		51 50 49 46 45 44 
38	6	4		50 46 45 43 
39	6	6		51 50 49 48 46 45 
40	6	3		46 45 43 
41	6	4		50 49 48 45 
42	6	3		50 45 44 
43	6	4		51 49 48 47 
44	6	2		48 47 
45	6	1		47 
46	6	1		47 
47	6	1		52 
48	6	1		52 
49	6	1		52 
50	6	1		52 
51	6	1		52 
52	1	0		
************************************************************************
REQUESTS/DURATIONS
jobnr.	mode	dur	R1	R2	N1	N2	
------------------------------------------------------------------------
1	1	0	0	0	0	0	
2	1	1	18	17	8	19	
	2	6	13	13	7	18	
	3	9	13	12	6	17	
	4	11	9	8	5	17	
	5	14	4	8	4	16	
	6	16	1	4	3	15	
3	1	2	14	14	20	18	
	2	9	12	13	14	18	
	3	10	8	12	13	18	
	4	11	7	11	9	17	
	5	19	4	9	6	17	
	6	20	3	8	4	17	
4	1	1	14	5	12	5	
	2	4	12	5	12	5	
	3	6	10	4	12	5	
	4	11	7	3	12	5	
	5	13	5	2	12	4	
	6	18	4	2	12	4	
5	1	2	8	14	12	19	
	2	3	6	12	12	19	
	3	6	6	11	12	19	
	4	12	6	9	12	18	
	5	13	5	7	12	18	
	6	16	4	7	12	18	
6	1	10	15	16	12	15	
	2	11	14	14	11	14	
	3	13	10	12	11	13	
	4	14	8	9	10	10	
	5	17	7	7	10	10	
	6	18	6	6	10	7	
7	1	6	8	10	19	15	
	2	8	8	9	17	14	
	3	10	7	8	16	10	
	4	14	6	8	15	7	
	5	15	4	6	14	4	
	6	16	4	6	14	3	
8	1	3	11	10	18	20	
	2	5	11	8	14	20	
	3	6	11	7	14	20	
	4	8	11	5	12	20	
	5	10	11	3	8	20	
	6	19	11	1	7	20	
9	1	5	19	14	16	11	
	2	13	14	14	16	11	
	3	14	14	13	16	11	
	4	15	11	13	16	11	
	5	18	7	11	16	10	
	6	19	4	11	16	10	
10	1	1	8	12	9	17	
	2	2	7	12	8	17	
	3	4	7	12	8	15	
	4	5	7	12	8	14	
	5	9	6	12	8	13	
	6	10	5	12	8	13	
11	1	1	15	9	13	18	
	2	2	11	9	13	14	
	3	3	9	9	13	14	
	4	4	9	9	13	12	
	5	8	7	9	13	9	
	6	13	5	9	13	8	
12	1	9	14	12	17	8	
	2	10	13	12	17	8	
	3	11	13	12	16	8	
	4	12	11	12	14	7	
	5	13	10	12	14	6	
	6	14	10	12	13	6	
13	1	8	8	19	14	6	
	2	13	7	15	12	6	
	3	14	6	13	11	5	
	4	15	6	8	11	3	
	5	19	4	6	10	3	
	6	20	4	4	9	2	
14	1	1	19	15	13	7	
	2	3	15	14	12	5	
	3	4	12	13	9	4	
	4	5	7	10	9	4	
	5	8	6	8	6	2	
	6	12	4	6	5	1	
15	1	7	6	15	7	13	
	2	8	6	15	6	13	
	3	9	5	15	5	11	
	4	10	4	15	4	10	
	5	11	3	14	4	9	
	6	15	1	14	3	9	
16	1	3	16	15	18	14	
	2	8	15	14	17	13	
	3	10	14	14	17	13	
	4	11	13	12	16	13	
	5	14	10	11	16	11	
	6	17	9	11	16	11	
17	1	1	18	19	10	7	
	2	3	15	17	8	7	
	3	4	13	14	7	7	
	4	15	11	14	7	6	
	5	18	10	11	5	6	
	6	19	6	11	5	6	
18	1	5	8	7	8	4	
	2	11	7	7	7	3	
	3	15	7	5	7	3	
	4	17	7	4	7	2	
	5	19	7	3	6	2	
	6	20	7	2	6	1	
19	1	2	12	16	15	15	
	2	4	11	15	13	14	
	3	5	9	11	12	14	
	4	9	9	9	11	14	
	5	13	6	8	9	14	
	6	18	5	6	7	14	
20	1	1	11	15	18	13	
	2	5	11	13	16	12	
	3	6	9	13	14	11	
	4	7	8	12	10	11	
	5	13	7	9	8	9	
	6	14	7	9	7	8	
21	1	1	1	17	13	7	
	2	2	1	16	12	6	
	3	3	1	15	11	6	
	4	4	1	15	10	5	
	5	5	1	14	9	5	
	6	11	1	14	9	4	
22	1	7	16	12	15	5	
	2	8	14	11	12	5	
	3	9	12	9	11	5	
	4	15	10	8	9	5	
	5	17	9	8	6	5	
	6	19	5	6	5	5	
23	1	3	10	11	10	8	
	2	4	8	9	9	8	
	3	8	7	8	9	8	
	4	9	6	5	8	8	
	5	17	2	3	8	8	
	6	20	2	2	8	8	
24	1	9	8	11	17	11	
	2	12	8	10	17	9	
	3	16	8	9	17	8	
	4	18	8	9	16	8	
	5	19	8	7	16	6	
	6	20	8	7	15	5	
25	1	4	9	13	3	7	
	2	12	9	12	2	5	
	3	14	9	12	2	4	
	4	16	8	12	1	4	
	5	17	8	11	1	2	
	6	20	7	11	1	2	
26	1	3	14	17	9	16	
	2	9	13	15	8	15	
	3	10	11	15	8	14	
	4	11	11	13	7	14	
	5	12	9	12	6	12	
	6	16	9	12	5	11	
27	1	4	12	16	12	10	
	2	6	11	15	11	8	
	3	8	9	14	9	6	
	4	18	8	11	9	5	
	5	19	8	10	8	5	
	6	20	6	9	7	4	
28	1	1	10	15	20	18	
	2	6	9	15	18	17	
	3	7	8	13	17	17	
	4	10	8	10	15	16	
	5	15	7	8	14	15	
	6	19	7	5	12	15	
29	1	2	15	17	9	7	
	2	10	14	17	8	6	
	3	12	12	15	8	6	
	4	13	12	12	5	4	
	5	16	10	11	5	3	
	6	18	10	8	4	2	
30	1	6	15	6	13	10	
	2	8	14	4	13	9	
	3	12	14	4	13	8	
	4	14	13	4	12	8	
	5	16	13	3	12	7	
	6	19	13	2	12	7	
31	1	2	7	17	13	15	
	2	3	6	16	12	15	
	3	5	6	14	9	15	
	4	6	5	11	7	15	
	5	8	5	9	7	15	
	6	10	5	9	4	15	
32	1	6	18	10	18	6	
	2	7	18	7	17	5	
	3	12	18	6	16	4	
	4	13	18	5	16	3	
	5	16	18	3	14	3	
	6	17	18	2	13	2	
33	1	2	9	17	16	20	
	2	6	9	17	14	19	
	3	8	7	15	12	19	
	4	9	6	14	10	19	
	5	14	5	13	10	19	
	6	15	4	13	8	19	
34	1	1	5	13	12	12	
	2	2	5	12	11	11	
	3	3	4	11	11	11	
	4	8	3	11	11	11	
	5	9	1	10	11	11	
	6	14	1	10	11	10	
35	1	1	13	20	16	8	
	2	2	11	19	13	7	
	3	7	11	19	12	6	
	4	17	9	19	9	5	
	5	18	9	18	6	4	
	6	20	8	18	1	4	
36	1	2	18	9	19	16	
	2	5	17	9	16	16	
	3	9	17	7	14	15	
	4	11	16	7	13	13	
	5	14	16	5	10	13	
	6	19	15	4	9	11	
37	1	3	7	17	13	16	
	2	10	6	17	10	12	
	3	15	6	17	10	11	
	4	16	5	16	6	10	
	5	17	5	16	4	7	
	6	18	4	16	1	7	
38	1	2	19	6	12	15	
	2	5	19	6	10	14	
	3	10	18	5	8	14	
	4	12	17	4	7	14	
	5	17	15	3	4	15	
	6	18	15	3	4	14	
39	1	3	15	5	12	12	
	2	4	15	5	11	10	
	3	5	15	4	9	9	
	4	6	15	2	5	6	
	5	15	15	2	4	6	
	6	20	15	1	3	4	
40	1	4	3	18	14	10	
	2	5	3	17	10	10	
	3	12	3	16	8	8	
	4	15	3	15	5	5	
	5	16	3	15	5	3	
	6	20	3	13	1	3	
41	1	5	15	19	12	14	
	2	10	14	16	12	12	
	3	13	12	15	11	12	
	4	15	10	13	10	8	
	5	18	9	10	9	6	
	6	19	7	9	8	4	
42	1	3	11	8	5	18	
	2	5	9	8	5	15	
	3	7	7	6	5	15	
	4	8	7	5	5	12	
	5	12	6	3	4	10	
	6	20	3	3	4	10	
43	1	1	15	15	14	14	
	2	9	14	14	12	13	
	3	10	14	14	10	13	
	4	15	14	14	6	13	
	5	16	14	13	6	13	
	6	17	14	12	4	13	
44	1	3	10	14	15	10	
	2	5	10	12	15	9	
	3	6	10	10	14	9	
	4	8	10	9	13	9	
	5	9	10	5	13	9	
	6	10	10	5	12	9	
45	1	2	12	12	11	12	
	2	8	11	11	11	10	
	3	9	8	10	9	6	
	4	11	7	9	9	5	
	5	14	7	6	7	3	
	6	15	4	5	6	1	
46	1	1	3	18	8	15	
	2	3	3	17	8	14	
	3	4	3	16	7	14	
	4	6	3	16	5	14	
	5	10	3	16	5	13	
	6	17	3	15	4	13	
47	1	4	10	16	16	4	
	2	7	9	14	15	4	
	3	8	8	13	14	4	
	4	16	6	13	13	3	
	5	18	5	12	12	3	
	6	20	3	10	10	3	
48	1	2	18	14	18	18	
	2	4	15	12	17	16	
	3	5	14	10	17	16	
	4	6	14	9	16	14	
	5	15	12	7	16	13	
	6	16	10	6	16	10	
49	1	4	20	19	14	7	
	2	8	18	18	10	7	
	3	10	16	16	10	5	
	4	17	16	14	6	5	
	5	18	14	11	4	3	
	6	19	14	10	2	3	
50	1	2	7	18	16	16	
	2	5	6	16	13	15	
	3	7	6	14	12	14	
	4	13	6	14	8	11	
	5	14	6	12	6	10	
	6	19	6	10	4	8	
51	1	2	18	13	8	11	
	2	3	13	9	7	9	
	3	7	13	8	7	9	
	4	9	8	5	6	8	
	5	15	7	5	5	7	
	6	18	4	2	5	7	
52	1	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2
	37	40	591	561

************************************************************************
