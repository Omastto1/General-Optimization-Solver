jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 4 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	5		2 3 5 7 9 
2	6	2		6 4 
3	6	2		11 4 
4	6	3		13 12 10 
5	6	3		22 14 10 
6	6	3		15 14 13 
7	6	1		8 
8	6	4		25 22 19 15 
9	6	3		25 15 13 
10	6	3		25 19 15 
11	6	2		25 13 
12	6	6		27 25 22 21 20 17 
13	6	4		22 21 18 17 
14	6	2		21 16 
15	6	3		21 18 17 
16	6	4		27 25 20 17 
17	6	5		33 29 28 26 24 
18	6	3		27 24 20 
19	6	5		38 34 33 30 29 
20	6	2		32 23 
21	6	3		38 33 26 
22	6	1		23 
23	6	6		45 38 37 34 33 30 
24	6	5		38 37 34 32 30 
25	6	5		38 37 34 32 30 
26	6	4		37 34 32 31 
27	6	5		40 38 36 35 32 
28	6	6		43 42 40 38 36 35 
29	6	4		43 37 35 32 
30	6	2		36 31 
31	6	5		44 43 42 40 35 
32	6	4		45 44 42 39 
33	6	3		47 42 36 
34	6	2		39 36 
35	6	3		51 47 39 
36	6	3		50 44 41 
37	6	2		47 41 
38	6	4		51 50 48 46 
39	6	2		50 41 
40	6	2		47 46 
41	6	2		48 46 
42	6	2		48 46 
43	6	2		48 47 
44	6	2		51 48 
45	6	2		49 48 
46	6	1		49 
47	6	1		50 
48	6	1		52 
49	6	1		52 
50	6	1		52 
51	6	1		52 
52	1	0		
************************************************************************
REQUESTS/DURATIONS
jobnr.	mode	dur	R1	R2	N1	N2	N3	N4	
------------------------------------------------------------------------
1	1	0	0	0	0	0	0	0	
2	1	13	15	16	7	6	11	3	
	2	14	14	16	5	6	11	3	
	3	15	14	13	5	6	11	3	
	4	16	14	7	4	5	11	3	
	5	17	13	6	4	5	11	2	
	6	19	13	3	3	4	11	2	
3	1	1	14	16	17	17	16	10	
	2	8	13	13	16	17	16	10	
	3	9	13	12	15	16	15	10	
	4	14	13	12	13	16	13	10	
	5	17	12	9	13	15	12	9	
	6	18	12	8	12	14	11	9	
4	1	2	14	19	17	16	19	16	
	2	4	13	18	14	13	17	12	
	3	17	12	15	14	12	16	11	
	4	18	12	15	10	7	14	9	
	5	19	11	14	8	5	13	8	
	6	20	10	11	4	3	11	5	
5	1	3	19	6	18	10	5	19	
	2	4	17	5	14	10	5	17	
	3	7	14	5	13	10	5	14	
	4	8	14	5	8	9	4	11	
	5	17	12	5	8	9	4	8	
	6	19	10	5	6	9	4	6	
6	1	2	10	14	20	10	18	6	
	2	5	8	11	19	8	18	6	
	3	6	8	8	19	8	16	6	
	4	13	5	6	19	6	15	6	
	5	16	4	3	19	5	14	6	
	6	19	3	2	19	5	14	6	
7	1	1	2	11	19	9	12	7	
	2	2	2	8	16	9	11	5	
	3	4	2	8	11	9	10	5	
	4	11	2	5	9	8	8	4	
	5	17	2	3	7	7	7	3	
	6	18	2	3	3	7	6	1	
8	1	2	10	13	17	5	15	15	
	2	3	9	12	15	3	13	14	
	3	7	7	11	13	3	12	13	
	4	8	5	11	13	3	11	13	
	5	16	2	9	11	2	9	13	
	6	17	2	9	10	1	9	12	
9	1	7	12	16	6	18	13	17	
	2	9	11	14	5	17	13	16	
	3	13	8	11	5	17	13	14	
	4	15	6	8	5	16	12	14	
	5	18	4	6	4	16	12	12	
	6	20	1	6	3	16	11	10	
10	1	7	19	19	20	6	9	10	
	2	9	19	19	19	6	9	10	
	3	10	18	18	18	6	7	10	
	4	11	16	18	18	6	6	10	
	5	12	16	17	17	6	6	10	
	6	18	15	16	17	6	5	10	
11	1	4	14	11	17	13	7	5	
	2	5	12	9	16	10	7	5	
	3	6	9	9	16	9	7	5	
	4	7	6	7	15	7	7	4	
	5	13	5	7	14	4	6	3	
	6	14	3	6	13	1	6	3	
12	1	11	17	14	14	14	16	15	
	2	12	17	14	13	14	16	15	
	3	17	15	14	12	12	16	15	
	4	18	15	13	11	9	16	15	
	5	19	13	13	10	7	16	15	
	6	20	13	13	10	3	16	15	
13	1	3	20	9	17	19	4	15	
	2	6	18	9	16	17	4	13	
	3	11	16	9	13	16	4	11	
	4	12	12	8	10	16	4	10	
	5	16	10	7	10	15	4	10	
	6	19	10	7	8	14	4	8	
14	1	5	7	10	19	17	10	1	
	2	6	5	8	16	13	10	1	
	3	8	5	8	14	11	9	1	
	4	9	4	7	14	7	7	1	
	5	17	3	7	12	4	6	1	
	6	20	3	6	9	4	5	1	
15	1	2	12	4	10	12	15	19	
	2	4	9	4	10	9	15	19	
	3	5	8	4	10	9	15	18	
	4	13	8	3	10	5	15	16	
	5	14	6	3	10	4	15	15	
	6	17	5	2	10	1	15	15	
16	1	2	14	20	9	14	9	15	
	2	7	13	18	8	13	7	15	
	3	8	11	17	8	9	6	14	
	4	9	10	16	6	9	5	14	
	5	17	8	16	5	5	4	14	
	6	20	7	15	5	4	2	13	
17	1	6	18	20	10	14	9	8	
	2	7	18	15	9	14	8	7	
	3	8	17	11	8	14	7	7	
	4	11	17	11	8	14	7	6	
	5	12	16	8	7	14	6	7	
	6	14	15	3	6	14	6	7	
18	1	4	15	3	18	6	14	10	
	2	6	15	2	18	6	13	9	
	3	8	14	2	17	6	13	9	
	4	12	13	2	17	6	12	8	
	5	15	12	2	15	6	11	8	
	6	18	12	2	15	6	10	7	
19	1	1	13	19	6	14	8	16	
	2	2	13	15	5	11	7	16	
	3	6	13	12	4	10	7	16	
	4	16	13	9	3	9	7	15	
	5	19	12	9	2	7	6	14	
	6	20	12	6	2	6	6	14	
20	1	3	15	16	13	17	2	13	
	2	7	14	16	11	14	1	11	
	3	8	14	11	10	12	1	7	
	4	9	13	9	6	11	1	6	
	5	12	13	5	6	9	1	3	
	6	15	12	4	3	7	1	2	
21	1	5	5	12	17	9	9	20	
	2	6	4	11	16	9	9	16	
	3	11	3	9	14	9	8	13	
	4	14	2	6	12	9	6	11	
	5	15	1	4	12	9	5	8	
	6	19	1	3	9	9	5	8	
22	1	3	6	19	16	15	16	10	
	2	5	6	17	16	14	13	10	
	3	6	6	17	12	13	10	9	
	4	14	5	15	10	11	8	7	
	5	15	4	14	8	11	8	6	
	6	19	4	12	7	10	5	6	
23	1	6	14	14	7	12	3	14	
	2	10	12	12	7	12	3	13	
	3	13	12	12	5	10	3	10	
	4	14	11	11	4	10	3	8	
	5	15	9	9	4	9	2	6	
	6	20	9	9	3	8	2	5	
24	1	2	15	13	12	14	14	14	
	2	7	15	11	11	11	14	12	
	3	10	14	9	9	11	13	12	
	4	17	14	6	8	8	11	10	
	5	18	12	5	7	5	10	9	
	6	19	12	2	7	4	8	9	
25	1	4	5	11	16	8	10	18	
	2	10	5	11	16	8	8	14	
	3	13	5	11	15	7	8	14	
	4	17	5	10	15	7	7	9	
	5	19	4	9	15	7	6	9	
	6	20	4	9	14	6	5	6	
26	1	4	11	7	16	14	18	6	
	2	5	10	7	15	13	15	4	
	3	14	8	7	14	13	13	4	
	4	15	8	7	14	12	13	2	
	5	16	6	7	13	12	11	1	
	6	17	4	7	12	11	9	1	
27	1	2	9	6	20	14	10	17	
	2	13	9	4	19	13	10	17	
	3	14	8	4	17	13	10	17	
	4	17	7	4	16	12	9	17	
	5	18	6	3	15	10	9	17	
	6	19	5	2	15	9	8	17	
28	1	7	15	9	18	8	15	13	
	2	11	14	9	16	7	12	11	
	3	13	14	9	15	6	10	7	
	4	16	13	9	14	6	9	6	
	5	18	12	9	14	5	4	5	
	6	19	11	9	12	5	1	3	
29	1	3	14	10	14	19	12	15	
	2	6	13	9	13	18	9	14	
	3	7	12	9	13	18	7	13	
	4	9	11	8	13	16	4	12	
	5	12	10	8	12	15	3	12	
	6	16	9	7	12	15	2	11	
30	1	1	16	19	13	12	9	19	
	2	2	12	13	10	10	9	13	
	3	3	11	12	9	8	8	11	
	4	8	9	8	6	7	6	8	
	5	12	8	6	3	5	5	5	
	6	20	6	5	3	2	5	4	
31	1	1	8	12	14	11	19	14	
	2	2	8	11	11	10	16	13	
	3	6	6	10	10	10	14	13	
	4	13	6	7	7	10	10	13	
	5	17	5	6	6	10	7	12	
	6	18	4	5	1	10	5	11	
32	1	1	12	3	17	17	15	16	
	2	5	12	3	15	14	13	13	
	3	6	12	3	11	12	11	12	
	4	8	12	3	9	11	10	11	
	5	9	12	3	8	9	7	9	
	6	15	12	3	5	4	5	8	
33	1	6	14	13	6	9	17	17	
	2	10	13	12	6	8	14	15	
	3	11	12	9	5	7	13	13	
	4	15	11	8	5	6	9	12	
	5	18	10	6	4	4	5	10	
	6	20	9	5	4	3	4	7	
34	1	6	3	9	10	14	20	20	
	2	7	3	8	10	14	20	17	
	3	8	3	6	6	14	20	12	
	4	9	2	6	6	14	20	11	
	5	16	2	4	4	14	20	7	
	6	18	1	4	1	14	20	2	
35	1	1	16	18	14	11	10	11	
	2	3	15	17	14	11	9	9	
	3	7	15	17	14	11	8	9	
	4	9	15	16	13	11	8	5	
	5	17	14	15	12	11	7	3	
	6	20	14	13	12	11	6	2	
36	1	9	3	11	8	17	15	4	
	2	11	3	9	8	16	13	3	
	3	12	3	9	8	16	8	3	
	4	16	2	6	7	16	6	3	
	5	17	2	5	7	16	6	3	
	6	18	2	5	7	16	3	3	
37	1	2	18	14	16	16	16	13	
	2	4	17	14	15	16	15	13	
	3	7	17	13	11	12	14	11	
	4	8	14	11	10	12	12	11	
	5	10	13	11	7	10	12	9	
	6	11	11	10	4	7	11	8	
38	1	9	12	9	7	17	16	11	
	2	11	9	8	7	17	15	8	
	3	14	8	8	7	17	15	6	
	4	15	7	7	6	17	14	6	
	5	19	3	7	6	17	14	4	
	6	20	2	7	5	17	14	2	
39	1	2	6	8	11	17	19	14	
	2	3	5	8	11	17	19	14	
	3	8	5	8	11	13	19	13	
	4	10	4	7	10	12	18	12	
	5	13	4	6	10	7	18	12	
	6	20	4	6	10	4	18	11	
40	1	4	16	15	20	15	1	12	
	2	5	16	14	14	14	1	11	
	3	14	14	14	12	14	1	11	
	4	18	11	13	9	12	1	11	
	5	19	9	11	7	11	1	11	
	6	20	9	11	4	10	1	11	
41	1	8	16	15	14	18	19	16	
	2	14	16	13	14	16	19	13	
	3	16	15	9	14	15	19	12	
	4	17	15	7	14	15	19	9	
	5	18	15	5	14	13	19	6	
	6	19	14	5	14	10	19	6	
42	1	7	14	15	11	19	17	14	
	2	8	12	14	10	17	15	11	
	3	11	12	12	10	17	12	9	
	4	13	12	12	9	15	8	7	
	5	15	10	10	7	14	7	5	
	6	16	10	9	7	13	4	4	
43	1	7	20	11	13	16	17	4	
	2	12	17	10	13	13	13	3	
	3	13	16	7	13	10	12	3	
	4	14	15	4	13	8	10	2	
	5	15	14	3	13	6	8	1	
	6	16	12	1	13	6	5	1	
44	1	6	5	15	17	13	18	19	
	2	8	5	15	17	13	16	19	
	3	14	5	14	16	11	16	18	
	4	15	5	13	15	11	14	18	
	5	16	5	11	14	9	12	17	
	6	19	5	10	13	8	10	17	
45	1	3	3	9	19	2	13	14	
	2	6	2	9	18	2	12	13	
	3	8	2	9	17	2	12	13	
	4	11	1	9	17	2	12	12	
	5	19	1	9	15	2	12	11	
	6	20	1	9	15	2	12	10	
46	1	6	15	19	7	13	15	13	
	2	9	12	18	6	11	14	13	
	3	10	12	17	4	10	12	13	
	4	11	10	14	3	8	10	13	
	5	14	4	14	3	6	9	13	
	6	18	4	12	2	4	6	13	
47	1	1	17	11	11	16	4	15	
	2	2	17	10	10	16	3	13	
	3	10	15	10	9	14	2	11	
	4	13	14	10	8	13	2	11	
	5	16	13	8	5	12	2	9	
	6	19	13	8	5	12	1	8	
48	1	4	10	2	5	18	20	17	
	2	6	10	2	5	15	18	17	
	3	8	9	2	5	13	17	15	
	4	10	7	1	4	12	16	12	
	5	12	7	1	4	10	15	11	
	6	15	6	1	3	9	14	10	
49	1	1	7	10	9	5	10	18	
	2	6	6	8	8	4	9	15	
	3	8	6	7	8	3	9	15	
	4	9	5	5	8	3	9	11	
	5	10	4	3	7	2	9	10	
	6	19	4	2	6	2	9	7	
50	1	1	16	19	15	11	8	19	
	2	3	12	18	14	8	8	15	
	3	4	11	17	12	8	8	12	
	4	5	8	16	9	5	8	11	
	5	6	5	13	7	3	8	8	
	6	14	3	12	4	3	8	7	
51	1	1	14	12	4	18	1	8	
	2	2	13	11	4	13	1	5	
	3	7	12	10	4	12	1	4	
	4	10	10	8	3	8	1	4	
	5	17	10	6	2	5	1	3	
	6	20	9	5	2	4	1	2	
52	1	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2	N 3	N 4
	35	36	459	443	438	438

************************************************************************
