jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 4 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	9		2 3 4 5 6 7 8 9 11 
2	6	5		18 16 13 12 10 
3	6	6		34 31 18 16 14 12 
4	6	11		37 35 30 26 23 22 21 20 19 17 16 
5	6	8		36 35 34 23 22 17 16 14 
6	6	7		35 34 31 24 22 16 14 
7	6	5		37 31 27 15 13 
8	6	7		36 35 25 23 22 16 14 
9	6	7		35 34 32 30 25 23 18 
10	6	6		37 35 34 29 24 17 
11	6	8		37 36 35 30 29 27 25 23 
12	6	5		35 30 26 23 17 
13	6	5		35 30 28 22 21 
14	6	6		37 32 30 29 28 27 
15	6	5		35 32 30 29 25 
16	6	6		41 40 32 29 28 27 
17	6	5		51 40 32 27 25 
18	6	5		43 37 36 33 22 
19	6	9		51 50 43 42 41 39 38 36 34 
20	6	5		43 41 39 29 27 
21	6	6		51 50 43 39 33 32 
22	6	4		48 41 40 29 
23	6	3		40 38 28 
24	6	7		51 50 48 43 42 40 38 
25	6	3		47 41 28 
26	6	5		50 49 46 42 33 
27	6	4		49 46 42 33 
28	6	5		50 49 45 43 39 
29	6	6		51 50 47 46 45 42 
30	6	4		50 46 40 38 
31	6	4		49 47 42 36 
32	6	5		49 47 46 45 42 
33	6	3		48 44 38 
34	6	2		47 40 
35	6	3		51 49 48 
36	6	3		46 45 44 
37	6	1		39 
38	6	2		47 45 
39	6	2		48 44 
40	6	2		45 44 
41	6	2		49 46 
42	6	1		44 
43	6	1		46 
44	6	1		52 
45	6	1		52 
46	6	1		52 
47	6	1		52 
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
2	1	1	4	5	15	11	14	18	
	2	2	4	4	14	9	13	14	
	3	4	3	4	12	7	13	14	
	4	5	3	4	11	5	11	11	
	5	14	2	4	11	5	9	7	
	6	15	2	4	10	2	7	6	
3	1	1	3	1	14	10	18	12	
	2	5	2	1	12	9	15	9	
	3	13	2	1	12	8	13	8	
	4	14	2	1	10	6	9	7	
	5	15	2	1	10	6	4	6	
	6	16	2	1	9	5	1	6	
4	1	2	2	3	17	10	12	11	
	2	3	2	2	16	9	10	10	
	3	7	2	2	15	9	10	10	
	4	11	2	1	15	9	10	8	
	5	12	2	1	12	9	8	6	
	6	13	2	1	11	9	8	5	
5	1	1	1	3	9	13	19	4	
	2	2	1	2	9	12	17	3	
	3	8	1	2	9	8	15	3	
	4	17	1	2	9	6	14	2	
	5	18	1	2	9	4	11	2	
	6	19	1	2	9	4	10	2	
6	1	6	4	3	17	9	19	20	
	2	8	4	2	15	9	16	19	
	3	9	4	2	14	9	15	18	
	4	16	4	2	11	9	14	18	
	5	18	4	2	11	9	11	18	
	6	19	4	2	9	9	10	17	
7	1	1	4	5	15	7	19	17	
	2	9	3	4	13	7	18	15	
	3	11	3	4	9	7	17	13	
	4	14	2	4	7	7	16	13	
	5	18	2	3	5	7	16	10	
	6	19	1	3	2	7	15	9	
8	1	5	5	4	13	10	19	18	
	2	7	4	4	11	9	15	15	
	3	9	4	4	11	9	13	15	
	4	10	4	3	9	9	10	13	
	5	12	3	3	8	9	6	9	
	6	20	3	3	6	9	1	9	
9	1	1	2	3	16	17	5	11	
	2	2	2	3	16	13	4	11	
	3	3	2	3	15	13	4	11	
	4	4	2	2	14	11	4	11	
	5	7	2	2	13	7	3	11	
	6	13	2	1	13	5	3	11	
10	1	1	3	4	17	10	6	9	
	2	3	3	3	16	10	6	7	
	3	10	3	3	14	8	6	6	
	4	14	3	3	13	7	5	5	
	5	15	2	2	11	6	5	4	
	6	16	2	2	8	4	5	3	
11	1	4	2	2	16	14	14	16	
	2	12	2	1	12	12	14	14	
	3	13	2	1	11	10	14	12	
	4	14	2	1	10	8	13	8	
	5	17	2	1	7	3	13	8	
	6	18	2	1	5	3	12	4	
12	1	2	1	4	13	15	13	10	
	2	3	1	4	11	12	12	8	
	3	4	1	4	10	12	10	7	
	4	11	1	3	9	10	9	5	
	5	17	1	2	8	8	8	3	
	6	19	1	2	7	6	6	1	
13	1	6	2	4	14	11	19	16	
	2	8	1	3	14	11	17	14	
	3	9	1	3	12	9	16	14	
	4	13	1	3	8	5	15	13	
	5	14	1	3	7	5	14	12	
	6	19	1	3	5	3	11	11	
14	1	5	2	5	17	8	6	16	
	2	6	2	4	16	7	5	16	
	3	7	2	4	16	7	5	14	
	4	12	1	4	16	6	4	13	
	5	15	1	4	16	6	3	10	
	6	20	1	4	16	5	3	8	
15	1	4	1	4	19	16	19	16	
	2	8	1	4	18	12	16	13	
	3	9	1	3	17	10	15	12	
	4	10	1	3	17	7	15	11	
	5	12	1	2	16	6	12	10	
	6	19	1	2	16	5	12	8	
16	1	6	4	4	9	3	14	17	
	2	7	4	3	7	3	12	17	
	3	9	4	3	6	3	10	17	
	4	10	4	3	6	2	6	16	
	5	19	4	3	5	2	5	15	
	6	20	4	3	3	1	3	15	
17	1	1	4	4	14	8	14	16	
	2	8	3	4	13	7	9	16	
	3	9	3	4	13	7	9	15	
	4	10	2	4	12	5	7	14	
	5	11	1	3	11	4	5	12	
	6	13	1	3	11	4	2	11	
18	1	1	1	2	13	11	5	11	
	2	7	1	2	13	11	5	10	
	3	8	1	2	12	11	5	9	
	4	9	1	2	9	11	5	9	
	5	11	1	2	9	11	5	7	
	6	16	1	2	7	11	5	5	
19	1	1	3	3	19	20	20	13	
	2	8	3	3	17	19	19	12	
	3	11	3	3	14	19	18	12	
	4	15	2	2	13	19	17	12	
	5	16	2	2	12	19	16	11	
	6	17	2	1	10	19	16	11	
20	1	6	4	4	5	15	19	8	
	2	8	4	4	5	13	16	8	
	3	10	3	4	5	13	12	8	
	4	12	3	4	5	11	11	8	
	5	16	2	3	5	9	10	8	
	6	18	2	3	5	9	8	8	
21	1	6	3	4	4	17	20	17	
	2	7	3	4	3	16	19	15	
	3	8	3	4	2	11	19	12	
	4	9	3	4	2	9	19	11	
	5	12	2	4	2	6	18	10	
	6	17	2	4	1	4	18	8	
22	1	3	4	3	13	20	3	17	
	2	7	4	3	12	15	3	16	
	3	8	3	3	11	11	2	14	
	4	9	3	3	10	8	2	13	
	5	12	2	3	9	8	2	13	
	6	19	2	3	9	5	1	12	
23	1	1	4	2	18	9	17	15	
	2	2	4	2	17	8	16	14	
	3	6	3	2	16	8	16	12	
	4	7	3	2	15	8	16	8	
	5	15	2	2	12	8	15	8	
	6	18	1	2	11	8	14	6	
24	1	3	1	5	16	18	12	13	
	2	4	1	4	15	17	11	11	
	3	5	1	4	12	15	8	11	
	4	6	1	3	10	15	5	10	
	5	10	1	3	9	13	4	10	
	6	16	1	3	5	12	3	9	
25	1	3	4	3	10	19	20	17	
	2	6	4	3	10	19	19	16	
	3	10	4	3	10	19	17	16	
	4	11	4	3	9	19	16	16	
	5	17	3	3	8	19	16	15	
	6	18	3	3	8	19	15	15	
26	1	5	3	5	2	15	13	12	
	2	6	3	5	2	14	12	12	
	3	9	3	5	2	13	12	11	
	4	10	3	5	2	13	10	11	
	5	15	3	5	2	13	10	10	
	6	20	3	5	2	12	8	9	
27	1	2	2	2	15	15	1	19	
	2	13	1	1	13	14	1	19	
	3	14	1	1	11	14	1	19	
	4	16	1	1	10	12	1	19	
	5	17	1	1	6	11	1	18	
	6	19	1	1	5	10	1	18	
28	1	2	3	3	1	8	12	4	
	2	4	3	3	1	7	11	3	
	3	7	3	3	1	6	11	3	
	4	11	2	3	1	6	11	3	
	5	13	2	3	1	4	11	1	
	6	15	1	3	1	4	11	1	
29	1	1	5	5	3	9	17	20	
	2	2	4	5	3	9	16	16	
	3	3	3	5	3	7	15	14	
	4	9	2	5	3	7	13	14	
	5	12	2	5	3	6	10	12	
	6	16	1	5	3	5	9	9	
30	1	7	3	2	15	17	2	12	
	2	10	3	1	13	17	2	11	
	3	12	3	1	10	15	2	10	
	4	13	2	1	8	14	2	10	
	5	17	2	1	6	11	2	8	
	6	18	1	1	5	9	2	7	
31	1	4	4	3	14	13	15	15	
	2	6	4	3	13	11	12	13	
	3	15	4	3	11	10	9	12	
	4	16	4	3	9	8	7	11	
	5	19	4	2	5	5	6	11	
	6	20	4	2	3	5	4	10	
32	1	2	4	3	15	12	18	17	
	2	5	3	3	14	9	17	16	
	3	7	3	3	14	8	17	16	
	4	9	2	3	14	7	14	15	
	5	10	2	3	14	6	14	15	
	6	18	2	3	14	4	11	15	
33	1	8	3	4	16	17	9	14	
	2	9	3	4	16	14	8	14	
	3	13	3	4	14	12	7	14	
	4	16	3	3	13	12	6	14	
	5	17	3	3	12	10	5	14	
	6	18	3	2	11	8	4	14	
34	1	7	3	4	20	13	16	15	
	2	8	2	4	18	12	15	15	
	3	11	2	4	18	10	15	15	
	4	12	1	4	17	6	14	15	
	5	14	1	4	16	6	13	15	
	6	15	1	4	15	3	11	15	
35	1	3	5	3	3	16	17	11	
	2	4	4	3	3	14	17	11	
	3	5	3	3	3	13	15	10	
	4	7	2	3	3	13	15	10	
	5	16	1	2	3	11	14	10	
	6	18	1	2	3	11	12	9	
36	1	1	3	5	3	13	10	15	
	2	7	3	5	3	12	9	11	
	3	14	3	5	3	12	8	9	
	4	15	3	5	3	11	8	9	
	5	16	3	5	3	11	6	5	
	6	17	3	5	3	11	6	2	
37	1	3	3	4	5	17	17	11	
	2	9	2	3	4	16	17	11	
	3	12	2	3	4	16	14	9	
	4	15	2	2	3	14	13	8	
	5	16	1	2	3	13	12	8	
	6	17	1	2	2	11	9	7	
38	1	7	5	4	11	12	6	10	
	2	11	4	4	10	11	5	9	
	3	13	4	4	10	11	5	8	
	4	17	3	4	9	11	5	7	
	5	18	3	3	8	11	5	5	
	6	19	3	3	7	11	5	3	
39	1	2	4	5	18	14	15	16	
	2	3	3	5	14	12	14	15	
	3	4	3	5	11	11	13	14	
	4	9	3	5	9	7	11	14	
	5	11	3	5	7	5	10	13	
	6	17	3	5	4	4	9	12	
40	1	2	4	4	17	18	16	8	
	2	3	3	4	16	17	13	5	
	3	5	3	4	16	14	13	5	
	4	8	2	4	15	14	11	3	
	5	11	2	3	13	12	7	3	
	6	19	1	3	12	9	6	2	
41	1	2	3	2	11	8	3	17	
	2	3	3	2	11	8	3	14	
	3	6	3	2	11	8	3	12	
	4	7	3	1	11	8	3	10	
	5	8	3	1	11	7	3	8	
	6	10	3	1	11	7	3	6	
42	1	7	4	5	10	15	8	16	
	2	10	4	4	7	11	7	13	
	3	11	3	3	6	8	5	12	
	4	16	3	3	5	8	5	9	
	5	18	2	3	3	6	3	6	
	6	19	2	2	2	4	2	3	
43	1	2	4	5	15	8	17	11	
	2	3	4	4	14	8	16	10	
	3	7	4	3	13	8	16	8	
	4	11	3	3	11	8	16	7	
	5	16	3	2	11	8	15	6	
	6	17	2	2	9	8	15	6	
44	1	6	2	2	15	14	18	20	
	2	8	2	2	14	13	16	19	
	3	9	2	2	11	13	14	19	
	4	11	2	2	11	13	11	19	
	5	12	2	2	8	13	9	19	
	6	17	2	2	4	13	7	19	
45	1	10	5	5	6	8	5	13	
	2	11	4	4	5	7	4	13	
	3	12	3	4	4	5	4	13	
	4	13	3	4	4	4	4	13	
	5	14	2	3	2	3	3	13	
	6	16	2	3	2	3	3	12	
46	1	4	3	4	16	17	10	17	
	2	5	3	4	12	15	9	15	
	3	7	3	4	11	12	8	13	
	4	11	3	3	10	10	7	10	
	5	12	3	3	6	9	6	6	
	6	18	3	2	5	7	6	6	
47	1	4	4	3	15	18	15	19	
	2	5	4	3	13	18	12	15	
	3	14	4	3	12	18	12	14	
	4	15	4	3	11	18	10	11	
	5	18	4	3	9	18	7	10	
	6	20	4	3	8	18	6	6	
48	1	1	5	4	18	18	12	8	
	2	2	4	3	15	16	11	7	
	3	8	4	3	11	14	10	6	
	4	12	4	3	10	12	10	5	
	5	17	3	2	8	11	9	2	
	6	18	3	1	5	10	9	1	
49	1	3	5	4	8	16	15	16	
	2	4	3	3	6	16	14	16	
	3	6	3	3	5	16	13	16	
	4	7	2	3	4	16	11	15	
	5	8	2	2	3	16	11	15	
	6	9	1	2	3	16	9	14	
50	1	2	4	4	15	19	17	10	
	2	3	3	3	14	18	15	9	
	3	12	3	3	10	17	14	9	
	4	14	2	2	9	16	12	8	
	5	15	1	2	8	16	11	6	
	6	19	1	1	3	15	11	6	
51	1	1	3	2	12	12	17	16	
	2	2	2	1	12	12	15	12	
	3	10	2	1	9	12	15	11	
	4	13	1	1	6	12	15	9	
	5	14	1	1	4	12	14	4	
	6	16	1	1	4	12	13	2	
52	1	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2	N 3	N 4
	29	33	559	596	595	628

************************************************************************
