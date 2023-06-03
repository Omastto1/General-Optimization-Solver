jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	13		2 3 4 5 6 7 8 9 10 12 13 15 17 
2	6	7		29 22 21 20 18 16 14 
3	6	5		24 21 18 16 11 
4	6	11		34 31 30 28 27 24 23 22 20 19 18 
5	6	7		35 32 27 24 22 20 14 
6	6	5		27 24 23 16 14 
7	6	9		34 31 30 28 27 25 20 19 16 
8	6	8		35 30 26 24 23 22 21 20 
9	6	8		35 33 31 30 28 27 25 19 
10	6	6		34 32 31 30 19 16 
11	6	3		35 27 14 
12	6	3		27 26 14 
13	6	6		33 30 28 26 22 18 
14	6	10		51 39 37 36 34 33 31 30 28 25 
15	6	6		35 33 32 30 28 20 
16	6	6		51 37 36 35 33 26 
17	6	5		38 35 33 30 27 
18	6	7		51 43 40 38 37 36 35 
19	6	5		51 39 37 36 26 
20	6	8		51 49 44 43 40 39 37 36 
21	6	5		43 42 40 32 28 
22	6	4		51 50 39 25 
23	6	8		51 49 44 43 42 41 40 33 
24	6	7		49 43 42 41 40 37 33 
25	6	7		49 47 44 43 42 41 40 
26	6	7		48 47 43 42 41 40 38 
27	6	6		49 47 44 42 40 37 
28	6	6		50 47 45 44 41 38 
29	6	5		49 47 44 40 37 
30	6	6		49 48 43 42 41 40 
31	6	6		50 48 47 45 41 38 
32	6	4		50 47 45 38 
33	6	5		50 48 47 46 45 
34	6	4		50 46 42 40 
35	6	3		47 44 42 
36	6	3		50 45 41 
37	6	3		50 46 45 
38	6	2		49 46 
39	6	2		47 45 
40	6	1		45 
41	6	1		46 
42	6	1		45 
43	6	1		45 
44	6	1		48 
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
jobnr.	mode	dur	R1	R2	N1	N2	
------------------------------------------------------------------------
1	1	0	0	0	0	0	
2	1	1	4	1	18	12	
	2	3	4	1	18	10	
	3	4	4	1	15	9	
	4	10	3	1	12	7	
	5	11	3	1	10	5	
	6	16	3	1	8	1	
3	1	5	4	2	11	18	
	2	11	3	2	8	18	
	3	13	3	2	8	16	
	4	14	3	2	6	15	
	5	15	2	2	5	13	
	6	17	2	2	4	13	
4	1	10	5	4	17	15	
	2	11	4	3	17	14	
	3	12	4	3	16	13	
	4	16	4	2	16	13	
	5	17	3	2	15	11	
	6	18	3	1	15	10	
5	1	1	4	4	13	12	
	2	5	3	3	12	8	
	3	8	3	3	11	7	
	4	10	2	2	9	5	
	5	15	2	2	7	4	
	6	17	1	2	4	2	
6	1	7	2	2	8	14	
	2	11	2	2	6	13	
	3	12	2	2	6	12	
	4	14	2	1	4	13	
	5	15	2	1	3	13	
	6	17	2	1	2	13	
7	1	1	2	3	16	18	
	2	2	2	3	16	16	
	3	6	2	3	15	14	
	4	7	2	3	14	13	
	5	8	2	3	13	11	
	6	16	2	3	12	10	
8	1	1	4	5	10	18	
	2	3	3	4	9	17	
	3	7	3	4	8	16	
	4	9	2	3	6	15	
	5	13	2	3	5	14	
	6	16	2	3	4	14	
9	1	1	3	4	14	17	
	2	7	2	4	13	14	
	3	8	2	4	12	11	
	4	9	2	3	9	8	
	5	11	2	3	9	5	
	6	12	2	2	6	3	
10	1	1	1	2	18	9	
	2	6	1	2	17	8	
	3	10	1	2	16	7	
	4	12	1	2	13	4	
	5	16	1	2	13	3	
	6	20	1	2	12	2	
11	1	1	5	4	13	15	
	2	3	4	4	13	14	
	3	8	4	4	11	11	
	4	9	4	3	11	11	
	5	13	3	3	10	7	
	6	18	3	3	9	7	
12	1	1	4	5	18	17	
	2	3	3	3	18	17	
	3	4	2	3	17	17	
	4	8	2	3	16	16	
	5	9	1	2	14	15	
	6	15	1	1	14	15	
13	1	7	2	3	18	13	
	2	8	2	2	14	13	
	3	13	2	2	11	13	
	4	16	2	1	7	13	
	5	18	2	1	5	13	
	6	20	2	1	2	13	
14	1	3	2	2	6	8	
	2	5	2	2	5	8	
	3	11	2	2	5	7	
	4	12	2	1	4	6	
	5	15	2	1	4	5	
	6	17	2	1	4	4	
15	1	10	5	4	12	17	
	2	11	4	4	9	16	
	3	12	4	4	9	15	
	4	16	3	4	7	15	
	5	17	2	4	6	15	
	6	18	2	4	6	14	
16	1	4	4	3	9	20	
	2	5	4	3	9	15	
	3	11	4	3	9	13	
	4	12	3	3	9	13	
	5	16	2	3	9	10	
	6	17	2	3	9	7	
17	1	3	1	4	14	16	
	2	7	1	3	14	14	
	3	8	1	3	14	12	
	4	9	1	3	13	10	
	5	13	1	3	13	7	
	6	20	1	3	12	6	
18	1	1	2	2	4	15	
	2	7	1	2	4	11	
	3	10	1	2	3	10	
	4	11	1	2	3	9	
	5	14	1	2	1	5	
	6	16	1	2	1	4	
19	1	2	5	3	4	5	
	2	3	4	3	3	4	
	3	4	3	3	3	4	
	4	6	3	3	2	4	
	5	7	1	3	2	3	
	6	16	1	3	2	2	
20	1	1	4	2	13	17	
	2	2	3	2	12	15	
	3	5	3	2	10	13	
	4	6	3	1	6	11	
	5	16	3	1	6	10	
	6	19	3	1	2	10	
21	1	2	3	2	20	4	
	2	11	3	1	19	3	
	3	13	2	1	19	3	
	4	15	2	1	18	2	
	5	16	1	1	18	2	
	6	17	1	1	17	1	
22	1	2	1	5	12	9	
	2	5	1	4	12	8	
	3	9	1	4	9	7	
	4	11	1	3	7	7	
	5	15	1	3	5	7	
	6	19	1	3	2	6	
23	1	2	4	5	10	15	
	2	5	3	4	10	15	
	3	11	3	4	10	14	
	4	16	2	3	10	11	
	5	17	2	3	9	11	
	6	18	2	3	9	9	
24	1	1	3	3	9	11	
	2	2	3	2	8	7	
	3	7	3	2	7	7	
	4	10	2	2	5	6	
	5	14	2	1	4	4	
	6	20	2	1	4	2	
25	1	4	5	2	20	14	
	2	5	3	2	20	14	
	3	6	3	2	20	13	
	4	13	2	2	20	14	
	5	16	2	2	20	13	
	6	19	1	2	20	14	
26	1	1	5	4	10	5	
	2	13	4	3	9	4	
	3	15	4	3	9	3	
	4	16	4	2	9	3	
	5	17	4	2	9	2	
	6	18	4	1	9	2	
27	1	2	5	2	16	10	
	2	3	5	2	14	9	
	3	6	5	2	13	9	
	4	8	5	1	12	7	
	5	12	5	1	9	7	
	6	17	5	1	9	6	
28	1	1	5	3	18	19	
	2	7	4	3	16	16	
	3	10	3	3	16	12	
	4	11	3	2	14	11	
	5	12	1	2	13	9	
	6	20	1	1	13	6	
29	1	2	5	4	9	12	
	2	11	4	4	6	11	
	3	12	4	4	5	10	
	4	13	3	4	3	8	
	5	14	2	4	3	6	
	6	18	2	4	2	6	
30	1	3	1	3	19	5	
	2	4	1	3	18	4	
	3	6	1	2	17	3	
	4	15	1	2	16	3	
	5	16	1	1	15	2	
	6	18	1	1	14	1	
31	1	1	4	4	11	3	
	2	3	3	4	10	3	
	3	4	3	4	8	3	
	4	5	3	4	6	3	
	5	17	2	4	5	3	
	6	20	2	4	2	3	
32	1	3	5	3	5	17	
	2	8	3	3	4	17	
	3	10	3	2	3	13	
	4	14	3	2	2	12	
	5	15	1	2	2	9	
	6	17	1	1	1	7	
33	1	3	3	2	18	17	
	2	4	3	2	14	17	
	3	5	3	2	14	12	
	4	6	3	2	10	10	
	5	9	3	2	10	6	
	6	13	3	2	8	5	
34	1	1	3	2	20	11	
	2	4	2	2	16	9	
	3	7	2	2	14	9	
	4	10	2	2	9	8	
	5	14	1	1	7	8	
	6	16	1	1	5	7	
35	1	3	2	2	16	8	
	2	7	2	2	15	8	
	3	8	2	2	15	7	
	4	17	2	2	15	6	
	5	18	2	2	14	6	
	6	19	2	2	14	5	
36	1	5	4	1	17	14	
	2	7	3	1	14	12	
	3	8	3	1	10	8	
	4	9	2	1	7	6	
	5	10	2	1	5	4	
	6	19	2	1	2	3	
37	1	1	4	2	16	18	
	2	2	3	2	15	18	
	3	9	3	2	14	18	
	4	11	3	2	11	17	
	5	13	3	2	11	16	
	6	19	3	2	8	17	
38	1	1	2	4	20	9	
	2	2	1	4	20	8	
	3	4	1	3	20	7	
	4	10	1	3	20	6	
	5	11	1	2	20	5	
	6	13	1	2	20	4	
39	1	7	1	5	16	11	
	2	12	1	4	13	10	
	3	13	1	4	11	10	
	4	14	1	3	10	10	
	5	15	1	2	7	9	
	6	17	1	2	4	9	
40	1	5	2	1	19	19	
	2	7	2	1	18	19	
	3	8	2	1	17	19	
	4	11	2	1	15	19	
	5	15	2	1	15	18	
	6	16	2	1	13	19	
41	1	9	4	4	11	19	
	2	10	4	4	9	19	
	3	11	4	4	8	17	
	4	12	4	3	7	17	
	5	13	4	3	6	16	
	6	17	4	3	3	15	
42	1	2	5	4	19	17	
	2	9	4	3	17	16	
	3	10	4	3	15	15	
	4	13	4	2	15	13	
	5	15	4	2	14	12	
	6	16	4	2	11	11	
43	1	1	3	4	11	13	
	2	8	2	3	10	13	
	3	9	2	2	9	11	
	4	10	2	2	7	10	
	5	11	1	1	7	10	
	6	14	1	1	6	8	
44	1	1	4	4	20	18	
	2	2	4	3	19	17	
	3	5	4	3	19	16	
	4	6	3	2	18	15	
	5	13	3	2	18	14	
	6	18	3	1	18	14	
45	1	2	4	2	16	8	
	2	3	3	2	15	8	
	3	6	3	2	14	7	
	4	7	3	1	14	7	
	5	12	2	1	13	6	
	6	14	2	1	13	5	
46	1	2	2	3	9	16	
	2	5	2	2	8	16	
	3	6	2	2	8	15	
	4	8	2	2	7	15	
	5	12	1	2	6	13	
	6	13	1	2	6	12	
47	1	3	4	1	13	14	
	2	4	4	1	12	13	
	3	6	4	1	10	12	
	4	9	4	1	9	12	
	5	14	3	1	7	11	
	6	18	3	1	6	10	
48	1	2	4	2	19	10	
	2	5	3	2	16	8	
	3	12	2	2	15	8	
	4	14	2	2	13	6	
	5	19	2	1	13	6	
	6	20	1	1	11	5	
49	1	4	2	3	12	20	
	2	8	2	3	9	17	
	3	14	2	3	8	16	
	4	16	2	3	7	14	
	5	19	2	3	3	11	
	6	20	2	3	3	10	
50	1	2	4	2	13	17	
	2	3	3	2	13	16	
	3	10	2	2	12	13	
	4	11	2	2	11	10	
	5	16	1	1	9	8	
	6	19	1	1	8	5	
51	1	6	4	3	9	5	
	2	7	4	3	8	4	
	3	9	4	3	8	3	
	4	10	3	3	5	3	
	5	11	3	2	4	2	
	6	13	2	2	2	2	
52	1	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2
	35	33	540	523

************************************************************************
