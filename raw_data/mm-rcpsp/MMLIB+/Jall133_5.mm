jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	5		2 3 4 5 9 
2	6	7		18 17 14 13 8 7 6 
3	6	7		17 14 13 12 8 7 6 
4	6	7		20 18 17 14 13 12 10 
5	6	7		20 18 16 14 13 11 10 
6	6	4		26 20 16 11 
7	6	3		16 11 10 
8	6	3		16 15 10 
9	6	2		16 10 
10	6	6		26 24 23 22 21 19 
11	6	6		30 24 23 22 21 19 
12	6	3		32 26 16 
13	6	4		26 24 21 19 
14	6	5		30 28 25 23 21 
15	6	5		30 28 25 23 21 
16	6	4		30 24 23 22 
17	6	4		32 26 23 22 
18	6	4		28 27 25 21 
19	6	5		32 31 28 27 25 
20	6	2		30 23 
21	6	6		40 35 34 32 31 29 
22	6	3		31 28 25 
23	6	3		40 29 27 
24	6	4		40 34 31 29 
25	6	4		40 38 34 29 
26	6	4		40 38 34 29 
27	6	3		36 35 34 
28	6	3		43 38 33 
29	6	3		43 39 33 
30	6	4		43 39 37 36 
31	6	3		43 38 36 
32	6	3		43 38 36 
33	6	3		51 37 36 
34	6	3		43 41 39 
35	6	2		43 38 
36	6	3		45 42 41 
37	6	3		45 42 41 
38	6	1		39 
39	6	4		51 50 49 42 
40	6	3		50 44 43 
41	6	4		50 49 47 44 
42	6	2		47 44 
43	6	3		49 47 46 
44	6	2		48 46 
45	6	1		46 
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
2	1	1	12	17	6	15	
	2	2	9	16	5	14	
	3	12	8	15	4	13	
	4	13	7	15	4	12	
	5	14	4	14	3	11	
	6	18	4	13	3	9	
3	1	4	16	7	12	6	
	2	8	15	7	11	6	
	3	11	12	6	11	6	
	4	13	10	6	11	5	
	5	14	8	4	11	5	
	6	18	7	4	11	4	
4	1	2	20	18	16	18	
	2	3	20	15	16	15	
	3	6	20	13	14	12	
	4	7	20	12	13	8	
	5	9	20	8	13	7	
	6	17	20	8	12	4	
5	1	2	12	20	20	6	
	2	4	11	16	18	6	
	3	7	11	15	15	6	
	4	10	11	12	13	6	
	5	13	11	12	10	6	
	6	14	11	10	10	6	
6	1	6	16	6	16	17	
	2	9	14	6	16	13	
	3	10	13	6	16	12	
	4	11	13	5	16	10	
	5	13	11	5	16	5	
	6	15	10	4	16	4	
7	1	9	20	8	17	16	
	2	10	16	6	16	15	
	3	11	14	6	14	14	
	4	17	12	4	10	14	
	5	19	8	3	8	13	
	6	20	5	2	6	13	
8	1	1	13	18	14	18	
	2	2	12	14	13	17	
	3	12	12	12	11	14	
	4	14	11	10	8	14	
	5	15	10	6	5	10	
	6	16	9	4	5	9	
9	1	4	16	12	20	12	
	2	5	16	12	17	12	
	3	7	15	11	12	12	
	4	9	13	9	11	11	
	5	17	10	9	8	11	
	6	18	9	8	5	11	
10	1	1	19	20	3	18	
	2	10	19	19	3	15	
	3	12	18	17	3	15	
	4	15	18	16	2	12	
	5	16	16	16	2	11	
	6	17	16	15	2	10	
11	1	1	10	14	9	19	
	2	4	7	13	9	18	
	3	12	6	11	9	15	
	4	13	4	8	8	14	
	5	14	4	5	7	11	
	6	17	3	4	7	10	
12	1	1	16	15	13	9	
	2	2	16	10	12	8	
	3	3	14	8	11	6	
	4	9	12	6	10	5	
	5	10	11	6	9	5	
	6	15	11	4	8	4	
13	1	7	18	9	13	15	
	2	11	13	9	11	13	
	3	13	13	9	10	12	
	4	14	8	8	8	10	
	5	18	6	8	6	8	
	6	20	3	8	5	6	
14	1	3	12	16	12	13	
	2	9	12	15	12	11	
	3	10	10	14	11	11	
	4	11	9	13	11	10	
	5	14	9	12	10	8	
	6	20	8	12	10	6	
15	1	1	12	14	17	4	
	2	6	12	13	15	3	
	3	8	11	13	12	3	
	4	9	10	11	10	2	
	5	16	8	10	6	1	
	6	18	8	7	4	1	
16	1	3	8	18	14	16	
	2	8	8	17	11	14	
	3	9	8	16	11	12	
	4	13	8	14	9	9	
	5	15	7	14	7	6	
	6	18	7	12	7	4	
17	1	3	5	12	4	11	
	2	6	4	12	4	10	
	3	13	4	11	4	10	
	4	14	3	10	4	9	
	5	18	1	9	4	9	
	6	19	1	8	4	8	
18	1	1	13	5	15	11	
	2	5	10	4	15	9	
	3	8	9	4	15	9	
	4	9	5	3	15	8	
	5	10	4	1	15	7	
	6	11	2	1	15	5	
19	1	1	20	14	8	18	
	2	6	18	13	8	17	
	3	7	17	12	8	16	
	4	8	15	12	8	14	
	5	11	13	12	8	12	
	6	12	12	11	8	11	
20	1	4	14	8	14	16	
	2	9	14	8	13	14	
	3	11	10	8	12	14	
	4	12	8	7	11	12	
	5	16	7	7	9	11	
	6	18	4	7	9	11	
21	1	2	16	9	18	14	
	2	12	14	8	14	11	
	3	13	14	8	13	8	
	4	18	11	7	12	7	
	5	19	9	7	10	4	
	6	20	5	7	8	1	
22	1	1	16	6	18	13	
	2	8	14	5	15	11	
	3	9	14	4	13	11	
	4	16	13	3	12	10	
	5	18	11	3	8	8	
	6	19	11	2	7	8	
23	1	5	11	13	16	20	
	2	9	10	13	16	15	
	3	10	10	13	13	13	
	4	11	9	13	11	10	
	5	15	9	13	10	8	
	6	16	8	13	9	4	
24	1	1	14	18	13	18	
	2	4	14	15	13	15	
	3	6	12	14	9	11	
	4	11	11	10	9	9	
	5	12	11	10	7	8	
	6	15	9	6	3	7	
25	1	4	20	17	18	15	
	2	5	18	16	12	14	
	3	7	15	15	12	12	
	4	8	13	15	10	6	
	5	10	12	15	5	4	
	6	20	10	14	3	1	
26	1	1	5	10	18	18	
	2	6	4	9	15	13	
	3	8	4	9	15	10	
	4	14	3	9	12	8	
	5	18	3	9	12	7	
	6	19	3	9	9	3	
27	1	1	16	15	6	9	
	2	10	15	14	5	7	
	3	12	12	13	5	6	
	4	13	7	12	5	4	
	5	15	4	11	5	4	
	6	19	4	11	5	2	
28	1	3	17	15	19	13	
	2	6	13	13	17	13	
	3	9	11	12	17	13	
	4	15	9	11	17	13	
	5	15	4	7	15	14	
	6	16	4	7	15	13	
29	1	4	9	8	14	7	
	2	5	7	8	13	7	
	3	6	6	7	12	7	
	4	7	6	6	10	6	
	5	11	4	5	10	5	
	6	14	3	4	8	5	
30	1	5	12	10	9	16	
	2	11	10	7	7	14	
	3	13	9	7	7	14	
	4	14	8	6	6	12	
	5	17	4	5	6	12	
	6	18	3	3	5	10	
31	1	3	11	19	16	17	
	2	4	9	15	14	17	
	3	7	7	11	14	17	
	4	11	6	10	13	17	
	5	12	3	7	12	17	
	6	19	3	5	11	17	
32	1	5	3	19	13	19	
	2	6	2	17	13	16	
	3	7	2	13	13	15	
	4	9	2	12	13	14	
	5	10	2	10	12	12	
	6	19	2	9	12	10	
33	1	1	11	17	16	18	
	2	3	10	16	13	16	
	3	8	9	12	12	15	
	4	12	8	9	10	14	
	5	14	5	8	9	13	
	6	15	5	5	6	11	
34	1	2	12	14	9	13	
	2	11	11	13	7	11	
	3	15	7	13	7	10	
	4	16	6	12	7	7	
	5	18	3	12	5	4	
	6	20	2	11	5	2	
35	1	4	11	17	7	20	
	2	5	9	16	6	18	
	3	6	8	14	5	17	
	4	8	6	14	4	16	
	5	9	5	13	2	15	
	6	20	4	12	2	13	
36	1	11	13	7	17	5	
	2	12	11	7	17	5	
	3	13	11	6	12	5	
	4	16	9	5	10	4	
	5	17	9	5	7	4	
	6	18	8	4	6	3	
37	1	5	17	13	4	12	
	2	10	16	12	3	10	
	3	12	15	11	3	9	
	4	14	14	11	3	8	
	5	19	12	9	3	7	
	6	20	11	9	3	6	
38	1	8	7	6	17	19	
	2	10	7	5	17	18	
	3	11	7	5	16	18	
	4	15	7	4	16	17	
	5	16	7	4	14	16	
	6	19	7	4	14	15	
39	1	1	17	16	3	17	
	2	9	15	14	3	17	
	3	11	13	13	3	17	
	4	12	12	13	3	17	
	5	16	12	11	3	17	
	6	17	9	10	3	17	
40	1	2	7	19	14	18	
	2	3	7	17	11	16	
	3	4	7	16	9	16	
	4	12	6	14	8	16	
	5	18	6	13	6	15	
	6	20	5	12	3	14	
41	1	1	8	15	13	16	
	2	3	8	13	12	15	
	3	15	8	13	11	14	
	4	16	7	12	10	14	
	5	17	7	10	8	13	
	6	18	6	9	8	12	
42	1	3	18	16	6	13	
	2	4	15	13	6	11	
	3	5	15	13	5	9	
	4	11	11	11	4	8	
	5	12	10	10	2	8	
	6	16	8	8	2	6	
43	1	2	20	6	18	17	
	2	5	15	5	17	17	
	3	6	13	5	17	17	
	4	7	10	4	16	16	
	5	8	5	4	16	16	
	6	9	5	3	16	16	
44	1	3	17	15	9	12	
	2	5	16	14	8	11	
	3	6	14	13	7	9	
	4	7	10	13	7	8	
	5	9	9	11	6	7	
	6	11	7	9	5	7	
45	1	1	18	18	10	3	
	2	4	17	14	8	3	
	3	8	16	11	7	2	
	4	9	16	7	7	2	
	5	15	14	7	5	2	
	6	18	14	3	5	1	
46	1	4	10	17	10	9	
	2	8	9	17	10	8	
	3	13	9	13	10	8	
	4	16	8	11	10	8	
	5	17	8	10	10	8	
	6	19	8	9	10	8	
47	1	4	18	6	19	18	
	2	5	18	6	18	16	
	3	12	17	4	17	15	
	4	13	17	4	16	13	
	5	14	17	3	15	12	
	6	20	16	2	13	11	
48	1	5	12	10	17	19	
	2	7	9	9	17	18	
	3	15	7	7	16	17	
	4	16	4	7	16	16	
	5	17	4	6	16	13	
	6	20	2	5	15	12	
49	1	1	16	16	6	17	
	2	2	13	12	5	16	
	3	3	12	11	4	15	
	4	4	8	10	3	15	
	5	10	5	7	2	14	
	6	18	2	6	1	13	
50	1	2	6	15	20	11	
	2	3	5	14	18	10	
	3	4	4	10	18	10	
	4	7	4	9	17	10	
	5	14	3	3	16	10	
	6	17	1	1	15	10	
51	1	1	17	16	15	1	
	2	4	16	15	14	1	
	3	8	14	15	14	1	
	4	13	14	14	12	1	
	5	18	13	14	12	1	
	6	19	11	14	11	1	
52	1	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2
	80	74	451	470

************************************************************************
