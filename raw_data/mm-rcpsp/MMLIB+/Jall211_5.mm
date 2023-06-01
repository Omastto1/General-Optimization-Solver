jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 4 R
- nonrenewable              : 4 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	7		2 3 4 5 6 10 13 
2	6	4		14 12 8 7 
3	6	5		16 15 14 12 11 
4	6	4		14 12 11 9 
5	6	4		26 16 15 12 
6	6	2		11 8 
7	6	3		21 15 11 
8	6	5		26 21 18 17 16 
9	6	5		21 19 18 17 16 
10	6	5		26 21 19 18 17 
11	6	4		26 19 18 17 
12	6	4		21 19 18 17 
13	6	3		26 18 15 
14	6	6		26 24 23 22 21 19 
15	6	4		24 23 22 19 
16	6	4		27 24 22 20 
17	6	5		32 30 27 24 22 
18	6	3		25 23 20 
19	6	3		27 25 20 
20	6	8		41 36 34 32 31 30 29 28 
21	6	6		41 36 32 31 29 27 
22	6	5		41 34 31 28 25 
23	6	4		41 36 29 27 
24	6	5		46 38 36 31 29 
25	6	4		46 38 36 29 
26	6	2		36 27 
27	6	2		38 28 
28	6	3		46 42 33 
29	6	2		37 33 
30	6	4		51 46 37 35 
31	6	1		33 
32	6	4		46 45 39 38 
33	6	3		51 40 35 
34	6	2		40 35 
35	6	3		45 43 39 
36	6	3		45 43 39 
37	6	4		50 45 43 42 
38	6	4		51 50 47 43 
39	6	3		50 49 44 
40	6	3		50 47 43 
41	6	3		50 45 44 
42	6	3		49 48 47 
43	6	2		49 48 
44	6	2		48 47 
45	6	1		47 
46	6	1		49 
47	6	1		52 
48	6	1		52 
49	6	1		52 
50	6	1		52 
51	6	1		52 
52	1	0		
************************************************************************
REQUESTS/DURATIONS
jobnr.	mode	dur	R1	R2	R3	R4	N1	N2	N3	N4	
------------------------------------------------------------------------
1	1	0	0	0	0	0	0	0	0	0	
2	1	1	24	12	17	20	27	24	27	25	
	2	13	20	10	17	17	26	22	23	23	
	3	15	19	9	16	16	24	19	20	19	
	4	22	17	5	15	12	23	13	17	15	
	5	26	16	3	14	11	21	12	14	13	
	6	30	13	2	12	8	18	5	13	12	
3	1	8	29	16	20	14	18	25	21	18	
	2	17	24	16	17	14	16	21	19	16	
	3	20	22	16	17	10	15	20	18	16	
	4	25	21	16	16	10	10	15	15	15	
	5	28	18	16	13	7	8	11	14	13	
	6	30	16	16	13	4	7	8	14	13	
4	1	5	22	16	11	26	27	20	10	29	
	2	19	22	16	10	23	26	20	8	21	
	3	20	21	14	10	21	26	16	8	19	
	4	25	20	13	10	21	24	14	8	13	
	5	29	18	12	8	19	23	12	7	13	
	6	30	18	12	8	17	23	9	6	7	
5	1	1	15	19	17	30	17	27	7	17	
	2	4	14	18	16	28	15	25	5	13	
	3	5	12	18	16	28	13	20	5	10	
	4	6	9	18	15	27	13	19	4	8	
	5	8	9	18	15	25	10	13	2	7	
	6	18	6	18	14	25	8	9	1	2	
6	1	4	16	21	27	11	15	24	26	17	
	2	8	11	21	21	11	14	23	25	15	
	3	13	11	20	18	11	13	23	25	13	
	4	16	7	20	13	11	12	21	24	10	
	5	21	3	19	8	11	12	20	24	7	
	6	22	2	18	5	11	11	20	23	4	
7	1	1	12	29	12	24	21	11	29	11	
	2	3	11	26	9	23	20	9	28	11	
	3	13	10	24	7	20	19	9	26	9	
	4	14	9	19	7	15	19	6	25	9	
	5	15	9	19	4	12	18	5	24	6	
	6	18	8	14	3	11	18	5	23	6	
8	1	6	4	9	11	5	23	30	20	8	
	2	7	3	8	10	4	20	29	19	8	
	3	10	3	6	9	4	16	29	16	7	
	4	14	3	6	7	4	13	28	13	7	
	5	21	2	4	5	3	10	28	8	7	
	6	26	2	2	3	3	4	28	5	6	
9	1	4	23	15	18	22	27	20	8	24	
	2	5	21	12	16	20	24	20	8	24	
	3	7	19	12	16	18	22	18	8	22	
	4	9	12	11	14	15	18	18	7	21	
	5	13	7	7	14	14	14	16	7	21	
	6	16	4	7	13	12	10	15	7	20	
10	1	6	19	18	21	18	22	13	19	10	
	2	12	16	13	21	14	19	12	19	9	
	3	13	15	13	21	13	19	11	18	8	
	4	16	12	7	20	9	17	9	16	6	
	5	19	9	6	19	8	15	6	14	5	
	6	27	9	4	19	4	13	6	12	5	
11	1	1	30	23	27	19	30	22	12	10	
	2	6	28	22	27	18	29	19	12	8	
	3	13	27	21	27	17	29	14	11	6	
	4	20	27	20	27	13	29	10	10	6	
	5	25	26	18	26	10	29	10	8	3	
	6	29	25	17	26	10	29	5	8	2	
12	1	7	18	23	28	20	25	16	22	22	
	2	15	18	22	26	19	22	16	22	22	
	3	16	17	21	26	17	19	16	22	22	
	4	24	17	21	25	11	19	16	22	22	
	5	29	17	20	23	10	15	16	21	22	
	6	30	16	18	23	8	13	16	21	22	
13	1	1	10	4	15	19	30	10	10	10	
	2	7	10	4	14	17	28	9	9	9	
	3	9	10	3	14	17	26	7	9	8	
	4	10	10	2	13	16	24	5	6	7	
	5	25	10	2	13	15	20	4	6	7	
	6	29	10	1	13	13	20	3	4	6	
14	1	3	14	12	11	28	22	24	9	26	
	2	5	13	10	10	26	20	22	7	24	
	3	6	12	7	9	25	15	16	6	21	
	4	13	12	6	9	22	12	14	5	20	
	5	24	9	4	8	22	8	8	4	18	
	6	29	8	4	8	21	5	3	2	17	
15	1	1	19	25	27	26	23	5	26	10	
	2	2	18	23	25	25	20	4	23	10	
	3	4	16	22	25	25	18	4	22	9	
	4	9	16	21	24	24	14	3	19	7	
	5	12	15	20	23	23	14	3	18	7	
	6	25	14	20	21	23	10	2	16	6	
16	1	5	23	8	17	26	18	28	7	8	
	2	11	22	7	16	21	17	26	6	7	
	3	13	19	7	13	16	14	26	6	6	
	4	16	18	7	10	11	12	24	6	6	
	5	17	17	7	6	8	8	23	6	5	
	6	18	16	7	5	4	7	20	6	4	
17	1	5	27	18	24	4	24	1	21	14	
	2	10	25	18	20	3	23	1	21	13	
	3	11	24	16	19	2	21	1	20	12	
	4	12	22	14	17	2	21	1	18	10	
	5	15	20	13	17	2	20	1	15	10	
	6	24	19	13	15	1	19	1	15	8	
18	1	2	28	15	22	9	30	29	27	19	
	2	3	27	11	21	8	27	25	24	18	
	3	10	27	11	20	8	25	20	23	15	
	4	12	27	7	19	8	24	15	22	14	
	5	19	27	5	18	7	24	10	21	14	
	6	29	27	2	18	7	22	7	20	12	
19	1	1	10	30	5	26	26	25	27	14	
	2	6	9	20	4	25	25	24	19	13	
	3	17	8	17	4	23	17	19	15	13	
	4	20	8	15	4	23	16	14	13	13	
	5	24	7	11	4	22	12	13	10	12	
	6	27	7	4	4	21	5	8	2	12	
20	1	6	29	28	30	12	28	5	16	29	
	2	16	29	25	23	12	24	4	15	26	
	3	17	27	22	22	12	23	3	15	24	
	4	21	26	18	18	11	21	3	15	24	
	5	22	26	14	14	11	21	2	15	21	
	6	30	25	12	12	11	19	1	15	21	
21	1	6	9	27	19	28	23	11	17	5	
	2	7	9	26	16	26	18	10	16	5	
	3	9	8	25	16	26	13	10	14	5	
	4	10	5	24	14	25	12	9	13	4	
	5	21	5	23	10	23	5	9	12	4	
	6	26	4	22	9	23	5	8	12	3	
22	1	1	27	18	26	20	29	11	29	25	
	2	9	24	15	21	17	24	11	28	25	
	3	19	21	14	16	15	22	11	28	25	
	4	26	20	13	12	13	21	11	28	25	
	5	29	17	11	6	10	18	10	28	24	
	6	30	14	9	3	6	15	10	28	24	
23	1	2	30	18	8	25	28	27	17	24	
	2	6	24	15	5	24	26	25	16	17	
	3	11	17	12	4	23	26	25	16	16	
	4	17	13	9	3	22	22	24	15	12	
	5	18	10	5	2	21	20	23	15	5	
	6	27	5	3	2	21	20	23	15	3	
24	1	7	28	14	9	16	28	23	27	7	
	2	8	28	11	7	13	23	19	23	7	
	3	9	28	10	7	11	21	15	17	6	
	4	10	27	6	7	8	16	13	12	6	
	5	11	27	4	5	7	10	9	10	6	
	6	25	27	4	5	6	10	7	5	5	
25	1	13	22	29	17	30	9	4	29	13	
	2	18	20	25	13	29	9	4	28	11	
	3	20	19	17	12	28	9	4	28	11	
	4	21	18	17	9	28	9	4	28	10	
	5	22	17	13	9	27	9	4	26	9	
	6	27	16	8	6	27	9	4	26	7	
26	1	4	16	25	29	23	22	11	23	28	
	2	8	13	25	27	19	19	9	19	25	
	3	13	13	22	25	18	14	8	17	21	
	4	16	8	21	23	16	10	6	9	15	
	5	18	5	21	21	14	7	5	8	10	
	6	22	3	19	18	12	6	3	3	4	
27	1	1	23	30	26	11	11	27	18	13	
	2	4	23	29	26	10	8	22	15	12	
	3	5	23	29	25	10	6	17	15	11	
	4	11	23	28	25	9	6	15	14	10	
	5	23	23	28	25	8	4	11	13	9	
	6	27	23	28	24	6	3	8	12	8	
28	1	5	14	2	26	23	20	9	12	11	
	2	7	14	2	25	23	20	9	12	11	
	3	12	14	2	22	23	20	8	9	11	
	4	18	14	2	21	22	20	8	6	11	
	5	29	14	1	18	22	20	7	4	11	
	6	30	14	1	14	21	20	7	3	11	
29	1	3	15	25	22	27	23	20	4	25	
	2	4	12	23	19	27	20	16	4	22	
	3	10	11	21	18	27	20	16	4	22	
	4	15	8	17	15	27	17	12	4	21	
	5	16	7	15	13	26	17	9	4	19	
	6	27	5	13	13	26	15	7	4	18	
30	1	3	16	22	9	27	25	13	28	16	
	2	8	15	20	8	26	23	12	28	13	
	3	13	13	20	8	25	18	12	28	13	
	4	20	12	20	7	24	16	11	28	10	
	5	27	11	19	7	23	14	11	28	9	
	6	28	10	18	6	23	10	11	28	8	
31	1	6	8	11	20	15	8	26	12	16	
	2	8	7	11	18	14	8	24	11	15	
	3	13	5	11	15	10	7	23	10	15	
	4	14	4	11	13	8	6	23	8	15	
	5	27	2	11	10	7	5	21	7	14	
	6	29	2	11	6	3	5	21	6	14	
32	1	1	30	28	22	17	13	24	30	24	
	2	3	25	22	20	15	10	17	28	22	
	3	5	18	18	18	15	8	14	28	17	
	4	6	13	14	14	14	5	14	27	14	
	5	14	11	11	11	13	3	8	26	11	
	6	28	6	5	7	13	3	5	25	7	
33	1	9	24	24	27	25	26	27	19	12	
	2	10	23	24	22	23	25	27	19	12	
	3	18	20	24	19	22	24	27	15	11	
	4	19	19	24	11	21	24	27	12	10	
	5	21	16	24	5	18	23	27	8	8	
	6	27	16	24	2	18	22	27	6	8	
34	1	2	13	14	17	24	10	20	10	21	
	2	6	12	14	15	22	8	17	9	18	
	3	13	12	14	13	21	7	15	9	16	
	4	14	12	13	12	18	4	9	9	12	
	5	17	12	13	11	17	2	4	8	12	
	6	23	12	13	10	16	2	4	8	8	
35	1	1	19	25	11	26	9	27	22	21	
	2	15	18	22	10	23	9	25	21	19	
	3	16	17	19	9	20	9	22	19	17	
	4	17	17	18	9	20	9	17	16	15	
	5	22	16	17	9	16	9	17	13	13	
	6	23	14	14	8	15	9	13	12	12	
36	1	1	24	17	17	22	19	18	22	10	
	2	12	22	14	16	20	19	18	22	10	
	3	18	19	12	14	16	19	18	21	10	
	4	19	15	9	12	14	18	17	19	10	
	5	20	10	7	11	10	18	17	17	10	
	6	21	8	2	9	7	18	17	16	10	
37	1	1	19	15	23	8	17	29	21	18	
	2	3	17	10	23	7	15	22	21	17	
	3	8	11	8	22	6	15	19	21	16	
	4	14	9	8	20	6	13	16	21	16	
	5	16	6	5	19	4	9	9	21	15	
	6	23	4	2	19	4	9	5	21	14	
38	1	2	14	18	26	15	26	23	16	10	
	2	4	14	16	24	13	20	23	15	10	
	3	5	12	14	23	13	19	22	12	10	
	4	21	8	12	22	11	16	22	9	10	
	5	28	7	10	21	10	13	22	7	10	
	6	30	3	10	21	10	12	21	5	10	
39	1	15	7	22	24	29	17	12	24	28	
	2	16	6	21	19	28	17	12	21	27	
	3	17	5	16	16	28	17	12	18	27	
	4	22	4	15	14	28	16	12	18	26	
	5	24	2	11	11	28	16	12	14	25	
	6	29	2	5	9	28	16	12	14	24	
40	1	1	18	24	21	18	25	17	19	13	
	2	2	17	23	17	18	23	17	19	12	
	3	9	16	23	17	15	22	16	19	12	
	4	14	15	23	15	15	19	16	19	11	
	5	22	15	23	11	12	16	15	19	10	
	6	28	14	23	11	12	11	14	19	10	
41	1	2	19	25	19	14	29	24	27	25	
	2	13	19	19	14	11	28	24	22	24	
	3	17	17	16	11	10	26	21	20	24	
	4	18	14	13	10	9	25	17	16	23	
	5	19	11	12	9	8	24	15	13	21	
	6	20	10	8	4	8	24	15	8	20	
42	1	13	21	23	20	28	16	12	8	14	
	2	15	20	23	18	27	16	10	8	12	
	3	17	19	19	16	23	14	8	8	9	
	4	20	17	17	12	23	13	5	7	7	
	5	25	16	14	11	19	12	3	6	6	
	6	29	16	14	8	18	11	1	6	5	
43	1	5	15	28	17	23	6	21	22	12	
	2	6	14	27	16	21	6	20	22	10	
	3	8	13	24	15	18	4	16	17	9	
	4	15	12	23	13	16	4	15	14	5	
	5	16	12	20	12	15	2	14	12	3	
	6	17	11	17	9	12	1	9	11	1	
44	1	11	30	27	11	5	26	25	14	12	
	2	14	22	25	11	4	24	25	12	10	
	3	15	20	17	11	4	22	24	9	9	
	4	17	13	13	11	3	20	23	7	8	
	5	23	12	12	11	2	16	22	5	7	
	6	29	8	8	11	2	16	21	3	6	
45	1	1	20	16	28	11	22	26	24	25	
	2	3	17	12	25	9	17	23	24	25	
	3	10	16	10	22	7	16	17	23	22	
	4	11	16	9	15	7	11	14	21	19	
	5	12	14	5	14	6	10	11	19	17	
	6	24	13	3	9	5	7	5	19	17	
46	1	8	15	26	28	16	9	10	22	27	
	2	9	14	26	22	16	7	7	20	25	
	3	12	14	25	18	16	7	6	18	20	
	4	13	14	23	15	16	5	5	15	16	
	5	22	14	23	13	16	5	2	13	12	
	6	24	14	22	9	16	4	1	10	10	
47	1	1	29	21	19	22	22	24	28	26	
	2	6	22	20	19	21	21	21	28	26	
	3	7	20	20	17	21	18	16	24	23	
	4	20	15	19	16	21	15	16	23	18	
	5	22	11	17	15	20	13	11	21	15	
	6	26	10	16	15	20	10	11	19	13	
48	1	1	16	20	19	30	26	21	24	24	
	2	5	12	17	19	26	25	17	21	22	
	3	17	10	17	18	25	25	14	19	20	
	4	20	9	15	16	24	25	13	13	20	
	5	22	8	13	14	20	25	11	13	16	
	6	23	5	11	12	20	25	7	8	13	
49	1	5	30	28	25	28	28	29	2	17	
	2	6	26	22	24	26	26	28	1	16	
	3	7	25	22	24	24	24	27	1	16	
	4	8	20	19	23	22	22	27	1	15	
	5	21	20	15	21	21	21	27	1	14	
	6	30	18	14	19	19	18	26	1	14	
50	1	1	17	14	18	14	14	17	22	20	
	2	2	17	13	18	13	14	15	21	18	
	3	12	16	12	17	13	12	12	21	17	
	4	13	16	10	16	11	10	10	20	17	
	5	17	16	8	15	11	8	8	20	15	
	6	30	15	8	14	10	7	7	20	13	
51	1	1	19	25	11	12	12	24	22	14	
	2	4	16	20	10	10	10	19	21	12	
	3	16	14	19	9	9	7	19	20	11	
	4	17	10	17	9	8	4	16	20	10	
	5	24	8	14	8	7	4	13	19	7	
	6	27	7	13	7	6	2	11	18	6	
52	1	0	0	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	R 3	R 4	N 1	N 2	N 3	N 4
	86	82	79	90	717	626	692	602

************************************************************************
