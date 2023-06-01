jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 4 R
- nonrenewable              : 4 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	8		2 4 5 6 7 8 10 11 
2	6	4		19 13 9 3 
3	6	6		22 18 17 16 15 12 
4	6	6		24 19 18 17 16 12 
5	6	4		22 19 15 13 
6	6	4		24 22 18 14 
7	6	4		24 22 19 14 
8	6	3		24 18 13 
9	6	4		24 22 21 14 
10	6	3		24 18 14 
11	6	4		28 25 20 19 
12	6	2		21 14 
13	6	1		14 
14	6	4		28 27 25 20 
15	6	4		28 27 24 23 
16	6	4		28 27 21 20 
17	6	5		31 29 28 27 25 
18	6	2		25 21 
19	6	2		27 21 
20	6	2		34 23 
21	6	2		31 26 
22	6	2		29 25 
23	6	3		31 30 29 
24	6	1		25 
25	6	1		26 
26	6	3		36 34 30 
27	6	5		36 35 34 33 32 
28	6	2		36 30 
29	6	4		41 36 35 33 
30	6	3		35 33 32 
31	6	3		37 35 33 
32	6	6		51 43 42 41 40 39 
33	6	3		43 39 38 
34	6	2		45 38 
35	6	5		51 49 45 42 40 
36	6	3		51 40 39 
37	6	5		51 45 44 43 42 
38	6	4		51 49 42 40 
39	6	5		50 49 48 47 45 
40	6	4		50 48 47 44 
41	6	4		50 49 48 46 
42	6	3		48 47 46 
43	6	3		49 47 46 
44	6	1		46 
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
jobnr.	mode	dur	R1	R2	R3	R4	N1	N2	N3	N4	
------------------------------------------------------------------------
1	1	0	0	0	0	0	0	0	0	0	
2	1	2	9	26	28	19	16	25	21	24	
	2	3	9	21	28	18	14	24	21	22	
	3	16	9	18	27	16	12	24	20	19	
	4	17	9	14	27	15	10	24	19	18	
	5	21	9	8	26	14	8	24	17	17	
	6	28	9	4	25	12	8	24	17	14	
3	1	6	5	15	10	27	23	26	8	22	
	2	7	3	15	10	21	18	24	8	19	
	3	13	3	14	9	18	15	22	8	18	
	4	17	3	13	8	12	14	22	7	16	
	5	18	2	12	8	9	12	20	7	15	
	6	27	1	11	7	4	7	19	7	15	
4	1	2	24	19	12	23	24	21	10	25	
	2	3	20	15	11	21	19	17	9	21	
	3	13	19	14	11	19	16	15	7	19	
	4	19	12	13	10	18	16	11	6	12	
	5	25	8	10	10	15	12	5	5	9	
	6	26	7	9	10	14	9	2	5	7	
5	1	1	20	25	11	15	18	25	18	27	
	2	4	16	22	8	14	15	24	16	25	
	3	13	13	17	8	14	12	23	16	20	
	4	14	13	12	7	14	10	23	14	17	
	5	15	9	8	5	14	7	21	13	16	
	6	19	9	4	5	14	3	21	13	12	
6	1	2	14	24	18	23	23	28	16	23	
	2	5	13	24	18	23	22	26	15	21	
	3	7	12	22	18	22	21	21	11	20	
	4	15	9	20	18	21	21	17	8	19	
	5	20	8	17	17	19	20	13	3	16	
	6	28	7	16	17	19	19	7	1	15	
7	1	3	26	11	19	20	21	28	25	27	
	2	9	23	11	19	16	16	27	24	22	
	3	14	23	10	13	14	15	27	24	20	
	4	20	21	10	12	14	13	27	22	16	
	5	25	19	9	6	10	10	26	21	15	
	6	27	19	8	5	10	10	26	20	12	
8	1	4	23	16	23	26	13	27	4	10	
	2	8	22	15	23	25	13	26	3	7	
	3	9	18	14	21	21	13	26	3	7	
	4	11	17	14	18	20	13	25	2	5	
	5	19	14	12	15	16	12	25	1	4	
	6	23	9	12	13	15	12	24	1	4	
9	1	1	26	29	16	23	9	19	26	16	
	2	11	22	28	15	23	9	17	22	15	
	3	12	15	27	14	19	9	16	18	15	
	4	13	13	25	14	15	9	14	12	12	
	5	14	9	25	12	11	9	12	9	11	
	6	18	4	24	11	9	9	11	5	10	
10	1	4	23	29	10	27	24	23	23	19	
	2	14	22	24	8	23	21	21	20	18	
	3	15	22	18	7	18	20	17	16	17	
	4	16	22	17	5	11	18	11	14	17	
	5	17	21	13	2	8	16	9	10	16	
	6	18	21	8	1	6	12	6	9	16	
11	1	15	26	16	26	16	9	22	26	17	
	2	21	25	15	26	15	9	19	24	16	
	3	22	23	13	26	13	9	18	24	15	
	4	25	22	13	26	12	8	16	23	15	
	5	26	21	12	26	12	8	15	22	14	
	6	30	20	11	26	11	8	14	22	13	
12	1	4	29	5	21	25	25	24	27	21	
	2	14	27	5	20	24	24	18	25	17	
	3	15	25	5	19	19	24	17	24	12	
	4	17	24	5	19	16	22	14	21	8	
	5	18	23	5	19	9	20	9	18	6	
	6	27	22	5	18	5	20	5	12	2	
13	1	5	27	15	6	12	28	17	4	24	
	2	10	27	13	5	12	23	13	4	22	
	3	16	27	13	5	8	22	11	4	20	
	4	27	27	12	4	8	19	10	4	16	
	5	29	27	11	2	6	14	9	4	14	
	6	30	27	10	1	2	12	7	4	13	
14	1	3	5	12	21	30	19	21	19	28	
	2	8	4	11	20	26	18	21	17	26	
	3	15	4	11	19	20	14	17	15	26	
	4	19	4	10	15	15	12	15	11	25	
	5	20	4	10	15	11	12	11	10	24	
	6	27	4	10	12	8	9	7	7	23	
15	1	3	18	16	27	22	7	9	12	20	
	2	4	15	15	22	19	7	7	11	19	
	3	5	14	13	20	18	5	6	10	18	
	4	13	11	8	19	12	5	6	9	17	
	5	14	11	6	17	9	2	5	8	16	
	6	28	8	2	13	9	2	4	8	15	
16	1	3	23	29	19	18	23	18	4	26	
	2	4	23	29	17	18	21	17	4	25	
	3	5	23	28	17	16	19	11	4	20	
	4	6	23	26	15	13	15	9	3	14	
	5	10	23	26	15	12	15	4	3	10	
	6	13	23	25	14	10	11	4	2	8	
17	1	12	25	25	15	21	5	27	27	27	
	2	19	22	24	12	16	4	24	24	25	
	3	21	20	23	9	14	4	16	20	24	
	4	23	18	23	6	8	4	13	19	23	
	5	26	16	22	6	8	4	9	17	20	
	6	27	15	21	3	2	4	7	14	19	
18	1	2	17	16	22	30	23	25	30	19	
	2	3	16	15	19	27	19	25	28	14	
	3	4	16	15	17	25	19	22	27	12	
	4	14	16	14	13	23	18	18	25	11	
	5	19	16	14	11	19	16	18	24	9	
	6	28	16	13	9	17	14	15	24	5	
19	1	1	30	26	12	21	30	24	27	24	
	2	5	22	21	11	20	28	23	26	23	
	3	6	21	17	7	15	28	20	26	23	
	4	12	17	14	5	12	28	15	25	22	
	5	24	12	11	5	8	26	14	24	22	
	6	30	7	6	2	6	26	10	24	21	
20	1	6	15	28	14	13	20	10	17	26	
	2	8	13	27	14	12	16	10	14	24	
	3	13	13	27	13	9	12	8	13	21	
	4	14	12	27	13	9	12	6	9	16	
	5	24	12	27	12	5	7	3	9	14	
	6	28	11	27	12	3	6	2	7	11	
21	1	4	17	13	23	12	21	10	19	22	
	2	15	16	10	23	12	20	10	17	22	
	3	19	16	8	21	11	17	10	14	18	
	4	23	14	8	21	9	15	10	12	14	
	5	24	13	6	18	8	14	10	10	13	
	6	30	13	3	18	6	9	10	10	11	
22	1	3	9	27	22	11	25	24	10	16	
	2	11	7	25	20	10	24	21	8	14	
	3	12	6	23	19	9	19	19	8	12	
	4	26	5	23	18	9	15	17	7	10	
	5	28	3	21	16	8	6	17	6	8	
	6	30	1	21	15	7	4	14	5	8	
23	1	2	28	16	23	28	21	20	25	18	
	2	12	25	15	22	27	21	14	18	17	
	3	13	24	14	19	26	21	10	17	16	
	4	14	22	13	16	24	21	10	11	16	
	5	27	19	11	15	23	21	4	6	16	
	6	29	17	9	13	22	21	2	3	15	
24	1	2	6	17	11	21	20	7	25	28	
	2	3	5	16	8	18	20	6	22	27	
	3	5	5	14	7	17	17	6	19	25	
	4	21	5	14	6	15	16	5	16	24	
	5	25	5	13	6	15	13	5	10	24	
	6	29	5	11	4	13	12	4	9	23	
25	1	2	18	23	19	29	22	28	3	26	
	2	9	16	19	17	22	22	25	3	23	
	3	15	14	18	15	17	19	18	3	18	
	4	18	14	14	9	12	18	17	3	15	
	5	21	12	9	7	9	14	13	3	8	
	6	25	11	4	3	4	14	9	3	2	
26	1	2	18	19	19	19	14	19	27	28	
	2	3	17	16	18	17	14	18	26	27	
	3	7	17	15	18	16	12	14	26	26	
	4	12	17	14	17	13	9	10	24	26	
	5	19	17	13	16	11	4	7	23	25	
	6	23	17	12	15	11	4	7	23	25	
27	1	1	5	21	8	16	24	11	27	25	
	2	5	5	18	8	15	18	10	26	25	
	3	14	4	16	7	13	16	10	25	25	
	4	18	3	15	6	9	12	9	23	25	
	5	25	3	13	5	8	6	8	22	25	
	6	29	2	13	3	3	3	8	21	25	
28	1	5	18	22	6	9	20	28	16	19	
	2	6	16	20	5	9	17	23	16	16	
	3	12	16	20	4	9	14	21	16	16	
	4	14	15	17	3	8	12	17	16	14	
	5	21	13	14	2	8	10	16	16	13	
	6	24	10	13	2	8	6	11	16	11	
29	1	6	8	16	26	7	11	21	14	19	
	2	10	8	16	21	6	8	19	12	15	
	3	15	7	15	21	4	6	19	11	13	
	4	17	7	15	19	3	4	17	9	12	
	5	18	5	14	15	2	3	16	7	8	
	6	30	5	14	13	2	2	15	4	7	
30	1	6	26	23	11	28	14	11	30	12	
	2	10	25	22	10	22	12	9	29	11	
	3	19	24	22	10	17	11	8	29	11	
	4	24	24	21	8	14	11	7	28	11	
	5	27	22	20	8	10	9	4	27	10	
	6	30	22	19	7	5	9	3	27	10	
31	1	3	5	17	11	13	15	7	25	5	
	2	8	4	14	9	12	15	7	25	5	
	3	17	4	12	9	12	13	5	25	4	
	4	22	3	8	6	12	12	4	25	4	
	5	27	2	7	5	10	12	2	25	2	
	6	30	2	6	4	10	10	2	25	2	
32	1	8	28	18	16	16	29	26	15	26	
	2	12	27	17	16	12	23	25	12	26	
	3	13	24	14	16	11	15	23	10	26	
	4	21	23	13	16	10	11	19	9	26	
	5	22	20	11	16	7	10	18	7	25	
	6	25	18	9	16	4	1	17	6	25	
33	1	8	18	25	24	29	17	28	23	14	
	2	16	16	24	23	29	16	23	21	12	
	3	20	13	22	23	29	16	23	20	11	
	4	21	11	17	23	29	16	18	17	9	
	5	23	10	15	22	28	16	14	16	8	
	6	29	9	13	22	28	16	12	14	5	
34	1	7	11	28	18	8	23	26	21	20	
	2	11	9	26	17	8	23	23	19	19	
	3	19	7	24	14	8	20	21	15	16	
	4	25	4	20	13	8	18	16	13	16	
	5	28	4	19	9	8	15	15	11	13	
	6	29	1	16	7	8	13	12	8	13	
35	1	2	25	23	24	18	25	11	22	22	
	2	10	25	23	18	16	24	10	22	20	
	3	16	17	23	16	13	15	8	20	18	
	4	21	14	22	10	13	13	6	18	17	
	5	22	6	21	10	9	7	4	17	15	
	6	24	2	21	5	9	4	4	15	13	
36	1	1	11	15	29	7	16	24	7	22	
	2	2	9	14	26	5	16	24	7	22	
	3	3	7	14	25	5	15	21	5	22	
	4	11	7	12	24	5	14	16	5	22	
	5	23	5	11	23	3	12	13	3	22	
	6	30	4	11	22	3	11	12	3	22	
37	1	3	30	24	26	18	14	22	25	19	
	2	5	29	21	18	17	13	19	23	18	
	3	6	29	18	16	17	12	17	20	17	
	4	7	29	15	11	17	12	15	15	15	
	5	14	29	14	8	16	12	15	12	12	
	6	27	29	10	4	16	11	13	9	10	
38	1	2	11	30	14	22	28	19	25	22	
	2	3	10	29	13	20	25	19	22	21	
	3	5	9	29	12	20	25	19	17	17	
	4	11	9	28	11	18	23	19	14	17	
	5	25	7	28	10	16	21	19	12	15	
	6	28	7	27	9	14	20	19	9	12	
39	1	9	28	23	16	26	12	9	18	16	
	2	16	27	18	15	22	8	6	17	13	
	3	21	27	15	12	18	7	6	17	12	
	4	23	26	14	11	15	5	4	17	11	
	5	25	25	7	8	14	3	2	17	11	
	6	29	24	6	7	10	2	2	17	10	
40	1	6	13	23	23	11	12	9	25	15	
	2	22	11	23	22	9	12	7	22	13	
	3	25	11	22	19	8	12	7	19	10	
	4	27	10	21	18	7	12	6	14	9	
	5	29	8	18	14	5	12	5	10	6	
	6	30	8	18	13	1	12	5	7	5	
41	1	2	22	14	20	9	19	20	9	10	
	2	4	19	12	17	9	18	20	8	10	
	3	13	15	9	16	8	17	19	7	8	
	4	16	13	8	14	8	12	18	7	8	
	5	17	12	6	10	8	11	17	6	5	
	6	27	7	6	8	7	8	15	6	5	
42	1	8	23	24	29	17	15	13	19	17	
	2	13	22	23	29	16	13	12	18	15	
	3	14	15	20	29	15	11	10	14	10	
	4	15	14	19	29	13	10	9	11	9	
	5	17	10	17	29	12	7	5	9	5	
	6	19	5	16	29	11	5	5	4	1	
43	1	2	19	11	28	28	22	3	20	19	
	2	5	17	11	28	27	20	2	19	19	
	3	6	15	7	28	27	16	2	12	19	
	4	12	12	5	28	26	16	1	11	19	
	5	20	11	3	28	25	13	1	9	19	
	6	21	7	1	28	23	8	1	3	19	
44	1	6	17	28	10	28	14	26	21	20	
	2	15	17	28	8	24	13	26	19	18	
	3	16	16	28	7	20	10	24	18	15	
	4	22	15	28	5	17	9	24	16	11	
	5	28	14	28	5	15	5	23	16	6	
	6	29	14	28	3	11	4	22	14	6	
45	1	4	15	28	20	23	12	14	22	26	
	2	5	14	28	19	18	11	14	21	26	
	3	14	13	28	17	13	8	14	16	23	
	4	15	11	27	16	8	5	14	14	22	
	5	16	10	27	15	6	5	14	14	22	
	6	29	10	27	14	4	3	14	11	19	
46	1	1	18	16	27	25	10	7	5	22	
	2	4	16	14	25	18	9	5	4	20	
	3	14	16	10	22	17	7	5	4	16	
	4	18	15	9	20	14	5	4	3	12	
	5	22	14	8	16	10	4	3	3	8	
	6	24	14	5	11	6	1	2	2	8	
47	1	2	29	20	24	10	13	20	28	12	
	2	3	21	20	17	9	13	18	24	11	
	3	8	20	20	14	8	13	17	23	11	
	4	9	14	19	13	7	13	17	20	8	
	5	11	10	19	8	6	13	15	20	7	
	6	22	9	19	4	6	13	15	17	7	
48	1	8	8	29	26	15	14	24	24	8	
	2	9	7	27	25	15	12	24	24	6	
	3	10	6	23	25	15	11	17	18	6	
	4	13	5	19	24	15	10	11	16	6	
	5	15	5	17	24	15	8	10	13	5	
	6	23	4	14	23	15	5	3	11	4	
49	1	2	28	15	21	19	27	25	17	21	
	2	6	21	14	21	14	22	24	14	20	
	3	8	16	14	21	14	16	24	13	20	
	4	10	14	14	20	11	13	24	13	19	
	5	15	10	14	20	8	8	23	11	19	
	6	16	7	14	20	4	5	23	10	19	
50	1	4	9	15	21	24	9	25	11	13	
	2	5	7	13	20	21	8	24	11	13	
	3	6	7	11	18	19	6	24	11	13	
	4	7	5	10	15	14	6	22	10	13	
	5	19	5	7	14	10	5	21	10	13	
	6	29	4	4	10	5	3	21	10	13	
51	1	4	25	8	10	23	19	23	10	24	
	2	5	25	7	9	20	18	19	8	23	
	3	15	24	5	7	19	15	18	5	22	
	4	27	24	5	7	16	12	15	5	22	
	5	28	23	3	4	15	9	13	2	21	
	6	29	22	3	3	13	6	12	2	20	
52	1	0	0	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	R 3	R 4	N 1	N 2	N 3	N 4
	66	64	61	64	563	641	627	710

************************************************************************
