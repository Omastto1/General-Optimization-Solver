jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 4 R
- nonrenewable              : 4 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	7		2 3 4 5 6 7 10 
2	6	5		17 14 12 9 8 
3	6	5		17 15 14 13 11 
4	6	3		17 11 8 
5	6	4		18 15 14 11 
6	6	3		18 12 8 
7	6	3		18 15 11 
8	6	3		21 15 13 
9	6	3		19 18 16 
10	6	3		17 16 15 
11	6	2		19 16 
12	6	3		21 20 19 
13	6	3		25 20 19 
14	6	3		22 21 20 
15	6	3		25 22 20 
16	6	3		23 21 20 
17	6	4		26 25 24 23 
18	6	3		26 25 22 
19	6	2		24 23 
20	6	4		40 28 27 26 
21	6	3		29 28 25 
22	6	2		40 23 
23	6	3		29 28 27 
24	6	3		40 28 27 
25	6	2		40 27 
26	6	4		34 31 30 29 
27	6	5		38 36 33 32 31 
28	6	4		37 36 34 30 
29	6	3		39 35 33 
30	6	5		43 42 41 39 38 
31	6	4		47 41 37 35 
32	6	6		51 49 48 47 43 42 
33	6	3		50 47 37 
34	6	5		49 48 47 46 41 
35	6	5		51 49 48 45 42 
36	6	3		50 46 39 
37	6	3		49 43 42 
38	6	5		50 49 47 45 44 
39	6	4		51 48 47 44 
40	6	2		48 42 
41	6	3		50 45 44 
42	6	2		46 44 
43	6	1		45 
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
jobnr.	mode	dur	R1	R2	R3	R4	N1	N2	N3	N4	
------------------------------------------------------------------------
1	1	0	0	0	0	0	0	0	0	0	
2	1	1	18	24	21	24	26	27	24	27	
	2	3	17	17	18	24	24	27	23	25	
	3	6	15	15	14	24	23	27	20	25	
	4	9	14	13	9	24	21	26	18	24	
	5	16	14	8	7	23	19	26	17	23	
	6	17	13	6	4	23	17	25	13	23	
3	1	8	17	10	19	22	13	13	26	18	
	2	9	17	9	16	22	10	12	26	15	
	3	10	17	8	15	21	8	10	26	13	
	4	11	17	7	13	21	7	8	26	10	
	5	16	17	7	11	21	3	5	25	6	
	6	20	17	6	7	20	3	5	25	6	
4	1	9	30	28	16	24	25	23	16	23	
	2	13	28	27	13	23	22	22	16	23	
	3	15	27	27	13	15	15	21	16	21	
	4	22	26	27	9	11	15	20	15	18	
	5	23	24	27	4	11	11	18	14	15	
	6	26	23	27	4	5	7	17	14	14	
5	1	11	15	18	27	24	20	24	25	27	
	2	16	14	15	23	22	16	19	21	24	
	3	19	14	12	22	20	16	16	19	21	
	4	20	13	7	20	15	13	15	14	17	
	5	21	13	5	17	13	8	11	8	15	
	6	24	13	1	15	10	6	10	1	10	
6	1	10	29	27	24	24	20	10	11	18	
	2	12	21	21	23	22	19	9	9	17	
	3	15	16	20	23	21	17	9	8	16	
	4	16	15	15	22	17	16	9	7	14	
	5	20	7	13	21	14	15	9	4	14	
	6	29	5	12	21	10	15	9	4	11	
7	1	1	17	19	4	19	19	13	3	9	
	2	9	17	15	4	19	19	12	3	7	
	3	11	16	12	4	18	17	11	2	6	
	4	15	14	10	4	17	17	11	2	4	
	5	17	13	7	4	15	15	11	1	2	
	6	22	13	4	4	15	13	10	1	2	
8	1	10	24	29	23	22	23	10	18	9	
	2	18	19	26	23	21	18	10	13	9	
	3	24	19	20	22	18	18	9	11	8	
	4	27	17	18	22	18	14	9	8	8	
	5	28	12	16	22	16	11	8	7	8	
	6	29	12	12	21	14	10	7	4	7	
9	1	7	13	11	19	7	20	19	11	26	
	2	9	12	8	17	5	18	17	9	24	
	3	16	9	6	15	5	16	14	7	24	
	4	20	8	4	15	4	11	12	7	22	
	5	21	7	4	13	3	8	11	4	22	
	6	29	5	1	10	1	2	8	3	21	
10	1	9	29	10	25	11	17	23	24	23	
	2	11	25	9	23	11	14	20	20	21	
	3	13	21	8	22	7	14	16	19	18	
	4	24	14	7	22	7	13	13	17	18	
	5	27	10	6	21	3	10	7	15	16	
	6	28	8	6	20	2	9	3	12	13	
11	1	8	18	28	25	5	16	15	22	22	
	2	10	17	27	22	5	14	11	19	22	
	3	11	17	27	21	5	10	11	16	19	
	4	12	16	26	21	5	9	7	10	18	
	5	13	16	25	17	5	8	7	9	14	
	6	14	15	23	17	5	5	4	5	14	
12	1	3	26	14	25	13	18	12	29	21	
	2	4	20	14	23	13	17	10	22	20	
	3	13	15	14	19	13	13	9	21	18	
	4	16	11	13	15	12	8	9	13	15	
	5	29	6	13	11	12	7	7	10	13	
	6	30	4	13	9	12	5	6	6	12	
13	1	7	5	11	21	19	23	28	12	23	
	2	11	5	10	20	19	23	28	12	23	
	3	15	4	8	19	16	22	28	11	23	
	4	16	3	8	18	11	22	28	11	22	
	5	25	2	6	18	10	21	28	10	22	
	6	30	2	6	17	6	20	28	9	22	
14	1	3	28	27	24	25	24	23	21	21	
	2	7	25	25	22	21	22	23	21	18	
	3	8	25	22	22	16	22	23	21	12	
	4	15	24	22	20	10	20	23	20	10	
	5	18	22	21	17	9	17	23	19	8	
	6	27	20	18	15	1	16	23	19	6	
15	1	3	23	26	4	20	22	10	6	24	
	2	8	21	23	3	20	22	9	6	20	
	3	14	17	19	3	15	22	7	6	15	
	4	17	16	19	3	12	21	6	6	12	
	5	21	14	14	3	9	21	5	5	12	
	6	27	8	12	3	4	21	4	5	8	
16	1	7	27	10	8	27	13	15	26	17	
	2	10	25	10	8	27	12	14	25	17	
	3	17	23	8	7	27	12	12	24	17	
	4	20	20	7	6	27	12	11	22	17	
	5	22	15	6	5	27	12	11	22	17	
	6	24	14	6	4	27	12	9	20	17	
17	1	7	25	24	24	14	12	24	29	5	
	2	8	20	20	23	12	10	22	29	4	
	3	9	19	18	23	8	9	20	28	4	
	4	12	14	13	21	6	8	18	26	3	
	5	21	9	8	19	4	8	14	25	3	
	6	25	5	6	18	3	7	12	25	3	
18	1	6	15	16	19	11	25	24	29	18	
	2	15	14	15	18	9	22	21	25	15	
	3	19	13	12	18	9	22	20	20	14	
	4	20	13	9	17	9	18	19	18	11	
	5	21	12	8	17	7	16	16	12	9	
	6	22	11	5	17	7	15	12	11	9	
19	1	11	26	9	30	20	25	14	24	21	
	2	17	23	8	27	17	24	13	22	20	
	3	21	22	5	24	16	22	12	21	19	
	4	22	22	5	22	14	21	12	19	18	
	5	26	21	4	21	10	21	12	18	17	
	6	29	19	1	19	9	20	11	15	15	
20	1	1	10	16	21	12	13	12	22	24	
	2	2	9	11	21	11	11	11	21	19	
	3	3	9	10	16	8	10	11	20	16	
	4	7	8	8	15	7	9	10	20	15	
	5	11	8	6	11	5	6	9	20	8	
	6	16	8	5	7	4	5	9	19	7	
21	1	7	29	21	24	16	25	1	18	16	
	2	9	25	19	20	16	21	1	15	14	
	3	10	21	14	14	12	19	1	13	13	
	4	13	15	10	13	11	16	1	9	10	
	5	23	9	6	6	8	14	1	7	9	
	6	24	5	4	6	4	11	1	2	7	
22	1	1	29	17	20	25	17	19	30	18	
	2	10	26	17	17	22	17	16	27	15	
	3	13	25	17	13	19	15	12	23	12	
	4	15	23	17	10	14	14	9	22	10	
	5	22	22	17	8	9	13	8	20	7	
	6	29	18	17	8	1	13	2	19	5	
23	1	17	23	18	18	25	19	15	11	10	
	2	20	22	17	15	21	18	11	10	7	
	3	21	18	15	12	20	18	10	9	6	
	4	24	16	11	8	18	18	9	9	5	
	5	25	13	11	6	15	18	7	7	3	
	6	28	12	8	2	14	18	4	7	1	
24	1	1	11	20	29	5	27	21	26	17	
	2	5	11	14	24	3	24	20	25	14	
	3	7	7	13	15	3	21	20	16	12	
	4	10	6	10	10	2	17	20	15	11	
	5	16	5	7	10	2	16	20	10	7	
	6	17	1	3	4	1	13	20	6	5	
25	1	10	16	7	15	30	15	24	25	6	
	2	18	16	7	14	28	15	19	20	5	
	3	23	15	5	11	28	15	19	18	5	
	4	24	14	5	10	27	15	14	15	4	
	5	26	12	4	8	26	15	12	7	3	
	6	30	12	2	7	26	15	10	3	2	
26	1	6	24	30	23	24	24	23	20	10	
	2	9	23	30	21	22	19	22	18	8	
	3	21	20	30	17	22	16	21	15	7	
	4	22	18	30	14	17	15	19	11	5	
	5	26	14	30	13	17	11	18	8	5	
	6	27	10	30	9	12	11	16	6	3	
27	1	6	18	25	23	21	27	23	15	17	
	2	11	18	21	22	18	27	20	14	17	
	3	13	18	16	17	16	27	19	10	17	
	4	19	18	10	14	11	27	17	10	16	
	5	23	18	6	12	9	27	17	7	16	
	6	28	18	2	9	7	27	14	5	15	
28	1	5	24	23	15	12	14	8	25	27	
	2	6	20	19	14	12	13	8	20	24	
	3	11	17	17	13	10	13	8	18	19	
	4	15	16	15	11	10	12	7	13	15	
	5	17	14	13	10	9	10	6	11	15	
	6	22	11	13	10	8	10	6	7	9	
29	1	1	28	25	18	25	20	17	15	24	
	2	10	22	24	17	19	19	16	14	16	
	3	12	21	19	16	17	19	16	10	16	
	4	13	16	13	16	13	18	16	8	13	
	5	19	13	10	13	11	17	16	7	6	
	6	25	10	6	12	7	16	16	6	4	
30	1	3	4	23	28	16	16	20	3	5	
	2	6	4	21	26	15	14	16	3	4	
	3	12	4	21	24	15	13	13	3	4	
	4	16	3	19	20	14	12	10	3	3	
	5	26	3	15	17	13	12	9	2	3	
	6	29	3	15	16	12	11	5	2	3	
31	1	4	27	21	5	21	17	23	23	26	
	2	17	26	20	4	18	16	20	21	21	
	3	19	25	20	4	16	16	18	19	20	
	4	24	25	20	4	11	15	14	17	17	
	5	25	24	20	4	8	14	7	15	15	
	6	26	23	20	4	7	14	5	13	10	
32	1	1	19	26	17	20	25	16	27	8	
	2	9	18	25	16	16	22	15	23	8	
	3	17	15	24	15	14	20	15	18	8	
	4	19	13	22	10	8	15	13	13	7	
	5	20	12	19	10	8	13	13	9	7	
	6	22	10	19	7	3	10	12	7	7	
33	1	2	29	12	12	22	22	25	21	26	
	2	7	27	9	11	21	21	23	20	25	
	3	12	25	8	10	21	18	16	18	25	
	4	13	25	8	10	20	13	15	15	24	
	5	14	21	7	8	20	12	9	15	24	
	6	21	20	5	8	19	8	4	12	24	
34	1	5	25	14	21	23	19	22	17	2	
	2	7	22	14	20	21	16	21	15	2	
	3	13	15	13	19	18	11	21	11	2	
	4	19	12	10	19	16	7	19	9	1	
	5	20	9	9	18	15	6	18	3	1	
	6	30	4	7	18	14	3	17	3	1	
35	1	5	28	9	7	24	18	26	26	28	
	2	8	26	7	6	23	18	22	25	25	
	3	15	26	6	6	23	16	21	25	23	
	4	21	24	5	6	23	16	16	23	18	
	5	28	23	4	6	22	13	14	22	17	
	6	29	23	1	6	22	12	13	21	14	
36	1	7	22	26	20	22	22	14	18	13	
	2	11	18	26	14	17	20	14	17	12	
	3	20	17	24	11	16	19	13	14	11	
	4	25	14	22	9	12	15	13	14	10	
	5	27	12	20	5	12	14	12	12	9	
	6	29	9	19	4	9	10	12	9	9	
37	1	5	19	19	23	17	21	19	8	9	
	2	7	15	18	20	12	18	16	7	8	
	3	10	11	16	17	11	17	15	6	7	
	4	18	10	13	13	8	11	14	6	6	
	5	19	5	12	6	7	10	12	6	3	
	6	27	4	10	4	5	8	10	5	3	
38	1	2	28	19	11	24	9	7	29	30	
	2	10	26	17	10	23	8	6	25	29	
	3	11	24	14	10	20	8	5	25	28	
	4	14	23	11	10	16	7	5	22	28	
	5	28	23	6	10	15	6	3	20	27	
	6	30	21	4	10	13	6	3	16	26	
39	1	1	11	19	17	4	12	24	9	24	
	2	4	10	19	16	3	12	23	8	21	
	3	13	9	18	15	3	12	21	8	19	
	4	14	8	16	15	2	11	20	7	11	
	5	18	7	16	14	2	11	20	7	6	
	6	21	7	15	13	2	11	19	6	5	
40	1	4	13	21	27	14	24	28	6	21	
	2	15	12	19	23	14	22	25	5	18	
	3	17	9	18	19	13	21	23	5	17	
	4	18	7	15	15	11	20	16	5	17	
	5	23	6	12	8	10	19	14	4	14	
	6	26	4	9	3	7	17	11	4	14	
41	1	11	26	22	6	20	26	22	26	13	
	2	14	24	21	6	17	24	22	25	11	
	3	20	20	18	6	14	22	22	22	10	
	4	22	18	15	6	11	21	22	21	7	
	5	25	18	13	5	9	19	22	19	6	
	6	26	15	10	5	6	17	22	17	3	
42	1	2	15	26	9	13	10	19	14	20	
	2	7	15	24	9	10	10	18	14	17	
	3	19	13	20	9	9	9	15	14	13	
	4	26	13	14	9	8	8	13	14	9	
	5	28	11	12	9	7	8	10	14	5	
	6	29	10	7	9	5	7	8	14	2	
43	1	3	29	14	28	14	16	3	20	17	
	2	6	26	13	26	14	14	2	18	16	
	3	12	25	13	26	14	14	2	17	15	
	4	18	23	13	25	13	10	1	16	13	
	5	20	23	13	25	13	10	1	13	10	
	6	28	22	13	24	13	6	1	13	8	
44	1	8	22	27	11	24	28	19	26	12	
	2	13	16	26	11	22	24	16	22	8	
	3	20	13	24	10	21	18	14	19	8	
	4	25	12	22	9	18	15	10	18	5	
	5	26	8	22	8	15	10	9	17	4	
	6	29	6	20	6	14	9	3	13	2	
45	1	6	26	8	16	20	22	11	26	20	
	2	8	24	8	13	14	19	10	22	19	
	3	14	23	7	11	14	16	8	21	16	
	4	16	21	5	10	9	14	6	19	16	
	5	21	21	4	9	8	13	3	15	13	
	6	22	19	4	6	3	10	1	13	10	
46	1	1	15	28	27	9	9	12	8	22	
	2	8	14	27	22	8	8	10	7	21	
	3	16	14	27	17	8	7	9	5	14	
	4	18	14	25	14	8	5	8	4	10	
	5	23	14	25	12	8	5	8	4	8	
	6	27	14	24	9	8	4	6	2	1	
47	1	20	25	24	25	15	16	21	18	7	
	2	21	25	20	21	14	14	19	15	6	
	3	22	25	19	20	13	13	16	13	5	
	4	23	25	13	19	12	12	13	13	5	
	5	24	24	9	16	10	11	13	12	4	
	6	30	24	6	15	10	8	10	10	4	
48	1	1	29	21	23	16	30	11	27	27	
	2	11	25	19	23	16	29	11	25	27	
	3	12	23	14	20	16	27	11	23	27	
	4	18	21	10	19	16	26	10	22	26	
	5	23	20	7	16	16	26	10	19	26	
	6	28	16	1	16	16	24	10	17	25	
49	1	7	20	16	21	22	13	23	14	6	
	2	11	19	13	17	21	9	18	11	5	
	3	12	18	12	13	21	9	16	11	5	
	4	13	18	12	11	20	8	15	9	3	
	5	17	16	9	7	18	4	12	6	3	
	6	23	16	7	7	18	4	11	5	2	
50	1	13	15	18	17	14	29	21	30	29	
	2	16	13	17	13	12	29	20	26	26	
	3	21	11	17	13	10	29	19	20	26	
	4	25	10	16	10	8	28	19	16	24	
	5	28	8	16	7	7	28	18	12	21	
	6	30	8	15	4	4	28	17	7	19	
51	1	6	9	28	29	18	24	23	29	21	
	2	14	9	27	26	18	20	23	24	20	
	3	21	9	24	21	18	17	18	19	20	
	4	23	9	22	14	18	12	14	18	20	
	5	26	9	20	14	18	12	11	14	20	
	6	28	9	18	10	18	7	9	9	20	
52	1	0	0	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	R 3	R 4	N 1	N 2	N 3	N 4
	61	60	57	59	679	607	614	589

************************************************************************
