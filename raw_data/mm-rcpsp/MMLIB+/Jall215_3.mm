jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 4 R
- nonrenewable              : 4 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	6		2 3 4 5 6 9 
2	6	7		18 15 13 12 11 8 7 
3	6	7		17 16 15 14 13 12 10 
4	6	7		18 17 16 15 14 12 10 
5	6	6		18 16 13 12 11 10 
6	6	6		25 18 17 12 10 8 
7	6	5		25 21 17 16 10 
8	6	4		20 19 16 14 
9	6	4		25 24 19 16 
10	6	4		24 22 20 19 
11	6	4		24 23 20 17 
12	6	3		24 22 21 
13	6	3		25 24 21 
14	6	2		24 21 
15	6	3		25 24 20 
16	6	2		23 22 
17	6	3		33 27 22 
18	6	4		33 32 27 24 
19	6	3		30 26 23 
20	6	5		33 32 30 29 26 
21	6	2		29 23 
22	6	4		32 30 29 26 
23	6	4		33 32 28 27 
24	6	2		30 26 
25	6	3		36 35 28 
26	6	2		36 28 
27	6	6		42 40 38 37 36 35 
28	6	3		40 34 31 
29	6	5		40 38 37 36 35 
30	6	4		42 39 38 36 
31	6	3		41 38 37 
32	6	3		41 38 37 
33	6	3		41 38 37 
34	6	5		51 43 42 41 39 
35	6	4		51 43 41 39 
36	6	5		51 49 44 43 41 
37	6	2		43 39 
38	6	5		51 49 48 47 46 
39	6	3		49 46 44 
40	6	3		48 46 43 
41	6	3		48 46 45 
42	6	3		47 46 45 
43	6	2		50 45 
44	6	2		47 45 
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
2	1	4	18	12	19	25	22	19	16	22	
	2	5	18	11	17	23	20	17	13	17	
	3	6	18	10	14	23	17	15	13	17	
	4	17	18	10	13	21	15	12	10	12	
	5	20	18	9	11	20	12	9	9	11	
	6	26	18	9	7	19	8	7	7	9	
3	1	5	10	22	7	17	24	6	10	26	
	2	6	7	22	6	13	23	5	9	24	
	3	8	7	21	6	13	22	5	9	19	
	4	9	6	20	5	9	22	5	9	12	
	5	13	4	19	5	7	20	3	9	7	
	6	18	4	19	5	6	20	3	9	1	
4	1	1	9	24	22	25	30	6	16	21	
	2	10	8	21	20	25	29	6	11	17	
	3	11	7	16	15	23	28	6	9	12	
	4	13	6	10	15	22	26	6	7	11	
	5	19	6	6	9	22	25	6	6	6	
	6	20	4	5	8	21	25	6	4	4	
5	1	1	22	14	23	21	23	19	26	27	
	2	5	20	11	23	19	20	19	25	22	
	3	22	17	8	21	19	20	17	24	15	
	4	28	15	7	19	18	17	14	22	13	
	5	29	12	4	17	18	11	14	21	7	
	6	30	10	2	14	17	10	12	21	1	
6	1	4	9	10	24	21	29	26	27	19	
	2	6	7	10	19	21	28	23	23	19	
	3	7	6	10	17	18	27	20	20	16	
	4	28	6	10	17	16	27	18	17	14	
	5	29	3	10	12	13	26	10	15	13	
	6	30	3	10	10	11	24	9	13	10	
7	1	3	18	9	26	21	21	8	29	26	
	2	12	16	8	25	17	19	6	29	24	
	3	18	15	7	25	15	15	6	29	22	
	4	27	8	7	24	13	13	5	29	20	
	5	29	5	5	24	7	10	4	29	19	
	6	30	1	5	24	5	4	2	29	18	
8	1	3	17	8	11	15	19	21	24	1	
	2	11	17	7	8	15	19	20	19	1	
	3	15	14	5	8	15	19	18	15	1	
	4	18	13	5	6	14	18	14	11	1	
	5	27	13	3	4	14	18	13	10	1	
	6	28	10	2	1	13	18	11	7	1	
9	1	7	2	16	20	15	23	12	8	18	
	2	10	1	15	17	14	18	11	8	16	
	3	12	1	14	17	13	14	9	7	14	
	4	14	1	14	13	11	12	8	5	13	
	5	26	1	12	12	8	9	7	4	10	
	6	27	1	12	11	8	1	7	4	9	
10	1	2	16	19	19	7	28	4	15	22	
	2	8	16	18	18	6	26	4	12	21	
	3	11	16	16	17	6	24	3	8	19	
	4	12	15	15	13	6	18	3	6	17	
	5	14	15	11	12	4	16	1	6	15	
	6	15	15	9	7	4	14	1	2	14	
11	1	2	13	6	25	4	27	18	18	4	
	2	3	9	6	22	3	26	15	15	4	
	3	8	8	6	18	3	25	13	10	3	
	4	10	6	6	15	2	23	10	10	3	
	5	23	6	6	14	2	22	8	7	2	
	6	24	3	6	10	2	21	8	2	2	
12	1	2	15	24	24	21	4	26	28	24	
	2	8	12	23	23	17	4	22	25	21	
	3	10	11	23	23	16	4	21	24	18	
	4	14	10	23	23	15	4	19	22	15	
	5	22	7	22	23	10	4	18	22	14	
	6	27	5	22	23	10	4	17	21	11	
13	1	3	21	21	24	19	10	7	21	22	
	2	5	19	18	21	16	10	5	20	20	
	3	9	18	17	21	15	9	5	18	17	
	4	11	17	17	19	14	6	3	17	12	
	5	19	14	15	15	11	6	2	17	10	
	6	21	13	12	14	9	5	2	16	5	
14	1	3	7	17	23	8	30	22	14	21	
	2	17	6	14	19	8	26	20	13	17	
	3	18	6	12	14	8	21	16	12	17	
	4	19	5	9	11	7	18	14	11	12	
	5	26	5	9	10	7	14	8	10	8	
	6	27	5	7	5	7	13	6	10	7	
15	1	2	26	12	14	26	25	16	20	24	
	2	4	25	11	12	20	25	13	19	23	
	3	15	25	9	12	16	24	11	18	17	
	4	16	25	7	10	10	22	11	17	11	
	5	19	25	5	10	6	21	8	16	9	
	6	22	25	5	9	1	21	5	15	4	
16	1	1	14	19	23	10	12	6	19	24	
	2	17	13	17	22	9	12	5	17	22	
	3	18	13	14	22	8	12	5	14	19	
	4	20	13	13	22	8	12	3	11	18	
	5	21	12	8	22	5	12	2	10	17	
	6	28	11	8	22	4	12	2	8	16	
17	1	3	24	13	23	24	22	17	19	29	
	2	6	22	12	21	24	22	16	18	24	
	3	10	22	11	18	21	22	16	18	16	
	4	11	20	11	16	20	22	15	18	12	
	5	17	18	9	16	18	22	15	17	6	
	6	18	16	8	12	18	22	14	17	5	
18	1	1	18	13	14	8	12	29	23	29	
	2	6	17	10	14	8	10	27	21	25	
	3	7	17	9	14	7	10	26	21	20	
	4	11	16	7	13	6	8	26	19	17	
	5	19	14	5	13	6	7	23	18	12	
	6	25	13	3	13	5	5	23	17	7	
19	1	2	26	10	22	21	19	16	26	7	
	2	4	22	9	18	21	14	13	22	7	
	3	6	19	9	16	21	12	10	19	7	
	4	9	18	9	14	20	9	6	17	7	
	5	28	15	9	11	20	9	6	10	7	
	6	29	12	9	10	20	7	2	7	7	
20	1	7	23	22	21	24	19	18	12	18	
	2	8	23	15	20	23	16	16	11	16	
	3	19	23	14	16	23	14	15	10	16	
	4	21	23	9	13	22	9	12	10	14	
	5	25	23	6	7	21	7	11	8	11	
	6	28	23	4	6	20	5	9	8	10	
21	1	2	8	21	19	15	28	22	27	16	
	2	12	7	21	19	12	23	20	25	12	
	3	13	5	19	19	10	19	16	24	11	
	4	19	4	19	19	8	16	12	22	7	
	5	20	3	17	19	5	14	5	20	5	
	6	21	2	17	19	3	13	1	19	3	
22	1	4	19	15	25	7	26	25	11	20	
	2	8	19	14	21	6	24	24	10	15	
	3	11	18	13	19	6	24	21	7	13	
	4	20	18	12	18	5	21	21	7	10	
	5	27	17	10	15	5	18	18	6	5	
	6	29	16	9	11	4	17	17	4	5	
23	1	3	12	7	22	21	20	18	29	23	
	2	4	11	7	22	15	19	17	27	22	
	3	5	9	5	22	12	18	17	26	21	
	4	6	6	5	21	12	17	16	25	19	
	5	10	5	3	20	8	16	14	21	18	
	6	23	4	2	20	5	15	14	21	18	
24	1	1	23	19	18	2	19	25	13	23	
	2	3	22	18	18	2	18	21	13	23	
	3	5	21	16	16	2	14	21	11	22	
	4	15	20	15	12	2	13	18	7	21	
	5	20	18	14	11	2	11	16	7	21	
	6	25	18	14	10	2	9	14	3	20	
25	1	1	24	18	5	10	21	13	11	23	
	2	11	23	16	5	9	20	12	11	23	
	3	12	23	15	5	9	19	9	11	18	
	4	18	22	14	5	9	17	6	11	15	
	5	23	22	14	5	8	16	6	11	10	
	6	28	22	13	5	8	15	4	11	6	
26	1	2	21	19	17	17	12	26	15	9	
	2	4	20	18	16	16	12	23	14	8	
	3	15	15	17	13	16	12	18	13	8	
	4	21	13	16	10	14	12	13	13	7	
	5	25	10	15	10	13	11	7	12	7	
	6	29	7	12	7	13	11	2	12	7	
27	1	4	22	21	25	6	14	27	3	28	
	2	5	22	21	19	6	12	27	3	27	
	3	14	20	21	16	5	12	26	3	26	
	4	15	18	21	14	4	11	24	3	25	
	5	19	17	21	7	3	10	23	3	24	
	6	20	15	21	7	1	10	23	3	24	
28	1	1	6	12	19	15	15	19	6	12	
	2	12	6	11	16	14	14	15	4	12	
	3	13	4	10	13	12	12	12	4	12	
	4	15	4	9	12	7	11	11	3	12	
	5	23	2	8	7	7	8	5	3	12	
	6	27	2	6	6	3	8	5	2	12	
29	1	4	18	9	26	10	20	25	16	22	
	2	7	14	8	23	10	18	21	15	19	
	3	14	13	7	22	9	15	21	15	18	
	4	15	7	5	20	9	15	16	14	11	
	5	21	4	3	20	8	11	13	13	7	
	6	28	3	1	19	8	7	10	13	5	
30	1	4	23	4	18	16	18	17	21	27	
	2	5	19	4	16	16	14	17	18	22	
	3	18	17	4	15	12	12	15	18	19	
	4	19	15	4	15	7	11	12	15	17	
	5	25	11	4	13	7	7	11	14	10	
	6	29	9	4	13	3	5	10	12	8	
31	1	2	25	12	24	22	26	18	3	29	
	2	7	22	10	24	21	24	18	3	27	
	3	8	19	7	24	18	24	16	3	26	
	4	9	18	6	24	17	24	15	3	26	
	5	10	17	4	23	15	22	13	3	23	
	6	15	14	3	23	13	22	11	3	22	
32	1	9	15	22	20	19	29	24	18	28	
	2	11	15	21	20	18	27	23	16	26	
	3	18	10	21	19	16	24	23	15	24	
	4	27	10	20	19	15	22	23	12	24	
	5	29	4	19	18	11	19	22	7	23	
	6	30	3	19	17	8	19	22	6	21	
33	1	1	21	22	23	19	15	28	6	4	
	2	12	18	19	20	17	13	28	5	3	
	3	15	16	19	16	16	10	27	4	3	
	4	18	14	17	12	15	7	27	3	3	
	5	23	11	17	11	14	5	26	3	2	
	6	29	8	16	6	12	4	26	2	1	
34	1	4	17	24	23	19	14	12	25	12	
	2	7	14	19	23	19	12	10	24	9	
	3	13	11	18	21	19	12	7	21	8	
	4	14	11	15	18	18	12	6	18	6	
	5	29	7	12	14	18	10	4	12	6	
	6	30	7	7	14	18	10	1	10	3	
35	1	1	9	23	11	20	28	19	21	19	
	2	2	8	23	9	19	27	16	19	18	
	3	12	7	21	9	16	24	15	15	18	
	4	19	6	20	9	16	22	12	15	17	
	5	27	5	20	8	14	21	11	13	17	
	6	28	5	19	7	11	18	8	10	17	
36	1	8	24	22	24	25	7	20	10	22	
	2	10	18	18	23	24	6	16	9	21	
	3	25	17	13	23	23	6	13	8	20	
	4	26	12	12	22	23	6	12	7	19	
	5	29	11	9	21	22	5	8	6	17	
	6	30	8	4	20	21	5	6	5	16	
37	1	7	19	21	11	20	26	23	6	28	
	2	10	17	20	9	18	23	18	6	24	
	3	12	14	20	8	15	19	14	6	21	
	4	19	12	19	7	15	17	9	6	14	
	5	25	10	18	7	11	15	6	6	12	
	6	30	10	18	6	10	11	3	6	8	
38	1	6	22	12	8	12	15	12	29	11	
	2	7	21	12	7	11	15	12	26	10	
	3	8	21	12	7	9	12	12	21	10	
	4	16	21	12	6	8	12	11	17	9	
	5	20	19	11	6	4	10	11	14	8	
	6	29	19	11	6	3	8	11	11	8	
39	1	1	24	7	11	9	15	23	12	7	
	2	5	17	6	8	8	13	17	11	7	
	3	6	15	6	8	8	12	17	9	6	
	4	10	14	5	7	7	10	13	7	6	
	5	26	8	4	5	7	8	9	6	5	
	6	28	8	3	4	7	7	7	6	5	
40	1	4	16	16	18	22	22	14	24	20	
	2	11	12	14	17	18	17	11	24	18	
	3	12	11	11	11	18	15	9	17	15	
	4	14	10	10	9	15	14	7	12	13	
	5	20	8	8	8	12	9	5	11	10	
	6	30	6	8	5	10	5	5	4	10	
41	1	11	15	23	18	23	29	26	18	24	
	2	13	12	21	18	21	27	23	15	21	
	3	22	8	20	18	18	26	17	11	20	
	4	23	6	20	18	18	25	15	10	17	
	5	24	4	18	18	15	24	11	7	14	
	6	28	1	16	18	11	21	11	3	11	
42	1	4	15	3	23	13	16	14	12	16	
	2	9	13	3	21	12	15	14	10	15	
	3	11	13	3	17	12	15	10	10	12	
	4	20	9	2	15	10	14	6	7	9	
	5	23	9	1	9	9	13	4	5	6	
	6	28	5	1	6	9	13	2	4	3	
43	1	3	16	3	22	3	26	24	25	26	
	2	5	13	3	21	2	26	19	25	24	
	3	7	13	3	20	2	22	15	21	22	
	4	9	10	2	20	2	20	14	20	21	
	5	15	9	2	18	2	18	11	18	21	
	6	25	9	2	18	2	14	4	17	19	
44	1	6	10	6	22	18	23	18	24	13	
	2	17	10	6	21	16	22	14	21	13	
	3	21	10	6	21	14	22	13	19	13	
	4	23	9	6	21	13	21	12	14	13	
	5	26	8	6	21	11	19	10	10	13	
	6	27	8	6	21	9	18	6	7	13	
45	1	5	4	9	15	15	6	24	14	12	
	2	6	3	8	14	15	5	19	12	11	
	3	7	3	8	13	15	5	13	11	10	
	4	16	3	8	12	15	5	9	8	9	
	5	21	3	7	8	15	5	6	6	8	
	6	29	3	7	8	15	5	1	5	8	
46	1	1	12	12	25	19	8	23	23	12	
	2	4	12	10	23	18	6	18	22	12	
	3	9	9	10	21	18	5	15	22	10	
	4	16	8	9	19	16	4	12	22	8	
	5	27	4	7	17	14	3	8	22	5	
	6	28	4	6	16	13	2	7	22	2	
47	1	10	24	8	11	23	7	12	23	19	
	2	15	23	7	9	22	6	11	21	16	
	3	17	22	6	7	21	6	10	20	13	
	4	21	19	5	7	19	6	10	17	11	
	5	22	18	5	5	18	6	9	15	9	
	6	29	17	3	5	17	6	8	15	7	
48	1	8	11	20	21	22	18	24	9	4	
	2	10	10	19	19	21	18	21	9	3	
	3	12	10	18	14	19	16	16	8	3	
	4	19	8	18	12	15	15	14	6	2	
	5	23	7	16	9	8	14	12	5	1	
	6	25	7	16	5	8	14	6	3	1	
49	1	12	23	25	23	6	11	25	20	5	
	2	15	23	25	19	5	10	23	18	3	
	3	18	22	24	15	5	9	23	16	3	
	4	20	22	23	13	5	7	23	15	3	
	5	27	21	22	9	4	6	21	14	2	
	6	29	21	21	8	4	6	21	12	1	
50	1	10	9	14	13	20	16	23	28	14	
	2	15	9	11	13	19	16	22	27	13	
	3	16	8	9	13	19	16	20	26	12	
	4	20	7	9	13	18	15	20	26	10	
	5	21	6	7	13	17	15	18	25	9	
	6	28	5	4	13	17	15	18	25	9	
51	1	6	17	19	15	20	16	13	24	19	
	2	18	14	17	13	17	15	13	23	17	
	3	19	12	14	12	13	15	11	23	17	
	4	20	8	11	6	7	14	11	22	15	
	5	21	5	8	4	6	14	9	21	15	
	6	24	2	8	3	2	14	9	21	13	
52	1	0	0	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	R 3	R 4	N 1	N 2	N 3	N 4
	98	100	119	101	775	685	705	689

************************************************************************
