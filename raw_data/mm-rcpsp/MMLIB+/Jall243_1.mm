jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	5		2 3 4 7 8 
2	9	2		6 5 
3	9	4		16 14 11 6 
4	9	3		16 11 6 
5	9	7		20 18 16 14 12 11 10 
6	9	6		20 18 15 13 12 10 
7	9	4		16 11 10 9 
8	9	5		20 18 16 15 10 
9	9	4		23 19 18 17 
10	9	4		23 21 19 17 
11	9	3		21 17 15 
12	9	3		26 23 17 
13	9	3		26 23 22 
14	9	3		25 21 19 
15	9	3		26 24 23 
16	9	2		25 19 
17	9	3		31 25 22 
18	9	3		31 25 21 
19	9	5		35 32 31 27 26 
20	9	4		32 31 27 25 
21	9	4		30 29 28 24 
22	9	3		28 27 24 
23	9	3		31 29 25 
24	9	4		38 35 33 32 
25	9	3		35 33 28 
26	9	4		38 36 34 30 
27	9	2		44 29 
28	9	6		44 43 40 38 37 36 
29	9	4		38 37 36 34 
30	9	4		51 42 41 33 
31	9	3		51 37 33 
32	9	5		46 44 42 39 36 
33	9	6		46 45 44 43 40 39 
34	9	6		49 46 45 43 42 40 
35	9	3		51 49 37 
36	9	4		51 50 48 41 
37	9	4		50 46 45 42 
38	9	3		50 45 39 
39	9	3		49 48 47 
40	9	3		50 48 47 
41	9	2		49 45 
42	9	2		48 47 
43	9	2		50 47 
44	9	2		49 47 
45	9	1		47 
46	9	1		47 
47	9	1		52 
48	9	1		52 
49	9	1		52 
50	9	1		52 
51	9	1		52 
52	1	0		
************************************************************************
REQUESTS/DURATIONS
jobnr.	mode	dur	R1	R2	N1	N2	
------------------------------------------------------------------------
1	1	0	0	0	0	0	
2	1	5	3	3	22	24	
	2	9	3	3	22	23	
	3	10	3	3	22	22	
	4	15	3	3	22	21	
	5	20	3	3	22	20	
	6	22	3	3	22	19	
	7	24	3	3	22	18	
	8	28	3	3	22	17	
	9	30	3	3	22	16	
3	1	2	3	3	21	12	
	2	6	3	2	20	9	
	3	12	3	2	19	9	
	4	18	3	2	16	8	
	5	19	3	1	15	6	
	6	24	3	1	15	5	
	7	25	3	1	14	4	
	8	26	3	1	11	4	
	9	27	3	1	11	2	
4	1	1	5	5	21	22	
	2	2	4	4	20	21	
	3	14	4	4	20	20	
	4	18	4	4	19	19	
	5	19	3	4	19	18	
	6	21	3	4	19	17	
	7	22	3	4	18	16	
	8	27	2	4	18	14	
	9	28	2	4	18	13	
5	1	3	2	4	29	15	
	2	4	2	4	26	14	
	3	5	2	4	26	12	
	4	9	2	4	25	11	
	5	19	1	4	23	10	
	6	21	1	4	21	9	
	7	23	1	4	21	6	
	8	27	1	4	19	5	
	9	28	1	4	18	5	
6	1	1	5	4	17	16	
	2	3	5	4	16	16	
	3	4	5	4	15	16	
	4	6	5	4	14	16	
	5	11	5	4	14	15	
	6	14	5	3	13	16	
	7	21	5	3	13	15	
	8	23	5	3	12	16	
	9	30	5	3	11	16	
7	1	5	4	3	24	27	
	2	13	3	3	22	26	
	3	18	3	3	22	25	
	4	19	3	2	20	25	
	5	20	3	2	20	23	
	6	21	3	2	18	23	
	7	22	3	2	18	22	
	8	24	3	1	17	20	
	9	27	3	1	15	20	
8	1	2	4	3	15	13	
	2	8	3	3	14	13	
	3	9	3	3	14	11	
	4	12	3	3	13	9	
	5	19	3	3	13	7	
	6	21	3	2	13	6	
	7	23	3	2	13	4	
	8	24	3	2	12	2	
	9	27	3	2	12	1	
9	1	5	2	5	19	14	
	2	6	2	4	18	13	
	3	10	2	4	18	12	
	4	11	2	4	18	11	
	5	15	2	3	16	13	
	6	16	2	3	16	12	
	7	17	2	3	16	11	
	8	27	2	3	14	13	
	9	28	2	3	14	12	
10	1	1	3	2	18	6	
	2	3	2	2	16	5	
	3	13	2	2	16	4	
	4	15	2	2	14	5	
	5	16	2	2	14	4	
	6	18	1	2	13	4	
	7	24	1	2	11	3	
	8	26	1	2	10	3	
	9	28	1	2	10	2	
11	1	1	1	4	21	15	
	2	6	1	4	20	15	
	3	16	1	4	18	14	
	4	19	1	3	16	13	
	5	21	1	3	15	11	
	6	23	1	3	12	10	
	7	24	1	2	11	9	
	8	25	1	2	10	6	
	9	28	1	2	7	6	
12	1	1	4	5	17	22	
	2	2	4	4	16	20	
	3	3	4	4	15	19	
	4	9	4	4	14	19	
	5	18	4	4	13	17	
	6	23	4	3	13	16	
	7	27	4	3	13	14	
	8	28	4	3	12	13	
	9	29	4	3	11	11	
13	1	7	3	4	29	30	
	2	8	3	4	28	28	
	3	12	3	4	26	27	
	4	13	3	4	24	26	
	5	20	2	4	22	26	
	6	21	2	4	21	25	
	7	22	2	4	21	23	
	8	23	2	4	18	22	
	9	26	2	4	18	21	
14	1	11	3	5	11	24	
	2	13	3	4	11	23	
	3	14	3	4	11	22	
	4	23	3	3	11	21	
	5	24	3	2	10	21	
	6	25	2	2	10	20	
	7	27	2	2	10	19	
	8	28	2	1	10	20	
	9	29	2	1	10	19	
15	1	4	1	4	21	23	
	2	8	1	4	20	20	
	3	10	1	4	20	19	
	4	16	1	4	20	18	
	5	19	1	3	19	17	
	6	22	1	3	19	14	
	7	24	1	2	18	14	
	8	27	1	2	18	12	
	9	30	1	2	18	10	
16	1	4	5	4	20	5	
	2	12	5	3	18	4	
	3	15	5	3	16	4	
	4	16	5	3	14	3	
	5	18	5	3	12	3	
	6	19	5	3	11	3	
	7	26	5	3	10	3	
	8	27	5	3	9	2	
	9	28	5	3	6	2	
17	1	6	5	5	13	12	
	2	7	4	5	12	10	
	3	9	4	5	12	8	
	4	12	4	5	10	8	
	5	16	4	5	10	5	
	6	18	4	5	7	4	
	7	24	4	5	6	4	
	8	25	4	5	5	2	
	9	28	4	5	5	1	
18	1	2	3	1	28	19	
	2	3	3	1	28	18	
	3	8	3	1	28	17	
	4	10	3	1	28	16	
	5	17	3	1	28	15	
	6	17	2	1	27	19	
	7	18	2	1	27	18	
	8	22	2	1	27	17	
	9	23	2	1	27	16	
19	1	11	3	2	19	18	
	2	12	2	1	16	18	
	3	13	2	1	15	18	
	4	18	2	1	11	17	
	5	19	1	1	11	17	
	6	23	1	1	9	16	
	7	25	1	1	7	15	
	8	28	1	1	5	15	
	9	29	1	1	3	15	
20	1	2	2	3	5	20	
	2	4	2	3	4	20	
	3	5	2	3	4	19	
	4	7	2	3	3	17	
	5	10	2	3	3	16	
	6	16	2	3	2	16	
	7	17	2	3	2	15	
	8	22	2	3	1	14	
	9	25	2	3	1	13	
21	1	8	2	2	19	14	
	2	14	2	2	19	12	
	3	15	2	2	17	11	
	4	18	2	2	17	9	
	5	19	2	2	16	6	
	6	20	2	1	15	6	
	7	23	2	1	14	4	
	8	26	2	1	13	3	
	9	27	2	1	13	2	
22	1	1	3	3	8	11	
	2	4	2	3	7	10	
	3	5	2	3	7	9	
	4	7	2	3	7	8	
	5	10	2	3	7	7	
	6	11	2	3	6	8	
	7	21	2	3	6	7	
	8	23	2	3	6	6	
	9	28	2	3	6	5	
23	1	1	4	1	24	27	
	2	4	3	1	24	27	
	3	8	3	1	24	26	
	4	11	3	1	24	25	
	5	14	2	1	23	27	
	6	18	2	1	23	26	
	7	23	2	1	22	27	
	8	28	2	1	22	26	
	9	29	2	1	22	25	
24	1	1	3	4	23	25	
	2	2	3	4	22	25	
	3	4	3	4	22	24	
	4	12	3	4	21	24	
	5	13	2	4	21	24	
	6	14	2	4	21	23	
	7	16	2	4	20	22	
	8	17	2	4	20	21	
	9	28	2	4	20	20	
25	1	2	3	2	24	14	
	2	5	3	2	21	13	
	3	7	3	2	20	13	
	4	8	3	2	19	12	
	5	10	2	2	16	12	
	6	11	2	1	15	12	
	7	19	2	1	14	11	
	8	20	2	1	11	10	
	9	30	2	1	10	10	
26	1	3	5	5	19	23	
	2	5	4	4	16	22	
	3	6	3	3	14	20	
	4	7	3	3	12	20	
	5	13	3	3	10	18	
	6	19	2	2	9	18	
	7	26	1	2	7	16	
	8	27	1	1	5	16	
	9	28	1	1	5	14	
27	1	2	4	1	10	29	
	2	7	3	1	10	24	
	3	9	3	1	10	21	
	4	21	2	1	10	18	
	5	24	2	1	10	15	
	6	25	2	1	10	13	
	7	26	1	1	10	13	
	8	27	1	1	10	9	
	9	30	1	1	10	6	
28	1	1	3	5	22	11	
	2	2	2	5	20	10	
	3	3	2	5	18	8	
	4	11	2	5	17	7	
	5	13	2	5	14	7	
	6	23	2	5	14	5	
	7	28	2	5	12	5	
	8	29	2	5	10	4	
	9	30	2	5	8	3	
29	1	1	4	4	22	23	
	2	4	4	3	20	21	
	3	5	4	3	18	21	
	4	8	4	3	16	19	
	5	18	4	3	15	19	
	6	22	4	3	14	19	
	7	23	4	3	13	18	
	8	24	4	3	12	16	
	9	27	4	3	9	16	
30	1	2	2	1	15	19	
	2	3	2	1	15	18	
	3	4	2	1	12	16	
	4	11	2	1	11	14	
	5	14	2	1	10	12	
	6	18	1	1	8	10	
	7	20	1	1	8	8	
	8	21	1	1	5	7	
	9	30	1	1	5	5	
31	1	5	4	5	18	17	
	2	6	4	4	16	16	
	3	8	4	4	14	16	
	4	13	3	3	12	16	
	5	23	3	3	11	15	
	6	25	3	3	11	14	
	7	27	3	2	10	14	
	8	28	2	2	7	12	
	9	30	2	2	7	11	
32	1	8	3	1	23	12	
	2	11	2	1	19	11	
	3	12	2	1	19	10	
	4	13	2	1	15	9	
	5	14	2	1	13	8	
	6	19	2	1	12	7	
	7	21	2	1	11	6	
	8	26	2	1	7	5	
	9	28	2	1	7	4	
33	1	2	3	2	4	21	
	2	7	3	1	4	19	
	3	15	3	1	4	18	
	4	17	3	1	4	17	
	5	19	2	1	4	18	
	6	22	2	1	4	17	
	7	25	2	1	4	16	
	8	28	2	1	4	15	
	9	30	2	1	4	14	
34	1	5	4	4	17	17	
	2	7	3	3	17	17	
	3	11	3	3	17	16	
	4	15	3	3	17	15	
	5	20	3	3	17	14	
	6	24	3	2	17	15	
	7	26	3	2	17	14	
	8	27	3	2	17	13	
	9	28	3	2	17	12	
35	1	3	2	2	25	27	
	2	4	2	1	24	26	
	3	5	2	1	21	24	
	4	7	2	1	20	24	
	5	8	2	1	20	22	
	6	11	2	1	17	22	
	7	17	2	1	17	20	
	8	18	2	1	15	19	
	9	28	2	1	14	19	
36	1	3	4	4	11	22	
	2	4	4	4	11	21	
	3	5	4	4	10	22	
	4	6	4	4	10	21	
	5	9	4	3	9	20	
	6	16	4	3	9	19	
	7	18	4	3	9	18	
	8	27	4	3	8	19	
	9	28	4	3	8	18	
37	1	1	1	2	25	17	
	2	4	1	2	21	15	
	3	8	1	2	19	15	
	4	11	1	2	18	14	
	5	12	1	2	14	12	
	6	13	1	1	12	11	
	7	23	1	1	9	10	
	8	24	1	1	6	10	
	9	29	1	1	4	9	
38	1	10	2	1	12	23	
	2	11	2	1	12	22	
	3	18	2	1	12	20	
	4	19	2	1	12	17	
	5	20	2	1	12	14	
	6	21	2	1	12	13	
	7	22	2	1	12	12	
	8	27	2	1	12	10	
	9	30	2	1	12	7	
39	1	3	3	4	26	23	
	2	4	3	3	26	23	
	3	7	3	3	26	22	
	4	10	3	3	25	23	
	5	11	2	2	25	23	
	6	12	2	2	24	23	
	7	14	2	2	23	23	
	8	20	2	2	23	22	
	9	22	2	2	23	21	
40	1	3	2	4	17	25	
	2	4	2	4	16	24	
	3	8	2	4	16	21	
	4	15	2	4	14	21	
	5	17	2	4	14	19	
	6	19	2	3	13	16	
	7	20	2	3	12	16	
	8	22	2	3	10	13	
	9	24	2	3	10	12	
41	1	2	1	2	21	29	
	2	6	1	2	19	25	
	3	8	1	2	19	23	
	4	12	1	2	17	17	
	5	16	1	2	16	15	
	6	18	1	2	15	14	
	7	20	1	2	14	11	
	8	24	1	2	13	8	
	9	27	1	2	12	5	
42	1	1	5	5	17	22	
	2	19	4	4	17	18	
	3	21	4	4	17	17	
	4	22	3	4	17	16	
	5	23	2	3	17	14	
	6	25	2	3	17	13	
	7	26	2	3	17	12	
	8	27	1	3	17	8	
	9	28	1	3	17	7	
43	1	5	1	5	20	23	
	2	5	1	4	17	24	
	3	6	1	4	17	23	
	4	7	1	4	13	23	
	5	10	1	4	11	23	
	6	13	1	3	9	23	
	7	17	1	3	7	23	
	8	18	1	3	7	22	
	9	27	1	3	5	23	
44	1	1	4	4	29	27	
	2	6	4	3	28	24	
	3	11	3	3	28	22	
	4	12	3	3	27	18	
	5	17	3	2	26	14	
	6	26	2	2	26	12	
	7	27	2	2	25	8	
	8	28	1	1	25	7	
	9	30	1	1	25	2	
45	1	7	5	3	21	22	
	2	13	4	3	21	20	
	3	14	4	3	21	19	
	4	15	4	3	21	16	
	5	16	3	3	21	15	
	6	17	3	3	21	12	
	7	19	2	3	21	12	
	8	23	2	3	21	10	
	9	29	2	3	21	8	
46	1	3	5	2	17	13	
	2	4	4	2	16	13	
	3	5	4	2	16	12	
	4	10	4	2	16	11	
	5	14	3	2	16	12	
	6	20	3	2	16	11	
	7	23	3	2	16	10	
	8	24	2	2	16	12	
	9	29	2	2	16	11	
47	1	2	2	2	24	22	
	2	7	2	2	24	20	
	3	12	2	2	22	20	
	4	13	2	2	21	19	
	5	15	2	2	21	17	
	6	16	1	1	20	17	
	7	19	1	1	20	15	
	8	24	1	1	18	14	
	9	25	1	1	18	13	
48	1	2	4	4	23	10	
	2	3	4	4	21	10	
	3	5	4	4	20	10	
	4	6	4	4	18	9	
	5	7	3	4	16	8	
	6	9	3	3	13	8	
	7	13	2	3	12	8	
	8	16	2	3	10	7	
	9	19	2	3	7	7	
49	1	2	2	3	26	20	
	2	6	2	3	24	19	
	3	9	2	3	22	18	
	4	10	2	3	20	13	
	5	12	1	3	18	12	
	6	15	1	3	17	9	
	7	16	1	3	15	9	
	8	17	1	3	12	4	
	9	24	1	3	11	2	
50	1	1	3	4	17	26	
	2	7	3	3	14	22	
	3	10	3	3	12	21	
	4	16	3	3	12	19	
	5	17	3	3	8	15	
	6	20	3	3	7	13	
	7	25	3	3	7	10	
	8	26	3	3	3	10	
	9	28	3	3	3	8	
51	1	6	3	4	18	21	
	2	7	3	4	16	20	
	3	15	3	4	16	19	
	4	16	3	4	14	20	
	5	17	3	4	13	20	
	6	18	3	4	12	20	
	7	24	3	4	11	20	
	8	29	3	4	11	19	
	9	30	3	4	10	20	
52	1	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2
	18	25	874	867

************************************************************************
