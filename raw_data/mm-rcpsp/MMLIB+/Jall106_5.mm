jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 4 R
- nonrenewable              : 4 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	7		2 3 5 6 8 9 10 
2	3	3		15 12 4 
3	3	4		18 15 14 7 
4	3	6		26 22 18 17 13 11 
5	3	2		11 7 
6	3	4		26 22 15 13 
7	3	5		27 26 22 17 16 
8	3	5		27 26 22 17 16 
9	3	5		27 26 22 18 16 
10	3	5		27 26 22 19 16 
11	3	3		27 19 16 
12	3	2		26 13 
13	3	4		28 27 23 20 
14	3	4		28 26 22 19 
15	3	2		27 17 
16	3	3		28 23 20 
17	3	2		28 19 
18	3	1		19 
19	3	1		21 
20	3	1		21 
21	3	4		37 33 25 24 
22	3	7		37 36 34 33 32 31 29 
23	3	6		39 37 36 34 31 30 
24	3	3		36 34 29 
25	3	3		43 32 31 
26	3	2		32 30 
27	3	5		40 39 38 37 36 
28	3	3		43 36 33 
29	3	2		39 30 
30	3	4		43 41 40 38 
31	3	3		40 38 35 
32	3	3		39 38 35 
33	3	3		40 38 35 
34	3	4		51 46 44 43 
35	3	3		51 44 41 
36	3	3		51 44 41 
37	3	3		51 44 43 
38	3	2		46 42 
39	3	2		44 41 
40	3	5		51 50 49 46 45 
41	3	4		50 49 46 45 
42	3	3		49 48 44 
43	3	3		49 48 45 
44	3	2		50 45 
45	3	1		47 
46	3	1		48 
47	3	1		52 
48	3	1		52 
49	3	1		52 
50	3	1		52 
51	3	1		52 
52	1	0		
************************************************************************
REQUESTS/DURATIONS
jobnr.	mode	dur	R1	R2	R3	R4	N1	N2	N3	N4	
------------------------------------------------------------------------
1	1	0	0	0	0	0	0	0	0	0	
2	1	1	2	4	5	4	20	15	18	25	
	2	8	2	4	4	3	19	12	16	21	
	3	26	2	3	3	2	19	7	16	15	
3	1	3	5	2	2	3	24	14	12	25	
	2	9	4	2	1	3	18	13	6	19	
	3	13	4	2	1	1	12	13	3	14	
4	1	3	5	2	5	4	4	27	16	13	
	2	8	3	2	4	4	3	26	10	9	
	3	16	3	2	4	3	3	26	9	7	
5	1	13	1	2	2	3	23	27	21	20	
	2	16	1	1	2	3	16	26	18	17	
	3	17	1	1	2	3	10	25	15	17	
6	1	7	3	4	1	4	20	13	22	18	
	2	16	3	2	1	2	14	13	12	11	
	3	18	1	2	1	1	12	6	7	9	
7	1	14	5	1	4	2	25	23	12	3	
	2	22	4	1	2	2	14	23	9	3	
	3	27	3	1	1	1	12	23	8	2	
8	1	13	5	4	2	5	20	28	17	23	
	2	23	5	3	2	4	9	28	10	20	
	3	26	5	3	1	3	8	27	4	18	
9	1	6	4	1	3	3	25	21	18	18	
	2	14	4	1	1	3	25	19	15	17	
	3	15	4	1	1	3	25	15	12	14	
10	1	6	4	2	3	4	25	19	15	17	
	2	7	4	1	3	3	22	19	11	11	
	3	16	3	1	3	3	21	15	6	3	
11	1	2	5	2	3	3	13	9	16	19	
	2	8	4	1	1	3	13	9	9	16	
	3	30	3	1	1	3	11	9	7	14	
12	1	3	3	5	2	2	16	24	23	30	
	2	6	3	5	2	1	11	21	18	27	
	3	26	2	5	2	1	7	15	15	25	
13	1	4	3	1	2	3	25	24	20	20	
	2	13	3	1	2	3	25	23	13	9	
	3	17	3	1	2	3	23	19	10	6	
14	1	11	4	3	5	5	15	5	27	16	
	2	25	2	3	2	5	15	4	23	12	
	3	30	1	3	1	5	7	3	14	10	
15	1	5	1	3	2	4	22	20	5	15	
	2	15	1	3	2	3	18	16	5	9	
	3	25	1	3	1	2	14	16	1	9	
16	1	1	4	3	2	3	15	24	17	28	
	2	12	3	2	1	3	13	11	11	24	
	3	13	1	2	1	3	6	6	6	19	
17	1	3	4	4	4	3	30	22	4	24	
	2	16	2	2	3	3	25	18	3	23	
	3	20	2	2	2	2	18	16	2	21	
18	1	8	4	1	4	3	9	11	20	17	
	2	14	3	1	3	2	5	7	15	9	
	3	27	3	1	2	1	5	2	10	1	
19	1	2	3	4	3	3	15	11	16	22	
	2	16	3	3	3	2	6	10	13	21	
	3	26	1	1	2	1	4	8	7	16	
20	1	3	3	2	3	2	28	18	24	18	
	2	15	3	1	3	1	27	14	19	17	
	3	29	3	1	2	1	26	11	18	16	
21	1	9	2	3	4	4	22	24	8	27	
	2	15	2	2	3	3	21	23	4	15	
	3	19	2	2	3	3	20	23	4	4	
22	1	20	3	4	4	4	16	27	21	18	
	2	21	3	4	4	3	14	19	12	15	
	3	29	3	3	4	2	9	19	9	5	
23	1	5	3	4	3	4	29	13	18	27	
	2	20	3	3	3	3	25	11	15	24	
	3	23	3	2	2	2	22	9	12	23	
24	1	16	4	5	4	4	21	22	25	27	
	2	19	4	4	3	3	17	16	23	26	
	3	25	4	3	3	1	12	14	20	24	
25	1	12	5	4	4	3	1	27	26	24	
	2	13	4	3	3	3	1	21	18	18	
	3	24	4	3	3	3	1	16	13	15	
26	1	15	2	4	3	2	20	22	25	16	
	2	25	2	4	2	1	17	19	22	15	
	3	29	2	4	1	1	15	17	21	15	
27	1	8	4	1	4	3	10	9	10	14	
	2	11	2	1	3	3	6	9	6	11	
	3	19	2	1	3	2	5	9	5	4	
28	1	10	3	3	4	3	4	23	13	27	
	2	21	3	2	3	2	4	21	9	26	
	3	26	3	2	3	1	3	21	5	26	
29	1	5	4	2	2	3	24	25	11	10	
	2	12	2	2	1	3	15	22	11	8	
	3	27	2	1	1	3	10	16	11	7	
30	1	17	4	3	2	3	20	18	26	20	
	2	23	4	2	2	3	15	10	24	20	
	3	28	3	2	2	3	12	4	24	19	
31	1	8	2	5	4	5	5	20	14	11	
	2	15	2	4	4	4	4	20	12	11	
	3	29	1	4	2	4	4	19	8	11	
32	1	4	2	4	3	3	22	29	19	14	
	2	6	1	4	2	2	16	29	17	14	
	3	23	1	4	2	2	14	29	7	14	
33	1	7	5	4	5	4	26	21	26	15	
	2	22	4	4	5	2	22	18	18	13	
	3	28	4	4	5	2	15	9	7	5	
34	1	11	4	2	4	3	22	13	20	22	
	2	12	2	1	4	3	18	12	10	20	
	3	25	2	1	4	3	11	8	4	20	
35	1	2	4	4	4	3	12	11	19	24	
	2	5	2	4	4	1	12	9	17	22	
	3	14	2	4	3	1	10	7	15	18	
36	1	16	3	4	3	5	16	17	29	16	
	2	18	3	4	2	3	9	12	24	6	
	3	26	2	3	2	2	8	8	22	2	
37	1	4	2	1	4	5	16	12	22	14	
	2	14	2	1	4	5	10	11	22	14	
	3	23	1	1	4	5	10	9	22	13	
38	1	11	3	5	5	4	12	13	16	17	
	2	22	3	4	3	3	11	13	11	13	
	3	24	3	4	3	3	11	13	4	12	
39	1	23	4	4	3	5	14	27	14	13	
	2	24	3	2	1	4	9	24	14	9	
	3	28	2	2	1	3	4	22	13	7	
40	1	3	1	3	4	4	22	16	7	12	
	2	26	1	2	2	4	20	16	4	11	
	3	30	1	2	2	4	11	9	1	7	
41	1	7	5	2	3	3	21	23	24	5	
	2	14	4	2	3	2	16	19	17	4	
	3	15	2	2	3	2	8	12	13	1	
42	1	2	5	4	1	3	24	11	7	25	
	2	12	5	3	1	2	20	6	5	23	
	3	23	5	2	1	2	4	3	5	22	
43	1	1	1	3	2	3	21	15	7	24	
	2	22	1	2	2	2	16	13	7	22	
	3	25	1	2	2	2	11	6	7	18	
44	1	2	4	3	5	3	30	27	4	19	
	2	4	4	3	4	3	30	14	3	9	
	3	24	4	3	2	2	30	5	3	7	
45	1	5	3	5	5	4	18	20	19	21	
	2	9	2	3	3	4	11	20	14	16	
	3	16	1	3	2	3	11	20	13	8	
46	1	15	3	3	5	3	24	23	29	18	
	2	20	3	3	3	3	22	16	25	17	
	3	27	3	3	1	3	21	11	19	11	
47	1	1	3	4	3	2	19	26	5	24	
	2	8	3	2	2	2	15	25	5	20	
	3	19	3	2	2	2	11	25	5	15	
48	1	19	4	5	4	3	20	21	12	27	
	2	23	3	4	3	2	16	19	9	27	
	3	25	2	4	2	2	12	7	3	24	
49	1	5	3	3	2	5	17	23	20	17	
	2	6	2	2	2	4	15	20	15	10	
	3	17	1	1	2	3	11	16	14	5	
50	1	1	4	5	5	3	11	13	19	17	
	2	18	4	4	3	3	9	12	13	8	
	3	19	4	3	2	3	7	7	11	7	
51	1	2	4	2	2	2	14	26	12	3	
	2	3	3	1	2	2	12	22	11	2	
	3	26	2	1	2	1	12	17	11	2	
52	1	0	0	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	R 3	R 4	N 1	N 2	N 3	N 4
	23	18	19	23	672	747	588	690

************************************************************************
