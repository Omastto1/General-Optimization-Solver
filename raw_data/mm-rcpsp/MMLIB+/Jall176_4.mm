jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 4 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	12		2 3 4 5 6 7 10 11 13 14 23 25 
2	6	5		22 18 15 9 8 
3	6	10		37 32 31 30 29 28 22 21 19 15 
4	6	4		24 17 16 12 
5	6	7		37 30 29 24 22 20 16 
6	6	6		31 30 24 22 20 18 
7	6	6		32 28 24 20 19 17 
8	6	9		51 32 31 30 29 28 26 24 19 
9	6	6		33 31 29 28 24 21 
10	6	5		37 30 26 24 18 
11	6	5		37 36 31 24 16 
12	6	6		37 33 30 28 27 21 
13	6	3		37 24 20 
14	6	7		51 36 33 32 29 28 22 
15	6	5		51 44 39 36 24 
16	6	8		51 45 44 38 35 33 32 28 
17	6	6		45 41 39 37 29 27 
18	6	6		51 44 41 35 32 28 
19	6	5		45 41 39 33 27 
20	6	5		51 45 35 33 26 
21	6	4		51 45 41 26 
22	6	7		45 44 43 42 39 35 34 
23	6	7		51 45 42 40 38 36 34 
24	6	3		45 41 27 
25	6	4		51 50 39 32 
26	6	5		44 42 40 36 34 
27	6	4		42 38 35 34 
28	6	4		49 43 42 39 
29	6	4		50 43 40 34 
30	6	4		48 45 43 42 
31	6	2		42 39 
32	6	2		43 34 
33	6	3		48 47 42 
34	6	4		49 48 47 46 
35	6	4		49 48 47 46 
36	6	4		49 48 47 46 
37	6	4		51 50 49 47 
38	6	2		49 43 
39	6	1		40 
40	6	3		48 47 46 
41	6	3		48 47 46 
42	6	2		50 46 
43	6	2		47 46 
44	6	2		47 46 
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
jobnr.	mode	dur	R1	R2	R3	R4	N1	N2	
------------------------------------------------------------------------
1	1	0	0	0	0	0	0	0	
2	1	5	5	5	4	3	19	15	
	2	9	4	4	4	2	16	13	
	3	10	3	3	3	2	13	11	
	4	11	3	2	3	1	9	8	
	5	14	2	1	1	1	7	5	
	6	25	1	1	1	1	2	2	
3	1	2	5	1	4	3	29	4	
	2	4	4	1	3	3	24	4	
	3	5	4	1	3	3	17	4	
	4	14	4	1	3	3	14	4	
	5	22	4	1	3	3	13	3	
	6	25	4	1	3	3	7	3	
4	1	14	3	4	1	2	26	29	
	2	15	2	4	1	2	24	28	
	3	16	2	4	1	2	24	26	
	4	18	1	4	1	2	22	25	
	5	20	1	3	1	2	19	23	
	6	24	1	3	1	2	17	22	
5	1	11	4	4	5	5	28	16	
	2	12	4	3	5	5	23	14	
	3	19	4	3	5	5	16	14	
	4	24	4	3	5	5	10	12	
	5	27	3	3	5	5	6	12	
	6	28	3	3	5	5	6	11	
6	1	1	3	4	1	5	24	22	
	2	7	2	3	1	4	23	18	
	3	13	2	3	1	4	21	13	
	4	20	2	3	1	4	20	10	
	5	24	2	3	1	4	20	8	
	6	25	2	3	1	4	17	6	
7	1	10	4	5	3	3	26	10	
	2	17	3	4	2	3	23	10	
	3	18	3	3	2	2	16	8	
	4	22	3	3	2	2	14	5	
	5	23	3	1	1	1	13	3	
	6	26	3	1	1	1	6	2	
8	1	1	5	2	3	3	17	17	
	2	4	3	2	3	2	12	17	
	3	8	3	2	3	2	10	15	
	4	18	3	2	2	1	8	13	
	5	20	1	1	2	1	6	9	
	6	24	1	1	2	1	2	9	
9	1	4	5	1	1	3	28	10	
	2	11	4	1	1	3	23	8	
	3	13	3	1	1	3	19	8	
	4	17	2	1	1	3	14	7	
	5	23	1	1	1	3	8	5	
	6	29	1	1	1	3	5	4	
10	1	1	4	3	2	2	20	20	
	2	2	3	3	2	2	16	18	
	3	8	3	3	2	2	12	15	
	4	10	3	3	2	2	10	11	
	5	11	3	3	2	2	7	9	
	6	20	3	3	2	2	1	9	
11	1	2	4	3	3	3	29	12	
	2	3	3	2	3	3	26	11	
	3	4	3	2	3	3	26	9	
	4	10	3	1	3	2	23	6	
	5	20	1	1	3	2	20	5	
	6	21	1	1	3	1	19	4	
12	1	1	4	4	2	1	25	25	
	2	2	3	3	1	1	21	24	
	3	4	3	3	1	1	20	24	
	4	22	3	2	1	1	13	23	
	5	25	1	2	1	1	8	22	
	6	26	1	1	1	1	5	20	
13	1	4	1	2	4	3	21	22	
	2	5	1	2	3	3	21	20	
	3	10	1	2	3	2	20	19	
	4	12	1	2	3	2	20	18	
	5	13	1	2	3	2	19	17	
	6	30	1	2	3	1	19	16	
14	1	6	5	3	3	3	16	20	
	2	7	3	2	3	3	16	15	
	3	12	3	2	3	2	16	13	
	4	16	3	2	3	2	16	11	
	5	26	2	2	2	1	16	8	
	6	27	1	2	2	1	16	8	
15	1	7	2	5	5	2	5	28	
	2	13	2	5	4	2	3	27	
	3	19	2	5	3	2	3	27	
	4	22	1	5	3	2	3	27	
	5	23	1	5	2	2	1	25	
	6	27	1	5	1	2	1	25	
16	1	7	3	4	4	5	29	12	
	2	10	3	3	4	5	27	10	
	3	21	3	3	4	5	26	9	
	4	22	3	2	4	5	23	7	
	5	26	3	1	4	5	21	7	
	6	27	3	1	4	5	20	5	
17	1	6	5	3	4	1	30	28	
	2	7	3	3	3	1	21	21	
	3	14	3	3	3	1	20	21	
	4	15	3	2	3	1	14	12	
	5	17	2	1	3	1	7	12	
	6	29	1	1	3	1	6	4	
18	1	3	4	5	4	3	20	25	
	2	4	3	4	3	2	20	21	
	3	9	3	4	2	2	16	17	
	4	15	2	3	2	2	15	13	
	5	23	1	2	2	1	13	11	
	6	28	1	2	1	1	12	8	
19	1	10	3	3	3	2	27	26	
	2	11	2	3	3	2	24	24	
	3	16	2	3	3	2	22	20	
	4	22	1	3	3	2	19	16	
	5	23	1	2	3	1	17	13	
	6	30	1	2	3	1	15	9	
20	1	5	5	2	4	1	24	10	
	2	12	5	2	3	1	20	10	
	3	14	5	2	3	1	16	10	
	4	19	5	2	3	1	11	10	
	5	20	5	2	1	1	7	9	
	6	28	5	2	1	1	3	9	
21	1	7	5	3	5	1	25	23	
	2	10	4	2	4	1	24	22	
	3	17	3	2	4	1	22	19	
	4	24	2	2	4	1	20	19	
	5	29	2	1	3	1	18	18	
	6	30	1	1	3	1	14	15	
22	1	5	4	3	4	4	6	22	
	2	7	4	3	4	4	6	21	
	3	12	4	3	4	3	6	15	
	4	18	3	3	4	3	6	14	
	5	23	3	2	4	1	6	9	
	6	27	2	2	4	1	6	6	
23	1	1	4	3	4	5	26	14	
	2	3	4	3	3	4	25	14	
	3	4	4	3	3	4	25	13	
	4	6	4	3	3	4	25	12	
	5	26	4	3	2	3	25	12	
	6	27	4	3	2	3	25	11	
24	1	1	4	4	4	4	21	26	
	2	14	4	3	4	4	19	25	
	3	19	4	3	3	4	18	24	
	4	20	4	3	3	3	18	23	
	5	21	3	2	2	3	16	23	
	6	28	3	1	2	3	14	22	
25	1	12	4	5	4	5	15	16	
	2	13	4	4	3	4	14	16	
	3	15	3	4	3	3	14	13	
	4	20	2	4	2	3	13	7	
	5	21	2	4	1	2	13	5	
	6	28	1	4	1	2	13	1	
26	1	2	2	3	1	5	18	17	
	2	5	2	3	1	4	17	17	
	3	6	2	3	1	4	17	14	
	4	10	2	3	1	3	17	12	
	5	14	2	2	1	3	17	7	
	6	15	2	2	1	2	17	4	
27	1	9	4	3	4	4	21	24	
	2	10	4	3	4	4	18	21	
	3	19	4	3	4	4	18	20	
	4	22	4	3	4	4	16	16	
	5	24	3	3	4	4	15	14	
	6	25	3	3	4	4	13	11	
28	1	10	3	5	4	2	28	13	
	2	11	3	5	4	2	23	11	
	3	18	3	5	3	2	19	10	
	4	20	2	5	2	2	14	10	
	5	27	2	5	1	2	12	7	
	6	29	1	5	1	2	7	6	
29	1	6	4	1	4	3	7	22	
	2	10	4	1	3	3	6	20	
	3	11	4	1	3	2	4	16	
	4	12	3	1	2	2	3	14	
	5	24	3	1	2	2	3	9	
	6	27	2	1	1	1	2	4	
30	1	13	2	1	3	2	28	26	
	2	16	2	1	3	2	25	25	
	3	19	2	1	3	2	22	25	
	4	20	2	1	3	2	18	25	
	5	24	2	1	3	2	18	24	
	6	28	2	1	3	2	14	25	
31	1	11	5	3	5	4	28	9	
	2	14	4	3	3	3	27	8	
	3	16	4	3	3	3	27	7	
	4	17	4	2	2	3	27	5	
	5	20	4	1	1	3	26	5	
	6	26	4	1	1	3	26	4	
32	1	3	3	4	4	3	15	24	
	2	10	2	3	4	3	15	24	
	3	20	2	3	4	3	15	23	
	4	21	2	2	4	3	15	22	
	5	26	2	2	4	3	15	21	
	6	27	2	2	4	3	15	20	
33	1	3	4	4	2	3	28	19	
	2	15	4	4	2	3	23	16	
	3	19	3	4	2	3	20	13	
	4	21	3	3	2	2	13	11	
	5	23	1	3	2	2	11	10	
	6	26	1	3	2	1	8	7	
34	1	9	4	4	3	2	19	16	
	2	10	4	3	2	2	14	15	
	3	15	3	3	2	2	12	15	
	4	19	3	3	2	2	9	13	
	5	20	3	3	2	2	6	13	
	6	29	2	3	2	2	3	12	
35	1	13	4	3	4	1	29	13	
	2	17	4	3	4	1	26	11	
	3	18	4	3	3	1	24	9	
	4	19	4	3	2	1	22	9	
	5	22	4	3	1	1	21	7	
	6	25	4	3	1	1	18	6	
36	1	9	4	5	1	3	14	25	
	2	17	3	4	1	3	14	20	
	3	19	3	4	1	3	13	19	
	4	26	2	3	1	3	12	16	
	5	27	2	3	1	3	10	12	
	6	29	1	2	1	3	8	9	
37	1	3	1	4	1	4	12	4	
	2	6	1	4	1	3	11	4	
	3	8	1	4	1	3	11	3	
	4	11	1	4	1	3	10	3	
	5	15	1	4	1	3	10	2	
	6	25	1	4	1	3	9	2	
38	1	3	2	2	3	5	19	18	
	2	12	2	1	3	4	18	16	
	3	13	2	1	3	4	18	15	
	4	24	1	1	3	3	17	13	
	5	25	1	1	2	3	17	13	
	6	26	1	1	2	2	17	12	
39	1	2	4	4	4	5	7	9	
	2	10	3	3	3	5	6	9	
	3	13	3	3	3	5	6	8	
	4	14	3	3	3	5	6	7	
	5	18	1	3	2	5	6	9	
	6	30	1	3	2	5	6	8	
40	1	7	1	4	2	4	8	18	
	2	11	1	3	1	3	8	17	
	3	13	1	3	1	3	7	15	
	4	14	1	2	1	3	5	14	
	5	16	1	2	1	3	4	14	
	6	17	1	1	1	3	4	13	
41	1	4	2	3	4	3	16	29	
	2	10	2	3	4	3	13	28	
	3	18	2	3	4	3	11	26	
	4	19	2	3	4	3	9	26	
	5	24	2	3	3	3	7	24	
	6	26	2	3	3	3	1	24	
42	1	1	1	5	2	3	27	25	
	2	7	1	4	2	3	23	24	
	3	10	1	4	2	3	22	24	
	4	11	1	4	2	2	20	24	
	5	14	1	4	2	2	18	24	
	6	29	1	4	2	2	17	24	
43	1	14	1	4	5	2	22	18	
	2	17	1	4	4	2	20	18	
	3	18	1	4	4	2	18	15	
	4	19	1	4	4	2	16	14	
	5	20	1	4	4	2	14	12	
	6	26	1	4	4	2	13	12	
44	1	4	1	3	5	4	17	20	
	2	6	1	3	4	3	17	19	
	3	7	1	3	4	3	15	17	
	4	8	1	2	3	3	13	14	
	5	17	1	1	3	3	10	12	
	6	18	1	1	3	3	8	12	
45	1	10	3	1	3	5	29	20	
	2	12	2	1	2	4	29	18	
	3	13	2	1	2	4	29	17	
	4	22	2	1	1	3	29	18	
	5	28	2	1	1	3	29	16	
	6	29	2	1	1	3	29	15	
46	1	15	3	5	3	4	25	10	
	2	17	3	4	2	4	20	9	
	3	22	3	4	2	4	17	8	
	4	26	3	4	2	4	13	8	
	5	29	3	4	1	4	11	8	
	6	30	3	4	1	4	8	7	
47	1	1	4	5	2	4	21	10	
	2	4	4	3	1	3	19	8	
	3	5	4	3	1	2	19	7	
	4	11	4	2	1	2	19	7	
	5	16	4	1	1	2	17	6	
	6	24	4	1	1	1	17	5	
48	1	1	5	3	4	4	18	21	
	2	3	4	3	4	4	18	19	
	3	11	3	3	4	3	16	16	
	4	15	3	3	4	3	15	14	
	5	16	2	3	4	2	13	14	
	6	19	1	3	4	2	13	12	
49	1	2	2	1	5	4	24	26	
	2	7	1	1	3	4	21	26	
	3	19	1	1	3	4	21	22	
	4	24	1	1	3	4	19	22	
	5	25	1	1	1	3	17	18	
	6	26	1	1	1	3	16	18	
50	1	10	4	4	4	3	20	23	
	2	11	4	3	4	3	20	21	
	3	13	4	3	4	3	20	19	
	4	18	3	3	4	3	19	15	
	5	22	3	2	4	3	18	13	
	6	23	3	1	4	3	18	11	
51	1	4	5	3	3	2	29	26	
	2	8	5	3	3	2	27	25	
	3	19	5	3	3	2	23	22	
	4	20	5	3	3	2	21	20	
	5	22	5	3	3	1	17	19	
	6	25	5	3	3	1	14	17	
52	1	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	R 3	R 4	N 1	N 2
	25	23	21	23	818	735

************************************************************************