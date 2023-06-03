jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 4 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	5		2 3 4 5 6 
2	3	5		14 12 10 9 7 
3	3	4		14 12 8 7 
4	3	4		15 14 13 10 
5	3	3		15 14 10 
6	3	3		15 12 10 
7	3	4		21 15 13 11 
8	3	2		15 10 
9	3	3		21 15 11 
10	3	2		21 11 
11	3	8		26 25 23 20 19 18 17 16 
12	3	6		26 25 23 21 17 16 
13	3	6		26 24 23 20 19 18 
14	3	6		35 28 26 25 24 18 
15	3	1		16 
16	3	6		35 31 30 28 24 22 
17	3	6		38 31 30 29 27 24 
18	3	4		31 30 27 22 
19	3	3		30 28 22 
20	3	6		38 35 34 33 30 27 
21	3	6		38 34 33 31 30 29 
22	3	4		38 34 33 29 
23	3	4		37 35 30 29 
24	3	4		39 34 33 32 
25	3	3		38 37 29 
26	3	2		33 29 
27	3	6		44 43 40 39 37 36 
28	3	6		44 43 40 39 37 36 
29	3	2		39 32 
30	3	2		44 32 
31	3	5		44 41 40 39 36 
32	3	4		43 41 40 36 
33	3	4		43 42 40 37 
34	3	5		45 44 43 41 40 
35	3	3		45 43 40 
36	3	2		48 42 
37	3	2		47 41 
38	3	4		51 48 47 46 
39	3	1		42 
40	3	4		51 50 48 46 
41	3	4		51 50 48 46 
42	3	3		47 46 45 
43	3	2		49 47 
44	3	2		50 46 
45	3	2		50 49 
46	3	1		49 
47	3	1		50 
48	3	1		49 
49	3	1		52 
50	3	1		52 
51	3	1		52 
52	1	0		
************************************************************************
REQUESTS/DURATIONS
jobnr.	mode	dur	R1	R2	R3	R4	N1	N2	
------------------------------------------------------------------------
1	1	0	0	0	0	0	0	0	
2	1	15	16	10	14	28	29	21	
	2	26	8	9	9	25	28	18	
	3	27	7	6	6	25	27	16	
3	1	10	15	8	23	9	11	24	
	2	12	12	7	19	7	6	22	
	3	15	10	7	5	7	3	13	
4	1	2	6	16	27	19	22	6	
	2	15	5	10	24	16	12	4	
	3	24	5	10	23	14	4	4	
5	1	4	29	6	20	27	20	20	
	2	9	25	4	20	24	20	10	
	3	29	24	2	14	22	20	5	
6	1	2	25	17	12	14	23	21	
	2	13	22	17	7	10	21	13	
	3	26	20	14	4	7	8	11	
7	1	18	2	12	20	3	18	17	
	2	24	1	11	20	2	17	11	
	3	26	1	6	19	1	17	10	
8	1	8	16	21	28	20	21	9	
	2	10	14	12	15	19	21	5	
	3	28	11	11	10	19	17	5	
9	1	19	11	20	25	22	12	17	
	2	20	8	20	23	17	7	12	
	3	29	4	13	23	14	1	9	
10	1	4	13	25	25	23	22	8	
	2	8	12	23	24	19	10	4	
	3	18	9	21	22	16	6	3	
11	1	15	20	21	11	25	21	20	
	2	20	18	20	10	23	18	11	
	3	25	17	20	9	23	8	10	
12	1	1	27	17	27	23	13	25	
	2	11	25	8	25	17	10	16	
	3	12	21	5	22	13	7	15	
13	1	9	19	16	24	14	26	8	
	2	12	13	12	22	12	19	8	
	3	16	9	9	21	9	14	6	
14	1	15	27	25	23	26	10	11	
	2	22	25	21	22	21	9	7	
	3	29	25	9	22	8	9	2	
15	1	12	21	18	17	25	24	22	
	2	17	18	13	16	25	23	12	
	3	26	10	6	15	19	21	2	
16	1	10	22	9	25	28	27	20	
	2	24	21	9	24	25	24	16	
	3	29	13	7	23	23	24	6	
17	1	3	16	24	27	28	15	19	
	2	12	16	16	24	26	15	14	
	3	23	16	11	23	23	15	8	
18	1	3	21	22	16	18	9	23	
	2	15	18	17	13	17	8	19	
	3	22	17	14	11	16	6	13	
19	1	1	12	15	26	16	26	2	
	2	3	10	13	21	14	11	2	
	3	20	8	12	19	13	9	2	
20	1	7	18	25	15	17	22	30	
	2	18	17	19	10	17	19	26	
	3	21	17	16	9	16	15	24	
21	1	12	27	12	25	19	29	28	
	2	22	24	8	24	17	29	26	
	3	29	23	4	22	17	27	23	
22	1	19	2	16	6	23	14	15	
	2	20	2	13	5	20	9	12	
	3	29	2	3	3	18	7	11	
23	1	5	29	24	18	6	20	26	
	2	19	17	21	14	4	18	14	
	3	26	3	19	12	2	13	10	
24	1	4	28	13	16	16	20	21	
	2	24	27	11	14	12	19	17	
	3	30	27	10	14	9	15	7	
25	1	6	19	25	24	23	14	23	
	2	25	19	19	16	22	9	14	
	3	27	19	16	10	21	8	3	
26	1	1	29	19	14	14	29	25	
	2	14	29	11	12	13	22	19	
	3	18	29	2	11	10	13	13	
27	1	3	2	18	23	20	30	26	
	2	10	1	18	14	19	27	24	
	3	21	1	18	9	10	26	21	
28	1	9	15	26	19	10	29	21	
	2	14	15	22	12	9	20	20	
	3	15	6	18	7	9	14	6	
29	1	4	14	12	13	15	18	21	
	2	12	11	8	10	7	13	19	
	3	27	4	5	8	6	3	17	
30	1	5	29	24	12	30	26	19	
	2	13	28	18	8	13	21	16	
	3	28	26	9	4	5	18	11	
31	1	2	26	27	28	16	27	22	
	2	13	26	23	22	13	23	19	
	3	21	25	17	21	2	21	11	
32	1	17	21	27	21	23	7	9	
	2	19	20	27	17	21	7	9	
	3	20	19	26	13	20	7	9	
33	1	3	19	6	22	16	21	26	
	2	14	14	4	21	14	13	25	
	3	18	10	4	20	14	6	23	
34	1	8	26	24	7	24	15	19	
	2	11	26	21	6	20	14	15	
	3	15	24	19	5	14	9	2	
35	1	20	27	23	19	19	15	11	
	2	25	26	20	17	17	15	10	
	3	26	25	13	16	2	15	9	
36	1	14	6	15	10	9	21	20	
	2	16	6	12	4	7	21	18	
	3	24	6	10	3	4	19	16	
37	1	12	27	29	19	13	14	23	
	2	17	16	27	10	11	10	21	
	3	19	3	22	8	10	7	18	
38	1	2	9	14	26	21	19	15	
	2	23	8	10	14	14	14	14	
	3	28	8	10	12	6	7	13	
39	1	5	24	29	26	4	21	28	
	2	14	21	26	24	3	18	20	
	3	17	9	26	19	2	14	16	
40	1	13	12	13	12	19	17	14	
	2	14	10	10	12	16	11	11	
	3	28	7	9	10	16	1	9	
41	1	5	18	20	18	15	29	5	
	2	8	15	19	9	14	25	4	
	3	11	10	18	7	10	24	4	
42	1	1	22	23	7	21	12	18	
	2	17	12	20	3	21	8	14	
	3	23	6	19	3	19	5	2	
43	1	4	16	8	11	12	4	23	
	2	7	13	6	10	7	4	16	
	3	16	12	4	8	6	3	11	
44	1	12	17	9	15	22	24	26	
	2	25	16	5	7	11	23	24	
	3	29	14	4	6	7	23	22	
45	1	1	23	6	30	15	23	15	
	2	8	22	5	23	7	21	11	
	3	23	22	5	19	3	20	1	
46	1	1	13	24	26	26	13	21	
	2	2	12	22	25	26	11	21	
	3	13	9	20	20	26	9	19	
47	1	10	16	17	8	22	22	3	
	2	14	8	15	6	16	22	2	
	3	21	5	12	6	9	21	2	
48	1	9	15	20	28	22	24	14	
	2	19	10	18	20	14	20	12	
	3	30	6	17	10	4	9	9	
49	1	5	10	26	24	29	20	17	
	2	21	8	19	18	14	17	8	
	3	29	8	19	11	10	14	6	
50	1	3	12	15	26	23	8	22	
	2	4	9	9	25	14	5	17	
	3	10	7	6	25	13	1	7	
51	1	13	27	22	16	20	16	20	
	2	14	21	21	14	9	12	20	
	3	16	15	19	14	2	5	19	
52	1	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	R 3	R 4	N 1	N 2
	97	96	97	90	793	716

************************************************************************
