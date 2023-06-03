jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 4 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	4		2 3 5 13 
2	9	3		9 7 4 
3	9	4		14 10 9 6 
4	9	3		16 14 6 
5	9	4		16 15 12 11 
6	9	2		11 8 
7	9	5		20 19 18 15 12 
8	9	4		20 19 18 12 
9	9	4		27 20 16 15 
10	9	4		27 19 16 15 
11	9	6		25 24 23 20 19 18 
12	9	5		27 26 23 22 17 
13	9	5		27 26 23 22 17 
14	9	4		27 19 18 17 
15	9	2		26 17 
16	9	2		24 18 
17	9	5		32 29 25 24 21 
18	9	3		32 22 21 
19	9	3		32 22 21 
20	9	3		32 22 21 
21	9	4		34 33 31 28 
22	9	3		34 29 28 
23	9	4		34 32 31 30 
24	9	3		34 33 30 
25	9	2		31 28 
26	9	2		29 28 
27	9	3		39 34 30 
28	9	4		39 37 35 30 
29	9	2		35 30 
30	9	6		51 44 42 40 38 36 
31	9	6		51 44 42 40 37 36 
32	9	6		51 44 42 41 40 36 
33	9	5		51 42 40 39 36 
34	9	3		51 37 35 
35	9	5		50 49 48 47 42 
36	9	5		50 49 47 46 43 
37	9	5		50 49 47 46 43 
38	9	3		50 48 41 
39	9	4		49 48 46 45 
40	9	3		48 46 45 
41	9	2		46 43 
42	9	2		46 45 
43	9	1		45 
44	9	1		47 
45	9	1		52 
46	9	1		52 
47	9	1		52 
48	9	1		52 
49	9	1		52 
50	9	1		52 
51	9	1		52 
52	1	0		
************************************************************************
REQUESTS/DURATIONS
jobnr.	mode	dur	R1	R2	R3	R4	N1	N2	
------------------------------------------------------------------------
1	1	0	0	0	0	0	0	0	
2	1	4	3	5	5	2	11	27	
	2	5	3	4	4	1	9	23	
	3	10	3	4	4	1	9	20	
	4	11	3	4	4	1	7	20	
	5	20	3	3	4	1	6	16	
	6	22	3	3	4	1	5	13	
	7	28	3	3	4	1	5	11	
	8	29	3	3	4	1	3	11	
	9	30	3	3	4	1	2	9	
3	1	4	4	4	1	5	28	22	
	2	5	4	4	1	4	25	20	
	3	8	4	4	1	4	23	19	
	4	23	4	4	1	4	17	18	
	5	25	3	4	1	4	14	18	
	6	26	3	4	1	4	12	17	
	7	27	3	4	1	4	10	17	
	8	28	3	4	1	4	7	16	
	9	29	3	4	1	4	2	15	
4	1	4	2	3	4	2	27	27	
	2	8	2	3	3	2	26	26	
	3	13	2	3	3	2	25	21	
	4	16	2	3	3	2	25	19	
	5	19	1	2	3	2	23	17	
	6	20	1	2	3	2	23	15	
	7	24	1	2	3	2	23	10	
	8	25	1	2	3	2	22	7	
	9	27	1	2	3	2	21	6	
5	1	3	4	5	5	5	17	15	
	2	4	4	4	4	4	14	15	
	3	5	3	4	4	4	13	13	
	4	6	3	4	4	4	12	12	
	5	11	2	4	4	4	10	12	
	6	15	2	4	3	4	8	10	
	7	26	2	4	3	4	8	9	
	8	27	1	4	3	4	4	8	
	9	29	1	4	3	4	4	7	
6	1	2	3	5	4	3	18	15	
	2	7	3	4	4	3	18	14	
	3	8	3	4	4	3	17	14	
	4	9	3	4	4	3	15	14	
	5	12	3	4	4	3	15	13	
	6	16	2	3	4	3	14	14	
	7	20	2	3	4	3	12	14	
	8	27	2	3	4	3	12	13	
	9	29	2	3	4	3	10	14	
7	1	1	5	4	2	5	24	23	
	2	4	4	3	2	5	22	22	
	3	9	4	3	2	5	20	21	
	4	16	4	3	2	5	19	20	
	5	17	3	2	2	5	17	20	
	6	18	3	2	2	5	16	19	
	7	19	3	2	2	5	14	19	
	8	22	2	2	2	5	14	18	
	9	28	2	2	2	5	13	17	
8	1	5	4	5	3	5	21	17	
	2	8	4	4	3	4	21	14	
	3	12	4	4	3	3	20	14	
	4	14	4	4	3	3	20	12	
	5	15	4	4	3	3	20	11	
	6	18	3	4	3	2	19	9	
	7	20	3	4	3	1	19	7	
	8	24	3	4	3	1	18	7	
	9	25	3	4	3	1	18	4	
9	1	2	5	4	3	2	11	20	
	2	3	4	4	2	2	11	20	
	3	4	4	4	2	2	9	18	
	4	6	3	3	2	2	8	17	
	5	7	3	3	2	2	8	16	
	6	8	3	2	2	2	7	14	
	7	21	3	1	2	2	6	13	
	8	26	2	1	2	2	6	12	
	9	29	2	1	2	2	5	11	
10	1	7	2	3	2	5	9	9	
	2	10	1	3	1	4	9	9	
	3	12	1	3	1	4	7	9	
	4	19	1	3	1	4	6	9	
	5	21	1	3	1	4	6	8	
	6	23	1	3	1	3	5	8	
	7	27	1	3	1	3	5	7	
	8	29	1	3	1	3	3	7	
	9	30	1	3	1	3	3	6	
11	1	2	2	3	4	3	18	11	
	2	3	2	2	4	2	17	11	
	3	4	2	2	4	2	17	10	
	4	5	2	2	3	2	17	9	
	5	10	2	2	2	2	17	8	
	6	12	2	2	2	1	17	7	
	7	15	2	2	2	1	17	6	
	8	18	2	2	1	1	17	4	
	9	25	2	2	1	1	17	3	
12	1	3	3	4	5	4	1	19	
	2	5	3	4	4	4	1	18	
	3	13	3	4	4	4	1	17	
	4	14	3	4	4	4	1	16	
	5	18	3	4	4	4	1	14	
	6	20	3	4	3	4	1	14	
	7	21	3	4	3	4	1	13	
	8	24	3	4	3	4	1	12	
	9	26	3	4	3	4	1	11	
13	1	4	4	4	3	5	25	12	
	2	7	4	3	3	5	23	10	
	3	8	4	3	3	5	18	10	
	4	9	4	3	3	5	14	7	
	5	10	4	2	2	5	14	6	
	6	21	3	2	2	5	12	6	
	7	23	3	2	2	5	9	4	
	8	24	3	1	2	5	5	2	
	9	25	3	1	2	5	2	1	
14	1	3	5	3	2	4	24	14	
	2	6	4	3	2	3	24	14	
	3	10	4	3	2	3	23	13	
	4	12	3	2	2	3	22	13	
	5	13	2	2	2	3	21	13	
	6	15	2	2	2	2	21	12	
	7	16	1	2	2	2	20	12	
	8	22	1	1	2	2	20	11	
	9	28	1	1	2	2	19	11	
15	1	2	4	4	4	3	13	23	
	2	8	4	3	4	3	12	23	
	3	10	4	3	4	3	12	22	
	4	11	4	3	4	3	12	21	
	5	20	4	3	4	3	10	23	
	6	21	4	2	4	3	10	23	
	7	26	4	2	4	3	10	22	
	8	29	4	2	4	3	8	24	
	9	30	4	2	4	3	8	23	
16	1	1	5	4	4	4	16	24	
	2	4	4	3	4	3	15	24	
	3	11	4	3	4	3	14	23	
	4	12	3	3	3	3	13	22	
	5	18	3	3	3	3	12	22	
	6	20	3	3	2	2	11	21	
	7	21	2	3	2	2	11	20	
	8	26	2	3	1	2	9	19	
	9	28	2	3	1	2	9	18	
17	1	1	4	4	3	3	16	16	
	2	6	4	4	3	3	15	15	
	3	9	4	4	3	3	15	14	
	4	11	3	4	3	3	15	14	
	5	12	3	3	3	2	14	13	
	6	13	2	3	3	2	13	12	
	7	16	2	3	3	2	13	11	
	8	17	1	3	3	2	12	11	
	9	22	1	3	3	2	12	10	
18	1	3	1	3	3	5	25	20	
	2	5	1	3	3	5	24	18	
	3	6	1	3	3	5	22	18	
	4	7	1	3	3	5	17	18	
	5	8	1	3	3	5	13	17	
	6	13	1	3	3	5	10	17	
	7	14	1	3	3	5	8	16	
	8	20	1	3	3	5	6	15	
	9	24	1	3	3	5	4	15	
19	1	14	4	3	2	4	21	22	
	2	15	4	2	1	3	19	20	
	3	16	4	2	1	3	17	18	
	4	17	4	2	1	3	14	14	
	5	18	4	2	1	3	13	12	
	6	19	4	2	1	3	11	9	
	7	20	4	2	1	3	11	6	
	8	25	4	2	1	3	9	4	
	9	30	4	2	1	3	7	3	
20	1	3	5	4	2	2	26	25	
	2	4	4	4	2	2	26	24	
	3	5	4	4	2	2	26	23	
	4	6	4	4	2	2	25	21	
	5	7	4	3	2	2	25	21	
	6	10	4	3	2	2	24	19	
	7	17	4	2	2	2	24	17	
	8	19	4	2	2	2	23	16	
	9	23	4	2	2	2	23	15	
21	1	6	2	5	4	2	15	25	
	2	12	2	5	4	2	14	21	
	3	13	2	5	4	2	13	20	
	4	14	2	5	4	2	13	16	
	5	21	2	5	4	1	12	16	
	6	26	2	5	4	1	11	12	
	7	28	2	5	4	1	10	11	
	8	29	2	5	4	1	9	9	
	9	30	2	5	4	1	9	5	
22	1	2	4	1	2	4	26	3	
	2	5	4	1	1	4	23	3	
	3	7	4	1	1	4	21	3	
	4	8	4	1	1	4	19	3	
	5	9	3	1	1	4	19	2	
	6	16	3	1	1	4	15	2	
	7	17	3	1	1	4	14	2	
	8	18	3	1	1	4	12	1	
	9	19	3	1	1	4	10	1	
23	1	3	5	1	2	4	27	23	
	2	4	4	1	1	4	25	20	
	3	5	4	1	1	4	25	17	
	4	6	4	1	1	3	24	15	
	5	17	4	1	1	3	23	14	
	6	18	3	1	1	3	21	14	
	7	19	3	1	1	2	20	11	
	8	26	3	1	1	2	20	10	
	9	27	3	1	1	2	19	7	
24	1	2	3	5	5	4	29	23	
	2	3	2	4	4	4	24	21	
	3	9	2	4	4	4	22	20	
	4	13	2	4	4	4	19	16	
	5	18	2	4	4	4	15	15	
	6	20	2	3	4	3	15	10	
	7	22	2	3	4	3	10	7	
	8	28	2	3	4	3	8	7	
	9	30	2	3	4	3	5	4	
25	1	2	1	1	4	4	11	27	
	2	3	1	1	4	4	10	26	
	3	4	1	1	4	4	9	24	
	4	5	1	1	4	4	9	21	
	5	6	1	1	4	3	9	19	
	6	9	1	1	4	3	8	15	
	7	16	1	1	4	3	8	14	
	8	20	1	1	4	3	7	13	
	9	27	1	1	4	3	7	11	
26	1	2	5	5	2	1	26	14	
	2	3	5	4	2	1	25	12	
	3	4	5	4	2	1	20	10	
	4	12	5	4	2	1	17	8	
	5	14	5	4	2	1	15	8	
	6	21	5	3	2	1	13	5	
	7	25	5	3	2	1	9	5	
	8	26	5	3	2	1	7	2	
	9	28	5	3	2	1	5	1	
27	1	3	3	2	3	5	22	20	
	2	6	3	2	3	4	21	20	
	3	12	3	2	3	4	20	20	
	4	17	3	2	3	4	20	19	
	5	18	3	2	3	3	19	19	
	6	22	3	2	2	3	19	19	
	7	23	3	2	2	3	18	18	
	8	28	3	2	2	3	18	17	
	9	29	3	2	2	3	18	16	
28	1	13	3	4	4	4	27	18	
	2	13	3	4	3	3	26	19	
	3	14	3	4	3	3	26	18	
	4	15	3	4	2	3	25	17	
	5	16	3	4	2	2	24	17	
	6	17	2	3	2	2	24	17	
	7	19	2	3	1	2	22	16	
	8	23	2	3	1	2	21	16	
	9	27	2	3	1	2	21	15	
29	1	5	2	4	4	3	22	16	
	2	6	2	3	3	3	21	15	
	3	8	2	3	3	3	21	12	
	4	9	2	3	3	3	21	11	
	5	12	2	3	3	2	20	9	
	6	18	2	2	3	2	20	8	
	7	19	2	2	3	1	20	7	
	8	26	2	2	3	1	19	6	
	9	28	2	2	3	1	19	5	
30	1	3	3	5	4	5	28	26	
	2	4	2	4	4	4	27	21	
	3	5	2	4	4	3	27	18	
	4	11	2	4	3	3	27	17	
	5	13	2	3	3	2	27	14	
	6	24	2	3	2	2	27	11	
	7	25	2	3	2	1	27	8	
	8	26	2	3	1	1	27	6	
	9	30	2	3	1	1	27	2	
31	1	3	4	3	4	4	17	17	
	2	10	4	3	4	3	17	15	
	3	11	4	3	3	3	13	14	
	4	13	4	3	3	3	11	12	
	5	15	4	3	2	3	11	11	
	6	16	3	3	2	3	9	9	
	7	23	3	3	2	3	5	9	
	8	26	3	3	1	3	4	7	
	9	27	3	3	1	3	2	5	
32	1	1	4	4	3	3	26	27	
	2	2	4	4	2	3	24	27	
	3	4	4	4	2	3	24	26	
	4	6	4	3	2	3	22	27	
	5	7	3	3	1	2	22	27	
	6	15	3	2	1	2	20	26	
	7	22	3	1	1	2	20	26	
	8	23	3	1	1	2	18	26	
	9	25	3	1	1	2	18	25	
33	1	1	3	2	3	5	17	21	
	2	4	3	1	3	4	16	21	
	3	5	3	1	3	3	16	21	
	4	7	3	1	3	3	14	21	
	5	10	3	1	3	3	13	21	
	6	26	3	1	3	2	13	20	
	7	27	3	1	3	1	11	20	
	8	28	3	1	3	1	10	20	
	9	29	3	1	3	1	10	19	
34	1	13	3	4	2	2	24	15	
	2	16	3	4	2	1	22	14	
	3	17	3	4	2	1	20	14	
	4	19	3	4	2	1	18	14	
	5	20	3	3	1	1	18	13	
	6	22	3	3	1	1	16	13	
	7	23	3	2	1	1	14	13	
	8	28	3	2	1	1	13	13	
	9	29	3	2	1	1	12	13	
35	1	1	3	4	3	4	24	26	
	2	3	3	3	3	3	24	26	
	3	7	3	3	3	3	23	23	
	4	10	3	3	3	3	22	19	
	5	15	3	3	2	2	22	18	
	6	22	2	3	2	2	22	15	
	7	25	2	3	2	1	21	15	
	8	28	2	3	2	1	21	11	
	9	30	2	3	2	1	20	9	
36	1	4	1	3	2	1	25	11	
	2	7	1	3	2	1	23	10	
	3	8	1	3	2	1	21	10	
	4	9	1	3	2	1	20	10	
	5	11	1	2	2	1	17	10	
	6	13	1	2	2	1	17	9	
	7	14	1	2	2	1	13	10	
	8	15	1	2	2	1	13	9	
	9	26	1	2	2	1	10	10	
37	1	2	5	2	4	3	25	26	
	2	6	4	1	4	3	24	25	
	3	8	4	1	4	3	23	23	
	4	10	4	1	4	3	21	22	
	5	12	3	1	4	3	20	21	
	6	17	3	1	4	3	18	20	
	7	19	3	1	4	3	18	19	
	8	27	2	1	4	3	16	18	
	9	30	2	1	4	3	14	18	
38	1	3	3	3	5	2	13	23	
	2	7	3	2	4	2	12	21	
	3	11	3	2	3	2	11	17	
	4	13	3	2	3	2	10	17	
	5	15	2	2	2	2	8	15	
	6	20	2	2	2	2	6	13	
	7	26	2	2	2	2	4	10	
	8	27	1	2	1	2	2	7	
	9	30	1	2	1	2	1	6	
39	1	4	3	2	2	4	9	26	
	2	5	3	2	2	3	7	22	
	3	9	3	2	2	3	7	21	
	4	16	3	2	2	3	7	19	
	5	19	3	2	2	3	5	17	
	6	20	2	2	2	3	5	16	
	7	24	2	2	2	3	4	12	
	8	28	2	2	2	3	4	11	
	9	29	2	2	2	3	3	10	
40	1	1	1	2	5	5	27	24	
	2	3	1	2	4	5	26	20	
	3	11	1	2	4	5	24	20	
	4	13	1	2	4	5	23	16	
	5	14	1	2	3	5	22	15	
	6	15	1	2	3	5	21	14	
	7	21	1	2	2	5	20	10	
	8	24	1	2	2	5	19	9	
	9	25	1	2	2	5	18	7	
41	1	4	3	3	5	2	7	29	
	2	8	3	3	4	1	7	26	
	3	9	3	3	4	1	7	25	
	4	11	3	3	3	1	7	23	
	5	12	3	2	3	1	7	19	
	6	16	3	2	2	1	7	18	
	7	17	3	2	2	1	7	15	
	8	18	3	2	1	1	7	14	
	9	19	3	2	1	1	7	11	
42	1	2	4	5	3	5	16	23	
	2	8	3	4	3	4	13	20	
	3	12	3	4	3	3	12	20	
	4	15	3	3	3	3	9	17	
	5	16	2	3	3	3	9	16	
	6	17	2	3	3	2	7	13	
	7	18	2	2	3	1	4	10	
	8	23	2	2	3	1	3	10	
	9	24	2	2	3	1	1	6	
43	1	2	4	4	4	2	24	18	
	2	4	3	4	4	1	23	18	
	3	20	3	4	4	1	20	17	
	4	21	3	4	4	1	17	17	
	5	22	2	4	4	1	14	16	
	6	23	2	4	4	1	11	16	
	7	27	2	4	4	1	8	16	
	8	29	2	4	4	1	6	15	
	9	30	2	4	4	1	3	15	
44	1	3	4	3	3	5	14	15	
	2	4	4	3	3	4	13	14	
	3	5	4	3	3	4	11	13	
	4	8	4	3	3	3	10	11	
	5	10	4	3	3	3	9	10	
	6	12	4	3	3	3	8	9	
	7	22	4	3	3	2	8	8	
	8	24	4	3	3	2	7	7	
	9	27	4	3	3	2	6	5	
45	1	1	3	3	5	4	21	27	
	2	2	2	2	4	4	16	26	
	3	4	2	2	4	4	15	25	
	4	18	2	2	4	4	12	25	
	5	20	2	2	4	3	11	24	
	6	26	2	2	4	3	8	24	
	7	28	2	2	4	3	8	23	
	8	29	2	2	4	3	4	23	
	9	30	2	2	4	3	2	23	
46	1	5	3	4	2	4	17	28	
	2	6	2	3	2	4	15	28	
	3	12	2	3	2	4	15	27	
	4	14	2	2	2	4	12	25	
	5	17	2	2	2	4	10	24	
	6	20	1	2	2	3	9	22	
	7	21	1	2	2	3	7	20	
	8	27	1	1	2	3	7	20	
	9	28	1	1	2	3	5	19	
47	1	4	3	4	3	2	25	12	
	2	11	2	3	3	2	24	11	
	3	13	2	3	3	2	23	10	
	4	14	2	3	3	2	23	9	
	5	15	2	2	3	2	23	10	
	6	20	1	2	2	2	22	9	
	7	27	1	2	2	2	21	8	
	8	29	1	1	2	2	21	8	
	9	30	1	1	2	2	21	7	
48	1	3	3	3	5	5	21	30	
	2	6	2	3	4	4	21	24	
	3	10	2	3	4	4	17	24	
	4	13	2	3	4	4	14	20	
	5	14	2	3	4	3	12	17	
	6	18	2	3	4	3	11	14	
	7	20	2	3	4	3	7	13	
	8	22	2	3	4	3	4	8	
	9	26	2	3	4	3	3	7	
49	1	10	4	3	1	4	11	24	
	2	14	4	3	1	4	11	19	
	3	16	4	3	1	4	11	17	
	4	19	4	3	1	4	11	16	
	5	20	4	2	1	4	11	14	
	6	21	3	2	1	4	11	10	
	7	22	3	2	1	4	11	9	
	8	23	3	2	1	4	11	7	
	9	24	3	2	1	4	11	4	
50	1	2	4	2	4	2	6	25	
	2	7	4	1	4	2	5	25	
	3	8	4	1	4	2	5	22	
	4	9	3	1	4	2	4	21	
	5	12	2	1	4	2	4	18	
	6	15	2	1	4	2	3	17	
	7	16	1	1	4	2	2	15	
	8	17	1	1	4	2	2	12	
	9	19	1	1	4	2	2	11	
51	1	5	2	2	4	2	16	23	
	2	6	2	2	4	2	14	20	
	3	8	2	2	4	2	14	18	
	4	9	2	2	4	2	13	16	
	5	10	2	2	4	2	12	15	
	6	18	1	2	4	1	11	13	
	7	19	1	2	4	1	10	13	
	8	20	1	2	4	1	9	11	
	9	26	1	2	4	1	7	8	
52	1	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	R 3	R 4	N 1	N 2
	20	16	19	19	732	770

************************************************************************
