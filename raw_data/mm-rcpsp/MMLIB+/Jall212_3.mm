jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 4 R
- nonrenewable              : 4 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	3		2 3 5 
2	6	4		8 7 6 4 
3	6	4		13 11 10 8 
4	6	5		21 15 13 12 10 
5	6	5		21 15 13 12 11 
6	6	7		20 19 18 17 15 14 12 
7	6	4		19 17 14 9 
8	6	5		19 18 15 14 12 
9	6	3		18 15 12 
10	6	3		24 18 17 
11	6	3		24 18 17 
12	6	3		23 22 16 
13	6	3		20 19 18 
14	6	3		30 22 21 
15	6	2		22 16 
16	6	6		35 31 28 26 25 24 
17	6	6		29 28 27 26 25 23 
18	6	3		29 28 22 
19	6	3		30 29 22 
20	6	2		30 22 
21	6	4		29 28 26 23 
22	6	4		35 27 26 25 
23	6	6		39 35 34 33 32 31 
24	6	3		32 30 29 
25	6	4		39 36 34 32 
26	6	3		39 33 32 
27	6	3		34 32 31 
28	6	3		39 33 32 
29	6	4		43 39 36 34 
30	6	6		43 42 40 39 38 37 
31	6	6		43 42 41 40 38 37 
32	6	6		51 45 43 42 41 40 
33	6	3		51 43 36 
34	6	5		51 49 42 41 40 
35	6	5		51 50 48 45 43 
36	6	3		50 42 38 
37	6	4		50 48 46 45 
38	6	4		49 48 47 46 
39	6	4		51 49 48 46 
40	6	2		50 44 
41	6	2		48 44 
42	6	1		44 
43	6	1		44 
44	6	2		47 46 
45	6	1		49 
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
2	1	12	28	20	18	21	21	14	28	28	
	2	13	27	16	18	21	20	13	25	24	
	3	14	25	14	18	21	19	12	24	22	
	4	15	23	12	18	21	18	10	20	21	
	5	20	20	9	17	21	17	10	20	18	
	6	23	19	4	17	21	17	9	17	15	
3	1	1	19	28	6	17	25	29	11	30	
	2	2	19	27	6	17	24	21	9	26	
	3	4	19	26	5	17	21	18	7	21	
	4	7	19	25	3	17	17	12	6	18	
	5	15	19	24	3	17	15	9	5	15	
	6	27	19	23	2	17	15	6	4	13	
4	1	4	27	27	28	26	18	17	26	12	
	2	7	21	19	28	24	16	15	23	10	
	3	13	20	18	25	19	14	14	18	8	
	4	19	17	14	23	18	10	14	16	8	
	5	22	12	7	23	12	10	11	15	5	
	6	29	9	1	21	12	5	9	13	3	
5	1	5	6	12	14	18	5	14	20	23	
	2	6	5	8	13	18	5	11	17	20	
	3	9	5	8	13	17	5	9	13	19	
	4	23	4	6	11	16	5	9	9	18	
	5	26	4	5	9	15	5	7	8	17	
	6	29	3	2	8	13	5	5	6	16	
6	1	6	15	25	18	11	19	9	12	23	
	2	7	15	22	17	11	18	8	11	21	
	3	12	12	20	16	11	17	8	11	21	
	4	18	11	16	16	10	16	8	11	18	
	5	22	10	12	15	9	15	7	11	18	
	6	25	9	11	13	9	14	6	11	15	
7	1	1	13	9	20	25	12	25	17	19	
	2	3	13	9	19	25	11	23	14	18	
	3	10	10	8	19	25	10	22	14	18	
	4	11	9	8	19	25	9	20	12	18	
	5	23	5	7	19	24	7	20	7	18	
	6	28	3	6	19	24	7	19	5	18	
8	1	10	17	21	22	30	15	17	17	3	
	2	11	14	21	22	28	13	15	16	2	
	3	21	14	19	22	26	12	11	16	2	
	4	22	12	17	21	24	11	10	16	2	
	5	23	9	13	21	20	10	7	16	2	
	6	28	9	13	20	20	9	4	16	2	
9	1	2	9	9	20	16	26	22	23	12	
	2	5	8	7	19	16	25	17	18	11	
	3	6	8	6	19	16	24	16	13	11	
	4	9	7	4	19	16	24	11	12	9	
	5	25	5	4	17	16	23	6	6	7	
	6	30	4	2	17	16	22	1	5	6	
10	1	4	19	25	23	23	24	26	14	7	
	2	8	15	22	22	19	20	26	11	6	
	3	10	13	22	18	15	16	22	11	6	
	4	21	12	15	18	13	16	20	8	6	
	5	24	9	11	13	9	12	18	7	6	
	6	27	5	10	13	7	11	18	7	6	
11	1	2	28	26	13	26	24	27	21	23	
	2	4	27	23	12	24	22	18	17	21	
	3	10	26	19	12	23	22	17	16	20	
	4	13	25	14	12	20	19	13	12	20	
	5	24	24	13	11	20	17	9	7	19	
	6	26	24	6	10	17	15	3	5	18	
12	1	4	19	25	22	26	16	24	13	21	
	2	14	19	19	21	26	15	23	10	17	
	3	19	19	15	13	24	15	18	10	14	
	4	20	19	15	10	21	14	15	6	13	
	5	26	19	10	9	18	12	11	4	11	
	6	27	19	8	3	17	12	11	3	7	
13	1	2	11	30	27	21	12	17	20	25	
	2	5	10	26	26	20	12	16	19	21	
	3	12	9	23	26	18	10	14	18	18	
	4	21	9	20	25	15	9	13	18	12	
	5	22	7	18	24	13	7	12	16	11	
	6	27	7	15	24	12	7	11	16	6	
14	1	4	25	18	6	22	16	28	23	2	
	2	5	24	15	6	20	15	27	21	2	
	3	13	24	14	6	17	13	26	21	2	
	4	17	23	12	6	16	9	26	21	2	
	5	24	23	12	6	13	8	25	20	2	
	6	28	22	11	6	12	4	25	19	2	
15	1	1	26	22	28	18	23	25	27	12	
	2	12	20	17	23	15	22	24	25	10	
	3	13	20	15	21	14	21	24	20	9	
	4	16	15	14	14	11	21	22	18	7	
	5	18	11	8	14	11	19	19	15	6	
	6	21	10	8	10	9	19	18	14	5	
16	1	8	6	23	12	26	14	15	26	23	
	2	12	6	23	10	25	14	15	19	21	
	3	17	5	22	10	22	12	13	17	20	
	4	26	5	20	8	16	11	12	12	19	
	5	28	4	20	8	15	10	10	7	19	
	6	29	4	19	7	12	9	10	6	18	
17	1	1	22	25	9	8	18	26	29	28	
	2	2	22	24	7	6	16	25	23	26	
	3	3	22	23	7	6	14	24	23	25	
	4	15	21	23	4	4	13	24	20	25	
	5	21	21	22	4	4	12	22	14	24	
	6	26	21	21	1	3	10	22	10	22	
18	1	2	16	10	5	7	4	20	22	27	
	2	3	12	9	5	6	3	20	22	26	
	3	7	9	6	5	6	3	19	22	26	
	4	18	6	4	4	6	2	19	22	25	
	5	23	3	4	4	6	2	19	22	25	
	6	24	3	2	3	6	1	18	22	25	
19	1	4	17	8	15	22	22	28	27	25	
	2	6	16	7	13	21	20	24	26	21	
	3	8	12	6	10	21	17	18	26	19	
	4	10	12	5	9	21	15	14	25	17	
	5	17	9	5	6	21	9	10	24	16	
	6	19	7	4	6	21	7	5	24	13	
20	1	3	21	14	29	17	27	21	6	13	
	2	8	20	13	25	17	25	20	5	12	
	3	12	19	11	20	17	24	20	4	9	
	4	22	17	8	17	17	24	19	4	6	
	5	24	15	7	16	17	22	19	3	3	
	6	26	13	4	12	17	22	18	2	1	
21	1	10	16	11	4	29	14	21	27	28	
	2	13	15	10	4	24	14	20	23	24	
	3	14	15	9	3	23	13	19	21	23	
	4	18	13	9	3	19	11	19	19	19	
	5	25	12	8	1	14	11	17	15	14	
	6	30	12	6	1	10	9	17	12	13	
22	1	2	20	22	15	17	4	23	20	13	
	2	4	20	20	11	16	3	22	18	12	
	3	6	19	19	9	15	3	22	18	11	
	4	13	19	17	8	14	3	21	17	8	
	5	15	18	16	6	13	3	19	16	8	
	6	24	17	14	5	11	3	19	15	6	
23	1	2	11	29	19	18	25	29	23	29	
	2	9	9	27	16	18	25	28	19	28	
	3	14	6	24	13	15	24	28	16	28	
	4	21	5	22	11	11	24	28	12	28	
	5	25	5	19	7	9	24	28	9	28	
	6	26	3	18	5	7	23	28	7	28	
24	1	6	11	29	20	18	22	25	16	21	
	2	7	10	29	18	14	18	23	12	20	
	3	19	9	29	17	13	16	22	9	18	
	4	21	7	29	16	12	15	21	8	17	
	5	24	6	29	16	10	12	20	5	17	
	6	25	4	29	15	9	12	19	5	16	
25	1	1	7	7	18	28	29	21	17	24	
	2	2	7	7	16	25	26	18	17	23	
	3	9	7	6	15	20	22	15	13	23	
	4	13	7	5	13	15	19	10	11	23	
	5	15	7	4	12	10	14	6	10	22	
	6	24	7	2	9	4	9	6	7	22	
26	1	2	20	28	15	24	25	4	13	19	
	2	11	19	26	15	20	22	3	13	16	
	3	12	19	25	11	15	19	3	13	14	
	4	13	19	21	11	11	17	3	12	10	
	5	15	19	21	8	10	16	3	11	8	
	6	17	19	18	6	6	15	3	11	5	
27	1	1	25	15	18	16	27	30	16	16	
	2	2	25	13	17	13	23	29	15	16	
	3	4	20	12	16	13	22	29	12	12	
	4	11	16	11	15	10	21	28	10	12	
	5	19	14	10	14	8	20	27	7	7	
	6	27	13	10	12	7	17	27	7	6	
28	1	6	26	21	26	27	18	26	28	26	
	2	10	23	19	23	25	17	25	26	26	
	3	13	22	19	17	23	16	19	26	26	
	4	20	20	17	16	22	15	19	26	26	
	5	28	19	16	13	20	15	14	24	25	
	6	29	16	13	8	19	14	12	24	25	
29	1	4	19	19	24	21	25	19	16	22	
	2	8	18	19	22	20	23	18	14	21	
	3	9	17	18	22	20	23	18	12	21	
	4	11	14	18	22	20	23	17	8	21	
	5	22	14	17	21	20	22	17	7	21	
	6	24	11	17	20	20	21	17	6	21	
30	1	5	8	19	21	28	13	26	6	18	
	2	16	8	18	21	27	12	24	4	14	
	3	17	6	18	21	25	9	19	4	12	
	4	18	5	17	20	22	9	11	2	11	
	5	20	3	16	20	21	5	5	1	5	
	6	23	2	15	20	19	5	4	1	5	
31	1	6	7	19	23	22	9	29	3	29	
	2	14	6	16	22	22	9	27	3	25	
	3	15	5	14	22	22	7	25	3	23	
	4	20	4	8	21	22	6	25	2	20	
	5	28	2	4	20	22	6	22	2	16	
	6	29	2	1	20	22	4	22	1	15	
32	1	5	19	27	21	19	25	9	12	10	
	2	15	18	24	21	19	22	9	10	10	
	3	16	17	23	21	16	21	9	8	10	
	4	20	17	23	21	10	21	9	7	10	
	5	23	16	20	20	6	19	9	5	9	
	6	30	15	19	20	4	18	9	4	9	
33	1	1	10	29	14	16	28	12	30	23	
	2	16	9	28	12	14	26	11	30	21	
	3	17	8	28	12	12	23	11	30	21	
	4	19	8	27	12	10	23	10	30	19	
	5	22	5	27	11	9	22	8	30	19	
	6	24	5	26	10	9	20	8	30	17	
34	1	2	22	7	16	23	16	5	12	24	
	2	3	21	6	13	22	16	5	12	22	
	3	5	17	6	11	22	15	4	12	20	
	4	8	10	5	7	21	15	4	11	19	
	5	13	7	4	7	21	14	3	10	14	
	6	23	4	3	5	20	14	3	10	13	
35	1	2	17	15	22	15	17	23	12	7	
	2	10	17	13	20	14	16	22	12	6	
	3	20	17	10	14	13	14	22	12	6	
	4	26	17	9	10	13	14	22	12	6	
	5	27	17	6	8	13	13	21	12	6	
	6	30	17	4	7	12	11	21	12	6	
36	1	3	24	4	22	15	27	27	27	14	
	2	9	21	4	20	13	27	26	27	11	
	3	13	21	4	20	11	27	25	25	10	
	4	20	18	3	17	9	27	24	25	6	
	5	22	16	2	16	8	27	23	23	4	
	6	29	14	2	14	6	27	22	22	3	
37	1	3	27	28	11	26	27	29	28	16	
	2	5	26	27	10	25	26	27	23	16	
	3	8	26	27	9	24	20	25	20	15	
	4	10	24	26	6	23	17	22	19	15	
	5	22	23	25	6	21	17	20	16	14	
	6	23	22	25	4	21	14	17	13	14	
38	1	8	20	26	29	26	19	5	16	14	
	2	15	17	24	27	24	16	3	16	13	
	3	25	16	20	26	21	15	3	13	13	
	4	27	15	18	25	19	13	2	10	12	
	5	29	13	14	24	16	10	2	10	10	
	6	30	13	11	22	14	9	1	8	10	
39	1	10	24	30	24	21	16	5	14	26	
	2	11	23	25	23	19	13	5	12	23	
	3	19	19	19	19	19	12	5	12	23	
	4	21	18	15	14	19	7	5	10	19	
	5	26	13	11	11	18	7	5	9	17	
	6	27	12	7	7	17	2	5	8	17	
40	1	3	11	27	16	27	28	17	24	29	
	2	15	10	22	15	26	27	16	24	29	
	3	17	10	19	14	26	26	16	21	29	
	4	18	10	18	14	25	22	15	18	29	
	5	22	9	12	13	25	20	15	16	29	
	6	28	9	12	13	25	19	15	12	29	
41	1	7	6	22	21	19	24	11	18	21	
	2	8	5	21	19	18	24	7	16	19	
	3	15	5	19	14	18	24	7	16	15	
	4	16	4	18	10	15	24	6	13	11	
	5	21	4	17	5	14	24	2	12	9	
	6	28	3	17	4	13	24	1	11	5	
42	1	3	10	28	20	26	15	26	19	24	
	2	4	7	27	20	21	15	20	14	22	
	3	5	6	26	20	18	14	15	13	20	
	4	9	6	24	19	15	13	13	12	18	
	5	17	5	23	19	8	10	7	6	15	
	6	23	3	22	19	7	9	4	4	15	
43	1	12	15	23	19	17	18	22	17	5	
	2	20	14	19	19	14	12	19	17	3	
	3	24	10	18	19	14	10	19	16	3	
	4	25	8	13	19	12	8	17	14	3	
	5	26	6	9	19	11	5	15	13	1	
	6	30	5	3	19	9	3	13	13	1	
44	1	7	21	21	22	26	26	2	27	18	
	2	16	17	17	21	25	25	2	25	13	
	3	26	17	17	21	25	21	2	23	10	
	4	28	12	12	21	25	19	2	19	8	
	5	29	11	8	20	25	16	2	16	6	
	6	30	8	4	19	25	11	2	15	2	
45	1	12	26	24	14	25	25	28	27	18	
	2	15	23	19	13	25	20	27	25	18	
	3	16	20	17	13	25	18	27	24	16	
	4	19	17	16	12	25	15	25	24	14	
	5	22	15	11	12	25	13	24	23	10	
	6	27	9	11	12	25	12	24	22	9	
46	1	2	12	25	17	19	25	11	24	16	
	2	9	10	23	16	18	20	10	24	14	
	3	12	10	23	14	14	19	10	22	12	
	4	20	9	22	14	13	12	9	18	11	
	5	22	8	20	13	11	10	9	18	7	
	6	28	8	19	10	10	5	8	15	3	
47	1	6	6	16	26	18	5	12	27	18	
	2	13	6	16	25	17	5	11	27	15	
	3	21	5	14	23	16	4	9	26	15	
	4	23	4	13	18	14	4	6	26	14	
	5	29	3	12	16	13	3	4	26	11	
	6	30	2	12	14	12	3	2	25	11	
48	1	5	30	17	15	20	20	26	13	22	
	2	6	29	17	13	17	20	24	13	18	
	3	13	28	13	13	14	20	20	13	17	
	4	16	26	11	9	14	20	11	13	17	
	5	19	26	9	8	11	20	8	13	14	
	6	26	24	8	7	8	20	5	13	12	
49	1	6	12	20	27	25	25	25	11	27	
	2	8	10	19	25	22	23	24	11	24	
	3	11	10	19	23	14	18	22	11	21	
	4	12	7	19	22	10	16	22	11	19	
	5	29	6	19	20	8	9	20	11	15	
	6	30	3	19	18	6	7	19	11	14	
50	1	5	23	17	20	12	16	17	17	25	
	2	8	22	13	18	9	15	17	16	19	
	3	9	22	11	17	8	15	17	16	17	
	4	21	22	8	10	6	15	16	15	16	
	5	22	22	6	7	4	15	15	14	14	
	6	29	22	3	3	3	15	15	14	10	
51	1	10	8	16	21	27	29	25	16	30	
	2	17	8	15	20	26	26	17	15	29	
	3	19	8	15	19	26	25	13	15	28	
	4	23	8	15	19	26	25	10	14	27	
	5	27	8	14	17	25	22	6	14	26	
	6	29	8	14	17	25	20	3	13	25	
52	1	0	0	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	R 3	R 4	N 1	N 2	N 3	N 4
	77	107	88	96	794	791	767	793

************************************************************************
