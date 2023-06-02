jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	6		2 3 4 5 7 11 
2	9	4		13 9 8 6 
3	9	5		17 13 12 10 9 
4	9	6		20 18 17 16 14 13 
5	9	6		20 18 17 16 14 13 
6	9	5		21 18 17 15 12 
7	9	5		20 18 16 14 13 
8	9	6		21 20 17 16 15 14 
9	9	6		21 20 19 18 16 15 
10	9	5		22 21 19 18 15 
11	9	5		23 20 18 17 15 
12	9	6		29 23 22 20 19 16 
13	9	4		22 21 19 15 
14	9	6		33 30 26 24 23 22 
15	9	7		33 30 29 28 27 25 24 
16	9	7		33 30 28 27 26 25 24 
17	9	3		29 27 19 
18	9	5		33 30 29 27 24 
19	9	5		33 32 30 28 26 
20	9	5		44 33 30 28 24 
21	9	3		30 28 23 
22	9	4		32 28 27 25 
23	9	2		32 25 
24	9	3		51 32 31 
25	9	4		44 37 35 31 
26	9	4		44 37 35 31 
27	9	4		38 37 35 34 
28	9	4		51 41 38 34 
29	9	4		44 38 37 35 
30	9	3		37 35 34 
31	9	3		41 38 34 
32	9	2		37 35 
33	9	3		41 39 36 
34	9	2		39 36 
35	9	3		50 41 40 
36	9	3		50 43 40 
37	9	2		42 39 
38	9	2		50 40 
39	9	5		50 49 47 46 43 
40	9	4		49 47 45 42 
41	9	3		47 43 42 
42	9	2		48 46 
43	9	1		45 
44	9	1		45 
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
jobnr.	mode	dur	R1	R2	N1	N2	
------------------------------------------------------------------------
1	1	0	0	0	0	0	
2	1	1	24	24	8	24	
	2	2	22	22	8	23	
	3	11	19	21	8	21	
	4	19	19	19	8	21	
	5	22	14	16	8	20	
	6	23	12	14	8	19	
	7	25	11	11	8	19	
	8	26	8	10	8	17	
	9	30	6	8	8	17	
3	1	4	27	6	18	5	
	2	5	25	5	18	5	
	3	8	25	5	17	5	
	4	9	24	4	17	5	
	5	11	23	4	16	5	
	6	12	22	4	16	5	
	7	17	20	3	15	5	
	8	22	19	3	15	5	
	9	28	19	3	15	4	
4	1	1	21	17	7	27	
	2	4	18	17	6	27	
	3	10	17	17	5	27	
	4	13	15	17	5	27	
	5	14	13	17	4	26	
	6	18	11	17	3	26	
	7	22	7	17	3	26	
	8	25	6	17	2	25	
	9	30	5	17	2	25	
5	1	1	8	9	28	7	
	2	2	8	8	28	5	
	3	12	8	7	28	5	
	4	16	8	5	28	5	
	5	17	8	5	28	3	
	6	21	8	4	28	3	
	7	22	8	4	28	2	
	8	25	8	3	28	2	
	9	27	8	2	28	1	
6	1	4	29	12	19	20	
	2	14	28	11	16	18	
	3	15	28	10	16	16	
	4	16	27	9	14	15	
	5	17	26	9	13	12	
	6	18	26	9	11	12	
	7	19	26	8	9	10	
	8	20	24	8	8	8	
	9	30	24	7	7	6	
7	1	6	23	16	30	28	
	2	11	23	15	26	24	
	3	15	22	15	20	21	
	4	17	21	15	19	19	
	5	20	21	15	15	17	
	6	21	20	15	10	15	
	7	22	20	15	8	14	
	8	24	19	15	7	11	
	9	29	19	15	4	9	
8	1	11	27	16	11	24	
	2	12	26	15	10	23	
	3	15	25	15	10	23	
	4	16	25	15	10	22	
	5	17	24	14	9	21	
	6	18	23	14	9	20	
	7	20	22	14	8	19	
	8	22	21	14	8	17	
	9	23	21	14	8	16	
9	1	9	22	30	28	26	
	2	14	18	28	25	25	
	3	16	17	28	22	21	
	4	17	15	26	20	20	
	5	18	13	26	17	18	
	6	22	10	25	13	14	
	7	23	6	24	10	13	
	8	25	4	23	8	9	
	9	28	4	23	6	9	
10	1	8	16	26	20	26	
	2	9	15	24	17	21	
	3	12	15	24	15	19	
	4	13	15	23	14	15	
	5	16	14	23	11	15	
	6	21	14	23	10	12	
	7	24	13	22	7	8	
	8	25	12	21	5	4	
	9	30	12	21	3	4	
11	1	4	18	28	27	22	
	2	6	16	26	24	22	
	3	11	15	23	23	21	
	4	14	14	19	22	19	
	5	15	11	14	19	17	
	6	17	10	12	18	16	
	7	22	7	11	16	13	
	8	24	6	6	12	13	
	9	26	5	3	10	11	
12	1	5	20	6	2	9	
	2	7	18	6	2	9	
	3	8	18	6	2	8	
	4	13	15	6	2	9	
	5	18	12	5	2	9	
	6	21	11	5	2	9	
	7	24	8	5	2	9	
	8	27	5	5	2	9	
	9	29	3	5	2	9	
13	1	1	29	27	21	29	
	2	2	29	26	21	29	
	3	6	27	26	21	29	
	4	9	26	26	20	28	
	5	17	26	26	20	27	
	6	18	24	25	20	28	
	7	20	24	25	19	28	
	8	21	23	25	19	27	
	9	27	22	25	19	27	
14	1	3	25	11	3	20	
	2	3	23	11	2	21	
	3	4	23	11	2	20	
	4	5	22	11	2	20	
	5	6	20	11	2	20	
	6	6	19	11	2	21	
	7	7	19	11	2	20	
	8	13	18	11	2	20	
	9	17	17	11	2	20	
15	1	2	24	26	7	24	
	2	6	22	22	6	22	
	3	8	22	22	5	22	
	4	10	22	19	5	18	
	5	12	21	19	5	15	
	6	13	21	17	4	14	
	7	15	20	15	4	11	
	8	19	20	13	3	8	
	9	20	19	12	3	8	
16	1	8	26	26	26	19	
	2	11	26	26	24	15	
	3	12	25	24	22	15	
	4	13	24	22	21	13	
	5	18	22	20	20	11	
	6	23	21	17	18	11	
	7	24	20	15	17	8	
	8	25	19	12	16	8	
	9	26	19	10	13	5	
17	1	5	26	19	16	27	
	2	7	25	18	16	27	
	3	8	25	16	16	26	
	4	9	24	12	16	24	
	5	10	23	12	16	23	
	6	11	23	8	16	23	
	7	21	22	6	16	21	
	8	28	22	4	16	20	
	9	29	21	4	16	20	
18	1	3	21	15	21	21	
	2	8	18	15	18	20	
	3	12	17	15	18	17	
	4	13	15	15	17	16	
	5	15	12	15	14	12	
	6	20	11	15	12	10	
	7	21	9	15	11	10	
	8	22	8	15	11	5	
	9	26	6	15	9	4	
19	1	4	24	17	29	28	
	2	5	19	16	25	27	
	3	6	18	15	24	25	
	4	7	17	14	21	24	
	5	9	11	14	20	23	
	6	11	11	13	20	22	
	7	24	9	12	17	19	
	8	25	5	12	15	18	
	9	28	2	11	14	18	
20	1	3	29	24	14	16	
	2	5	26	24	14	15	
	3	7	25	22	14	14	
	4	13	25	21	14	14	
	5	15	24	20	14	13	
	6	17	23	20	14	13	
	7	18	22	19	14	12	
	8	21	20	18	14	11	
	9	23	19	16	14	11	
21	1	6	20	13	15	25	
	2	7	18	12	15	24	
	3	11	17	12	13	24	
	4	13	14	12	13	24	
	5	17	14	11	12	24	
	6	19	11	11	10	24	
	7	22	10	11	10	24	
	8	27	7	11	8	24	
	9	30	5	11	7	24	
22	1	2	21	22	17	22	
	2	3	18	21	16	20	
	3	4	16	19	16	19	
	4	6	12	19	14	18	
	5	7	11	16	14	13	
	6	8	9	14	14	12	
	7	15	5	12	13	10	
	8	16	5	11	12	6	
	9	27	1	9	11	4	
23	1	7	9	7	10	18	
	2	8	9	5	10	18	
	3	9	9	5	9	16	
	4	11	8	5	8	16	
	5	13	8	4	7	13	
	6	15	7	4	6	12	
	7	21	7	3	6	11	
	8	23	6	2	4	9	
	9	28	6	2	4	8	
24	1	2	8	26	23	23	
	2	5	8	25	18	23	
	3	6	8	25	16	22	
	4	7	8	24	15	21	
	5	10	8	24	14	20	
	6	12	7	23	9	19	
	7	17	7	23	7	18	
	8	18	7	22	7	18	
	9	19	7	22	3	17	
25	1	2	23	30	3	19	
	2	5	23	28	2	18	
	3	6	18	27	2	17	
	4	7	17	25	2	15	
	5	9	16	23	2	13	
	6	23	12	22	2	10	
	7	24	11	22	2	9	
	8	29	6	20	2	6	
	9	30	6	19	2	4	
26	1	1	14	21	25	25	
	2	7	14	19	23	24	
	3	8	12	16	21	23	
	4	10	10	13	21	21	
	5	11	10	13	19	20	
	6	13	9	9	16	20	
	7	25	8	8	15	18	
	8	29	6	4	14	18	
	9	30	4	1	11	17	
27	1	4	14	27	18	18	
	2	10	14	24	16	17	
	3	15	14	21	13	16	
	4	19	14	18	13	15	
	5	20	14	17	10	12	
	6	23	14	13	8	10	
	7	26	14	10	7	9	
	8	29	14	9	3	9	
	9	30	14	7	1	7	
28	1	3	25	9	9	1	
	2	8	25	9	8	1	
	3	9	23	8	7	2	
	4	12	23	8	7	1	
	5	15	22	6	5	1	
	6	23	21	6	4	1	
	7	28	20	5	4	1	
	8	29	18	4	2	1	
	9	30	18	4	1	1	
29	1	1	20	15	22	26	
	2	3	19	15	21	24	
	3	5	19	13	20	20	
	4	7	19	13	19	16	
	5	18	18	11	19	16	
	6	19	18	10	18	12	
	7	20	17	10	17	9	
	8	26	17	8	15	6	
	9	27	17	8	15	1	
30	1	10	27	13	23	30	
	2	11	27	12	22	27	
	3	16	26	11	22	23	
	4	17	26	10	22	22	
	5	19	25	10	21	20	
	6	21	25	10	20	16	
	7	22	24	9	20	13	
	8	25	24	8	19	10	
	9	30	24	8	19	9	
31	1	2	14	27	23	29	
	2	6	13	25	22	27	
	3	13	10	24	21	27	
	4	14	9	23	19	27	
	5	15	7	23	19	26	
	6	17	7	23	17	25	
	7	19	5	21	15	25	
	8	24	3	20	15	24	
	9	26	1	20	14	24	
32	1	1	19	18	25	13	
	2	2	18	17	22	13	
	3	3	18	14	21	13	
	4	8	16	13	20	13	
	5	16	15	11	16	12	
	6	25	14	8	15	12	
	7	27	13	7	14	12	
	8	28	12	3	12	11	
	9	29	12	1	11	11	
33	1	1	28	26	15	12	
	2	4	26	25	15	11	
	3	9	26	24	13	11	
	4	13	24	24	13	10	
	5	14	24	23	12	10	
	6	15	24	21	10	10	
	7	16	23	21	10	9	
	8	17	21	19	8	8	
	9	27	21	18	8	8	
34	1	7	23	24	19	16	
	2	9	19	24	19	15	
	3	11	17	23	16	15	
	4	18	15	21	15	14	
	5	19	10	20	13	13	
	6	21	8	19	12	11	
	7	23	5	18	12	10	
	8	29	4	17	9	9	
	9	30	1	17	8	9	
35	1	1	18	24	24	25	
	2	2	18	23	22	24	
	3	3	16	23	21	24	
	4	4	14	21	15	24	
	5	6	13	21	14	22	
	6	9	12	21	11	22	
	7	11	9	19	8	21	
	8	18	9	18	6	20	
	9	19	6	18	3	20	
36	1	7	15	14	2	26	
	2	10	15	12	1	26	
	3	12	12	11	1	26	
	4	15	11	10	1	26	
	5	16	9	10	1	26	
	6	17	8	8	1	26	
	7	20	6	8	1	26	
	8	25	4	6	1	26	
	9	26	3	6	1	26	
37	1	2	28	12	15	12	
	2	7	25	12	13	12	
	3	9	24	12	13	12	
	4	10	23	12	11	12	
	5	11	21	12	10	12	
	6	12	17	12	9	12	
	7	16	16	12	7	12	
	8	26	14	12	7	12	
	9	27	11	12	5	12	
38	1	6	14	22	28	25	
	2	8	13	21	27	24	
	3	15	13	21	26	23	
	4	16	13	21	23	22	
	5	18	12	20	20	22	
	6	23	12	20	19	21	
	7	24	11	20	17	21	
	8	25	11	20	15	20	
	9	30	11	20	14	19	
39	1	4	8	14	15	29	
	2	5	8	14	13	27	
	3	6	8	13	12	25	
	4	7	8	12	11	23	
	5	17	7	11	10	21	
	6	20	7	9	9	19	
	7	22	7	9	8	16	
	8	25	7	8	7	14	
	9	30	7	6	5	13	
40	1	7	23	18	13	17	
	2	8	23	18	13	14	
	3	9	23	16	11	13	
	4	13	23	16	9	11	
	5	21	23	15	8	10	
	6	22	23	14	7	10	
	7	23	23	12	7	8	
	8	24	23	12	5	7	
	9	29	23	11	3	4	
41	1	2	28	25	20	20	
	2	5	26	24	19	20	
	3	7	24	24	19	20	
	4	13	20	23	19	20	
	5	15	18	21	18	19	
	6	17	15	19	18	19	
	7	22	12	19	18	19	
	8	24	11	17	18	19	
	9	28	10	16	18	19	
42	1	3	25	21	13	17	
	2	4	23	19	12	16	
	3	6	22	19	11	14	
	4	13	22	18	10	12	
	5	17	21	16	9	12	
	6	23	19	14	9	11	
	7	24	18	14	7	10	
	8	26	17	11	7	8	
	9	30	16	10	6	6	
43	1	3	11	26	23	18	
	2	6	10	23	20	17	
	3	7	9	19	20	16	
	4	10	8	17	16	16	
	5	11	7	15	15	15	
	6	13	7	14	13	15	
	7	14	7	11	11	14	
	8	15	6	9	9	13	
	9	26	5	8	9	13	
44	1	2	20	13	23	26	
	2	12	19	11	20	24	
	3	14	17	11	16	23	
	4	15	16	10	14	21	
	5	16	15	9	10	20	
	6	18	15	7	8	20	
	7	26	14	5	6	18	
	8	27	13	5	4	16	
	9	28	12	4	2	14	
45	1	2	28	17	5	25	
	2	3	28	17	5	24	
	3	6	27	15	5	24	
	4	8	27	14	5	24	
	5	13	26	14	5	23	
	6	15	26	13	5	23	
	7	19	25	12	5	22	
	8	24	25	11	5	22	
	9	29	25	11	5	21	
46	1	4	12	10	24	29	
	2	5	11	9	23	29	
	3	8	10	7	18	29	
	4	11	10	7	18	28	
	5	15	9	6	14	29	
	6	19	6	4	13	29	
	7	20	6	3	8	29	
	8	22	5	2	8	29	
	9	27	3	1	4	29	
47	1	5	22	20	20	21	
	2	6	18	20	19	17	
	3	9	18	19	18	17	
	4	12	17	17	16	16	
	5	17	14	16	16	14	
	6	18	13	15	13	12	
	7	24	10	14	13	10	
	8	25	9	13	12	10	
	9	30	8	13	10	7	
48	1	1	23	18	28	26	
	2	5	22	18	25	26	
	3	8	20	18	22	25	
	4	9	19	17	19	23	
	5	15	16	16	19	22	
	6	22	16	16	17	22	
	7	23	13	16	13	21	
	8	28	12	15	12	20	
	9	30	10	15	10	19	
49	1	3	24	26	21	23	
	2	6	24	23	17	21	
	3	8	24	19	17	19	
	4	13	24	19	15	18	
	5	14	24	14	11	17	
	6	18	24	10	8	15	
	7	20	24	9	7	13	
	8	21	24	4	4	10	
	9	28	24	2	2	9	
50	1	1	29	21	26	22	
	2	10	25	21	23	21	
	3	11	21	17	21	20	
	4	12	21	17	18	18	
	5	13	17	15	16	17	
	6	22	15	13	15	16	
	7	26	13	11	13	14	
	8	28	8	8	10	13	
	9	29	6	8	10	13	
51	1	3	27	10	26	21	
	2	4	25	9	22	19	
	3	7	25	9	20	18	
	4	10	24	9	17	16	
	5	11	22	8	13	15	
	6	17	22	7	13	14	
	7	20	21	7	9	12	
	8	26	19	6	7	11	
	9	28	18	5	2	10	
52	1	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2
	100	82	657	843

************************************************************************
