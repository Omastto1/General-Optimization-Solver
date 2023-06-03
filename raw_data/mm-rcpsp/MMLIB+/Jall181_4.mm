jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 4 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	6		2 3 4 5 7 9 
2	6	4		12 11 8 6 
3	6	4		15 11 10 8 
4	6	2		10 6 
5	6	2		15 8 
6	6	6		22 21 19 16 15 14 
7	6	6		25 22 21 19 16 14 
8	6	5		23 22 19 18 13 
9	6	6		27 23 22 19 18 17 
10	6	4		25 23 19 16 
11	6	2		16 14 
12	6	5		27 25 22 19 17 
13	6	3		25 21 16 
14	6	4		27 23 18 17 
15	6	3		27 25 17 
16	6	2		27 17 
17	6	3		26 24 20 
18	6	5		37 30 29 26 24 
19	6	5		37 31 30 29 28 
20	6	4		37 30 29 28 
21	6	3		37 28 27 
22	6	6		44 41 37 34 33 31 
23	6	7		44 41 37 36 34 33 32 
24	6	3		36 32 28 
25	6	3		36 32 28 
26	6	4		44 41 33 31 
27	6	3		44 33 31 
28	6	5		44 41 40 34 33 
29	6	5		44 41 36 35 32 
30	6	6		44 43 41 40 36 35 
31	6	3		36 35 32 
32	6	5		51 50 43 40 38 
33	6	3		45 43 35 
34	6	4		50 45 43 39 
35	6	3		51 50 38 
36	6	3		51 50 39 
37	6	2		45 38 
38	6	1		39 
39	6	3		48 47 42 
40	6	3		48 47 45 
41	6	3		50 47 46 
42	6	2		49 46 
43	6	2		47 46 
44	6	2		51 47 
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
2	1	12	14	20	24	20	6	17	
	2	13	12	20	22	20	5	14	
	3	15	12	20	17	20	4	13	
	4	16	9	20	15	20	3	10	
	5	21	9	19	13	20	1	7	
	6	27	8	19	9	20	1	3	
3	1	7	20	28	29	14	17	23	
	2	12	15	27	26	14	15	22	
	3	22	13	25	21	11	11	21	
	4	24	11	23	21	10	11	21	
	5	27	6	22	16	6	9	20	
	6	28	4	19	14	5	7	19	
4	1	2	15	22	19	17	15	24	
	2	3	13	21	19	14	13	24	
	3	5	9	20	19	13	11	22	
	4	13	7	19	19	13	11	19	
	5	15	4	16	19	12	9	18	
	6	18	2	15	19	10	8	16	
5	1	7	27	21	18	21	28	17	
	2	11	27	15	17	21	26	15	
	3	20	26	13	16	20	23	13	
	4	22	26	11	16	20	20	12	
	5	24	24	9	15	20	19	11	
	6	25	24	6	14	19	17	8	
6	1	2	22	15	20	26	28	18	
	2	12	17	14	19	25	21	17	
	3	15	16	13	19	24	21	16	
	4	21	14	12	18	19	17	15	
	5	25	10	11	17	18	10	13	
	6	26	6	9	17	17	8	12	
7	1	7	9	18	25	27	21	29	
	2	8	9	16	20	27	17	25	
	3	15	7	13	17	26	14	25	
	4	20	6	8	16	23	11	22	
	5	25	4	6	12	22	10	20	
	6	27	1	3	10	21	4	19	
8	1	8	7	24	18	10	23	24	
	2	9	6	21	18	10	23	21	
	3	16	4	20	17	10	23	18	
	4	21	4	15	16	10	23	14	
	5	22	2	12	15	10	23	10	
	6	26	1	11	14	10	23	8	
9	1	6	27	20	19	8	7	26	
	2	8	25	19	19	8	7	24	
	3	15	22	16	19	7	7	21	
	4	23	19	16	19	6	6	19	
	5	26	19	14	19	6	6	18	
	6	30	16	13	19	5	5	15	
10	1	3	24	22	22	3	26	10	
	2	6	21	18	22	3	25	9	
	3	14	19	17	15	2	24	7	
	4	25	16	16	14	2	22	7	
	5	28	11	13	9	1	21	5	
	6	30	10	12	3	1	21	4	
11	1	1	24	14	8	29	26	21	
	2	11	20	13	8	28	25	20	
	3	15	14	13	8	28	25	20	
	4	20	10	12	8	28	25	19	
	5	27	5	10	8	28	25	19	
	6	29	4	10	8	28	25	18	
12	1	3	16	19	22	11	19	12	
	2	7	15	17	17	10	17	12	
	3	16	15	17	14	10	16	11	
	4	18	14	16	12	9	14	9	
	5	19	14	14	7	8	12	8	
	6	22	14	14	5	7	10	8	
13	1	7	10	10	17	15	27	10	
	2	8	9	10	15	13	20	7	
	3	9	7	10	14	12	18	6	
	4	13	5	9	11	10	16	5	
	5	20	2	8	6	9	13	2	
	6	21	2	8	1	8	7	1	
14	1	4	18	10	27	26	5	14	
	2	8	13	9	22	25	4	12	
	3	9	10	8	19	24	4	11	
	4	17	10	7	16	23	3	8	
	5	24	4	6	10	23	3	6	
	6	25	4	6	7	22	3	4	
15	1	8	26	13	30	17	19	25	
	2	11	25	11	30	14	19	24	
	3	12	25	11	30	10	12	23	
	4	13	24	8	30	10	11	23	
	5	15	24	7	30	6	5	22	
	6	28	24	6	30	2	4	21	
16	1	4	5	25	27	23	26	23	
	2	8	5	19	24	20	26	23	
	3	15	4	13	16	20	24	23	
	4	20	4	11	13	16	24	23	
	5	24	4	8	8	11	23	23	
	6	30	3	5	6	9	21	23	
17	1	4	23	5	3	24	27	18	
	2	5	23	4	2	23	27	18	
	3	7	23	4	2	22	23	17	
	4	8	23	4	1	21	23	16	
	5	15	23	3	1	20	21	13	
	6	27	23	2	1	20	18	13	
18	1	1	22	26	22	23	28	25	
	2	14	18	22	22	20	27	20	
	3	19	17	19	21	18	27	16	
	4	20	17	15	19	16	27	12	
	5	21	13	14	19	13	26	10	
	6	22	12	8	18	11	26	6	
19	1	2	25	28	27	25	15	24	
	2	3	24	23	27	25	14	23	
	3	6	22	22	24	20	13	22	
	4	9	17	19	22	17	12	20	
	5	21	16	16	20	16	11	20	
	6	24	14	13	19	13	8	18	
20	1	11	20	13	15	21	22	26	
	2	23	13	11	12	16	20	23	
	3	24	11	10	9	12	14	21	
	4	25	8	10	8	11	12	16	
	5	26	7	9	6	6	8	15	
	6	28	4	8	5	2	5	13	
21	1	8	9	7	27	11	22	8	
	2	10	9	7	26	9	21	7	
	3	18	9	7	26	7	16	7	
	4	21	9	6	24	6	14	7	
	5	24	9	6	23	2	8	6	
	6	29	9	6	21	2	3	6	
22	1	3	15	20	19	9	26	17	
	2	10	15	19	17	9	25	15	
	3	13	15	15	15	7	24	14	
	4	16	15	13	15	7	24	11	
	5	17	15	10	11	5	23	8	
	6	18	15	6	11	4	22	8	
23	1	6	27	15	19	27	21	24	
	2	21	24	14	18	25	18	18	
	3	23	20	12	15	24	14	15	
	4	24	15	10	15	24	12	13	
	5	28	13	8	13	22	8	10	
	6	29	12	5	11	22	1	3	
24	1	3	16	21	16	23	29	9	
	2	9	12	18	14	20	29	7	
	3	14	11	16	13	15	29	6	
	4	15	8	14	13	10	28	5	
	5	21	7	12	11	10	28	3	
	6	30	5	7	11	5	27	1	
25	1	6	25	23	20	29	21	21	
	2	8	24	22	19	25	21	18	
	3	9	24	19	18	21	21	15	
	4	10	22	18	17	18	21	11	
	5	13	21	16	14	16	21	11	
	6	26	21	12	13	12	21	8	
26	1	6	14	29	16	30	24	15	
	2	12	12	26	14	27	23	14	
	3	17	11	25	12	25	23	13	
	4	19	10	24	9	25	22	12	
	5	27	9	24	8	23	22	11	
	6	28	6	22	7	20	22	10	
27	1	3	25	24	7	26	23	29	
	2	10	24	21	7	23	17	23	
	3	11	24	21	6	21	16	21	
	4	20	24	17	5	16	9	16	
	5	26	24	17	3	16	8	10	
	6	27	24	14	3	12	3	10	
28	1	9	14	17	14	15	23	13	
	2	10	14	16	12	12	20	11	
	3	12	13	16	12	12	15	9	
	4	15	13	16	10	8	12	8	
	5	16	13	15	9	6	7	8	
	6	19	12	14	7	4	3	5	
29	1	7	14	21	15	30	10	24	
	2	21	13	19	15	29	7	23	
	3	25	13	16	14	29	6	22	
	4	26	13	15	12	29	4	22	
	5	28	13	12	12	28	4	22	
	6	29	13	8	11	28	2	21	
30	1	4	26	25	17	17	18	12	
	2	8	25	24	15	15	13	10	
	3	9	25	24	15	11	11	10	
	4	10	23	24	14	10	8	8	
	5	11	22	24	13	7	5	5	
	6	30	21	24	12	1	4	3	
31	1	5	26	12	24	25	25	10	
	2	14	24	10	23	25	23	9	
	3	21	22	10	20	25	22	9	
	4	22	18	7	15	25	20	9	
	5	24	16	5	14	25	19	9	
	6	27	15	4	10	25	19	9	
32	1	8	16	20	25	28	24	19	
	2	13	15	19	25	26	19	18	
	3	17	15	18	24	24	17	15	
	4	18	14	16	21	23	12	11	
	5	20	13	16	19	19	9	8	
	6	26	12	15	19	19	7	5	
33	1	2	20	17	8	5	23	24	
	2	4	18	17	7	4	23	23	
	3	6	18	15	6	3	22	21	
	4	8	17	14	5	3	20	17	
	5	9	16	13	3	2	20	16	
	6	30	16	13	1	2	18	15	
34	1	8	15	2	24	20	15	18	
	2	10	12	2	23	19	15	16	
	3	11	11	2	23	15	15	12	
	4	13	9	2	22	13	14	8	
	5	18	6	2	22	7	14	4	
	6	26	6	2	22	7	13	4	
35	1	5	19	5	27	27	17	18	
	2	7	19	4	24	25	14	16	
	3	22	18	3	22	24	12	16	
	4	23	17	3	18	21	9	15	
	5	25	16	3	14	21	7	14	
	6	29	16	2	13	19	2	14	
36	1	4	29	3	18	26	28	24	
	2	7	21	2	15	24	28	23	
	3	15	21	2	15	23	27	20	
	4	18	16	1	14	20	25	19	
	5	23	13	1	12	19	25	16	
	6	27	5	1	11	18	23	14	
37	1	6	19	16	23	24	17	28	
	2	7	17	14	23	22	14	21	
	3	11	11	14	19	19	11	14	
	4	12	10	14	11	17	11	11	
	5	19	4	13	8	15	7	10	
	6	23	2	12	5	14	5	3	
38	1	1	22	15	12	22	17	7	
	2	4	21	13	10	21	14	6	
	3	9	20	13	9	20	12	6	
	4	17	20	12	6	20	9	5	
	5	20	18	10	6	19	4	3	
	6	25	18	8	5	19	4	3	
39	1	11	15	18	10	19	22	19	
	2	13	14	16	10	19	16	17	
	3	15	14	12	10	16	16	15	
	4	25	13	10	10	15	11	14	
	5	27	12	8	10	12	9	12	
	6	28	12	4	10	12	7	11	
40	1	3	16	24	26	27	15	23	
	2	12	14	23	21	24	14	21	
	3	16	12	23	20	24	12	20	
	4	23	11	23	14	23	12	19	
	5	29	9	23	14	20	10	18	
	6	30	8	23	8	19	9	17	
41	1	4	13	28	7	29	19	24	
	2	8	11	19	7	25	18	23	
	3	12	11	18	7	23	14	22	
	4	14	11	14	7	23	12	22	
	5	17	9	7	6	20	7	21	
	6	20	9	1	6	19	6	20	
42	1	13	18	27	24	17	25	26	
	2	14	18	26	21	17	25	21	
	3	17	16	26	15	17	20	18	
	4	24	15	25	12	17	20	15	
	5	26	14	25	11	17	15	14	
	6	28	14	24	9	17	13	11	
43	1	3	26	4	14	6	29	13	
	2	9	25	3	11	5	28	12	
	3	10	20	3	11	5	28	8	
	4	13	19	3	10	5	27	6	
	5	14	16	1	8	3	27	4	
	6	17	13	1	6	3	26	3	
44	1	1	10	10	25	25	28	26	
	2	13	9	9	23	24	27	24	
	3	14	8	9	19	23	25	23	
	4	17	7	8	12	23	24	22	
	5	24	3	8	9	22	23	20	
	6	29	3	7	5	21	23	19	
45	1	2	27	18	21	21	23	22	
	2	19	25	18	20	18	19	19	
	3	21	20	18	20	18	16	19	
	4	24	14	18	18	16	13	18	
	5	25	12	18	17	14	11	16	
	6	29	6	18	17	13	10	16	
46	1	2	26	23	21	29	28	17	
	2	9	26	21	21	24	28	16	
	3	10	26	16	21	22	25	16	
	4	14	25	12	20	21	23	16	
	5	22	25	8	20	17	22	16	
	6	30	25	5	20	14	20	16	
47	1	1	13	26	9	18	27	20	
	2	16	10	23	8	15	27	18	
	3	17	10	22	7	12	24	15	
	4	20	9	17	6	9	23	14	
	5	24	7	16	6	9	22	11	
	6	27	7	12	5	6	21	10	
48	1	2	11	27	6	6	25	12	
	2	6	11	27	6	6	24	11	
	3	19	10	25	5	5	20	11	
	4	20	7	25	5	5	18	8	
	5	23	6	24	4	5	15	7	
	6	28	6	23	4	4	13	7	
49	1	2	28	16	28	16	19	23	
	2	5	27	16	25	15	18	19	
	3	8	25	16	21	15	16	13	
	4	23	25	16	15	14	12	9	
	5	24	24	16	13	14	12	7	
	6	27	23	16	7	14	9	1	
50	1	6	13	22	20	3	25	28	
	2	12	11	21	19	3	21	26	
	3	13	10	17	19	3	20	25	
	4	16	6	15	19	3	14	24	
	5	17	4	11	19	3	11	21	
	6	18	3	10	19	3	9	21	
51	1	3	30	24	6	25	27	20	
	2	20	27	19	5	25	24	20	
	3	22	25	15	5	22	24	20	
	4	23	24	9	4	16	21	20	
	5	25	20	8	3	13	19	19	
	6	29	19	4	3	12	18	19	
52	1	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	R 3	R 4	N 1	N 2
	53	54	56	55	723	651

************************************************************************
