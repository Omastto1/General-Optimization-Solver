jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	7		2 3 5 6 14 15 18 
2	9	3		10 7 4 
3	9	4		19 13 12 9 
4	9	3		13 12 8 
5	9	2		12 8 
6	9	2		12 8 
7	9	2		19 11 
8	9	4		29 24 17 16 
9	9	4		29 24 17 16 
10	9	4		29 24 17 16 
11	9	5		29 24 21 20 17 
12	9	3		24 17 16 
13	9	3		24 17 16 
14	9	5		29 24 22 21 20 
15	9	4		29 24 21 20 
16	9	3		22 21 20 
17	9	5		30 28 26 23 22 
18	9	5		30 28 26 23 22 
19	9	1		20 
20	9	6		35 30 28 27 26 23 
21	9	7		40 35 33 30 28 27 25 
22	9	7		40 37 36 35 33 31 27 
23	9	6		40 37 33 32 31 25 
24	9	6		36 35 34 33 32 30 
25	9	4		41 39 36 34 
26	9	4		39 38 37 32 
27	9	3		39 34 32 
28	9	3		36 32 31 
29	9	3		37 34 31 
30	9	4		42 39 38 37 
31	9	3		42 41 38 
32	9	4		46 43 42 41 
33	9	3		44 39 38 
34	9	2		42 38 
35	9	3		44 41 39 
36	9	2		44 38 
37	9	6		51 50 49 46 45 44 
38	9	4		50 46 45 43 
39	9	6		51 50 49 48 46 45 
40	9	3		46 45 43 
41	9	4		50 49 48 45 
42	9	3		50 45 44 
43	9	4		51 49 48 47 
44	9	2		48 47 
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
2	1	1	25	25	8	22	
	2	10	24	23	8	20	
	3	11	19	21	8	19	
	4	12	17	20	8	18	
	5	14	16	18	8	17	
	6	15	11	17	8	14	
	7	17	7	16	8	14	
	8	23	7	13	8	12	
	9	30	3	12	8	11	
3	1	8	15	25	5	21	
	2	9	14	25	5	18	
	3	12	13	23	5	17	
	4	13	13	19	4	14	
	5	14	12	17	4	14	
	6	19	12	15	4	12	
	7	21	11	13	4	10	
	8	22	11	11	3	6	
	9	25	11	10	3	6	
4	1	1	22	27	27	15	
	2	2	22	27	27	14	
	3	7	22	26	23	13	
	4	8	22	25	21	13	
	5	9	22	24	20	11	
	6	10	22	22	16	9	
	7	14	22	22	15	8	
	8	17	22	21	13	8	
	9	28	22	20	11	6	
5	1	1	28	16	24	10	
	2	5	25	14	24	9	
	3	6	21	12	24	8	
	4	7	18	12	24	8	
	5	8	18	9	24	7	
	6	9	15	9	24	6	
	7	10	11	7	24	5	
	8	20	11	3	24	4	
	9	30	7	3	24	4	
6	1	3	27	23	11	17	
	2	9	26	23	11	16	
	3	11	26	23	11	15	
	4	16	25	23	11	13	
	5	19	25	23	11	11	
	6	20	25	23	11	10	
	7	22	25	23	11	7	
	8	26	24	23	11	6	
	9	29	24	23	11	3	
7	1	3	18	12	24	13	
	2	7	18	11	20	12	
	3	8	18	11	17	11	
	4	9	18	10	17	11	
	5	10	18	9	14	9	
	6	16	18	9	11	9	
	7	22	18	8	10	8	
	8	29	18	8	8	6	
	9	30	18	8	4	6	
8	1	1	15	15	23	24	
	2	5	13	15	21	24	
	3	6	12	15	21	24	
	4	15	11	15	20	24	
	5	19	11	14	19	24	
	6	20	10	14	19	24	
	7	23	10	13	17	24	
	8	27	8	13	16	24	
	9	30	8	13	16	23	
9	1	2	18	26	4	28	
	2	4	18	24	3	25	
	3	8	18	23	3	21	
	4	13	18	21	3	16	
	5	19	18	20	2	16	
	6	20	18	20	2	11	
	7	21	18	19	2	10	
	8	26	18	18	2	4	
	9	27	18	16	2	2	
10	1	1	23	18	11	20	
	2	2	21	16	10	19	
	3	9	20	14	10	19	
	4	13	16	13	10	18	
	5	16	14	11	10	18	
	6	19	14	9	10	18	
	7	22	12	9	10	18	
	8	26	8	6	10	17	
	9	29	6	4	10	17	
11	1	2	29	18	30	23	
	2	10	26	18	27	21	
	3	11	25	16	27	20	
	4	15	24	12	26	19	
	5	17	22	11	23	19	
	6	18	22	9	22	17	
	7	19	21	9	20	16	
	8	24	19	7	20	16	
	9	29	18	5	18	14	
12	1	1	22	14	12	8	
	2	2	20	14	12	7	
	3	12	18	12	12	6	
	4	14	18	12	12	5	
	5	16	13	11	12	5	
	6	17	13	9	12	4	
	7	18	8	8	12	2	
	8	22	7	7	12	2	
	9	26	4	7	12	1	
13	1	2	16	20	24	21	
	2	3	16	19	20	17	
	3	4	15	18	19	16	
	4	5	15	14	17	12	
	5	6	14	14	16	10	
	6	15	14	12	15	9	
	7	18	14	11	12	5	
	8	22	13	9	10	5	
	9	29	13	6	10	2	
14	1	3	25	27	22	28	
	2	4	25	26	21	27	
	3	16	25	25	20	27	
	4	17	25	25	19	25	
	5	18	25	23	17	25	
	6	23	25	23	16	23	
	7	28	25	23	15	22	
	8	29	25	22	15	22	
	9	30	25	21	13	21	
15	1	7	24	13	25	24	
	2	11	21	11	23	23	
	3	17	18	11	21	23	
	4	18	15	11	18	22	
	5	20	14	10	17	22	
	6	21	12	10	13	22	
	7	22	8	9	12	21	
	8	27	5	9	11	21	
	9	28	3	8	8	21	
16	1	1	22	9	12	3	
	2	4	22	9	12	2	
	3	5	21	8	11	3	
	4	6	19	7	10	3	
	5	11	19	7	10	2	
	6	14	17	6	9	3	
	7	18	16	6	9	3	
	8	26	15	6	9	3	
	9	27	15	5	8	3	
17	1	1	14	12	7	17	
	2	4	14	12	5	15	
	3	5	12	9	5	13	
	4	8	11	8	5	12	
	5	11	9	6	4	10	
	6	15	7	6	3	9	
	7	17	6	3	3	5	
	8	20	5	2	3	4	
	9	27	3	2	2	3	
18	1	2	23	23	14	27	
	2	3	22	23	13	26	
	3	5	21	22	13	26	
	4	6	20	21	13	25	
	5	7	20	20	13	24	
	6	12	18	20	13	23	
	7	16	18	18	13	22	
	8	24	17	18	13	20	
	9	25	15	17	13	20	
19	1	12	27	23	13	26	
	2	15	24	19	11	25	
	3	16	23	18	10	24	
	4	17	21	13	10	24	
	5	18	17	13	8	22	
	6	19	13	8	8	21	
	7	20	11	6	6	21	
	8	23	9	5	6	19	
	9	26	7	1	5	19	
20	1	1	28	30	6	8	
	2	2	26	29	6	8	
	3	7	22	28	5	8	
	4	8	19	28	4	7	
	5	9	15	27	4	7	
	6	10	11	27	3	6	
	7	11	8	26	3	5	
	8	14	5	26	3	5	
	9	20	2	26	2	5	
21	1	6	26	27	20	13	
	2	8	23	26	19	12	
	3	10	22	26	18	10	
	4	11	20	24	16	9	
	5	16	20	24	15	8	
	6	18	18	23	14	8	
	7	20	17	22	11	7	
	8	23	15	21	10	5	
	9	29	15	21	9	4	
22	1	1	5	21	23	17	
	2	4	5	21	19	16	
	3	6	4	20	19	16	
	4	15	3	19	18	16	
	5	24	3	18	16	16	
	6	25	3	18	15	16	
	7	26	2	18	12	16	
	8	28	1	16	10	16	
	9	30	1	16	10	15	
23	1	1	28	28	26	11	
	2	2	27	27	26	11	
	3	10	27	27	26	10	
	4	11	24	27	25	8	
	5	13	23	26	25	8	
	6	25	22	26	25	6	
	7	26	20	25	24	5	
	8	27	19	24	24	4	
	9	29	17	24	24	3	
24	1	4	29	14	17	29	
	2	6	28	12	14	25	
	3	13	28	11	13	22	
	4	20	27	11	11	21	
	5	21	27	8	9	18	
	6	22	27	8	8	16	
	7	24	27	5	7	12	
	8	25	26	5	6	11	
	9	30	26	3	5	7	
25	1	5	15	5	20	25	
	2	6	14	5	18	22	
	3	7	14	4	15	20	
	4	8	13	4	13	19	
	5	14	13	3	13	13	
	6	20	13	3	11	13	
	7	26	12	2	8	8	
	8	29	12	2	5	4	
	9	30	12	2	4	3	
26	1	1	11	8	15	7	
	2	5	11	8	14	6	
	3	8	9	8	14	6	
	4	9	8	8	14	5	
	5	12	6	8	13	5	
	6	13	5	8	13	5	
	7	20	5	8	13	4	
	8	23	3	8	13	3	
	9	30	2	8	13	3	
27	1	3	26	19	29	17	
	2	9	25	17	27	16	
	3	10	23	16	26	16	
	4	11	23	16	23	15	
	5	19	21	15	21	15	
	6	23	20	14	21	14	
	7	24	17	14	19	13	
	8	25	16	13	17	13	
	9	26	15	12	16	13	
28	1	2	29	24	27	21	
	2	7	28	21	26	19	
	3	8	27	20	25	18	
	4	9	25	17	25	17	
	5	19	25	16	23	16	
	6	20	23	13	22	15	
	7	21	23	8	21	12	
	8	22	21	7	21	11	
	9	29	20	5	20	10	
29	1	7	24	9	26	30	
	2	9	24	8	25	28	
	3	10	24	6	22	28	
	4	16	24	5	21	27	
	5	17	24	5	20	26	
	6	20	24	4	18	24	
	7	25	24	4	18	23	
	8	26	24	2	16	23	
	9	30	24	2	14	22	
30	1	6	3	15	18	15	
	2	8	3	14	17	14	
	3	9	3	14	17	13	
	4	14	3	12	16	10	
	5	18	3	11	16	9	
	6	20	3	10	16	8	
	7	22	3	9	16	5	
	8	24	3	7	15	4	
	9	29	3	6	15	3	
31	1	7	19	27	28	21	
	2	8	16	25	25	19	
	3	11	16	25	25	18	
	4	13	15	25	23	17	
	5	14	12	23	22	16	
	6	23	11	23	21	13	
	7	25	10	22	19	13	
	8	28	8	22	18	10	
	9	29	7	21	16	9	
32	1	2	26	7	22	27	
	2	4	26	6	21	27	
	3	8	26	6	19	27	
	4	12	26	6	17	27	
	5	13	25	6	14	26	
	6	15	25	6	12	26	
	7	20	24	6	8	25	
	8	21	24	6	7	25	
	9	27	24	6	4	25	
33	1	1	14	7	22	30	
	2	8	14	6	22	29	
	3	12	13	5	18	27	
	4	13	11	5	16	27	
	5	15	11	4	14	26	
	6	16	10	3	10	26	
	7	21	10	3	7	24	
	8	22	8	2	6	23	
	9	26	8	1	2	23	
34	1	2	10	29	27	23	
	2	10	10	27	25	23	
	3	11	10	25	25	22	
	4	13	10	23	24	22	
	5	14	10	22	24	21	
	6	20	9	21	23	20	
	7	22	9	18	23	20	
	8	24	9	18	23	19	
	9	28	9	17	22	19	
35	1	4	21	25	28	26	
	2	6	21	22	26	25	
	3	10	20	18	22	25	
	4	12	20	16	20	24	
	5	16	19	14	20	24	
	6	17	19	12	19	24	
	7	25	18	10	16	24	
	8	27	17	8	14	23	
	9	30	17	7	12	23	
36	1	4	14	23	18	30	
	2	5	13	23	15	27	
	3	8	13	23	13	22	
	4	10	13	22	13	19	
	5	15	12	22	9	19	
	6	19	12	22	7	14	
	7	28	11	21	5	14	
	8	29	11	21	5	10	
	9	30	10	21	2	8	
37	1	2	27	9	9	27	
	2	6	27	7	7	24	
	3	7	24	7	7	22	
	4	8	23	6	5	18	
	5	9	22	5	5	16	
	6	14	21	5	5	16	
	7	16	18	4	4	14	
	8	19	17	2	3	9	
	9	20	17	2	2	7	
38	1	7	24	11	9	24	
	2	8	24	10	9	23	
	3	9	22	10	8	21	
	4	10	21	8	8	19	
	5	13	21	8	7	18	
	6	16	21	6	6	16	
	7	21	19	6	6	14	
	8	22	18	4	4	13	
	9	27	18	3	4	11	
39	1	1	9	12	21	23	
	2	2	8	11	20	19	
	3	5	8	10	20	17	
	4	10	7	10	18	17	
	5	11	5	10	18	13	
	6	12	4	9	16	11	
	7	17	4	9	16	9	
	8	23	3	8	14	9	
	9	24	2	8	14	5	
40	1	2	2	14	24	23	
	2	3	2	12	20	22	
	3	4	2	11	17	21	
	4	9	2	9	17	20	
	5	10	2	9	12	19	
	6	12	2	6	12	18	
	7	14	2	6	8	17	
	8	16	2	3	7	17	
	9	28	2	2	4	16	
41	1	2	29	6	30	29	
	2	6	29	5	29	25	
	3	12	29	5	29	21	
	4	16	29	5	29	19	
	5	20	29	4	29	15	
	6	21	29	4	29	14	
	7	22	29	3	29	9	
	8	23	29	3	29	6	
	9	27	29	2	29	5	
42	1	4	20	9	10	25	
	2	15	19	9	9	22	
	3	16	18	8	8	20	
	4	21	16	7	8	20	
	5	24	14	6	7	17	
	6	25	13	6	7	15	
	7	26	12	5	6	13	
	8	27	12	4	6	11	
	9	30	10	4	5	11	
43	1	1	14	18	27	11	
	2	2	13	16	26	9	
	3	3	13	16	26	8	
	4	4	11	15	26	6	
	5	6	11	14	26	5	
	6	11	10	14	26	4	
	7	15	8	13	26	3	
	8	18	7	12	26	3	
	9	29	7	12	26	1	
44	1	1	19	28	9	13	
	2	5	17	26	9	11	
	3	6	16	24	9	10	
	4	10	16	20	9	10	
	5	12	13	18	9	8	
	6	14	13	16	8	5	
	7	15	11	14	8	4	
	8	16	9	13	8	3	
	9	30	8	12	8	1	
45	1	3	24	29	15	17	
	2	12	23	28	14	16	
	3	15	22	26	13	15	
	4	18	22	24	13	14	
	5	19	22	23	11	14	
	6	20	21	22	10	13	
	7	21	21	22	10	11	
	8	22	20	19	9	10	
	9	26	20	19	8	10	
46	1	3	28	19	19	24	
	2	4	27	18	17	20	
	3	7	23	18	17	20	
	4	13	21	18	14	18	
	5	17	20	17	14	14	
	6	18	19	17	13	13	
	7	20	15	16	10	12	
	8	24	15	16	10	10	
	9	30	11	16	8	8	
47	1	4	27	16	10	27	
	2	9	25	16	10	27	
	3	11	25	15	10	26	
	4	12	23	15	10	25	
	5	13	23	15	10	23	
	6	21	21	14	10	22	
	7	23	20	14	10	21	
	8	29	19	13	10	21	
	9	30	19	13	10	20	
48	1	1	19	22	3	29	
	2	6	18	20	3	25	
	3	7	17	19	3	24	
	4	8	16	15	3	21	
	5	10	15	14	3	20	
	6	11	13	12	3	18	
	7	12	13	11	3	16	
	8	22	12	7	3	14	
	9	30	10	6	3	12	
49	1	7	15	21	27	6	
	2	8	15	20	26	5	
	3	9	14	19	20	5	
	4	12	12	18	17	5	
	5	15	10	16	14	4	
	6	19	9	15	14	3	
	7	20	9	14	10	2	
	8	24	6	13	7	2	
	9	26	6	13	2	1	
50	1	2	22	16	28	14	
	2	9	21	15	28	13	
	3	13	21	14	28	12	
	4	14	21	14	28	11	
	5	15	20	13	28	11	
	6	16	20	12	28	10	
	7	22	20	11	28	10	
	8	26	20	10	28	8	
	9	28	20	10	28	7	
51	1	5	22	17	27	17	
	2	6	18	16	27	15	
	3	7	16	15	22	14	
	4	9	16	13	18	14	
	5	12	11	13	15	13	
	6	17	10	11	12	11	
	7	19	8	11	9	10	
	8	26	6	9	9	10	
	9	29	4	9	5	9	
52	1	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2
	103	99	627	629

************************************************************************
