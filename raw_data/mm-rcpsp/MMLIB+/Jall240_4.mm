jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	4		2 3 4 5 
2	9	5		13 11 9 8 6 
3	9	3		9 8 6 
4	9	2		13 7 
5	9	1		6 
6	9	5		18 17 16 15 12 
7	9	5		18 17 16 15 12 
8	9	4		17 16 15 12 
9	9	3		16 14 10 
10	9	5		26 22 21 17 15 
11	9	5		26 21 19 16 15 
12	9	3		22 21 14 
13	9	3		22 21 14 
14	9	7		31 28 27 26 24 20 19 
15	9	8		32 31 28 27 25 24 23 20 
16	9	5		32 27 25 24 23 
17	9	5		31 29 28 27 20 
18	9	5		32 30 28 27 24 
19	9	4		32 29 25 23 
20	9	5		39 36 35 33 30 
21	9	5		39 36 35 33 32 
22	9	3		36 33 29 
23	9	6		44 39 37 36 35 34 
24	9	3		37 34 29 
25	9	3		39 33 30 
26	9	3		37 34 29 
27	9	4		42 39 34 33 
28	9	5		44 42 40 37 36 
29	9	3		44 39 35 
30	9	3		42 37 34 
31	9	3		42 39 34 
32	9	6		51 44 42 41 40 38 
33	9	4		51 44 40 37 
34	9	4		51 41 40 38 
35	9	5		50 48 42 41 40 
36	9	3		51 41 38 
37	9	2		41 38 
38	9	5		50 49 48 47 43 
39	9	4		51 48 47 43 
40	9	3		49 47 43 
41	9	1		43 
42	9	1		43 
43	9	1		45 
44	9	1		45 
45	9	1		46 
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
2	1	5	21	14	2	16	
	2	12	21	12	1	14	
	3	20	21	11	1	14	
	4	21	21	10	1	13	
	5	22	20	9	1	12	
	6	23	20	9	1	10	
	7	24	20	7	1	10	
	8	25	20	7	1	8	
	9	26	20	6	1	7	
3	1	4	18	24	23	12	
	2	8	18	22	23	11	
	3	10	18	22	23	10	
	4	11	18	21	23	9	
	5	13	18	20	23	7	
	6	16	18	19	23	7	
	7	20	18	18	23	6	
	8	22	18	17	23	4	
	9	25	18	17	23	3	
4	1	1	12	5	25	25	
	2	3	11	4	24	23	
	3	5	11	3	24	19	
	4	11	11	3	23	18	
	5	15	10	3	22	16	
	6	17	10	2	21	13	
	7	21	10	1	20	11	
	8	28	9	1	19	10	
	9	29	9	1	18	9	
5	1	11	23	23	26	27	
	2	12	21	22	26	26	
	3	14	20	20	25	25	
	4	18	19	18	25	24	
	5	20	19	16	24	23	
	6	23	18	13	24	23	
	7	24	17	11	24	21	
	8	27	15	10	23	21	
	9	28	14	6	23	20	
6	1	3	21	6	28	16	
	2	6	21	5	27	16	
	3	16	19	5	25	15	
	4	20	18	4	24	15	
	5	21	17	4	23	14	
	6	22	16	4	23	14	
	7	26	15	3	21	14	
	8	27	15	3	20	13	
	9	30	14	3	20	13	
7	1	2	26	23	21	27	
	2	3	24	23	20	23	
	3	4	24	21	17	23	
	4	6	23	21	17	19	
	5	17	21	20	15	16	
	6	20	21	19	11	14	
	7	27	20	17	10	12	
	8	29	18	16	6	10	
	9	30	17	15	5	9	
8	1	1	16	20	23	22	
	2	5	14	20	20	21	
	3	6	13	19	17	19	
	4	7	12	17	16	18	
	5	14	12	17	14	15	
	6	22	12	17	10	13	
	7	25	10	16	8	12	
	8	26	10	14	6	10	
	9	28	9	14	5	9	
9	1	4	16	1	29	6	
	2	5	16	1	27	5	
	3	11	15	1	27	5	
	4	15	15	1	25	5	
	5	16	14	1	25	4	
	6	18	14	1	23	4	
	7	19	13	1	22	3	
	8	23	13	1	22	2	
	9	26	12	1	21	3	
10	1	2	14	29	26	10	
	2	14	13	23	22	10	
	3	18	13	20	19	10	
	4	19	13	17	15	10	
	5	21	13	15	13	9	
	6	24	13	10	11	9	
	7	25	13	9	8	9	
	8	26	13	7	3	8	
	9	27	13	3	2	8	
11	1	6	18	18	22	28	
	2	8	18	18	21	26	
	3	9	18	18	20	22	
	4	12	18	17	20	19	
	5	19	18	17	19	17	
	6	21	18	17	19	16	
	7	26	18	17	19	12	
	8	29	18	16	18	9	
	9	30	18	16	18	7	
12	1	2	24	28	28	23	
	2	5	22	28	27	22	
	3	6	20	27	25	21	
	4	9	20	27	22	20	
	5	19	17	25	21	20	
	6	20	17	25	19	20	
	7	24	15	25	18	19	
	8	25	14	23	15	19	
	9	30	12	23	14	18	
13	1	10	21	10	19	9	
	2	14	17	9	17	9	
	3	16	15	9	17	9	
	4	19	15	7	13	9	
	5	20	11	7	10	9	
	6	21	11	6	9	9	
	7	25	9	5	7	9	
	8	26	7	5	4	9	
	9	28	5	4	4	9	
14	1	10	17	21	20	12	
	2	21	16	19	19	11	
	3	22	16	18	19	9	
	4	23	16	16	19	9	
	5	24	15	15	18	6	
	6	25	15	13	18	6	
	7	28	14	12	17	4	
	8	29	14	12	17	3	
	9	30	14	10	17	1	
15	1	2	18	21	29	27	
	2	3	17	17	28	26	
	3	9	17	15	28	25	
	4	10	17	12	28	24	
	5	19	17	10	28	22	
	6	26	17	10	28	21	
	7	27	17	6	28	20	
	8	28	17	5	28	20	
	9	30	17	2	28	19	
16	1	2	20	24	27	20	
	2	7	20	21	27	19	
	3	9	20	17	26	18	
	4	22	20	16	26	15	
	5	23	20	12	25	15	
	6	25	19	9	25	12	
	7	27	19	7	24	11	
	8	28	19	5	23	9	
	9	30	19	4	23	8	
17	1	9	15	26	12	25	
	2	10	14	23	10	25	
	3	12	13	21	8	23	
	4	15	10	20	8	22	
	5	16	9	18	7	22	
	6	23	8	16	4	20	
	7	28	7	13	4	19	
	8	29	5	12	3	18	
	9	30	3	10	2	17	
18	1	3	14	29	26	16	
	2	6	12	28	25	15	
	3	11	11	24	25	13	
	4	16	10	24	24	10	
	5	18	9	22	24	9	
	6	22	8	19	24	9	
	7	26	5	18	23	5	
	8	27	4	17	23	5	
	9	28	2	14	22	2	
19	1	4	15	20	26	12	
	2	5	15	20	20	12	
	3	6	15	20	17	11	
	4	10	15	20	14	10	
	5	12	15	20	12	10	
	6	13	14	20	11	9	
	7	18	14	20	7	9	
	8	20	14	20	3	8	
	9	22	14	20	1	7	
20	1	1	22	27	5	13	
	2	6	18	26	5	12	
	3	8	15	26	5	12	
	4	9	13	26	5	12	
	5	10	12	25	5	12	
	6	11	9	24	5	12	
	7	13	7	24	5	12	
	8	15	3	23	5	12	
	9	16	3	23	5	11	
21	1	8	29	24	26	30	
	2	12	27	23	24	25	
	3	16	25	22	22	20	
	4	21	23	20	20	20	
	5	24	21	20	17	17	
	6	25	19	18	14	13	
	7	26	18	17	11	7	
	8	27	15	16	11	4	
	9	28	14	15	6	2	
22	1	7	13	12	9	22	
	2	13	13	12	8	21	
	3	16	12	12	8	21	
	4	17	12	12	7	21	
	5	23	11	12	7	20	
	6	24	11	12	6	20	
	7	25	10	12	5	20	
	8	27	9	12	5	19	
	9	30	9	12	5	18	
23	1	2	27	25	26	19	
	2	10	27	21	25	16	
	3	11	26	19	25	16	
	4	17	26	15	24	14	
	5	18	25	14	22	14	
	6	19	24	10	21	12	
	7	20	24	9	19	11	
	8	27	23	6	19	11	
	9	30	23	4	18	10	
24	1	2	25	22	20	30	
	2	3	21	19	19	30	
	3	4	17	18	18	30	
	4	12	14	16	18	30	
	5	13	12	15	16	30	
	6	20	11	15	15	30	
	7	23	8	12	15	30	
	8	27	4	11	14	30	
	9	29	2	11	13	30	
25	1	7	26	26	24	29	
	2	12	26	22	20	27	
	3	13	25	21	20	24	
	4	16	25	19	18	23	
	5	18	24	19	16	21	
	6	19	24	16	11	20	
	7	20	23	16	10	18	
	8	21	23	14	8	14	
	9	25	23	12	5	14	
26	1	4	28	21	23	28	
	2	5	26	19	22	25	
	3	11	25	17	20	22	
	4	17	24	15	16	22	
	5	18	24	12	14	18	
	6	20	23	11	12	17	
	7	24	21	7	8	16	
	8	28	21	5	8	12	
	9	29	20	4	5	12	
27	1	5	20	27	17	16	
	2	10	20	22	17	15	
	3	13	18	20	17	14	
	4	17	17	16	17	14	
	5	20	16	15	17	13	
	6	22	16	14	16	13	
	7	23	15	10	16	12	
	8	29	13	8	16	12	
	9	30	13	6	16	12	
28	1	7	27	27	8	3	
	2	13	26	25	8	3	
	3	14	26	25	7	3	
	4	18	24	24	7	3	
	5	19	23	24	6	3	
	6	20	22	23	5	3	
	7	21	21	22	4	3	
	8	23	20	22	3	3	
	9	30	19	21	3	3	
29	1	1	4	25	17	10	
	2	2	4	24	16	10	
	3	4	4	22	14	10	
	4	14	4	21	14	10	
	5	17	4	21	13	10	
	6	21	4	19	12	10	
	7	23	4	19	10	10	
	8	29	4	17	9	10	
	9	30	4	17	8	10	
30	1	1	9	10	13	12	
	2	2	7	9	12	12	
	3	3	7	8	11	12	
	4	6	7	6	11	11	
	5	12	6	5	10	10	
	6	13	6	4	9	10	
	7	14	5	3	9	10	
	8	24	5	2	9	9	
	9	25	4	1	8	9	
31	1	11	23	17	26	12	
	2	12	19	16	25	12	
	3	16	17	14	22	12	
	4	17	17	13	21	12	
	5	22	15	12	19	12	
	6	24	12	12	17	11	
	7	26	10	10	17	11	
	8	27	9	9	15	11	
	9	29	8	9	13	11	
32	1	9	26	16	27	17	
	2	10	25	14	26	14	
	3	11	25	11	24	14	
	4	12	24	11	19	12	
	5	13	23	8	16	11	
	6	17	22	8	15	10	
	7	25	21	6	9	7	
	8	26	20	5	6	7	
	9	29	19	4	5	6	
33	1	4	27	12	29	28	
	2	5	26	12	24	25	
	3	7	22	12	24	22	
	4	8	20	12	21	20	
	5	11	16	11	18	16	
	6	12	14	11	17	14	
	7	14	11	10	15	14	
	8	15	8	10	14	9	
	9	25	7	10	10	6	
34	1	4	29	9	26	23	
	2	13	29	9	25	22	
	3	14	27	8	25	22	
	4	17	26	7	24	22	
	5	18	26	7	24	21	
	6	22	26	6	24	21	
	7	24	25	6	23	21	
	8	25	24	5	23	21	
	9	30	23	5	22	21	
35	1	1	20	25	18	17	
	2	3	19	22	16	17	
	3	11	18	21	15	17	
	4	17	15	18	15	17	
	5	19	15	16	12	17	
	6	20	13	14	11	16	
	7	24	12	13	9	16	
	8	25	10	11	7	16	
	9	27	10	9	7	16	
36	1	3	27	18	27	23	
	2	5	24	17	23	20	
	3	6	22	15	23	19	
	4	7	20	15	21	19	
	5	8	20	14	19	18	
	6	11	17	12	17	16	
	7	14	16	11	17	15	
	8	21	14	10	13	14	
	9	25	13	10	12	12	
37	1	1	24	16	27	13	
	2	7	21	13	25	12	
	3	9	20	13	24	12	
	4	10	17	11	22	12	
	5	14	16	10	18	12	
	6	15	14	9	16	11	
	7	16	11	7	16	11	
	8	28	6	6	12	11	
	9	29	6	5	11	11	
38	1	3	29	30	13	10	
	2	5	27	26	11	9	
	3	6	27	23	10	8	
	4	17	25	22	9	8	
	5	18	25	20	9	7	
	6	20	24	17	9	7	
	7	21	22	15	8	6	
	8	28	21	15	6	6	
	9	29	21	13	6	6	
39	1	1	27	5	29	29	
	2	4	26	5	28	28	
	3	8	25	5	28	27	
	4	14	24	5	27	26	
	5	15	21	5	27	25	
	6	18	19	4	27	25	
	7	20	17	4	26	24	
	8	23	17	4	26	22	
	9	24	15	4	25	22	
40	1	1	12	29	30	16	
	2	8	12	29	27	15	
	3	13	11	29	25	15	
	4	15	9	29	21	13	
	5	16	9	29	17	13	
	6	17	9	29	16	12	
	7	27	8	29	14	12	
	8	28	7	29	11	11	
	9	29	6	29	10	10	
41	1	1	30	13	20	24	
	2	2	27	11	19	24	
	3	5	27	11	18	21	
	4	8	25	9	16	18	
	5	19	23	8	16	15	
	6	20	23	8	15	10	
	7	21	20	6	13	9	
	8	22	20	5	12	4	
	9	28	19	5	12	2	
42	1	9	27	15	16	9	
	2	13	27	15	15	8	
	3	20	25	13	13	7	
	4	22	25	12	10	7	
	5	25	23	12	10	6	
	6	27	22	10	6	5	
	7	28	22	10	4	5	
	8	29	21	8	4	4	
	9	30	20	8	2	4	
43	1	2	16	29	5	21	
	2	4	15	27	4	19	
	3	5	14	26	4	19	
	4	12	12	25	3	17	
	5	14	12	25	3	16	
	6	27	11	23	2	15	
	7	28	10	21	2	15	
	8	29	8	20	2	14	
	9	30	7	20	1	13	
44	1	10	25	26	15	30	
	2	15	21	25	11	28	
	3	21	21	21	11	28	
	4	22	18	17	9	28	
	5	23	13	16	7	26	
	6	27	10	11	7	26	
	7	28	9	8	4	26	
	8	29	6	8	2	24	
	9	30	3	4	1	24	
45	1	4	24	22	25	24	
	2	6	22	20	23	24	
	3	7	19	17	23	24	
	4	8	17	17	21	24	
	5	11	15	13	20	23	
	6	13	13	11	19	23	
	7	15	10	9	18	23	
	8	16	9	7	18	23	
	9	27	6	5	17	23	
46	1	7	22	22	19	29	
	2	11	18	20	18	26	
	3	12	15	20	15	25	
	4	13	14	17	15	23	
	5	14	12	16	12	23	
	6	19	9	14	10	22	
	7	20	8	13	10	20	
	8	26	6	11	9	19	
	9	27	4	10	7	18	
47	1	2	23	22	25	16	
	2	5	22	22	24	14	
	3	9	22	22	23	14	
	4	15	22	22	21	13	
	5	20	21	22	21	12	
	6	21	21	21	20	12	
	7	22	21	21	19	11	
	8	26	21	21	17	11	
	9	29	21	21	16	10	
48	1	7	29	14	23	24	
	2	8	27	13	22	24	
	3	13	27	13	22	23	
	4	16	26	12	22	24	
	5	22	26	11	21	23	
	6	26	26	11	21	22	
	7	27	25	11	20	23	
	8	27	24	10	20	24	
	9	28	24	10	20	23	
49	1	1	27	23	13	28	
	2	2	25	22	12	27	
	3	7	24	21	11	26	
	4	8	23	21	10	26	
	5	14	20	19	8	25	
	6	15	20	19	8	24	
	7	16	19	19	6	23	
	8	27	16	17	5	23	
	9	28	16	17	5	22	
50	1	12	19	21	28	21	
	2	14	17	20	28	21	
	3	15	16	18	27	20	
	4	16	16	17	25	20	
	5	17	15	16	25	19	
	6	21	14	15	23	18	
	7	24	13	15	22	18	
	8	28	12	14	22	17	
	9	30	11	12	21	17	
51	1	5	13	15	23	23	
	2	8	13	14	21	22	
	3	10	10	14	20	20	
	4	13	8	13	17	20	
	5	14	8	12	16	18	
	6	15	5	12	14	18	
	7	21	5	11	11	16	
	8	22	4	11	8	15	
	9	30	1	11	6	15	
52	1	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2
	96	95	940	885

************************************************************************
