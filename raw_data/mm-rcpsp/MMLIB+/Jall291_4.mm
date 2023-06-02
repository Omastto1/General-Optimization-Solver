jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 4 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	6		2 4 5 6 7 10 
2	9	3		15 14 3 
3	9	5		22 17 12 9 8 
4	9	4		17 15 14 8 
5	9	4		22 17 14 8 
6	9	4		22 17 11 9 
7	9	3		17 15 13 
8	9	2		16 11 
9	9	5		24 21 20 18 16 
10	9	1		11 
11	9	5		29 24 21 20 18 
12	9	3		24 21 16 
13	9	3		24 21 16 
14	9	7		31 29 27 25 24 20 19 
15	9	4		31 25 22 18 
16	9	5		31 29 27 23 19 
17	9	4		29 27 20 19 
18	9	3		27 23 19 
19	9	3		38 28 26 
20	9	3		32 26 23 
21	9	3		32 31 25 
22	9	2		24 23 
23	9	4		38 36 34 30 
24	9	4		38 36 32 30 
25	9	4		38 36 34 30 
26	9	3		36 34 30 
27	9	1		28 
28	9	4		37 36 33 32 
29	9	3		36 33 32 
30	9	4		46 37 35 33 
31	9	3		35 34 33 
32	9	5		51 46 45 40 35 
33	9	6		51 50 45 44 41 39 
34	9	5		51 50 46 45 40 
35	9	4		49 44 42 41 
36	9	3		44 42 40 
37	9	5		51 50 48 45 43 
38	9	2		44 40 
39	9	1		40 
40	9	3		49 48 43 
41	9	2		48 43 
42	9	2		50 47 
43	9	1		47 
44	9	1		47 
45	9	1		47 
46	9	1		48 
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
2	1	4	22	17	26	15	22	26	
	2	5	22	16	25	15	22	25	
	3	10	22	16	25	14	20	24	
	4	14	21	15	25	13	19	24	
	5	15	20	14	24	13	17	22	
	6	19	20	14	24	12	16	22	
	7	20	19	14	24	12	16	21	
	8	21	19	13	23	11	13	20	
	9	29	19	13	23	11	12	20	
3	1	2	14	18	11	22	5	17	
	2	9	14	17	11	22	5	17	
	3	10	14	17	11	21	5	16	
	4	17	14	17	11	19	5	16	
	5	18	14	15	11	19	5	15	
	6	19	13	15	10	18	5	14	
	7	27	13	14	10	18	5	13	
	8	29	13	13	10	17	5	12	
	9	30	13	13	10	16	5	12	
4	1	5	24	11	21	18	30	6	
	2	7	21	11	21	15	28	6	
	3	8	17	10	20	13	26	5	
	4	9	15	9	20	11	24	5	
	5	10	13	8	19	9	23	5	
	6	17	13	6	19	7	21	4	
	7	24	11	6	19	7	18	3	
	8	29	9	4	18	4	17	3	
	9	30	7	3	18	2	16	3	
5	1	5	10	16	14	14	28	29	
	2	6	9	16	13	13	27	27	
	3	7	7	16	12	13	27	23	
	4	8	6	16	10	13	27	19	
	5	10	5	15	9	13	27	19	
	6	16	5	15	9	13	27	14	
	7	19	3	15	6	13	27	12	
	8	20	2	14	6	13	27	10	
	9	26	1	14	4	13	27	8	
6	1	1	27	28	19	28	29	5	
	2	2	23	25	18	25	28	4	
	3	3	23	24	18	24	27	4	
	4	6	20	20	18	23	26	4	
	5	9	20	19	17	22	25	3	
	6	17	18	17	17	21	24	3	
	7	18	16	14	16	18	24	3	
	8	22	15	14	16	18	23	2	
	9	30	13	12	16	15	22	2	
7	1	2	26	28	24	29	28	24	
	2	6	26	28	24	28	25	23	
	3	7	26	26	23	28	25	21	
	4	11	26	25	22	28	21	20	
	5	16	26	25	21	28	20	20	
	6	20	26	25	20	28	18	18	
	7	22	26	23	19	28	17	17	
	8	23	26	23	19	28	15	16	
	9	24	26	22	18	28	11	15	
8	1	3	30	7	25	27	22	24	
	2	5	29	7	22	27	20	20	
	3	17	27	7	21	23	20	19	
	4	19	27	7	17	22	20	18	
	5	20	26	6	14	20	19	16	
	6	25	24	6	11	18	18	14	
	7	26	24	6	9	17	17	12	
	8	27	23	6	6	15	17	12	
	9	29	22	6	4	13	16	10	
9	1	1	20	17	9	22	12	3	
	2	8	18	15	7	20	11	3	
	3	13	16	13	7	19	11	3	
	4	17	16	12	6	17	11	3	
	5	19	14	11	5	16	11	3	
	6	21	14	9	5	15	11	3	
	7	27	12	7	4	13	11	3	
	8	28	11	5	4	11	11	3	
	9	29	11	4	3	11	11	3	
10	1	1	21	23	20	15	29	25	
	2	5	21	22	19	15	26	24	
	3	13	16	22	17	13	26	22	
	4	14	15	22	16	10	24	22	
	5	15	13	21	13	8	24	21	
	6	16	9	21	13	7	22	20	
	7	21	6	20	10	5	21	20	
	8	28	4	20	9	3	21	19	
	9	30	2	20	9	1	20	18	
11	1	4	23	20	26	30	2	24	
	2	8	20	19	23	27	1	22	
	3	10	19	16	20	23	1	22	
	4	15	19	14	16	21	1	18	
	5	18	16	13	15	19	1	17	
	6	20	14	12	12	14	1	14	
	7	21	14	9	7	14	1	11	
	8	27	12	7	4	10	1	7	
	9	30	10	4	3	7	1	7	
12	1	10	13	5	24	19	27	14	
	2	11	13	4	23	17	26	13	
	3	16	12	4	22	16	25	13	
	4	17	11	4	22	14	23	13	
	5	18	10	3	22	14	22	12	
	6	27	10	3	21	13	22	12	
	7	28	9	3	20	11	20	11	
	8	29	8	3	20	8	19	11	
	9	30	7	3	20	7	17	11	
13	1	9	9	22	29	19	26	14	
	2	12	9	22	29	18	23	11	
	3	14	8	20	29	16	21	11	
	4	15	8	19	29	15	19	10	
	5	16	7	19	29	12	19	8	
	6	24	7	18	28	11	17	8	
	7	26	6	18	28	9	15	7	
	8	27	6	16	28	8	11	6	
	9	29	5	16	28	6	11	4	
14	1	3	17	15	8	7	27	8	
	2	4	17	14	6	5	27	8	
	3	7	17	14	6	5	27	7	
	4	14	16	14	6	4	27	8	
	5	15	16	13	4	3	27	8	
	6	24	16	13	4	3	26	8	
	7	26	16	13	3	2	26	8	
	8	27	15	12	3	1	26	8	
	9	30	15	12	2	1	26	8	
15	1	2	24	22	13	23	21	24	
	2	6	22	20	12	22	21	24	
	3	8	21	20	10	22	21	24	
	4	13	20	20	9	22	21	24	
	5	14	19	19	9	22	21	24	
	6	15	18	19	7	22	20	23	
	7	20	17	18	6	22	20	23	
	8	24	16	17	6	22	20	23	
	9	25	15	17	5	22	20	23	
16	1	1	21	5	7	11	8	19	
	2	6	20	4	7	11	6	17	
	3	14	20	4	6	11	6	16	
	4	16	18	4	5	11	5	15	
	5	20	15	4	5	11	4	14	
	6	23	14	4	5	11	4	13	
	7	26	13	4	4	11	4	12	
	8	28	12	4	4	11	2	10	
	9	29	11	4	3	11	2	10	
17	1	1	20	13	19	26	16	25	
	2	11	19	12	18	24	15	22	
	3	17	18	10	18	18	15	20	
	4	18	17	9	17	16	15	17	
	5	19	17	9	16	12	15	12	
	6	20	16	9	16	12	15	11	
	7	21	16	8	16	7	15	6	
	8	22	15	7	15	6	15	5	
	9	24	14	6	15	3	15	2	
18	1	1	14	4	11	17	21	13	
	2	3	13	4	10	17	20	13	
	3	7	10	4	9	17	19	11	
	4	9	10	4	8	17	19	11	
	5	14	7	3	6	17	18	8	
	6	17	5	3	6	17	17	7	
	7	18	5	2	4	17	17	6	
	8	22	3	2	4	17	17	5	
	9	24	1	2	3	17	16	3	
19	1	3	14	29	27	20	29	27	
	2	5	14	29	26	17	26	26	
	3	10	14	29	23	14	23	20	
	4	11	14	29	23	12	22	19	
	5	14	13	29	20	10	21	16	
	6	19	13	29	16	9	17	11	
	7	20	13	29	14	8	16	11	
	8	23	13	29	12	5	13	5	
	9	28	13	29	10	3	13	5	
20	1	9	18	13	29	6	27	22	
	2	11	16	10	26	6	26	22	
	3	13	16	10	26	5	23	18	
	4	18	14	9	23	4	21	16	
	5	19	14	6	21	4	19	14	
	6	20	13	6	19	3	18	12	
	7	21	12	5	18	3	16	10	
	8	24	10	4	17	2	15	7	
	9	25	9	3	14	2	13	6	
21	1	4	3	24	13	21	17	18	
	2	12	3	23	13	20	17	14	
	3	13	3	23	12	19	15	14	
	4	14	3	22	11	19	14	12	
	5	18	3	22	11	17	14	11	
	6	19	3	21	10	16	13	8	
	7	20	3	21	9	16	13	7	
	8	27	3	20	8	14	11	5	
	9	28	3	20	8	14	11	4	
22	1	1	6	8	25	18	23	16	
	2	4	6	7	22	17	21	13	
	3	5	5	7	21	17	19	12	
	4	8	5	7	17	16	16	11	
	5	10	5	6	16	14	16	9	
	6	11	4	5	11	13	13	8	
	7	15	3	4	8	12	13	6	
	8	21	3	3	6	11	11	5	
	9	26	3	3	4	11	9	4	
23	1	4	17	23	7	15	28	19	
	2	5	14	18	7	14	27	18	
	3	6	14	18	7	14	25	18	
	4	14	12	14	6	14	21	18	
	5	15	11	13	6	13	20	18	
	6	18	11	9	5	13	19	18	
	7	19	10	6	5	13	16	18	
	8	26	9	4	4	12	16	18	
	9	28	7	3	4	12	14	18	
24	1	4	13	11	10	14	18	12	
	2	5	13	11	10	13	15	9	
	3	17	13	9	10	13	13	9	
	4	18	13	7	10	13	11	7	
	5	20	12	6	10	11	10	5	
	6	23	12	5	9	11	6	4	
	7	28	12	5	9	11	4	3	
	8	29	12	2	9	9	2	3	
	9	30	12	2	9	9	2	2	
25	1	5	24	9	16	11	21	7	
	2	6	22	9	14	10	21	7	
	3	8	21	9	14	9	20	7	
	4	12	19	9	14	9	19	7	
	5	15	19	9	13	8	19	7	
	6	18	18	9	13	8	18	7	
	7	19	16	9	12	7	18	7	
	8	20	14	9	12	7	18	7	
	9	28	12	9	11	7	17	7	
26	1	1	16	12	17	12	26	5	
	2	2	16	11	17	11	23	4	
	3	8	15	10	15	10	22	4	
	4	14	14	10	14	10	20	4	
	5	15	14	9	11	10	18	4	
	6	17	14	7	10	9	17	4	
	7	18	13	6	8	9	16	4	
	8	19	12	6	6	8	14	4	
	9	27	12	5	6	8	12	4	
27	1	4	17	23	20	19	20	13	
	2	5	15	21	17	17	18	13	
	3	9	15	19	16	17	17	13	
	4	13	14	16	13	16	16	13	
	5	21	11	16	12	14	13	12	
	6	25	10	13	10	13	11	12	
	7	26	10	12	10	13	9	12	
	8	29	9	11	7	12	7	12	
	9	30	7	8	6	10	6	12	
28	1	3	23	20	26	13	17	20	
	2	7	22	17	24	11	16	19	
	3	8	21	17	20	11	16	18	
	4	9	21	16	20	10	16	15	
	5	12	21	13	14	9	16	14	
	6	16	20	12	10	7	16	13	
	7	17	19	12	7	7	16	12	
	8	21	19	10	4	5	16	11	
	9	23	19	9	3	5	16	10	
29	1	3	9	15	10	25	22	16	
	2	6	7	14	9	24	22	14	
	3	19	6	12	9	23	20	13	
	4	21	6	12	7	23	19	12	
	5	23	5	10	7	22	19	12	
	6	24	4	10	6	22	17	9	
	7	25	4	9	4	21	17	9	
	8	26	3	8	3	21	16	8	
	9	28	2	6	3	20	15	7	
30	1	9	5	6	21	30	10	29	
	2	10	5	4	20	28	10	27	
	3	11	5	4	20	27	10	25	
	4	13	4	4	20	25	9	24	
	5	16	4	3	20	25	9	21	
	6	18	4	3	19	23	9	21	
	7	19	3	2	19	21	8	17	
	8	21	3	1	19	20	8	16	
	9	30	3	1	19	20	8	15	
31	1	4	17	17	27	14	11	18	
	2	7	17	17	26	14	10	17	
	3	9	16	15	24	14	10	17	
	4	17	15	13	24	14	9	17	
	5	19	12	11	22	14	9	17	
	6	20	12	7	21	14	9	17	
	7	25	10	5	18	14	8	17	
	8	26	9	3	17	14	8	17	
	9	27	9	2	16	14	7	17	
32	1	2	25	19	20	26	11	20	
	2	3	24	18	20	23	11	20	
	3	7	23	17	18	20	11	20	
	4	8	22	14	15	20	11	20	
	5	9	20	14	14	16	11	20	
	6	10	20	11	11	14	11	20	
	7	21	19	10	10	10	11	20	
	8	22	18	8	7	6	11	20	
	9	29	17	8	6	2	11	20	
33	1	2	13	8	23	24	14	16	
	2	7	11	7	19	23	14	16	
	3	12	10	7	19	21	14	15	
	4	14	9	7	18	17	14	15	
	5	17	6	7	16	17	14	15	
	6	20	5	6	14	13	14	14	
	7	21	4	6	12	11	14	13	
	8	25	2	6	12	11	14	13	
	9	29	2	6	9	8	14	13	
34	1	2	11	26	28	15	25	16	
	2	3	11	24	28	15	22	14	
	3	11	10	21	27	15	21	14	
	4	14	10	21	26	14	20	12	
	5	15	9	18	24	14	19	10	
	6	16	9	18	23	14	18	9	
	7	18	9	16	21	14	15	7	
	8	28	8	14	21	13	15	4	
	9	29	8	13	19	13	14	4	
35	1	1	17	19	30	15	11	18	
	2	2	14	18	29	14	11	17	
	3	12	14	18	27	13	9	16	
	4	15	13	18	27	13	9	14	
	5	16	11	18	26	12	8	14	
	6	17	9	18	24	11	8	13	
	7	19	9	18	24	11	7	12	
	8	23	8	18	23	10	6	11	
	9	28	6	18	21	9	5	10	
36	1	4	27	18	29	17	20	22	
	2	6	26	17	28	16	16	18	
	3	9	25	16	28	16	16	15	
	4	11	25	15	27	16	13	14	
	5	13	24	15	26	15	12	11	
	6	14	24	14	26	14	10	9	
	7	15	24	14	25	14	10	7	
	8	16	23	13	25	13	6	4	
	9	17	23	12	24	13	6	3	
37	1	10	8	21	2	20	27	18	
	2	14	8	19	2	18	27	14	
	3	17	7	17	2	16	25	13	
	4	18	5	17	2	15	25	11	
	5	20	5	15	2	12	24	10	
	6	24	5	13	2	11	23	8	
	7	25	3	12	2	8	22	7	
	8	26	2	11	2	8	21	7	
	9	27	2	11	2	6	20	5	
38	1	2	27	24	17	24	12	13	
	2	4	21	23	14	20	12	11	
	3	6	19	21	13	18	11	10	
	4	12	15	21	11	17	11	9	
	5	14	13	18	9	16	10	9	
	6	19	10	18	7	11	9	8	
	7	21	10	16	5	10	9	8	
	8	22	5	15	2	9	8	6	
	9	29	3	14	2	6	8	6	
39	1	2	28	16	26	25	10	13	
	2	5	24	15	23	25	9	11	
	3	10	24	15	23	25	8	10	
	4	14	21	14	21	25	7	8	
	5	17	19	14	21	25	6	7	
	6	19	19	13	19	25	5	5	
	7	23	15	13	17	25	4	4	
	8	24	14	12	17	25	3	2	
	9	26	12	12	15	25	3	1	
40	1	1	29	21	16	28	14	20	
	2	5	27	21	16	26	12	19	
	3	12	27	21	16	24	11	19	
	4	13	24	21	16	22	10	18	
	5	19	23	21	15	21	10	16	
	6	20	22	20	15	20	9	16	
	7	24	19	20	14	17	8	15	
	8	27	19	20	14	15	6	13	
	9	29	17	20	14	14	6	13	
41	1	6	16	16	18	27	22	17	
	2	8	14	16	18	25	21	16	
	3	14	13	16	15	25	21	16	
	4	17	13	16	15	23	19	15	
	5	18	12	15	13	20	19	14	
	6	20	10	15	12	17	17	14	
	7	25	9	14	9	14	16	13	
	8	27	8	14	7	11	15	12	
	9	28	6	14	6	9	13	12	
42	1	22	25	9	23	5	23	22	
	2	23	25	9	23	4	23	20	
	3	24	25	9	20	4	23	19	
	4	25	25	9	15	3	23	18	
	5	26	25	9	15	3	23	17	
	6	27	24	9	12	2	23	17	
	7	28	24	9	8	2	23	15	
	8	29	24	9	6	2	23	15	
	9	30	24	9	4	1	23	14	
43	1	2	21	10	13	11	22	11	
	2	4	20	8	13	9	22	9	
	3	13	17	8	13	9	22	9	
	4	14	16	7	12	7	22	9	
	5	19	13	5	12	6	22	7	
	6	21	10	4	11	6	22	7	
	7	23	10	4	10	5	22	6	
	8	25	7	2	10	4	22	6	
	9	27	5	2	10	3	22	5	
44	1	1	25	18	23	13	28	21	
	2	4	25	16	21	13	26	20	
	3	7	25	13	16	13	21	19	
	4	8	25	13	15	13	19	19	
	5	10	25	11	13	13	17	17	
	6	14	25	9	8	13	13	17	
	7	26	25	7	6	13	10	16	
	8	29	25	5	5	13	8	15	
	9	30	25	1	1	13	7	15	
45	1	6	1	13	26	16	19	27	
	2	12	1	12	25	15	19	26	
	3	14	1	11	22	14	19	26	
	4	18	1	11	20	13	19	26	
	5	24	1	10	20	12	19	25	
	6	25	1	9	17	12	19	25	
	7	26	1	9	15	10	19	24	
	8	28	1	8	13	9	19	24	
	9	30	1	8	13	9	19	23	
46	1	3	22	16	18	23	24	22	
	2	9	21	16	16	20	22	21	
	3	14	20	15	16	20	20	18	
	4	15	18	13	14	17	16	16	
	5	19	17	12	14	17	13	14	
	6	27	16	11	12	15	11	13	
	7	28	15	10	11	14	9	9	
	8	29	13	9	11	13	5	8	
	9	30	12	9	10	12	5	6	
47	1	1	28	13	11	7	21	27	
	2	6	26	11	10	5	19	26	
	3	7	25	11	8	5	18	26	
	4	9	23	8	8	4	17	26	
	5	10	22	6	7	4	16	25	
	6	12	21	6	6	4	14	25	
	7	13	21	3	5	3	14	25	
	8	27	20	3	5	3	12	25	
	9	28	18	1	4	2	10	25	
48	1	1	28	17	28	3	27	26	
	2	4	24	17	27	3	26	26	
	3	11	22	16	22	3	25	26	
	4	14	17	15	19	3	23	26	
	5	24	15	14	15	3	22	26	
	6	25	11	12	15	3	22	26	
	7	26	8	12	10	3	21	26	
	8	27	5	10	7	3	20	26	
	9	28	5	10	4	3	19	26	
49	1	1	26	22	21	27	19	24	
	2	2	26	20	20	27	18	23	
	3	7	25	18	20	27	15	22	
	4	8	23	15	19	26	14	20	
	5	9	23	15	18	26	12	18	
	6	12	21	11	18	26	10	17	
	7	14	21	10	18	25	7	14	
	8	18	20	9	17	25	6	14	
	9	30	18	6	17	25	5	13	
50	1	13	23	17	28	7	29	22	
	2	14	23	13	27	7	24	20	
	3	16	21	13	26	7	23	19	
	4	17	18	11	25	7	21	17	
	5	18	16	9	25	7	18	17	
	6	22	13	7	24	7	16	15	
	7	23	12	6	24	7	11	14	
	8	29	9	5	23	7	11	13	
	9	30	7	2	23	7	9	13	
51	1	6	23	25	16	20	16	29	
	2	8	18	24	15	19	15	28	
	3	11	18	21	15	18	15	28	
	4	19	14	19	14	15	14	28	
	5	20	13	19	14	13	14	28	
	6	21	9	18	14	12	13	28	
	7	26	6	16	14	12	12	28	
	8	27	5	13	13	10	12	28	
	9	29	4	12	13	9	12	28	
52	1	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	R 3	R 4	N 1	N 2
	55	52	54	58	917	826

************************************************************************
