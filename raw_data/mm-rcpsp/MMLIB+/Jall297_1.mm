jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 4 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	5		2 4 5 6 8 
2	9	2		11 3 
3	9	6		15 13 12 10 9 7 
4	9	5		15 14 13 11 9 
5	9	4		19 18 14 11 
6	9	3		19 13 11 
7	9	5		23 19 17 16 14 
8	9	4		23 16 14 12 
9	9	4		25 19 18 16 
10	9	3		21 17 16 
11	9	3		23 21 16 
12	9	3		21 20 17 
13	9	3		24 23 17 
14	9	5		28 25 24 22 21 
15	9	2		24 17 
16	9	3		30 22 20 
17	9	4		30 28 25 22 
18	9	3		27 24 23 
19	9	2		24 21 
20	9	4		31 28 27 24 
21	9	6		36 33 31 30 27 26 
22	9	6		38 36 33 31 27 26 
23	9	5		36 31 30 28 26 
24	9	4		38 36 33 26 
25	9	3		38 31 26 
26	9	4		37 34 32 29 
27	9	4		37 34 32 29 
28	9	4		43 41 35 32 
29	9	5		51 45 43 41 35 
30	9	5		43 41 40 39 38 
31	9	6		51 50 45 43 40 39 
32	9	4		51 50 45 40 
33	9	4		51 50 45 40 
34	9	3		50 40 39 
35	9	3		50 48 40 
36	9	2		45 40 
37	9	5		50 49 48 44 42 
38	9	5		50 49 47 46 45 
39	9	4		49 48 46 44 
40	9	3		49 44 42 
41	9	2		44 42 
42	9	2		47 46 
43	9	2		48 46 
44	9	1		47 
45	9	1		48 
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
2	1	1	1	4	3	1	14	9	
	2	4	1	3	3	1	12	9	
	3	6	1	3	3	1	11	8	
	4	13	1	3	3	1	10	8	
	5	15	1	3	2	1	9	7	
	6	16	1	2	2	1	8	6	
	7	22	1	2	2	1	7	6	
	8	23	1	2	2	1	6	5	
	9	28	1	2	2	1	6	4	
3	1	6	5	5	5	3	24	23	
	2	7	4	4	4	3	23	20	
	3	10	4	4	3	3	21	19	
	4	12	4	4	3	3	20	18	
	5	13	3	3	2	3	18	16	
	6	15	3	3	2	2	17	13	
	7	19	3	3	2	2	16	13	
	8	21	3	3	1	2	14	10	
	9	26	3	3	1	2	13	8	
4	1	4	4	4	2	2	17	23	
	2	10	3	4	2	1	15	23	
	3	15	3	4	2	1	14	22	
	4	16	3	4	2	1	13	22	
	5	17	3	4	1	1	12	21	
	6	18	3	4	1	1	11	21	
	7	23	3	4	1	1	10	21	
	8	25	3	4	1	1	10	20	
	9	30	3	4	1	1	9	20	
5	1	7	4	2	3	4	29	26	
	2	8	4	2	3	3	28	23	
	3	9	4	2	3	3	25	23	
	4	13	4	2	3	3	22	20	
	5	15	3	2	3	3	22	20	
	6	18	3	2	3	3	19	18	
	7	26	2	2	3	3	17	16	
	8	28	2	2	3	3	16	16	
	9	30	2	2	3	3	14	15	
6	1	2	4	1	3	4	10	21	
	2	5	4	1	2	3	8	21	
	3	8	4	1	2	3	8	20	
	4	9	4	1	2	3	7	21	
	5	11	4	1	2	2	7	21	
	6	16	3	1	2	2	6	21	
	7	17	3	1	2	2	6	20	
	8	21	3	1	2	2	5	21	
	9	22	3	1	2	2	4	21	
7	1	2	3	3	2	3	22	15	
	2	5	2	3	2	3	18	15	
	3	9	2	3	2	3	17	15	
	4	12	2	2	2	3	15	15	
	5	15	2	2	1	3	10	15	
	6	16	1	2	1	3	9	15	
	7	17	1	2	1	3	6	15	
	8	21	1	1	1	3	4	15	
	9	27	1	1	1	3	3	15	
8	1	3	2	3	5	4	18	18	
	2	8	2	2	4	3	18	17	
	3	9	2	2	4	3	16	17	
	4	10	2	2	4	3	16	16	
	5	12	2	1	3	2	14	15	
	6	13	1	1	3	2	14	15	
	7	25	1	1	3	1	12	14	
	8	26	1	1	2	1	11	13	
	9	28	1	1	2	1	11	12	
9	1	2	3	3	2	5	25	29	
	2	6	2	3	2	5	25	29	
	3	7	2	3	2	5	25	28	
	4	10	2	3	2	5	25	26	
	5	12	2	3	2	5	25	25	
	6	13	1	3	2	5	25	25	
	7	17	1	3	2	5	25	24	
	8	23	1	3	2	5	25	23	
	9	25	1	3	2	5	25	22	
10	1	10	3	4	4	1	1	27	
	2	11	3	4	4	1	1	25	
	3	13	3	4	4	1	1	24	
	4	14	3	4	4	1	1	23	
	5	17	3	4	3	1	1	23	
	6	20	3	4	3	1	1	22	
	7	26	3	4	3	1	1	21	
	8	27	3	4	3	1	1	20	
	9	30	3	4	3	1	1	19	
11	1	1	5	2	5	4	14	16	
	2	6	5	2	4	4	13	14	
	3	8	5	2	4	4	11	14	
	4	9	5	2	4	4	11	13	
	5	18	5	2	4	4	10	12	
	6	20	5	1	3	4	10	12	
	7	26	5	1	3	4	8	11	
	8	27	5	1	3	4	7	11	
	9	28	5	1	3	4	7	10	
12	1	2	4	5	3	4	23	11	
	2	4	4	4	3	4	23	10	
	3	8	4	4	3	4	19	10	
	4	14	4	4	3	4	19	9	
	5	21	3	4	2	4	16	9	
	6	23	3	4	2	3	14	8	
	7	25	3	4	2	3	12	7	
	8	29	3	4	2	3	7	6	
	9	30	3	4	2	3	5	6	
13	1	7	2	4	5	4	28	8	
	2	16	1	4	4	4	26	8	
	3	20	1	3	4	4	25	8	
	4	21	1	3	4	4	23	8	
	5	22	1	3	4	4	20	8	
	6	23	1	2	4	4	19	9	
	7	24	1	2	4	4	19	8	
	8	25	1	1	4	4	16	8	
	9	26	1	1	4	4	15	8	
14	1	3	3	5	2	5	27	23	
	2	5	3	4	1	4	27	20	
	3	10	3	4	1	4	27	19	
	4	12	3	4	1	4	27	18	
	5	16	3	4	1	4	27	16	
	6	19	2	3	1	3	27	16	
	7	24	2	3	1	3	27	15	
	8	26	2	3	1	3	27	14	
	9	29	2	3	1	3	27	12	
15	1	7	1	4	5	5	24	28	
	2	8	1	3	4	5	24	26	
	3	13	1	3	4	5	20	24	
	4	17	1	3	4	5	20	23	
	5	21	1	2	3	5	18	21	
	6	25	1	2	3	5	14	21	
	7	26	1	2	3	5	13	19	
	8	27	1	1	2	5	10	19	
	9	28	1	1	2	5	10	18	
16	1	2	4	1	5	4	17	28	
	2	7	4	1	4	4	16	27	
	3	12	4	1	4	4	16	26	
	4	15	4	1	4	4	15	26	
	5	16	4	1	4	3	15	24	
	6	22	4	1	3	3	14	23	
	7	24	4	1	3	3	13	22	
	8	26	4	1	3	2	13	20	
	9	28	4	1	3	2	13	19	
17	1	9	5	4	5	4	11	13	
	2	15	4	4	5	4	11	10	
	3	16	4	4	5	3	11	10	
	4	18	4	4	5	3	11	9	
	5	20	4	3	5	2	11	7	
	6	21	4	3	5	2	11	6	
	7	24	4	3	5	1	11	4	
	8	28	4	3	5	1	11	3	
	9	29	4	3	5	1	11	2	
18	1	1	4	4	5	2	26	25	
	2	4	4	4	5	2	24	25	
	3	5	4	3	5	2	21	21	
	4	8	4	3	5	2	20	17	
	5	17	3	3	5	2	19	14	
	6	18	3	2	5	2	16	11	
	7	20	3	2	5	2	14	10	
	8	22	3	1	5	2	13	5	
	9	23	3	1	5	2	10	3	
19	1	1	3	3	1	4	2	13	
	2	5	3	2	1	4	2	12	
	3	6	3	2	1	4	2	10	
	4	7	2	2	1	4	2	9	
	5	23	2	2	1	4	2	8	
	6	24	2	2	1	4	2	7	
	7	25	2	2	1	4	2	6	
	8	27	1	2	1	4	2	6	
	9	30	1	2	1	4	2	5	
20	1	8	4	2	1	5	25	22	
	2	9	4	1	1	4	23	22	
	3	10	4	1	1	4	22	22	
	4	16	4	1	1	3	18	22	
	5	17	3	1	1	3	15	22	
	6	18	3	1	1	3	12	22	
	7	22	3	1	1	3	7	22	
	8	26	3	1	1	2	4	22	
	9	28	3	1	1	2	3	22	
21	1	6	2	5	4	3	13	12	
	2	7	2	4	4	3	12	11	
	3	8	2	4	4	3	12	10	
	4	9	2	3	4	3	11	11	
	5	13	2	2	4	3	10	11	
	6	16	1	2	4	3	8	11	
	7	21	1	1	4	3	7	11	
	8	26	1	1	4	3	7	10	
	9	27	1	1	4	3	6	11	
22	1	7	4	4	3	4	22	5	
	2	8	4	3	3	4	20	4	
	3	14	4	3	3	4	19	4	
	4	15	4	3	3	4	18	3	
	5	16	4	3	2	4	18	3	
	6	17	4	3	2	3	16	3	
	7	22	4	3	2	3	15	2	
	8	23	4	3	2	3	14	2	
	9	28	4	3	2	3	14	1	
23	1	5	3	3	4	4	19	21	
	2	6	3	2	4	4	19	20	
	3	9	3	2	4	4	18	19	
	4	14	3	2	4	3	18	18	
	5	16	3	2	4	3	16	18	
	6	23	3	1	3	2	16	17	
	7	24	3	1	3	1	15	16	
	8	26	3	1	3	1	14	15	
	9	27	3	1	3	1	14	14	
24	1	7	5	3	5	2	24	9	
	2	8	5	3	4	2	23	9	
	3	10	5	3	4	2	21	9	
	4	11	5	3	4	2	20	9	
	5	16	5	3	4	2	20	8	
	6	17	5	3	4	1	20	8	
	7	18	5	3	4	1	19	8	
	8	28	5	3	4	1	18	7	
	9	29	5	3	4	1	17	7	
25	1	5	4	4	4	5	28	20	
	2	8	4	3	4	4	28	19	
	3	9	4	3	4	4	26	18	
	4	10	3	3	3	4	25	18	
	5	11	2	2	3	3	25	17	
	6	12	2	2	2	3	24	17	
	7	17	1	2	1	2	23	16	
	8	18	1	2	1	2	21	16	
	9	23	1	2	1	2	21	15	
26	1	3	1	3	3	3	24	20	
	2	7	1	3	2	3	24	19	
	3	11	1	3	2	3	23	19	
	4	14	1	3	2	3	23	18	
	5	15	1	2	2	3	22	18	
	6	21	1	2	2	3	22	17	
	7	23	1	2	2	3	22	16	
	8	25	1	2	2	3	21	18	
	9	29	1	2	2	3	21	17	
27	1	2	4	4	5	2	22	8	
	2	3	4	4	4	1	20	7	
	3	8	4	4	4	1	18	6	
	4	11	4	4	4	1	16	6	
	5	14	4	4	3	1	15	5	
	6	15	4	4	3	1	14	5	
	7	17	4	4	3	1	11	4	
	8	20	4	4	3	1	11	3	
	9	30	4	4	3	1	8	3	
28	1	2	1	4	5	4	25	23	
	2	3	1	4	4	4	25	23	
	3	4	1	3	4	3	25	22	
	4	6	1	3	4	3	24	22	
	5	11	1	2	3	2	23	21	
	6	12	1	2	3	2	23	20	
	7	18	1	2	3	2	22	20	
	8	22	1	1	2	1	22	19	
	9	27	1	1	2	1	22	18	
29	1	2	4	4	5	2	24	30	
	2	7	4	3	4	2	23	28	
	3	16	4	3	4	2	23	25	
	4	20	4	2	4	2	21	23	
	5	21	4	2	3	2	21	22	
	6	24	3	2	3	2	20	21	
	7	25	3	1	3	2	18	19	
	8	28	3	1	3	2	18	16	
	9	30	3	1	3	2	17	14	
30	1	4	4	5	4	5	29	11	
	2	12	3	4	3	4	28	11	
	3	14	3	4	3	4	28	10	
	4	18	3	4	3	4	28	9	
	5	21	3	4	2	3	28	9	
	6	22	3	3	2	3	28	8	
	7	23	3	3	1	3	28	7	
	8	25	3	3	1	3	28	6	
	9	26	3	3	1	3	28	5	
31	1	1	3	3	2	2	14	23	
	2	4	2	3	2	2	14	21	
	3	5	2	3	2	2	14	20	
	4	10	2	3	2	2	14	18	
	5	19	2	3	2	2	14	15	
	6	21	1	3	1	1	14	15	
	7	23	1	3	1	1	14	11	
	8	25	1	3	1	1	14	9	
	9	28	1	3	1	1	14	7	
32	1	3	3	5	5	2	24	14	
	2	6	3	4	4	2	23	14	
	3	9	3	3	4	2	22	13	
	4	14	3	3	3	2	21	13	
	5	21	3	3	2	2	20	12	
	6	25	3	2	2	2	20	12	
	7	26	3	1	2	2	19	11	
	8	28	3	1	1	2	19	11	
	9	29	3	1	1	2	18	10	
33	1	4	1	4	1	3	25	22	
	2	11	1	3	1	3	22	22	
	3	12	1	3	1	3	21	20	
	4	13	1	2	1	3	17	17	
	5	14	1	2	1	2	17	16	
	6	15	1	2	1	2	13	15	
	7	16	1	1	1	1	11	14	
	8	22	1	1	1	1	9	12	
	9	24	1	1	1	1	9	10	
34	1	7	3	2	5	4	20	24	
	2	10	3	2	4	4	19	22	
	3	11	3	2	4	4	16	22	
	4	12	3	2	4	4	15	20	
	5	13	3	1	4	4	11	20	
	6	24	3	1	4	4	10	19	
	7	26	3	1	4	4	7	17	
	8	27	3	1	4	4	5	17	
	9	28	3	1	4	4	5	15	
35	1	1	4	2	2	4	21	27	
	2	4	4	2	2	4	17	26	
	3	6	4	2	2	4	17	25	
	4	8	4	2	2	4	15	24	
	5	10	3	2	1	4	13	23	
	6	17	3	2	1	3	10	23	
	7	22	3	2	1	3	7	22	
	8	24	3	2	1	3	6	21	
	9	29	3	2	1	3	4	19	
36	1	3	4	3	5	1	16	26	
	2	9	4	3	4	1	14	25	
	3	15	4	3	4	1	12	25	
	4	17	4	3	3	1	10	25	
	5	18	4	3	3	1	9	25	
	6	19	4	3	3	1	9	24	
	7	20	4	3	2	1	7	24	
	8	22	4	3	2	1	3	24	
	9	25	4	3	2	1	2	24	
37	1	1	4	4	2	2	26	15	
	2	7	4	3	1	2	26	15	
	3	9	4	3	1	2	25	14	
	4	20	4	3	1	2	24	12	
	5	21	4	3	1	2	22	11	
	6	23	4	3	1	2	21	11	
	7	24	4	3	1	2	21	10	
	8	26	4	3	1	2	19	9	
	9	27	4	3	1	2	18	8	
38	1	1	5	3	3	3	26	23	
	2	3	4	3	2	3	23	22	
	3	6	4	3	2	3	23	21	
	4	9	4	3	2	3	19	21	
	5	16	4	3	1	2	16	20	
	6	22	4	3	1	2	11	20	
	7	23	4	3	1	2	8	19	
	8	29	4	3	1	2	5	18	
	9	30	4	3	1	2	4	17	
39	1	1	2	4	4	4	26	29	
	2	3	2	3	3	4	23	26	
	3	5	2	3	3	4	22	21	
	4	12	2	3	3	4	18	17	
	5	13	2	3	3	3	14	17	
	6	15	2	2	2	3	14	14	
	7	17	2	2	2	2	10	8	
	8	24	2	2	2	2	7	7	
	9	26	2	2	2	2	6	5	
40	1	4	4	3	5	5	21	8	
	2	6	4	2	4	4	19	8	
	3	7	4	2	4	4	17	7	
	4	8	4	2	4	4	14	6	
	5	9	3	1	4	4	13	5	
	6	21	3	1	4	3	8	5	
	7	23	3	1	4	3	8	4	
	8	25	3	1	4	3	5	3	
	9	29	3	1	4	3	4	3	
41	1	4	5	4	2	3	4	29	
	2	10	4	3	1	3	3	29	
	3	11	4	3	1	3	3	28	
	4	12	3	3	1	3	3	29	
	5	14	3	3	1	2	2	28	
	6	15	3	2	1	2	2	28	
	7	16	3	2	1	1	1	27	
	8	17	2	2	1	1	1	27	
	9	22	2	2	1	1	1	26	
42	1	2	4	4	3	5	22	16	
	2	3	4	4	3	4	20	15	
	3	4	3	4	3	4	19	15	
	4	18	3	4	3	4	18	14	
	5	19	3	4	3	4	16	14	
	6	20	2	4	3	4	15	13	
	7	21	2	4	3	4	14	13	
	8	23	1	4	3	4	13	11	
	9	28	1	4	3	4	11	11	
43	1	2	2	5	2	4	20	19	
	2	5	2	4	2	3	17	18	
	3	7	2	4	2	3	16	16	
	4	8	2	3	2	2	14	16	
	5	14	2	2	2	2	13	14	
	6	15	2	2	2	2	10	13	
	7	21	2	1	2	2	9	13	
	8	23	2	1	2	1	8	12	
	9	24	2	1	2	1	6	10	
44	1	3	5	4	5	4	24	24	
	2	4	4	3	4	3	24	21	
	3	5	4	3	4	3	24	19	
	4	6	4	3	4	3	24	16	
	5	8	4	3	4	3	24	15	
	6	10	4	3	4	2	24	13	
	7	21	4	3	4	2	24	12	
	8	27	4	3	4	2	24	10	
	9	28	4	3	4	2	24	7	
45	1	4	3	2	5	5	15	12	
	2	9	3	1	4	4	14	12	
	3	10	3	1	4	4	12	11	
	4	22	3	1	4	4	10	11	
	5	24	3	1	4	4	9	9	
	6	25	3	1	4	4	7	9	
	7	26	3	1	4	4	6	9	
	8	28	3	1	4	4	4	7	
	9	30	3	1	4	4	4	6	
46	1	10	4	3	5	4	29	30	
	2	19	4	2	4	4	26	29	
	3	20	4	2	4	4	22	29	
	4	21	4	2	4	3	19	28	
	5	22	4	2	4	3	16	27	
	6	23	4	1	3	3	13	27	
	7	24	4	1	3	2	12	26	
	8	25	4	1	3	2	8	26	
	9	26	4	1	3	2	6	26	
47	1	1	4	5	3	5	28	23	
	2	2	4	4	3	5	25	22	
	3	3	4	4	3	5	25	20	
	4	4	4	3	3	5	24	18	
	5	5	3	3	3	5	22	17	
	6	6	3	2	3	5	22	17	
	7	21	2	1	3	5	20	15	
	8	25	2	1	3	5	20	14	
	9	26	2	1	3	5	19	11	
48	1	2	2	4	3	4	24	6	
	2	4	2	4	2	4	22	6	
	3	6	2	4	2	4	21	5	
	4	8	2	4	2	3	20	5	
	5	9	2	4	2	2	19	5	
	6	10	2	3	2	2	17	4	
	7	11	2	3	2	1	16	3	
	8	23	2	3	2	1	13	3	
	9	30	2	3	2	1	12	3	
49	1	3	4	3	3	4	24	17	
	2	10	3	2	2	3	24	14	
	3	11	3	2	2	3	21	14	
	4	16	3	2	2	3	20	11	
	5	17	3	2	2	2	20	8	
	6	19	2	2	1	2	18	6	
	7	24	2	2	1	2	16	4	
	8	27	2	2	1	1	14	3	
	9	28	2	2	1	1	13	1	
50	1	5	5	4	5	2	6	23	
	2	8	4	4	4	2	5	23	
	3	9	4	4	4	2	5	22	
	4	10	4	4	4	2	4	21	
	5	11	4	4	3	2	3	21	
	6	12	3	3	3	2	3	20	
	7	14	3	3	2	2	3	19	
	8	16	3	3	2	2	2	18	
	9	30	3	3	2	2	2	17	
51	1	1	3	4	2	3	27	10	
	2	3	2	4	1	3	26	9	
	3	4	2	4	1	3	20	9	
	4	5	2	4	1	3	20	8	
	5	6	2	3	1	2	17	7	
	6	15	1	3	1	2	10	5	
	7	23	1	3	1	2	9	4	
	8	29	1	3	1	2	7	3	
	9	30	1	3	1	2	1	2	
52	1	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	R 3	R 4	N 1	N 2
	17	20	20	20	906	867

************************************************************************
