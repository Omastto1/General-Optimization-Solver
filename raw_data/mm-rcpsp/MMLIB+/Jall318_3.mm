jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 4 R
- nonrenewable              : 4 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	4		2 4 5 11 
2	9	4		10 9 7 3 
3	9	4		21 16 8 6 
4	9	3		16 9 6 
5	9	5		21 17 13 12 8 
6	9	5		19 17 15 14 12 
7	9	6		24 21 19 18 15 14 
8	9	5		24 19 18 15 14 
9	9	5		24 21 17 15 14 
10	9	5		24 18 17 15 14 
11	9	5		24 22 19 17 15 
12	9	5		28 26 24 23 18 
13	9	4		24 23 22 15 
14	9	5		28 26 23 22 20 
15	9	4		28 27 26 20 
16	9	4		28 27 26 20 
17	9	5		30 28 27 26 25 
18	9	2		22 20 
19	9	5		33 32 30 29 28 
20	9	4		31 30 29 25 
21	9	5		34 33 31 30 26 
22	9	3		35 27 25 
23	9	6		35 34 33 32 31 30 
24	9	6		38 36 33 32 31 29 
25	9	5		41 38 34 33 32 
26	9	4		38 37 36 29 
27	9	4		45 36 34 31 
28	9	3		45 34 31 
29	9	4		45 42 40 35 
30	9	3		42 38 36 
31	9	3		41 39 37 
32	9	3		45 39 37 
33	9	2		40 39 
34	9	3		43 42 39 
35	9	3		51 43 39 
36	9	2		44 41 
37	9	3		51 43 42 
38	9	1		39 
39	9	3		47 46 44 
40	9	3		51 49 43 
41	9	2		51 43 
42	9	3		50 49 46 
43	9	2		47 46 
44	9	2		50 49 
45	9	1		46 
46	9	1		48 
47	9	1		48 
48	9	1		52 
49	9	1		52 
50	9	1		52 
51	9	1		52 
52	1	0		
************************************************************************
REQUESTS/DURATIONS
jobnr.	mode	dur	R1	R2	R3	R4	N1	N2	N3	N4	
------------------------------------------------------------------------
1	1	0	0	0	0	0	0	0	0	0	
2	1	2	16	3	18	26	17	26	27	20	
	2	3	13	3	17	25	17	23	25	19	
	3	8	11	3	17	23	17	21	25	19	
	4	17	10	3	16	19	17	19	23	19	
	5	19	8	3	16	17	17	15	20	19	
	6	23	8	3	16	15	17	15	19	18	
	7	25	5	3	16	14	17	11	17	18	
	8	26	4	3	15	13	17	9	14	18	
	9	27	3	3	15	10	17	6	13	18	
3	1	6	25	16	30	12	28	11	28	16	
	2	10	23	14	28	11	26	9	26	15	
	3	11	21	13	27	11	26	8	26	14	
	4	14	21	13	26	11	25	8	24	14	
	5	17	17	11	23	11	25	7	24	12	
	6	18	17	11	23	11	24	6	21	12	
	7	21	14	10	21	11	24	6	21	11	
	8	28	12	9	21	11	23	4	19	9	
	9	29	11	7	20	11	23	4	18	8	
4	1	5	22	28	25	26	26	5	14	22	
	2	8	20	28	22	23	24	5	14	21	
	3	9	20	27	19	21	23	5	13	20	
	4	10	16	27	19	15	19	5	12	20	
	5	14	15	26	16	12	17	5	12	19	
	6	16	12	25	12	12	15	5	12	19	
	7	18	12	25	11	7	12	5	11	18	
	8	27	8	24	7	6	10	5	11	18	
	9	30	8	24	5	1	10	5	10	18	
5	1	3	22	18	29	4	11	3	14	17	
	2	4	18	18	28	3	11	3	13	15	
	3	9	15	18	27	3	10	3	12	15	
	4	14	15	18	25	3	9	3	12	14	
	5	15	12	18	25	2	6	3	9	12	
	6	16	10	18	24	2	5	3	8	12	
	7	17	8	18	23	2	4	3	8	11	
	8	22	7	18	22	2	4	3	7	10	
	9	29	5	18	22	2	2	3	5	8	
6	1	7	19	18	29	7	25	18	20	29	
	2	8	18	18	27	6	25	17	20	27	
	3	17	18	18	25	6	25	16	20	23	
	4	18	18	17	25	6	24	12	20	22	
	5	19	17	16	24	4	24	12	20	19	
	6	20	17	16	22	4	24	10	20	18	
	7	24	16	15	20	3	24	6	20	17	
	8	27	16	15	19	2	23	6	20	14	
	9	28	16	15	18	2	23	4	20	12	
7	1	7	6	29	25	19	22	24	27	19	
	2	8	6	28	25	16	19	22	25	16	
	3	11	6	27	22	15	19	20	24	16	
	4	16	6	25	19	12	18	18	23	14	
	5	17	6	24	19	10	16	17	22	13	
	6	18	6	23	17	9	14	17	21	13	
	7	24	6	22	14	5	12	13	20	12	
	8	28	6	21	13	4	9	11	19	11	
	9	30	6	21	11	2	9	11	18	9	
8	1	7	27	21	21	19	19	25	28	11	
	2	10	27	19	20	18	18	23	25	10	
	3	14	25	18	19	17	17	23	21	9	
	4	20	24	18	19	16	15	22	18	8	
	5	22	24	16	18	15	15	22	18	6	
	6	23	22	16	17	15	13	22	14	6	
	7	25	21	14	17	14	12	21	12	4	
	8	28	20	13	16	14	10	20	10	4	
	9	30	20	13	16	13	8	20	7	3	
9	1	12	22	26	19	11	17	15	17	18	
	2	13	20	25	19	10	16	14	17	18	
	3	14	16	25	16	9	14	13	17	17	
	4	17	15	24	12	9	12	12	16	17	
	5	22	13	21	11	7	10	10	15	16	
	6	24	9	20	9	7	8	7	15	16	
	7	26	8	19	6	7	5	7	14	15	
	8	27	6	18	5	5	4	4	14	15	
	9	28	4	18	2	5	2	4	14	14	
10	1	7	24	30	7	28	18	15	26	24	
	2	8	22	28	6	27	16	14	26	24	
	3	9	21	25	6	27	14	13	24	24	
	4	10	21	24	6	27	12	12	24	23	
	5	11	19	21	5	26	12	12	22	23	
	6	17	18	20	5	25	8	11	21	22	
	7	18	18	20	4	25	7	11	20	21	
	8	25	16	16	4	24	3	11	18	21	
	9	26	16	15	4	23	2	10	17	21	
11	1	6	22	23	24	14	27	24	21	22	
	2	9	20	21	23	13	23	22	20	21	
	3	10	20	21	21	10	23	21	20	20	
	4	15	18	19	19	10	20	19	19	20	
	5	22	15	17	15	9	18	19	17	19	
	6	23	13	15	13	7	14	18	17	18	
	7	24	9	15	11	5	13	16	15	17	
	8	25	7	12	10	3	9	13	14	17	
	9	26	5	11	8	3	7	12	14	16	
12	1	8	11	27	5	21	14	25	25	18	
	2	9	11	27	4	21	14	24	24	18	
	3	12	10	27	4	21	13	22	24	18	
	4	17	8	26	4	21	11	22	24	17	
	5	18	7	26	4	20	11	19	24	17	
	6	19	6	25	4	20	9	19	24	17	
	7	22	6	25	4	20	8	18	24	17	
	8	28	4	24	4	19	6	15	24	16	
	9	30	3	24	4	19	5	15	24	16	
13	1	1	26	29	2	25	21	10	13	16	
	2	5	26	28	2	22	19	10	11	15	
	3	15	26	28	2	21	18	10	11	14	
	4	17	25	28	2	21	17	10	10	14	
	5	20	25	27	2	19	15	10	10	11	
	6	21	24	27	2	17	14	10	9	10	
	7	22	23	27	2	16	14	10	8	9	
	8	24	23	27	2	15	13	10	7	8	
	9	26	23	27	2	12	12	10	7	6	
14	1	2	28	23	27	14	24	4	13	10	
	2	5	27	21	26	14	23	4	12	9	
	3	7	26	19	23	13	20	4	11	8	
	4	11	26	17	20	12	17	4	10	8	
	5	14	25	17	19	11	15	3	9	7	
	6	20	25	13	16	9	13	3	9	6	
	7	25	25	13	14	9	10	3	8	4	
	8	27	24	11	9	7	8	2	6	3	
	9	28	24	9	9	6	4	2	6	3	
15	1	2	20	22	21	26	22	9	27	14	
	2	3	19	20	21	25	20	8	24	14	
	3	10	17	19	21	22	20	8	24	12	
	4	11	16	18	21	21	19	8	22	12	
	5	12	16	17	21	19	16	8	19	10	
	6	20	16	17	21	17	15	8	19	9	
	7	21	15	16	21	16	15	8	15	8	
	8	25	14	14	21	13	14	8	13	8	
	9	29	13	13	21	12	13	8	13	7	
16	1	3	25	19	13	16	3	28	20	25	
	2	6	23	18	12	14	3	26	20	24	
	3	9	21	16	12	13	3	26	20	22	
	4	10	18	15	11	12	3	25	20	21	
	5	11	18	13	11	10	2	23	20	20	
	6	14	16	13	11	9	2	23	20	20	
	7	16	12	11	11	8	2	22	20	19	
	8	28	11	8	10	7	1	20	20	18	
	9	29	9	8	10	7	1	20	20	16	
17	1	1	18	18	18	25	12	7	20	16	
	2	2	16	17	18	24	11	6	16	15	
	3	5	12	17	16	22	11	6	15	15	
	4	9	11	16	14	22	10	6	14	15	
	5	11	10	14	14	21	10	5	12	15	
	6	23	8	13	12	18	10	5	8	15	
	7	27	6	13	9	18	9	5	8	15	
	8	28	2	12	9	16	9	4	5	15	
	9	30	2	11	7	16	8	4	4	15	
18	1	3	27	27	27	6	19	23	12	6	
	2	5	27	26	25	6	19	23	9	6	
	3	8	26	26	24	6	15	21	9	6	
	4	14	26	25	21	6	15	19	7	6	
	5	18	25	25	20	5	11	17	7	5	
	6	25	25	25	18	5	11	15	5	5	
	7	26	25	24	17	4	8	14	4	4	
	8	29	24	24	17	4	5	13	2	4	
	9	30	24	24	15	4	4	10	2	4	
19	1	3	21	27	10	27	22	17	24	28	
	2	5	21	26	9	24	22	15	23	23	
	3	9	18	25	9	24	20	13	23	20	
	4	14	17	23	7	19	20	11	23	18	
	5	18	15	21	7	16	18	8	23	15	
	6	23	14	20	5	14	18	7	23	13	
	7	24	13	20	5	14	17	4	23	12	
	8	27	10	18	4	10	16	2	23	10	
	9	29	9	17	3	7	15	2	23	6	
20	1	2	25	21	30	7	17	14	21	28	
	2	9	25	20	26	7	17	11	19	24	
	3	11	25	18	24	7	17	11	18	24	
	4	13	24	18	22	7	17	8	16	21	
	5	19	24	14	19	7	16	6	15	19	
	6	24	24	14	17	7	16	6	13	19	
	7	27	23	12	16	7	15	4	10	16	
	8	28	23	11	12	7	15	3	9	16	
	9	30	23	8	12	7	15	2	9	14	
21	1	1	27	8	6	27	15	27	21	29	
	2	5	27	8	6	25	13	27	20	26	
	3	6	27	7	6	25	13	26	18	26	
	4	10	27	7	6	24	11	23	17	24	
	5	13	27	5	6	24	11	21	17	24	
	6	15	26	5	6	24	10	21	15	23	
	7	18	26	4	6	23	9	18	14	21	
	8	19	26	4	6	23	8	18	13	20	
	9	22	26	3	6	22	7	17	13	19	
22	1	7	29	12	21	21	8	23	15	10	
	2	10	27	11	19	20	8	22	14	10	
	3	22	27	11	19	19	7	20	14	9	
	4	23	27	11	15	17	7	18	14	7	
	5	24	26	11	15	17	7	13	14	7	
	6	26	25	10	14	15	6	11	14	6	
	7	27	25	10	12	14	5	8	14	4	
	8	28	25	10	10	13	5	6	14	3	
	9	29	24	10	8	13	5	4	14	2	
23	1	3	14	12	19	11	9	15	27	28	
	2	8	13	12	16	10	8	12	26	28	
	3	11	13	12	16	10	8	11	26	28	
	4	12	13	12	15	10	8	9	25	28	
	5	13	13	11	13	10	6	7	25	28	
	6	15	13	11	11	10	6	6	25	28	
	7	19	13	11	11	10	6	5	25	28	
	8	23	13	10	10	10	4	2	24	28	
	9	29	13	10	8	10	4	2	24	28	
24	1	4	2	12	5	15	22	22	14	22	
	2	5	1	11	4	13	21	20	13	21	
	3	7	1	10	4	13	21	18	13	21	
	4	13	1	8	4	11	20	18	13	21	
	5	15	1	7	3	10	20	16	12	21	
	6	21	1	7	3	10	19	13	12	20	
	7	22	1	4	3	8	19	13	12	20	
	8	23	1	3	3	8	18	11	11	20	
	9	24	1	2	3	7	18	9	11	20	
25	1	1	24	21	19	24	28	21	13	26	
	2	3	21	18	18	24	27	21	12	25	
	3	12	20	18	17	23	25	20	12	25	
	4	15	19	15	17	22	22	19	11	24	
	5	16	19	13	16	21	21	18	10	24	
	6	17	17	9	16	19	19	18	9	24	
	7	19	16	7	16	18	18	18	8	24	
	8	20	14	3	15	17	16	17	7	23	
	9	24	14	3	15	17	15	16	6	23	
26	1	1	18	21	26	21	29	12	20	28	
	2	6	16	20	25	21	25	10	20	27	
	3	12	14	16	24	21	24	9	18	27	
	4	14	14	15	23	21	19	8	15	27	
	5	24	13	10	20	21	17	7	14	26	
	6	25	11	10	19	21	16	6	13	25	
	7	28	10	5	18	21	12	6	11	25	
	8	29	10	3	17	21	11	5	9	24	
	9	30	8	1	15	21	7	3	9	23	
27	1	5	20	13	18	18	19	28	10	22	
	2	19	19	13	18	17	18	26	10	22	
	3	20	17	13	18	16	18	26	10	21	
	4	21	17	13	18	15	18	24	10	19	
	5	22	16	13	18	14	17	24	9	19	
	6	25	15	12	17	14	17	22	9	18	
	7	26	15	12	17	13	17	21	9	17	
	8	27	14	12	17	12	17	21	9	15	
	9	28	13	12	17	12	17	19	9	14	
28	1	1	24	23	29	26	28	17	30	26	
	2	5	21	22	26	25	27	15	28	23	
	3	7	16	21	25	23	27	15	24	21	
	4	17	15	21	24	21	26	15	22	20	
	5	20	11	19	23	19	24	14	22	19	
	6	21	9	19	22	16	24	13	21	17	
	7	25	6	18	22	15	23	12	19	17	
	8	27	3	17	21	13	22	12	16	15	
	9	29	2	17	20	11	21	11	15	13	
29	1	1	27	28	13	2	27	23	27	20	
	2	2	24	22	12	2	27	21	26	19	
	3	3	23	19	12	2	27	18	26	16	
	4	6	22	19	11	2	27	15	25	15	
	5	7	20	13	10	2	26	14	25	13	
	6	9	19	11	10	2	26	11	25	12	
	7	19	19	7	9	2	26	11	24	9	
	8	21	17	5	9	2	26	9	24	7	
	9	22	17	3	9	2	26	6	23	6	
30	1	8	20	18	24	27	17	22	30	23	
	2	16	18	18	22	27	15	22	28	20	
	3	19	17	16	21	25	15	21	25	19	
	4	20	15	15	20	24	14	19	23	16	
	5	21	12	14	19	23	11	19	23	16	
	6	22	11	14	18	22	11	18	20	13	
	7	24	10	12	18	21	8	17	19	12	
	8	28	8	11	16	20	7	16	19	10	
	9	29	5	11	16	20	6	15	16	9	
31	1	2	23	16	25	14	16	18	26	23	
	2	7	21	14	25	12	15	16	23	20	
	3	10	18	14	21	11	15	15	22	18	
	4	18	18	14	20	10	14	15	21	16	
	5	19	15	13	15	9	14	13	18	15	
	6	21	15	12	13	8	14	12	16	14	
	7	23	14	11	11	6	14	10	15	11	
	8	29	11	10	8	5	13	10	12	11	
	9	30	10	10	7	4	13	9	11	8	
32	1	2	23	29	29	25	16	28	27	11	
	2	16	23	28	27	24	16	28	27	10	
	3	18	21	28	24	23	16	26	25	10	
	4	20	21	28	22	23	16	25	23	10	
	5	24	19	27	19	22	16	23	22	8	
	6	25	18	27	19	21	16	23	21	8	
	7	26	17	26	16	20	16	22	19	7	
	8	27	16	26	14	19	16	20	18	6	
	9	30	14	26	13	18	16	19	17	6	
33	1	1	13	18	8	11	25	27	28	27	
	2	3	12	17	8	10	24	27	28	25	
	3	4	12	14	6	9	24	27	26	22	
	4	5	11	11	6	7	23	27	25	19	
	5	6	11	9	5	6	23	26	24	16	
	6	8	11	8	4	5	22	26	22	15	
	7	9	10	5	3	5	21	26	22	14	
	8	11	10	5	1	3	21	26	21	10	
	9	16	10	3	1	3	20	26	20	8	
34	1	3	24	21	27	11	23	25	19	19	
	2	4	19	18	25	9	22	22	17	18	
	3	5	18	17	24	9	22	18	17	17	
	4	6	15	16	21	8	21	17	15	17	
	5	13	15	14	19	6	20	12	15	16	
	6	18	11	13	17	4	20	11	15	14	
	7	22	11	9	16	3	19	7	14	14	
	8	26	8	8	14	3	19	4	13	12	
	9	30	5	7	12	2	19	3	12	11	
35	1	10	17	24	24	21	29	28	26	18	
	2	16	17	24	24	21	28	22	24	16	
	3	21	16	22	23	20	26	20	22	15	
	4	22	15	19	21	20	26	18	17	15	
	5	26	12	18	20	19	25	15	15	12	
	6	27	11	17	17	18	23	14	14	11	
	7	28	11	15	16	17	22	10	8	11	
	8	29	9	15	15	17	21	10	7	9	
	9	30	8	13	13	16	21	5	2	9	
36	1	1	28	25	6	16	13	22	30	28	
	2	4	24	25	6	14	13	19	28	25	
	3	5	23	25	6	14	13	17	27	22	
	4	12	19	25	6	12	13	16	25	21	
	5	22	18	25	6	12	13	15	24	20	
	6	23	12	24	6	11	13	11	24	17	
	7	26	12	24	6	10	13	11	22	14	
	8	29	8	24	6	9	13	8	22	13	
	9	30	5	24	6	7	13	5	20	9	
37	1	5	6	10	18	15	14	29	8	24	
	2	8	5	10	17	14	12	27	8	24	
	3	10	4	9	16	14	12	27	8	21	
	4	18	3	9	12	14	11	25	8	20	
	5	19	3	9	12	14	11	25	7	18	
	6	25	2	8	9	14	10	24	7	16	
	7	26	2	8	7	14	9	24	7	16	
	8	27	2	7	5	14	9	23	7	12	
	9	29	1	7	2	14	8	22	7	12	
38	1	1	21	28	10	27	22	26	22	19	
	2	3	19	26	9	26	21	25	21	18	
	3	6	17	22	9	26	20	24	19	16	
	4	15	13	20	9	25	20	24	19	15	
	5	18	12	18	8	25	19	21	18	12	
	6	19	9	16	8	24	18	21	18	11	
	7	20	8	12	8	24	18	18	16	9	
	8	26	5	8	8	24	17	18	16	8	
	9	30	1	6	8	23	17	17	15	6	
39	1	13	25	5	12	26	26	4	7	29	
	2	16	25	5	10	25	26	4	7	28	
	3	19	22	5	9	23	25	4	6	28	
	4	20	22	5	8	22	24	4	6	28	
	5	21	19	5	8	20	23	4	5	28	
	6	22	19	5	6	20	23	4	4	27	
	7	24	18	5	6	18	21	4	4	27	
	8	25	16	5	4	16	21	4	3	27	
	9	28	15	5	3	14	20	4	3	27	
40	1	5	16	29	18	13	25	28	28	7	
	2	11	14	24	16	11	22	27	27	5	
	3	15	14	21	16	10	21	25	27	5	
	4	16	12	18	15	9	17	22	26	4	
	5	24	11	17	14	7	15	21	26	4	
	6	25	10	12	14	5	15	20	25	4	
	7	26	10	9	14	5	11	18	25	3	
	8	27	9	7	13	3	9	15	24	3	
	9	28	8	1	12	1	6	14	24	2	
41	1	8	27	18	21	12	21	19	9	21	
	2	9	25	16	21	12	17	16	8	21	
	3	14	24	16	21	12	14	16	7	17	
	4	16	23	14	20	11	12	15	7	16	
	5	19	22	14	20	10	12	14	6	14	
	6	20	20	12	19	10	8	13	6	9	
	7	21	19	11	18	9	6	12	6	9	
	8	22	19	11	18	9	4	11	5	5	
	9	23	18	10	18	9	2	9	5	4	
42	1	3	16	17	6	23	29	16	22	19	
	2	5	15	16	6	18	26	16	21	19	
	3	10	14	16	6	18	25	16	21	15	
	4	18	13	15	6	16	22	16	18	14	
	5	20	11	15	6	14	21	16	16	11	
	6	21	10	14	6	11	20	16	16	9	
	7	23	9	14	6	8	17	16	13	9	
	8	25	9	13	6	7	16	16	13	5	
	9	28	8	13	6	5	14	16	12	3	
43	1	1	9	25	22	27	14	4	22	23	
	2	5	9	21	18	27	12	4	21	23	
	3	9	8	18	15	25	11	4	21	22	
	4	10	6	16	15	21	10	4	20	19	
	5	13	6	12	11	20	9	4	20	19	
	6	14	5	11	11	19	7	4	20	17	
	7	16	5	7	8	16	5	4	19	17	
	8	20	3	4	7	14	3	4	19	14	
	9	30	3	4	4	11	2	4	18	13	
44	1	1	18	30	21	24	15	12	18	21	
	2	4	18	27	18	22	14	10	17	20	
	3	5	17	27	17	20	14	10	17	19	
	4	6	17	25	15	18	13	8	17	17	
	5	7	17	24	12	18	11	6	16	15	
	6	15	16	23	12	15	10	5	16	13	
	7	20	16	20	11	13	9	5	15	9	
	8	23	15	20	9	9	9	2	15	7	
	9	25	15	18	6	8	8	2	14	7	
45	1	1	28	23	10	20	20	14	12	25	
	2	2	26	22	10	19	20	13	11	23	
	3	4	22	20	10	18	20	12	11	20	
	4	5	20	20	10	18	20	12	11	20	
	5	9	20	19	10	16	20	12	11	17	
	6	11	15	17	9	15	19	11	11	15	
	7	13	14	17	9	14	19	11	11	11	
	8	21	12	16	9	13	19	10	11	10	
	9	22	11	14	9	11	19	10	11	7	
46	1	4	16	11	16	5	25	18	26	14	
	2	9	15	11	15	5	25	16	25	13	
	3	11	14	10	14	5	24	14	25	12	
	4	12	12	8	14	4	24	11	25	10	
	5	16	11	7	13	4	23	11	25	10	
	6	17	10	6	13	4	22	9	24	8	
	7	20	9	3	12	4	22	7	24	7	
	8	25	9	3	12	3	21	5	24	6	
	9	27	8	2	12	3	21	3	24	6	
47	1	1	20	20	18	23	16	15	2	23	
	2	3	19	19	17	22	16	14	2	22	
	3	5	19	17	15	20	15	12	2	20	
	4	8	19	16	12	19	14	11	2	16	
	5	12	19	12	11	18	14	7	2	15	
	6	20	19	9	9	18	14	6	2	12	
	7	22	19	7	6	17	13	6	2	10	
	8	28	19	4	5	16	13	2	2	8	
	9	29	19	3	1	15	12	1	2	8	
48	1	5	26	28	18	27	25	27	20	20	
	2	6	24	24	16	22	24	26	19	18	
	3	13	21	22	15	22	20	26	19	17	
	4	15	19	21	14	17	19	26	18	17	
	5	18	16	19	14	16	18	26	17	15	
	6	20	15	14	13	12	14	26	17	14	
	7	21	12	13	11	7	12	26	17	12	
	8	22	10	11	11	5	11	26	15	11	
	9	26	8	7	10	1	8	26	15	10	
49	1	1	17	17	6	10	28	25	20	25	
	2	5	17	16	5	9	26	23	16	20	
	3	6	16	16	5	9	24	22	16	20	
	4	11	15	16	5	9	22	21	14	16	
	5	15	14	15	4	9	22	20	12	15	
	6	18	13	15	3	9	21	18	10	13	
	7	20	13	14	2	9	18	18	10	11	
	8	27	11	13	1	9	18	16	9	9	
	9	28	11	13	1	9	17	16	6	6	
50	1	1	22	28	12	17	13	23	12	24	
	2	3	19	25	12	17	13	20	12	23	
	3	4	18	23	10	17	12	20	11	23	
	4	10	17	22	9	17	11	17	10	23	
	5	12	17	21	7	16	11	13	9	21	
	6	13	15	18	6	16	10	11	9	21	
	7	19	15	16	6	16	9	10	7	21	
	8	23	13	14	4	16	9	7	7	19	
	9	25	13	14	4	16	8	3	6	19	
51	1	1	10	9	7	15	3	20	22	17	
	2	2	10	7	6	14	3	19	21	15	
	3	6	9	7	6	13	3	18	20	13	
	4	7	9	6	6	12	3	17	19	13	
	5	8	7	6	5	11	3	15	18	11	
	6	17	7	5	5	10	3	14	15	10	
	7	20	7	5	5	9	3	11	15	10	
	8	26	6	5	5	7	3	11	12	8	
	9	29	5	4	5	7	3	9	12	8	
52	1	0	0	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	R 3	R 4	N 1	N 2	N 3	N 4
	55	61	51	56	881	825	917	912

************************************************************************
