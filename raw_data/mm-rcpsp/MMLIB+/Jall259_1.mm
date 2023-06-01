jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 4 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	8		2 3 4 8 9 11 14 15 
2	9	4		16 10 7 5 
3	9	7		22 21 19 17 16 13 10 
4	9	4		39 27 22 6 
5	9	4		34 20 19 12 
6	9	4		28 21 20 13 
7	9	4		27 21 20 13 
8	9	4		51 34 19 12 
9	9	4		39 28 26 13 
10	9	10		50 38 34 32 28 27 25 24 23 20 
11	9	2		51 12 
12	9	10		50 49 39 38 30 29 28 26 25 21 
13	9	8		51 50 49 38 31 25 24 18 
14	9	8		51 50 49 38 34 26 23 21 
15	9	7		50 34 32 28 24 23 20 
16	9	6		50 38 28 24 23 20 
17	9	9		49 48 46 37 33 31 30 29 24 
18	9	5		48 35 34 32 23 
19	9	4		49 47 26 23 
20	9	8		51 49 47 46 35 33 31 26 
21	9	6		47 46 37 32 31 24 
22	9	8		49 46 44 37 35 31 30 29 
23	9	8		46 44 43 37 36 33 30 29 
24	9	5		44 43 41 36 35 
25	9	4		48 43 35 33 
26	9	6		48 44 43 42 41 40 
27	9	5		47 44 41 40 35 
28	9	4		43 41 40 35 
29	9	4		45 42 41 40 
30	9	3		42 41 40 
31	9	2		45 36 
32	9	1		33 
33	9	2		41 40 
34	9	2		41 40 
35	9	1		42 
36	9	1		40 
37	9	1		40 
38	9	1		47 
39	9	1		47 
40	9	1		52 
41	9	1		52 
42	9	1		52 
43	9	1		52 
44	9	1		52 
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
jobnr.	mode	dur	R1	R2	N1	N2	N3	N4	
------------------------------------------------------------------------
1	1	0	0	0	0	0	0	0	
2	1	3	3	5	22	11	29	28	
	2	4	2	4	20	9	28	23	
	3	10	2	4	18	8	26	22	
	4	11	2	3	18	8	26	18	
	5	12	2	3	17	7	24	17	
	6	17	2	3	15	6	24	15	
	7	20	2	3	15	6	22	11	
	8	28	2	2	14	4	21	9	
	9	29	2	2	12	4	21	9	
3	1	2	5	1	10	16	5	27	
	2	3	4	1	9	12	5	23	
	3	4	4	1	8	11	5	19	
	4	7	3	1	7	9	5	15	
	5	10	3	1	7	8	5	14	
	6	19	2	1	6	7	5	13	
	7	23	1	1	6	4	5	9	
	8	24	1	1	5	2	5	7	
	9	29	1	1	5	2	5	2	
4	1	1	4	2	18	17	24	6	
	2	2	3	2	17	14	22	6	
	3	3	3	2	15	14	21	6	
	4	4	3	2	15	12	19	6	
	5	5	2	1	14	11	19	5	
	6	10	2	1	12	10	17	5	
	7	14	2	1	12	9	15	5	
	8	26	2	1	10	6	14	5	
	9	30	2	1	10	6	13	5	
5	1	1	3	2	28	4	27	9	
	2	3	3	2	24	3	25	9	
	3	7	3	2	20	3	22	9	
	4	13	3	2	20	3	19	9	
	5	16	3	2	16	3	19	9	
	6	20	2	2	12	3	17	9	
	7	21	2	2	12	3	15	9	
	8	27	2	2	10	3	13	9	
	9	29	2	2	5	3	12	9	
6	1	2	4	3	15	10	4	20	
	2	3	4	3	11	9	4	20	
	3	6	4	3	11	8	4	16	
	4	18	4	3	8	6	4	15	
	5	19	4	2	8	5	3	14	
	6	20	3	2	6	4	3	11	
	7	23	3	2	4	4	3	9	
	8	25	3	1	4	2	3	8	
	9	27	3	1	2	2	3	7	
7	1	1	5	4	7	21	12	5	
	2	2	4	3	7	21	12	5	
	3	10	4	3	7	20	11	5	
	4	11	3	3	7	19	10	5	
	5	14	3	2	7	19	10	4	
	6	19	2	2	7	19	9	4	
	7	23	2	2	7	18	9	4	
	8	26	1	1	7	18	8	4	
	9	27	1	1	7	17	7	4	
8	1	3	2	3	28	24	25	21	
	2	10	2	2	28	22	24	18	
	3	13	2	2	28	22	21	17	
	4	15	2	2	28	21	16	17	
	5	16	2	2	28	21	14	15	
	6	17	1	1	28	20	12	15	
	7	18	1	1	28	20	9	13	
	8	19	1	1	28	20	6	13	
	9	20	1	1	28	19	5	11	
9	1	1	3	3	29	14	20	27	
	2	8	3	3	27	14	19	23	
	3	9	3	3	27	13	19	20	
	4	10	3	3	27	12	19	19	
	5	12	2	3	25	11	18	17	
	6	15	2	2	25	10	18	14	
	7	19	2	2	24	9	18	12	
	8	27	2	2	23	8	18	9	
	9	28	2	2	23	6	18	8	
10	1	5	2	5	22	15	11	26	
	2	7	2	4	20	12	11	24	
	3	8	2	4	18	11	11	21	
	4	9	2	4	14	11	10	18	
	5	10	2	4	14	10	10	15	
	6	11	1	4	12	8	10	12	
	7	13	1	4	9	7	10	12	
	8	18	1	4	6	6	9	7	
	9	24	1	4	5	6	9	5	
11	1	3	4	4	8	14	28	27	
	2	4	3	3	8	14	27	27	
	3	5	3	3	8	14	26	27	
	4	15	2	3	8	14	26	27	
	5	18	2	2	8	14	25	27	
	6	20	2	2	8	14	23	27	
	7	25	2	1	8	14	23	27	
	8	28	1	1	8	14	22	27	
	9	29	1	1	8	14	20	27	
12	1	4	4	3	12	27	21	7	
	2	5	4	2	11	23	21	7	
	3	8	4	2	10	21	20	7	
	4	15	3	2	9	19	19	7	
	5	16	3	2	9	16	17	7	
	6	17	2	1	8	15	16	7	
	7	25	1	1	8	12	16	7	
	8	26	1	1	6	9	15	7	
	9	28	1	1	6	6	13	7	
13	1	4	5	5	22	25	28	20	
	2	5	4	4	20	23	28	18	
	3	10	4	4	18	23	24	15	
	4	12	4	3	16	21	22	15	
	5	14	3	2	13	18	22	13	
	6	15	3	2	11	15	20	11	
	7	16	3	2	11	14	17	7	
	8	17	3	1	7	9	14	7	
	9	25	3	1	5	9	14	5	
14	1	2	4	2	3	13	8	5	
	2	9	4	2	3	13	8	4	
	3	12	4	2	3	11	8	5	
	4	13	4	2	2	11	8	5	
	5	18	3	2	2	10	8	4	
	6	23	3	2	2	10	7	4	
	7	24	3	2	1	8	7	4	
	8	27	3	2	1	8	7	3	
	9	28	3	2	1	7	7	4	
15	1	5	4	1	29	25	23	16	
	2	6	4	1	28	23	21	15	
	3	9	4	1	28	23	19	13	
	4	10	4	1	27	22	14	11	
	5	13	4	1	27	21	13	11	
	6	14	4	1	27	20	10	8	
	7	15	4	1	26	20	9	8	
	8	17	4	1	25	18	4	6	
	9	21	4	1	25	18	1	4	
16	1	2	4	2	28	28	16	25	
	2	3	4	2	26	28	16	25	
	3	18	4	2	24	28	15	25	
	4	21	4	2	22	27	13	25	
	5	22	4	2	21	27	12	25	
	6	23	4	2	18	26	11	25	
	7	24	4	2	15	26	10	25	
	8	27	4	2	14	25	9	25	
	9	28	4	2	13	25	8	25	
17	1	2	4	5	25	28	11	20	
	2	8	4	4	21	28	9	18	
	3	16	4	4	17	27	9	16	
	4	20	4	4	16	27	8	13	
	5	21	4	3	12	25	7	11	
	6	23	4	3	12	25	6	11	
	7	25	4	3	7	24	4	8	
	8	29	4	3	5	24	3	7	
	9	30	4	3	3	23	2	6	
18	1	2	4	1	15	28	15	29	
	2	3	4	1	15	25	14	28	
	3	9	4	1	15	22	12	28	
	4	13	4	1	15	19	9	27	
	5	16	4	1	15	16	9	26	
	6	17	3	1	15	14	7	26	
	7	22	3	1	15	10	4	25	
	8	23	3	1	15	6	4	23	
	9	25	3	1	15	5	2	23	
19	1	1	5	3	19	28	28	28	
	2	4	4	3	18	24	25	24	
	3	5	4	3	18	22	24	24	
	4	7	4	3	18	21	20	21	
	5	10	4	3	17	20	18	18	
	6	14	4	2	17	17	16	17	
	7	15	4	2	17	16	14	15	
	8	17	4	2	17	13	10	10	
	9	18	4	2	17	12	9	10	
20	1	7	4	3	9	19	12	24	
	2	13	4	3	8	18	12	23	
	3	17	4	3	7	15	10	23	
	4	19	4	3	7	15	10	22	
	5	22	3	3	6	12	7	20	
	6	23	3	3	5	11	7	19	
	7	24	3	3	5	9	5	18	
	8	25	3	3	4	9	4	15	
	9	26	3	3	4	8	4	15	
21	1	4	3	1	6	17	24	16	
	2	5	3	1	5	14	22	14	
	3	7	3	1	5	12	21	13	
	4	15	3	1	4	10	18	12	
	5	16	3	1	3	10	18	11	
	6	17	3	1	3	7	16	9	
	7	18	3	1	2	6	16	9	
	8	19	3	1	2	3	13	8	
	9	27	3	1	2	3	12	6	
22	1	1	3	3	14	20	28	7	
	2	2	2	3	12	18	27	6	
	3	6	2	3	12	17	27	6	
	4	8	2	3	11	16	25	6	
	5	20	2	3	9	11	23	5	
	6	21	1	3	9	10	22	4	
	7	23	1	3	8	8	22	4	
	8	28	1	3	6	4	20	2	
	9	30	1	3	6	4	20	1	
23	1	1	4	3	12	6	3	28	
	2	2	3	2	12	6	3	23	
	3	11	3	2	11	6	3	22	
	4	15	3	2	10	5	3	18	
	5	16	3	1	10	5	3	16	
	6	18	3	1	9	5	3	12	
	7	19	3	1	9	4	3	10	
	8	20	3	1	7	4	3	7	
	9	30	3	1	7	4	3	1	
24	1	2	3	4	19	11	24	18	
	2	5	3	3	17	10	23	16	
	3	6	3	3	16	9	20	15	
	4	7	3	3	13	9	19	15	
	5	15	3	3	11	8	17	14	
	6	16	3	2	8	8	17	13	
	7	17	3	2	8	6	15	12	
	8	24	3	2	4	6	11	11	
	9	25	3	2	4	5	10	10	
25	1	1	3	1	11	29	24	13	
	2	5	3	1	9	27	23	12	
	3	6	3	1	9	25	19	11	
	4	7	3	1	8	24	18	8	
	5	8	3	1	8	23	14	8	
	6	11	3	1	8	22	12	7	
	7	19	3	1	7	20	10	6	
	8	26	3	1	6	19	6	3	
	9	30	3	1	6	17	4	2	
26	1	2	5	4	24	21	21	11	
	2	5	5	3	24	20	20	11	
	3	6	5	3	24	20	20	10	
	4	9	5	3	24	20	20	9	
	5	10	5	2	24	19	18	12	
	6	11	5	2	24	19	18	11	
	7	12	5	2	24	18	18	11	
	8	17	5	1	24	18	16	11	
	9	22	5	1	24	18	16	10	
27	1	3	3	2	25	30	27	8	
	2	4	2	1	21	28	23	8	
	3	7	2	1	18	25	21	8	
	4	9	2	1	14	23	18	8	
	5	13	2	1	14	21	14	8	
	6	17	2	1	10	20	11	8	
	7	18	2	1	8	17	9	8	
	8	19	2	1	7	14	5	8	
	9	30	2	1	4	12	1	8	
28	1	11	5	5	11	14	24	16	
	2	12	4	5	11	13	21	15	
	3	14	4	5	9	11	20	15	
	4	16	4	5	9	10	19	14	
	5	19	3	5	7	10	18	13	
	6	25	3	5	6	9	18	11	
	7	26	3	5	6	7	17	10	
	8	27	2	5	4	7	16	9	
	9	28	2	5	3	5	15	9	
29	1	5	1	2	21	14	12	12	
	2	7	1	2	20	14	11	11	
	3	12	1	2	20	12	10	9	
	4	13	1	2	20	11	10	8	
	5	19	1	2	20	11	8	8	
	6	20	1	2	20	9	8	7	
	7	23	1	2	20	8	7	6	
	8	27	1	2	20	7	7	5	
	9	29	1	2	20	6	6	4	
30	1	3	4	4	15	16	7	5	
	2	4	3	4	15	16	6	4	
	3	10	3	4	13	15	6	4	
	4	14	3	3	13	13	5	4	
	5	15	3	2	12	11	5	4	
	6	25	3	2	11	10	4	4	
	7	26	3	1	11	10	4	4	
	8	29	3	1	9	8	4	4	
	9	30	3	1	9	7	3	4	
31	1	2	5	3	30	13	28	25	
	2	10	4	3	29	12	25	23	
	3	11	4	3	28	11	23	22	
	4	12	4	3	27	11	21	22	
	5	15	3	3	27	10	19	20	
	6	16	3	3	26	10	18	19	
	7	23	3	3	25	9	17	19	
	8	24	3	3	24	9	13	17	
	9	27	3	3	24	9	13	16	
32	1	1	1	3	25	29	11	15	
	2	3	1	2	23	24	11	13	
	3	4	1	2	21	23	11	11	
	4	12	1	2	20	20	11	10	
	5	17	1	2	19	18	11	7	
	6	18	1	2	18	13	11	7	
	7	19	1	2	16	10	11	4	
	8	22	1	2	15	7	11	3	
	9	23	1	2	14	5	11	1	
33	1	1	3	4	29	14	26	20	
	2	3	2	4	28	13	24	20	
	3	12	2	4	28	12	23	20	
	4	13	2	4	28	11	17	19	
	5	14	2	4	27	8	13	19	
	6	15	2	3	27	7	10	19	
	7	21	2	3	27	6	7	18	
	8	27	2	3	27	2	5	18	
	9	28	2	3	27	1	4	18	
34	1	2	3	2	5	29	21	26	
	2	5	3	1	5	29	20	25	
	3	6	3	1	5	27	18	23	
	4	9	3	1	5	26	17	21	
	5	10	3	1	5	26	17	17	
	6	12	3	1	5	25	16	15	
	7	18	3	1	5	23	15	15	
	8	19	3	1	5	23	14	10	
	9	23	3	1	5	22	14	8	
35	1	2	3	4	23	16	29	19	
	2	6	3	4	20	14	28	17	
	3	13	3	4	19	14	28	17	
	4	17	3	4	18	13	27	16	
	5	19	3	3	16	13	26	14	
	6	20	3	3	13	12	26	13	
	7	21	3	3	12	11	24	13	
	8	22	3	3	10	10	23	11	
	9	29	3	3	7	10	23	11	
36	1	4	2	5	28	22	30	22	
	2	7	2	4	26	21	29	20	
	3	12	2	3	25	21	28	20	
	4	13	2	3	23	20	28	17	
	5	15	2	2	20	20	27	17	
	6	21	2	2	16	20	27	14	
	7	22	2	2	15	20	26	12	
	8	24	2	1	12	19	26	11	
	9	27	2	1	9	19	25	10	
37	1	3	1	4	21	17	21	21	
	2	7	1	4	20	15	20	18	
	3	8	1	4	20	14	16	15	
	4	9	1	4	19	12	14	13	
	5	16	1	3	17	11	13	9	
	6	17	1	3	16	9	12	8	
	7	19	1	2	15	8	10	6	
	8	20	1	2	14	7	7	3	
	9	30	1	2	14	7	6	1	
38	1	2	3	3	7	11	19	28	
	2	3	2	2	6	10	16	25	
	3	4	2	2	6	10	15	21	
	4	6	2	2	5	10	14	17	
	5	13	2	2	5	9	13	13	
	6	14	1	1	5	9	12	10	
	7	15	1	1	4	9	12	8	
	8	16	1	1	3	9	11	7	
	9	26	1	1	3	9	10	3	
39	1	4	4	4	4	24	23	27	
	2	5	4	4	4	23	20	26	
	3	12	4	4	4	20	19	23	
	4	13	4	4	4	15	15	22	
	5	15	4	4	4	15	13	20	
	6	19	4	3	4	12	13	18	
	7	24	4	3	4	9	9	17	
	8	25	4	3	4	7	7	15	
	9	29	4	3	4	4	6	12	
40	1	2	3	2	8	26	24	2	
	2	9	2	2	7	24	21	1	
	3	10	2	2	6	24	19	1	
	4	13	2	2	6	20	15	1	
	5	20	2	2	5	19	15	1	
	6	22	1	1	4	17	13	1	
	7	24	1	1	3	17	9	1	
	8	26	1	1	3	14	8	1	
	9	30	1	1	2	12	5	1	
41	1	3	2	4	17	19	26	25	
	2	4	1	4	16	19	25	23	
	3	5	1	4	16	18	24	22	
	4	7	1	4	15	17	22	21	
	5	9	1	4	14	16	22	18	
	6	22	1	3	13	14	21	17	
	7	26	1	3	11	13	21	14	
	8	27	1	3	11	13	20	14	
	9	28	1	3	9	12	19	12	
42	1	3	2	4	28	16	9	28	
	2	5	1	4	27	15	8	24	
	3	6	1	4	27	15	7	23	
	4	16	1	4	26	14	7	19	
	5	19	1	3	26	13	6	16	
	6	22	1	3	26	13	6	15	
	7	24	1	3	26	13	4	14	
	8	25	1	2	25	11	3	9	
	9	29	1	2	25	11	3	8	
43	1	3	4	2	26	18	27	5	
	2	4	4	2	23	15	25	4	
	3	5	4	2	23	15	22	4	
	4	14	3	2	20	13	22	4	
	5	17	3	2	19	13	20	4	
	6	18	3	1	15	10	18	4	
	7	21	3	1	13	10	17	4	
	8	26	2	1	12	9	16	4	
	9	30	2	1	9	8	14	4	
44	1	1	3	5	25	11	28	20	
	2	2	3	4	25	10	27	19	
	3	11	3	4	22	10	27	19	
	4	12	3	3	19	10	27	18	
	5	15	3	3	17	10	27	17	
	6	17	3	3	16	9	27	15	
	7	20	3	3	12	9	27	15	
	8	28	3	2	10	9	27	14	
	9	29	3	2	9	9	27	13	
45	1	1	5	2	29	26	27	5	
	2	2	4	2	27	25	27	5	
	3	4	4	2	24	25	27	4	
	4	8	3	2	23	24	27	4	
	5	10	3	2	19	24	27	4	
	6	16	2	2	19	23	27	3	
	7	22	2	2	15	23	27	2	
	8	27	1	2	13	21	27	2	
	9	28	1	2	12	21	27	2	
46	1	5	2	1	5	26	14	22	
	2	9	2	1	4	24	13	19	
	3	11	2	1	4	24	13	17	
	4	14	2	1	4	23	13	16	
	5	19	2	1	4	21	13	14	
	6	21	2	1	4	20	13	12	
	7	25	2	1	4	19	13	9	
	8	28	2	1	4	18	13	7	
	9	30	2	1	4	18	13	6	
47	1	3	4	5	22	22	21	30	
	2	5	4	4	19	21	18	29	
	3	14	4	4	17	21	17	27	
	4	16	4	4	16	20	16	26	
	5	20	4	4	14	20	13	26	
	6	24	4	4	12	19	13	25	
	7	25	4	4	8	18	12	24	
	8	26	4	4	8	18	10	23	
	9	30	4	4	5	18	8	23	
48	1	11	2	5	20	13	20	24	
	2	17	1	4	20	13	18	24	
	3	18	1	4	18	12	18	24	
	4	20	1	4	15	11	16	24	
	5	21	1	4	13	10	16	24	
	6	22	1	4	10	10	16	24	
	7	25	1	4	10	9	14	24	
	8	27	1	4	8	9	14	24	
	9	28	1	4	6	8	13	24	
49	1	1	4	5	22	23	23	27	
	2	3	4	4	20	22	23	24	
	3	10	4	4	18	22	23	24	
	4	12	4	4	16	19	23	22	
	5	15	4	3	15	18	23	21	
	6	16	4	3	11	16	23	19	
	7	22	4	3	9	14	23	16	
	8	24	4	3	8	14	23	14	
	9	26	4	3	6	12	23	13	
50	1	3	5	1	24	4	18	9	
	2	5	4	1	23	4	18	9	
	3	7	4	1	22	4	16	9	
	4	8	4	1	20	3	13	9	
	5	11	4	1	19	3	12	8	
	6	19	4	1	19	3	9	8	
	7	22	4	1	17	2	7	7	
	8	23	4	1	17	2	4	7	
	9	24	4	1	15	2	4	7	
51	1	2	4	3	4	27	22	21	
	2	3	3	2	3	27	22	20	
	3	9	3	2	3	26	19	18	
	4	10	3	2	2	25	18	16	
	5	11	3	1	2	25	14	15	
	6	13	2	1	2	24	13	14	
	7	16	2	1	2	24	11	11	
	8	18	2	1	1	23	9	8	
	9	30	2	1	1	23	8	7	
52	1	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2	N 3	N 4
	29	31	594	622	656	565

************************************************************************
