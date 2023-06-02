jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 4 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	11		2 3 4 5 6 7 8 9 10 11 13 
2	9	8		28 23 22 20 19 18 15 12 
3	9	9		32 28 27 24 22 21 18 17 14 
4	9	7		28 27 24 22 18 16 14 
5	9	7		30 27 24 23 22 18 16 
6	9	4		28 18 15 12 
7	9	7		32 28 23 22 21 19 15 
8	9	7		35 30 28 27 20 18 16 
9	9	8		32 30 29 28 24 23 21 19 
10	9	8		33 32 28 27 23 22 20 18 
11	9	5		27 23 20 18 16 
12	9	4		27 25 24 16 
13	9	8		50 36 34 33 32 28 26 25 
14	9	5		33 29 25 23 19 
15	9	4		34 27 25 24 
16	9	7		50 44 36 34 33 32 26 
17	9	7		51 37 36 34 33 31 29 
18	9	4		49 46 29 25 
19	9	8		50 46 44 43 38 37 36 34 
20	9	4		50 46 39 25 
21	9	3		49 33 25 
22	9	7		50 49 46 43 37 34 31 
23	9	6		48 46 43 39 36 31 
24	9	6		50 48 41 35 33 31 
25	9	7		51 48 47 43 41 38 31 
26	9	4		47 46 45 29 
27	9	6		47 46 45 44 41 37 
28	9	6		48 47 46 44 43 39 
29	9	5		48 43 42 39 38 
30	9	6		49 45 44 42 41 40 
31	9	4		45 44 42 40 
32	9	4		46 43 42 40 
33	9	3		47 46 43 
34	9	3		48 45 41 
35	9	3		44 43 40 
36	9	2		49 47 
37	9	2		42 40 
38	9	1		40 
39	9	1		41 
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
2	1	2	2	2	16	28	3	8	
	2	3	2	2	14	25	2	8	
	3	6	2	2	14	25	2	7	
	4	10	2	2	14	24	2	8	
	5	11	2	2	13	23	2	8	
	6	17	2	1	12	21	1	8	
	7	23	2	1	11	19	1	8	
	8	27	2	1	10	19	1	8	
	9	28	2	1	10	18	1	8	
3	1	1	4	4	13	15	20	9	
	2	2	4	4	12	15	19	8	
	3	8	4	4	12	15	19	7	
	4	9	3	3	12	14	19	7	
	5	15	3	3	12	13	18	6	
	6	21	2	2	12	13	17	6	
	7	23	1	1	12	13	17	5	
	8	28	1	1	12	12	15	5	
	9	29	1	1	12	12	15	4	
4	1	1	2	3	24	25	13	26	
	2	4	2	2	23	22	12	23	
	3	7	2	2	23	21	12	22	
	4	10	2	2	22	19	12	20	
	5	12	2	2	22	16	11	14	
	6	13	1	1	22	14	11	12	
	7	17	1	1	22	10	11	11	
	8	21	1	1	21	9	11	5	
	9	22	1	1	21	6	11	4	
5	1	2	3	1	5	25	7	24	
	2	3	3	1	5	25	6	23	
	3	5	3	1	5	25	6	19	
	4	8	3	1	5	25	4	17	
	5	9	3	1	5	24	4	15	
	6	13	3	1	4	24	4	14	
	7	20	3	1	4	24	2	12	
	8	21	3	1	4	24	2	11	
	9	24	3	1	4	24	1	7	
6	1	2	2	5	24	20	9	14	
	2	8	1	5	24	18	9	14	
	3	10	1	5	24	17	9	11	
	4	16	1	5	24	15	9	11	
	5	18	1	5	24	13	9	9	
	6	20	1	5	24	13	9	7	
	7	23	1	5	24	12	9	7	
	8	24	1	5	24	10	9	4	
	9	30	1	5	24	9	9	4	
7	1	1	2	4	2	9	13	16	
	2	2	2	4	2	7	12	16	
	3	3	2	4	2	7	12	14	
	4	6	2	3	2	7	11	11	
	5	12	2	3	2	6	11	11	
	6	13	2	3	2	6	10	8	
	7	16	2	2	2	5	10	6	
	8	21	2	2	2	5	9	4	
	9	30	2	2	2	4	9	4	
8	1	3	2	1	20	20	17	12	
	2	5	2	1	16	19	15	12	
	3	7	2	1	16	18	13	12	
	4	11	2	1	15	13	11	12	
	5	13	2	1	13	11	11	12	
	6	16	2	1	12	8	9	12	
	7	19	2	1	10	7	7	12	
	8	22	2	1	9	3	5	12	
	9	29	2	1	7	3	4	12	
9	1	1	3	5	12	19	23	23	
	2	10	3	4	12	17	23	22	
	3	16	3	4	12	16	23	20	
	4	17	3	4	12	13	23	20	
	5	19	3	4	12	13	22	18	
	6	22	3	3	12	11	22	18	
	7	24	3	3	12	11	21	17	
	8	25	3	3	12	9	21	16	
	9	27	3	3	12	8	21	14	
10	1	1	4	1	26	18	12	8	
	2	2	4	1	26	17	11	8	
	3	10	4	1	26	16	11	8	
	4	11	3	1	26	15	11	8	
	5	14	3	1	26	15	10	8	
	6	17	3	1	26	15	10	7	
	7	22	2	1	26	14	10	8	
	8	25	2	1	26	13	9	8	
	9	28	2	1	26	12	9	8	
11	1	8	5	3	24	7	23	18	
	2	11	4	3	21	6	20	16	
	3	12	4	3	18	6	18	16	
	4	13	4	3	16	6	16	15	
	5	22	3	3	15	6	16	15	
	6	25	3	3	13	6	11	14	
	7	26	3	3	10	6	11	14	
	8	27	3	3	6	6	8	13	
	9	29	3	3	4	6	5	12	
12	1	3	4	3	28	17	27	24	
	2	7	4	3	27	14	23	21	
	3	9	4	3	27	12	20	21	
	4	14	3	2	27	12	19	20	
	5	20	2	2	27	8	17	18	
	6	22	2	2	26	6	13	15	
	7	23	2	2	26	4	10	13	
	8	25	1	1	26	3	10	11	
	9	26	1	1	26	2	7	11	
13	1	1	2	5	9	25	14	17	
	2	2	2	4	7	24	13	17	
	3	3	2	4	6	22	13	16	
	4	6	2	4	5	21	13	16	
	5	7	2	3	5	20	13	16	
	6	10	2	3	5	20	13	15	
	7	14	2	3	4	18	13	15	
	8	18	2	3	2	17	13	14	
	9	29	2	3	2	17	13	13	
14	1	7	4	3	28	27	10	20	
	2	8	3	2	28	27	9	20	
	3	12	3	2	26	27	9	18	
	4	13	3	2	23	26	9	15	
	5	14	2	1	23	26	8	15	
	6	17	2	1	20	25	8	14	
	7	18	2	1	19	24	8	11	
	8	19	1	1	16	24	8	11	
	9	26	1	1	15	24	8	8	
15	1	7	3	4	10	18	30	24	
	2	10	2	3	9	17	27	22	
	3	12	2	3	9	16	26	19	
	4	17	2	2	8	16	26	17	
	5	19	2	2	7	16	23	16	
	6	22	2	2	5	15	22	14	
	7	25	2	2	5	14	21	10	
	8	27	2	1	4	14	21	9	
	9	30	2	1	3	14	20	6	
16	1	3	2	4	4	26	26	15	
	2	8	1	4	4	25	25	15	
	3	10	1	4	4	22	25	14	
	4	13	1	3	4	22	25	14	
	5	18	1	3	4	19	24	13	
	6	20	1	3	4	17	23	12	
	7	21	1	3	4	15	22	12	
	8	26	1	2	4	13	21	12	
	9	28	1	2	4	13	21	11	
17	1	7	2	5	19	20	22	23	
	2	13	2	4	18	17	22	22	
	3	16	2	4	18	15	19	22	
	4	17	2	4	16	15	18	21	
	5	18	2	3	15	12	16	21	
	6	19	1	3	14	12	14	21	
	7	22	1	2	13	10	13	21	
	8	25	1	2	13	8	10	20	
	9	29	1	2	11	7	10	20	
18	1	2	3	5	13	28	18	28	
	2	3	3	4	13	28	17	25	
	3	11	3	4	13	28	16	22	
	4	13	3	4	13	28	14	19	
	5	14	3	3	13	27	12	15	
	6	15	3	3	13	27	10	12	
	7	18	3	3	13	26	9	9	
	8	21	3	2	13	26	8	6	
	9	28	3	2	13	26	6	3	
19	1	2	5	4	12	21	27	28	
	2	3	5	4	11	20	22	28	
	3	5	5	4	11	18	21	27	
	4	6	5	3	9	17	18	27	
	5	13	5	2	7	17	16	27	
	6	15	5	2	6	16	12	26	
	7	16	5	1	5	14	11	26	
	8	23	5	1	4	13	7	25	
	9	28	5	1	3	13	4	25	
20	1	5	5	3	13	29	27	22	
	2	6	4	3	12	28	27	22	
	3	7	3	3	11	24	27	22	
	4	12	3	3	10	22	27	22	
	5	13	2	2	9	22	26	22	
	6	17	2	2	8	20	26	22	
	7	18	1	2	7	18	26	22	
	8	19	1	2	5	15	26	22	
	9	22	1	2	4	12	26	22	
21	1	2	3	5	20	19	26	16	
	2	3	3	4	18	17	24	15	
	3	4	3	4	18	15	23	15	
	4	5	3	3	16	15	23	14	
	5	6	3	3	15	13	21	13	
	6	10	3	2	15	12	19	12	
	7	13	3	1	14	11	18	10	
	8	14	3	1	12	10	17	9	
	9	17	3	1	12	10	15	9	
22	1	4	4	4	13	30	27	19	
	2	5	3	4	12	26	25	17	
	3	11	3	4	12	26	25	15	
	4	15	3	4	12	24	24	14	
	5	18	3	4	12	21	23	14	
	6	21	3	4	12	20	21	12	
	7	22	3	4	12	17	20	11	
	8	23	3	4	12	15	19	11	
	9	28	3	4	12	14	18	9	
23	1	5	2	4	6	13	9	14	
	2	6	2	4	6	12	9	13	
	3	7	2	4	5	12	9	12	
	4	9	2	4	4	12	9	12	
	5	15	2	4	3	12	8	11	
	6	20	2	4	3	11	8	10	
	7	21	2	4	2	11	8	8	
	8	28	2	4	2	11	8	7	
	9	29	2	4	1	11	8	7	
24	1	7	2	1	21	20	6	15	
	2	13	2	1	21	19	5	13	
	3	17	2	1	20	18	5	12	
	4	18	2	1	18	18	4	11	
	5	22	2	1	18	17	4	9	
	6	23	2	1	16	15	3	9	
	7	24	2	1	15	14	3	8	
	8	27	2	1	15	13	2	6	
	9	29	2	1	14	13	2	5	
25	1	1	1	5	27	23	18	13	
	2	2	1	4	23	20	16	11	
	3	5	1	4	21	19	15	9	
	4	7	1	4	19	18	15	7	
	5	9	1	4	17	18	12	6	
	6	14	1	4	14	16	12	6	
	7	21	1	4	14	14	10	5	
	8	26	1	4	9	13	8	3	
	9	30	1	4	7	13	6	2	
26	1	2	4	5	26	2	15	25	
	2	4	3	4	23	1	12	24	
	3	5	3	4	22	1	11	23	
	4	7	3	3	20	1	11	22	
	5	9	3	3	14	1	8	20	
	6	13	3	3	11	1	7	20	
	7	15	3	2	10	1	6	18	
	8	19	3	2	8	1	5	17	
	9	24	3	2	5	1	4	17	
27	1	3	5	3	13	11	29	28	
	2	9	4	3	11	10	28	25	
	3	10	4	3	11	9	26	23	
	4	12	4	3	10	9	25	22	
	5	17	4	3	10	8	25	20	
	6	18	3	3	9	8	24	17	
	7	19	3	3	8	8	22	16	
	8	21	3	3	7	7	21	13	
	9	22	3	3	7	7	21	12	
28	1	3	3	5	15	26	28	26	
	2	4	2	4	15	24	27	24	
	3	7	2	4	15	24	26	24	
	4	8	2	4	14	24	24	20	
	5	12	2	4	13	23	23	18	
	6	15	2	3	13	23	21	16	
	7	24	2	3	13	22	21	14	
	8	25	2	3	12	21	18	12	
	9	28	2	3	12	21	18	11	
29	1	1	4	2	25	22	12	11	
	2	2	3	2	23	20	10	10	
	3	3	3	2	22	20	10	9	
	4	4	3	2	22	19	10	8	
	5	7	2	2	20	18	9	7	
	6	8	2	2	20	17	8	6	
	7	16	1	2	18	16	7	6	
	8	23	1	2	17	15	7	4	
	9	28	1	2	17	15	6	4	
30	1	3	4	4	8	7	23	21	
	2	7	3	4	7	6	21	17	
	3	9	3	3	7	6	18	17	
	4	10	3	3	6	5	17	15	
	5	12	2	3	4	5	16	11	
	6	16	2	2	3	5	14	9	
	7	19	2	2	3	4	13	7	
	8	25	2	1	2	4	9	3	
	9	28	2	1	1	3	9	3	
31	1	1	4	4	19	19	20	29	
	2	4	3	4	18	17	19	28	
	3	15	3	4	18	16	19	28	
	4	18	3	4	18	13	19	27	
	5	19	2	4	18	12	18	27	
	6	20	2	4	17	10	17	26	
	7	22	2	4	17	10	17	25	
	8	29	2	4	17	8	16	26	
	9	30	2	4	17	6	16	25	
32	1	2	2	3	16	24	22	18	
	2	10	2	2	15	23	20	18	
	3	16	2	2	15	22	17	18	
	4	17	2	2	14	21	15	18	
	5	18	2	2	14	21	13	18	
	6	20	2	2	13	20	13	18	
	7	25	2	2	12	20	11	18	
	8	29	2	2	12	19	9	18	
	9	30	2	2	12	19	7	18	
33	1	10	3	2	22	23	26	22	
	2	21	3	1	20	23	26	21	
	3	22	3	1	19	21	25	17	
	4	23	2	1	18	20	24	17	
	5	24	2	1	16	18	24	14	
	6	25	2	1	15	15	23	12	
	7	26	1	1	13	13	23	8	
	8	28	1	1	13	11	22	7	
	9	29	1	1	12	10	22	5	
34	1	13	4	5	18	15	22	26	
	2	15	4	4	17	15	22	25	
	3	17	4	4	17	14	18	20	
	4	22	4	4	16	12	16	19	
	5	24	3	3	14	11	14	15	
	6	25	3	3	13	11	12	12	
	7	26	3	3	13	10	9	7	
	8	27	2	3	12	9	6	6	
	9	28	2	3	11	8	6	3	
35	1	2	1	5	27	16	19	26	
	2	4	1	4	26	14	19	23	
	3	7	1	4	24	13	19	23	
	4	10	1	4	23	13	19	20	
	5	11	1	4	22	12	19	17	
	6	12	1	4	21	11	19	15	
	7	18	1	4	21	10	19	13	
	8	29	1	4	20	9	19	10	
	9	30	1	4	18	9	19	6	
36	1	7	1	5	24	17	3	30	
	2	8	1	4	21	17	3	26	
	3	14	1	4	20	17	3	23	
	4	18	1	4	17	17	3	20	
	5	20	1	3	15	16	3	15	
	6	25	1	3	13	16	3	14	
	7	27	1	3	9	16	3	11	
	8	28	1	2	7	15	3	9	
	9	30	1	2	5	15	3	6	
37	1	6	3	4	26	25	23	29	
	2	10	3	4	24	23	22	28	
	3	11	3	4	23	22	20	26	
	4	12	3	4	22	19	17	26	
	5	14	3	4	21	15	16	23	
	6	20	3	4	18	14	15	23	
	7	21	3	4	17	11	14	21	
	8	24	3	4	15	9	10	20	
	9	26	3	4	15	8	10	18	
38	1	6	4	4	17	25	22	23	
	2	7	4	4	16	24	20	23	
	3	14	4	4	16	23	20	23	
	4	15	4	3	15	23	19	23	
	5	16	3	2	12	21	19	23	
	6	22	3	2	11	20	18	22	
	7	24	3	2	9	20	17	22	
	8	28	3	1	9	18	17	22	
	9	29	3	1	7	17	16	22	
39	1	4	4	4	26	24	28	29	
	2	5	4	3	23	24	28	26	
	3	10	4	3	22	20	28	21	
	4	16	4	3	19	17	28	18	
	5	17	4	3	17	14	27	14	
	6	18	4	2	16	14	27	12	
	7	19	4	2	12	12	27	11	
	8	22	4	2	10	9	27	6	
	9	25	4	2	8	6	27	5	
40	1	3	2	4	24	22	26	19	
	2	9	2	4	23	20	25	19	
	3	15	2	4	23	19	24	17	
	4	16	2	3	22	18	22	15	
	5	17	2	3	22	18	17	12	
	6	22	2	2	22	18	16	9	
	7	23	2	1	21	16	15	5	
	8	27	2	1	21	16	11	4	
	9	28	2	1	21	15	9	1	
41	1	3	3	4	11	18	26	17	
	2	5	3	4	10	16	23	17	
	3	7	3	4	10	15	22	17	
	4	13	3	4	8	12	18	16	
	5	14	3	4	8	9	16	16	
	6	16	2	4	6	9	12	16	
	7	25	2	4	5	7	9	15	
	8	26	2	4	4	3	8	15	
	9	28	2	4	3	2	5	15	
42	1	1	5	5	18	23	8	29	
	2	3	4	4	17	23	7	28	
	3	13	4	4	13	23	6	26	
	4	17	4	4	13	23	6	25	
	5	18	4	3	11	23	4	25	
	6	22	4	3	9	23	4	24	
	7	24	4	3	7	23	4	24	
	8	27	4	3	4	23	3	23	
	9	28	4	3	3	23	2	22	
43	1	2	2	4	17	30	23	14	
	2	5	2	4	16	29	22	14	
	3	14	2	3	15	29	21	13	
	4	17	2	3	15	29	20	11	
	5	19	2	2	15	28	19	11	
	6	22	2	2	14	28	19	9	
	7	25	2	2	14	28	17	8	
	8	29	2	1	13	28	17	8	
	9	30	2	1	13	28	16	7	
44	1	1	4	3	19	29	7	15	
	2	8	3	2	18	28	7	13	
	3	10	3	2	16	28	7	13	
	4	16	3	2	16	28	7	12	
	5	18	3	2	14	27	6	8	
	6	19	2	2	12	27	6	8	
	7	20	2	2	12	27	6	7	
	8	21	2	2	10	26	5	4	
	9	27	2	2	9	26	5	2	
45	1	2	3	4	11	25	2	21	
	2	4	2	3	11	22	1	19	
	3	5	2	3	10	19	1	16	
	4	7	2	2	10	17	1	15	
	5	8	2	2	10	13	1	15	
	6	9	2	2	9	10	1	13	
	7	13	2	1	8	7	1	11	
	8	19	2	1	8	7	1	9	
	9	20	2	1	8	3	1	9	
46	1	6	2	4	26	26	26	7	
	2	8	2	3	26	25	24	7	
	3	10	2	3	26	21	23	6	
	4	16	2	3	25	19	22	5	
	5	19	2	2	24	18	20	4	
	6	22	2	2	24	18	19	4	
	7	24	2	2	24	15	15	3	
	8	27	2	2	23	12	13	2	
	9	28	2	2	23	12	13	1	
47	1	7	4	5	26	22	14	26	
	2	9	4	5	24	20	14	25	
	3	10	4	5	22	18	14	25	
	4	12	4	5	18	18	14	25	
	5	15	4	5	14	16	14	24	
	6	16	4	5	13	15	14	24	
	7	17	4	5	9	15	14	24	
	8	19	4	5	7	13	14	24	
	9	22	4	5	4	13	14	24	
48	1	1	2	3	26	15	22	27	
	2	4	2	2	26	15	21	27	
	3	8	2	2	26	13	19	26	
	4	9	2	2	26	12	14	26	
	5	12	2	2	25	11	13	25	
	6	13	2	2	25	9	12	25	
	7	16	2	2	25	6	7	24	
	8	19	2	2	25	5	6	24	
	9	20	2	2	25	4	5	24	
49	1	4	1	4	16	29	20	24	
	2	5	1	4	16	27	18	24	
	3	8	1	4	14	27	16	24	
	4	14	1	4	11	25	12	24	
	5	15	1	4	9	24	10	23	
	6	24	1	4	8	24	8	23	
	7	26	1	4	6	23	7	23	
	8	28	1	4	4	21	4	23	
	9	29	1	4	4	21	3	23	
50	1	2	5	4	15	19	29	18	
	2	4	4	4	15	18	29	17	
	3	5	4	4	14	15	29	15	
	4	10	4	4	12	14	29	12	
	5	21	3	3	11	12	29	12	
	6	24	3	3	10	11	29	10	
	7	26	3	2	9	9	29	8	
	8	29	3	2	7	7	29	6	
	9	30	3	2	5	7	29	5	
51	1	8	2	5	17	23	22	13	
	2	14	2	4	16	22	21	13	
	3	16	2	4	14	22	21	13	
	4	17	2	4	13	22	21	13	
	5	20	2	3	12	22	20	13	
	6	21	2	3	11	21	20	13	
	7	22	2	3	11	21	20	12	
	8	25	2	2	10	21	20	13	
	9	26	2	2	9	21	20	13	
52	1	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2	N 3	N 4
	18	23	803	932	846	889

************************************************************************
