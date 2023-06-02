jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 4 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	10		2 3 4 5 6 7 9 10 14 16 
2	9	7		28 26 22 21 19 18 8 
3	9	7		26 25 22 21 17 15 13 
4	9	5		29 26 22 21 11 
5	9	6		33 32 24 22 20 12 
6	9	4		27 22 15 12 
7	9	7		34 29 26 24 23 22 21 
8	9	2		17 13 
9	9	7		33 29 28 26 23 22 20 
10	9	10		36 34 33 32 31 30 29 27 26 22 
11	9	7		37 34 33 32 27 24 23 
12	9	7		37 34 31 30 29 28 25 
13	9	5		36 33 32 29 20 
14	9	5		51 37 33 28 20 
15	9	8		51 50 34 33 32 31 30 28 
16	9	9		51 50 42 37 35 33 31 30 25 
17	9	7		51 50 42 37 36 34 23 
18	9	6		51 37 36 34 27 25 
19	9	9		51 50 42 41 40 36 35 31 30 
20	9	7		50 42 41 35 34 31 30 
21	9	7		48 47 42 41 35 33 32 
22	9	8		51 50 49 47 41 40 39 38 
23	9	4		41 40 31 30 
24	9	7		51 50 49 45 40 39 36 
25	9	5		49 47 41 39 38 
26	9	6		51 49 47 44 41 39 
27	9	5		50 48 47 42 35 
28	9	3		42 36 35 
29	9	8		51 50 49 48 47 45 44 43 
30	9	5		49 47 44 43 39 
31	9	3		48 43 38 
32	9	3		49 43 38 
33	9	2		40 38 
34	9	3		46 45 40 
35	9	2		46 38 
36	9	3		47 44 43 
37	9	3		48 47 43 
38	9	2		45 44 
39	9	2		48 46 
40	9	1		43 
41	9	1		43 
42	9	1		46 
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
2	1	3	4	3	29	23	12	27	
	2	4	4	3	29	21	11	25	
	3	5	4	3	29	17	11	25	
	4	6	4	3	29	17	11	24	
	5	7	4	3	28	13	10	22	
	6	8	4	3	28	11	10	21	
	7	14	4	3	28	8	10	20	
	8	22	4	3	28	4	10	20	
	9	23	4	3	28	2	10	19	
3	1	1	5	2	14	9	28	14	
	2	15	4	2	13	9	24	14	
	3	17	4	2	13	9	24	13	
	4	22	4	2	10	8	22	14	
	5	24	3	2	8	8	20	13	
	6	25	3	2	7	8	17	13	
	7	27	2	2	7	7	13	13	
	8	28	2	2	4	7	12	12	
	9	30	2	2	3	7	11	12	
4	1	2	5	5	28	20	28	12	
	2	6	4	4	24	19	28	11	
	3	13	4	4	23	18	27	11	
	4	14	4	4	20	13	25	10	
	5	15	3	3	17	13	24	10	
	6	18	3	3	15	9	21	10	
	7	19	3	3	13	6	21	10	
	8	21	3	2	12	5	18	9	
	9	30	3	2	11	2	17	9	
5	1	2	5	5	26	28	17	19	
	2	3	4	5	24	26	15	18	
	3	4	3	5	21	23	15	16	
	4	5	3	5	20	22	13	12	
	5	8	2	5	18	21	13	11	
	6	11	2	5	17	18	13	8	
	7	12	1	5	17	17	11	6	
	8	21	1	5	14	14	10	6	
	9	27	1	5	14	13	10	4	
6	1	1	4	5	8	25	10	27	
	2	5	3	4	8	24	9	25	
	3	6	3	4	7	24	9	23	
	4	7	3	4	6	23	8	21	
	5	10	3	3	6	23	8	17	
	6	15	2	3	6	23	8	15	
	7	16	2	3	5	22	7	12	
	8	24	2	2	4	22	7	12	
	9	29	2	2	4	22	6	9	
7	1	1	4	4	30	25	18	9	
	2	2	4	4	26	25	17	7	
	3	4	4	4	21	24	16	7	
	4	5	4	4	21	21	15	6	
	5	8	4	4	16	21	13	5	
	6	9	4	4	14	19	13	4	
	7	17	4	4	9	17	11	3	
	8	25	4	4	9	17	10	2	
	9	29	4	4	6	15	9	2	
8	1	1	3	4	28	15	27	9	
	2	2	2	4	26	15	26	8	
	3	8	2	4	26	14	22	8	
	4	16	2	4	25	14	21	8	
	5	17	2	4	23	14	16	7	
	6	23	2	4	23	13	14	7	
	7	25	2	4	21	13	12	7	
	8	26	2	4	21	12	11	6	
	9	28	2	4	19	12	6	6	
9	1	3	5	2	20	21	5	28	
	2	15	4	2	19	20	4	26	
	3	17	4	2	18	20	4	25	
	4	18	4	2	15	19	4	22	
	5	19	3	2	13	18	3	20	
	6	20	3	2	12	18	3	20	
	7	22	2	2	12	17	3	17	
	8	26	2	2	10	16	2	15	
	9	30	2	2	9	16	2	13	
10	1	4	4	4	29	18	29	11	
	2	9	4	3	28	16	28	11	
	3	13	4	3	27	15	28	11	
	4	14	4	3	26	14	28	10	
	5	18	4	3	25	12	27	10	
	6	20	4	3	23	10	27	10	
	7	26	4	3	22	8	27	9	
	8	28	4	3	22	7	27	9	
	9	30	4	3	21	6	27	9	
11	1	4	2	3	8	26	8	11	
	2	5	2	3	7	26	7	11	
	3	6	2	3	7	25	7	11	
	4	8	2	2	7	23	7	11	
	5	9	2	2	7	23	6	10	
	6	14	2	2	6	21	5	10	
	7	16	2	1	6	19	5	10	
	8	18	2	1	6	19	3	10	
	9	23	2	1	6	17	3	10	
12	1	1	4	4	18	11	14	28	
	2	7	4	4	18	10	12	27	
	3	14	4	4	17	10	10	27	
	4	16	4	3	17	9	9	27	
	5	18	3	3	16	9	8	26	
	6	21	3	2	15	9	6	25	
	7	23	2	1	15	8	4	25	
	8	25	2	1	15	7	4	24	
	9	29	2	1	14	7	1	24	
13	1	4	5	5	20	28	16	28	
	2	5	4	5	20	27	16	25	
	3	8	4	5	18	25	16	24	
	4	9	4	5	16	24	15	22	
	5	16	4	5	15	24	15	22	
	6	17	4	5	14	23	14	20	
	7	19	4	5	11	21	13	17	
	8	29	4	5	10	21	13	15	
	9	30	4	5	9	19	13	15	
14	1	4	3	2	28	18	15	29	
	2	7	2	1	28	18	12	28	
	3	8	2	1	25	17	12	28	
	4	13	2	1	23	15	11	28	
	5	14	1	1	22	15	10	27	
	6	18	1	1	20	14	9	27	
	7	24	1	1	17	12	8	26	
	8	27	1	1	15	12	7	26	
	9	28	1	1	15	11	6	26	
15	1	8	2	5	17	17	13	8	
	2	10	1	4	15	16	10	8	
	3	11	1	4	13	16	10	8	
	4	12	1	3	12	15	8	8	
	5	13	1	2	9	15	7	8	
	6	17	1	2	9	15	6	8	
	7	18	1	1	6	14	3	8	
	8	23	1	1	5	14	3	8	
	9	27	1	1	3	14	1	8	
16	1	6	2	3	22	21	27	27	
	2	9	1	3	21	20	24	25	
	3	12	1	3	21	20	22	24	
	4	17	1	3	21	19	20	21	
	5	18	1	3	21	18	15	20	
	6	22	1	3	21	18	15	19	
	7	23	1	3	21	18	10	17	
	8	28	1	3	21	17	8	13	
	9	30	1	3	21	17	6	13	
17	1	1	3	5	22	24	22	21	
	2	5	3	4	20	21	21	16	
	3	9	3	4	20	19	21	16	
	4	10	3	4	18	18	21	14	
	5	11	3	4	17	16	21	11	
	6	19	3	4	15	15	21	7	
	7	20	3	4	15	13	21	6	
	8	21	3	4	14	10	21	4	
	9	30	3	4	13	9	21	2	
18	1	1	4	2	29	28	25	27	
	2	2	3	1	27	26	24	26	
	3	3	3	1	24	25	20	23	
	4	6	3	1	19	24	18	21	
	5	11	3	1	16	24	18	21	
	6	12	3	1	13	21	13	18	
	7	13	3	1	10	20	13	14	
	8	23	3	1	6	20	8	14	
	9	30	3	1	3	18	6	10	
19	1	5	3	4	18	13	8	8	
	2	6	2	4	16	12	7	8	
	3	7	2	4	16	10	7	8	
	4	17	2	4	15	8	7	8	
	5	18	1	4	15	7	7	8	
	6	20	1	3	14	5	7	8	
	7	27	1	3	13	5	7	8	
	8	29	1	3	12	3	7	8	
	9	30	1	3	12	2	7	8	
20	1	10	4	4	21	18	16	28	
	2	11	4	3	17	18	16	27	
	3	13	4	3	15	17	15	27	
	4	14	4	3	14	17	15	27	
	5	16	4	2	13	16	14	27	
	6	17	4	2	8	16	14	27	
	7	22	4	2	7	15	13	27	
	8	23	4	2	5	15	13	27	
	9	25	4	2	3	15	13	27	
21	1	5	2	5	30	14	17	25	
	2	6	2	4	29	14	16	23	
	3	8	2	4	28	14	15	23	
	4	9	2	4	27	14	14	20	
	5	10	1	4	26	14	13	18	
	6	19	1	4	25	14	12	15	
	7	28	1	4	24	14	11	14	
	8	29	1	4	24	14	10	9	
	9	30	1	4	23	14	9	8	
22	1	3	4	4	28	15	26	11	
	2	5	4	4	27	12	23	11	
	3	13	4	4	26	11	21	11	
	4	15	4	4	25	10	19	11	
	5	16	4	3	24	8	19	10	
	6	20	4	3	24	8	14	10	
	7	22	4	3	24	7	12	10	
	8	24	4	3	23	5	11	10	
	9	30	4	3	22	4	8	10	
23	1	2	4	2	15	21	28	5	
	2	5	4	2	14	21	26	5	
	3	13	4	2	13	19	26	5	
	4	18	4	2	12	19	25	4	
	5	20	3	2	11	17	22	4	
	6	21	3	2	9	17	20	3	
	7	24	3	2	8	15	19	3	
	8	29	3	2	8	15	16	2	
	9	30	3	2	7	13	15	2	
24	1	1	3	2	26	21	20	27	
	2	2	2	2	26	20	19	26	
	3	5	2	2	26	18	19	23	
	4	12	2	2	25	17	19	22	
	5	15	2	2	25	16	19	18	
	6	19	1	2	25	15	19	16	
	7	26	1	2	24	14	19	14	
	8	27	1	2	24	13	19	13	
	9	30	1	2	24	11	19	11	
25	1	6	5	5	23	30	23	25	
	2	9	4	4	23	29	19	24	
	3	11	4	4	21	29	16	23	
	4	16	3	4	19	29	16	22	
	5	17	3	4	14	29	11	22	
	6	18	2	4	13	29	10	21	
	7	23	2	4	12	29	7	19	
	8	27	1	4	8	29	5	18	
	9	28	1	4	6	29	4	18	
26	1	3	3	4	26	7	6	24	
	2	4	3	4	25	6	5	23	
	3	6	3	3	25	6	5	22	
	4	7	3	3	25	6	5	19	
	5	9	2	2	25	6	4	16	
	6	22	2	2	25	5	4	14	
	7	26	2	1	25	5	3	11	
	8	28	2	1	25	5	3	10	
	9	29	2	1	25	5	3	8	
27	1	1	4	3	15	11	17	18	
	2	3	3	2	13	11	14	17	
	3	4	3	2	13	11	13	17	
	4	15	3	2	12	10	13	17	
	5	18	3	2	11	10	12	17	
	6	19	3	2	9	9	11	16	
	7	23	3	2	9	9	10	16	
	8	28	3	2	7	8	8	16	
	9	29	3	2	7	8	8	15	
28	1	1	4	4	27	22	22	21	
	2	3	3	3	23	22	21	19	
	3	4	3	3	19	21	20	19	
	4	7	3	3	19	21	19	18	
	5	11	2	2	14	21	19	16	
	6	20	2	2	11	20	18	14	
	7	21	2	2	8	19	17	13	
	8	24	2	2	7	19	17	12	
	9	29	2	2	5	19	16	11	
29	1	3	4	4	14	27	20	15	
	2	7	3	3	11	26	20	14	
	3	8	3	3	11	26	17	14	
	4	10	3	2	10	25	17	13	
	5	11	2	2	7	24	16	13	
	6	14	2	2	7	24	14	13	
	7	20	2	2	5	23	11	12	
	8	24	1	1	4	23	11	12	
	9	25	1	1	2	22	10	12	
30	1	2	4	4	15	13	22	24	
	2	3	4	4	13	12	20	23	
	3	10	4	4	11	12	19	22	
	4	13	4	4	11	12	18	21	
	5	17	4	4	9	12	16	20	
	6	19	4	4	8	12	14	20	
	7	20	4	4	8	12	13	19	
	8	24	4	4	6	12	13	19	
	9	25	4	4	6	12	11	18	
31	1	1	3	5	21	28	28	12	
	2	11	3	4	20	27	27	11	
	3	12	3	4	20	27	25	11	
	4	16	3	3	19	26	23	11	
	5	18	2	3	17	26	20	11	
	6	20	2	3	15	26	20	10	
	7	21	2	2	14	26	18	10	
	8	25	2	2	13	25	16	10	
	9	26	2	2	11	25	14	10	
32	1	3	3	4	22	27	19	23	
	2	10	2	3	21	27	18	22	
	3	16	2	3	21	26	17	21	
	4	22	2	3	19	26	17	18	
	5	23	2	3	18	25	16	15	
	6	25	2	3	17	24	14	13	
	7	26	2	3	17	24	13	12	
	8	29	2	3	15	24	13	10	
	9	30	2	3	15	23	12	9	
33	1	6	4	2	23	12	27	8	
	2	7	4	1	20	11	25	7	
	3	15	4	1	18	9	24	6	
	4	19	4	1	14	9	21	5	
	5	23	3	1	12	7	20	4	
	6	25	3	1	12	5	18	4	
	7	26	3	1	7	5	17	3	
	8	27	3	1	4	4	16	3	
	9	28	3	1	2	2	14	2	
34	1	4	4	4	28	26	9	19	
	2	6	3	3	24	26	9	17	
	3	7	3	3	22	21	9	17	
	4	8	3	3	19	19	8	16	
	5	9	2	2	16	17	8	16	
	6	10	2	2	16	15	7	15	
	7	12	2	2	12	10	7	15	
	8	13	2	2	11	10	6	15	
	9	22	2	2	8	7	6	14	
35	1	5	5	4	12	23	14	29	
	2	6	4	4	12	21	11	27	
	3	9	4	4	11	20	11	22	
	4	17	4	4	11	16	10	21	
	5	20	3	4	10	15	8	17	
	6	21	3	4	9	12	8	15	
	7	28	3	4	9	9	6	15	
	8	29	3	4	8	8	5	11	
	9	30	3	4	8	5	5	9	
36	1	1	5	3	26	23	16	26	
	2	8	4	3	25	21	16	25	
	3	10	4	3	23	20	15	19	
	4	15	4	3	23	18	15	18	
	5	18	3	3	20	17	14	14	
	6	24	3	3	20	15	14	13	
	7	27	3	3	17	13	13	7	
	8	28	3	3	17	12	12	5	
	9	29	3	3	16	11	12	3	
37	1	8	3	1	26	14	28	27	
	2	11	2	1	25	11	27	24	
	3	12	2	1	24	9	23	21	
	4	13	2	1	23	8	21	21	
	5	20	2	1	21	8	18	18	
	6	23	2	1	20	5	16	17	
	7	24	2	1	20	3	12	16	
	8	26	2	1	19	2	9	14	
	9	27	2	1	17	1	7	12	
38	1	6	2	4	26	24	28	15	
	2	7	2	4	21	23	27	12	
	3	12	2	4	19	22	27	11	
	4	14	2	4	17	22	26	10	
	5	16	2	4	16	21	25	10	
	6	19	2	3	13	20	24	8	
	7	24	2	3	8	19	22	6	
	8	29	2	3	7	19	21	5	
	9	30	2	3	4	18	20	4	
39	1	6	3	5	26	25	26	19	
	2	9	3	4	25	25	23	18	
	3	11	3	4	23	23	21	17	
	4	12	3	4	21	21	18	17	
	5	19	3	4	20	17	13	16	
	6	23	3	3	20	15	13	16	
	7	24	3	3	17	13	8	15	
	8	29	3	3	17	11	6	15	
	9	30	3	3	15	9	2	14	
40	1	7	3	5	9	15	26	25	
	2	8	3	5	9	14	25	23	
	3	10	3	5	9	14	25	22	
	4	11	3	5	9	13	25	21	
	5	14	3	5	9	13	25	20	
	6	18	3	5	9	13	24	21	
	7	27	3	5	9	12	24	19	
	8	29	3	5	9	11	24	19	
	9	30	3	5	9	11	24	18	
41	1	3	5	3	25	22	14	11	
	2	5	4	3	24	22	13	10	
	3	8	4	3	23	19	11	8	
	4	11	4	3	23	17	11	7	
	5	14	3	3	22	12	9	6	
	6	15	3	2	21	12	7	5	
	7	20	3	2	21	8	5	4	
	8	28	2	2	20	4	2	2	
	9	30	2	2	19	4	1	2	
42	1	4	4	3	19	27	25	27	
	2	8	4	3	17	25	25	23	
	3	12	3	3	16	22	24	22	
	4	13	3	3	16	21	24	21	
	5	14	2	3	15	19	23	18	
	6	18	2	3	14	17	22	17	
	7	20	1	3	12	16	22	14	
	8	21	1	3	12	12	21	14	
	9	27	1	3	10	11	21	12	
43	1	3	3	5	21	26	18	27	
	2	15	3	4	21	25	14	25	
	3	16	3	4	21	24	13	22	
	4	19	3	4	21	23	13	21	
	5	21	3	4	20	22	10	18	
	6	22	3	4	20	21	8	13	
	7	23	3	4	20	21	6	11	
	8	26	3	4	19	19	4	9	
	9	28	3	4	19	18	4	4	
44	1	7	3	5	9	15	9	21	
	2	15	3	5	8	13	8	20	
	3	20	3	5	8	10	8	19	
	4	21	3	5	7	8	8	18	
	5	22	3	5	7	8	8	17	
	6	23	3	5	7	7	8	16	
	7	25	3	5	6	4	8	15	
	8	27	3	5	6	2	8	14	
	9	28	3	5	6	2	8	13	
45	1	6	2	2	20	19	18	8	
	2	10	2	2	19	19	16	8	
	3	16	2	2	19	19	14	8	
	4	17	2	2	18	19	10	8	
	5	19	2	2	17	19	10	8	
	6	23	1	2	15	19	8	8	
	7	25	1	2	15	19	7	8	
	8	28	1	2	14	19	5	8	
	9	29	1	2	13	19	3	8	
46	1	3	1	4	19	30	25	22	
	2	5	1	4	17	28	24	21	
	3	9	1	4	16	27	23	20	
	4	10	1	4	14	26	22	19	
	5	13	1	3	12	26	21	19	
	6	18	1	3	11	25	21	18	
	7	20	1	2	9	24	20	18	
	8	21	1	2	8	22	19	18	
	9	22	1	2	6	21	19	17	
47	1	1	4	5	21	25	15	28	
	2	3	3	4	20	25	13	28	
	3	4	3	4	20	25	12	26	
	4	5	3	4	20	25	11	26	
	5	10	3	4	20	24	11	24	
	6	13	2	4	20	24	10	23	
	7	16	2	4	20	24	9	22	
	8	20	2	4	20	24	8	22	
	9	29	2	4	20	24	8	21	
48	1	10	4	1	23	18	26	17	
	2	11	4	1	22	14	25	16	
	3	16	4	1	22	14	24	15	
	4	22	4	1	22	11	24	14	
	5	23	3	1	22	11	24	14	
	6	24	3	1	22	10	23	14	
	7	26	3	1	22	7	22	13	
	8	27	3	1	22	6	22	11	
	9	28	3	1	22	5	22	11	
49	1	3	5	3	23	26	23	26	
	2	9	4	2	21	22	21	24	
	3	10	4	2	20	20	20	24	
	4	11	4	2	16	19	19	24	
	5	12	4	1	16	16	19	22	
	6	13	4	1	14	13	19	22	
	7	14	4	1	12	11	18	22	
	8	27	4	1	10	10	17	21	
	9	29	4	1	8	7	16	20	
50	1	1	5	2	20	27	12	20	
	2	2	5	2	20	23	11	18	
	3	14	5	2	19	22	11	17	
	4	15	5	2	19	19	11	15	
	5	18	5	2	17	15	11	11	
	6	19	5	2	17	13	11	10	
	7	20	5	2	16	12	11	9	
	8	21	5	2	15	9	11	5	
	9	22	5	2	15	6	11	4	
51	1	2	4	4	24	21	18	16	
	2	3	4	4	24	19	17	16	
	3	5	4	4	24	17	16	15	
	4	17	3	3	23	14	14	13	
	5	18	3	2	23	11	12	13	
	6	21	3	2	23	9	11	12	
	7	25	2	2	22	7	11	11	
	8	28	2	1	22	7	9	10	
	9	29	2	1	22	4	7	9	
52	1	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2	N 3	N 4
	23	22	841	818	738	774

************************************************************************
