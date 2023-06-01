jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 4 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	1		2 
2	9	2		4 3 
3	9	7		14 13 12 10 9 8 7 
4	9	4		14 13 9 5 
5	9	3		11 8 6 
6	9	4		21 18 17 12 
7	9	4		21 20 18 11 
8	9	6		24 21 20 19 18 15 
9	9	6		24 21 20 19 18 15 
10	9	5		28 18 17 16 15 
11	9	5		28 25 24 22 16 
12	9	5		28 25 24 22 16 
13	9	5		28 24 22 20 16 
14	9	5		28 25 23 22 19 
15	9	4		27 26 25 23 
16	9	2		23 19 
17	9	5		36 31 27 26 25 
18	9	3		32 27 22 
19	9	5		36 32 31 27 26 
20	9	2		26 23 
21	9	4		36 28 26 25 
22	9	4		36 33 31 26 
23	9	5		36 33 32 31 29 
24	9	3		36 32 26 
25	9	4		37 32 30 29 
26	9	3		37 30 29 
27	9	5		40 37 35 33 30 
28	9	2		37 29 
29	9	4		43 38 35 34 
30	9	7		51 50 44 43 42 41 38 
31	9	5		44 43 42 39 37 
32	9	6		51 50 44 43 42 39 
33	9	6		50 44 43 42 41 38 
34	9	5		51 45 44 40 39 
35	9	5		51 50 48 42 39 
36	9	4		51 45 44 39 
37	9	4		51 50 49 46 
38	9	3		48 46 45 
39	9	2		46 41 
40	9	3		50 48 46 
41	9	2		49 47 
42	9	2		49 46 
43	9	2		48 47 
44	9	1		48 
45	9	1		47 
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
2	1	8	27	19	1	28	10	17	
	2	9	24	19	1	25	9	15	
	3	11	24	18	1	24	7	15	
	4	12	20	18	1	23	6	13	
	5	19	20	16	1	19	5	11	
	6	25	17	16	1	18	4	9	
	7	26	15	15	1	16	4	9	
	8	27	13	14	1	14	2	6	
	9	28	11	14	1	14	2	5	
3	1	2	24	20	29	25	13	28	
	2	3	20	19	29	25	13	28	
	3	4	19	18	28	23	13	28	
	4	16	18	17	27	22	13	28	
	5	21	13	17	26	21	13	27	
	6	23	11	16	26	20	13	27	
	7	25	10	15	24	20	13	27	
	8	26	9	13	23	19	13	26	
	9	28	5	12	23	18	13	26	
4	1	2	22	15	17	23	29	10	
	2	3	19	14	16	22	28	9	
	3	4	19	14	15	20	27	9	
	4	11	17	13	14	19	25	7	
	5	13	15	12	13	19	24	5	
	6	18	13	12	12	18	23	4	
	7	25	12	11	12	18	20	4	
	8	26	9	11	11	16	19	2	
	9	27	8	10	9	16	18	2	
5	1	2	19	28	27	30	30	21	
	2	9	17	27	26	27	30	19	
	3	14	13	26	25	26	30	17	
	4	15	12	23	24	22	30	16	
	5	17	11	20	24	21	30	15	
	6	19	7	18	22	19	30	13	
	7	21	7	17	21	18	30	11	
	8	22	5	14	21	14	30	9	
	9	24	2	12	19	13	30	8	
6	1	3	23	24	20	22	13	7	
	2	7	19	22	20	21	12	6	
	3	8	19	21	18	21	11	6	
	4	9	14	20	17	21	9	6	
	5	10	13	19	16	21	7	6	
	6	12	12	17	15	20	7	5	
	7	19	8	17	14	20	4	5	
	8	21	5	15	13	20	4	5	
	9	29	5	14	13	20	2	5	
7	1	2	19	27	11	22	12	9	
	2	6	18	25	9	19	11	9	
	3	15	17	23	9	19	9	8	
	4	16	16	22	7	17	9	8	
	5	17	15	19	7	17	7	8	
	6	27	15	18	5	15	7	7	
	7	28	15	16	4	14	5	7	
	8	29	14	16	3	13	5	6	
	9	30	13	13	1	13	4	6	
8	1	8	8	13	29	14	25	20	
	2	9	8	13	27	12	20	19	
	3	10	7	13	27	12	20	18	
	4	14	7	12	26	9	18	17	
	5	16	5	11	25	9	14	17	
	6	19	5	11	24	6	11	16	
	7	20	4	10	23	6	11	15	
	8	28	3	10	22	4	8	14	
	9	30	3	10	21	3	6	13	
9	1	2	21	12	27	16	26	17	
	2	5	20	10	23	16	26	17	
	3	7	18	9	21	16	26	13	
	4	10	18	8	19	16	25	12	
	5	14	16	5	14	16	25	10	
	6	15	15	4	14	16	24	9	
	7	19	15	3	9	16	24	6	
	8	21	13	2	8	16	23	3	
	9	25	13	2	4	16	23	3	
10	1	3	16	9	16	20	16	28	
	2	4	16	8	14	20	15	27	
	3	6	14	8	14	17	15	27	
	4	8	10	8	14	16	14	27	
	5	10	9	7	12	13	14	27	
	6	20	8	7	12	13	14	27	
	7	26	4	7	11	11	14	27	
	8	29	4	6	10	9	13	27	
	9	30	2	6	10	8	13	27	
11	1	2	19	19	25	19	27	21	
	2	4	18	17	21	18	25	17	
	3	12	16	14	19	16	23	17	
	4	13	15	12	18	15	18	13	
	5	16	14	10	17	14	18	13	
	6	17	14	9	15	14	13	12	
	7	21	13	6	11	12	9	10	
	8	25	12	5	9	11	9	7	
	9	27	11	2	8	11	5	5	
12	1	2	26	13	17	4	3	28	
	2	3	25	11	17	3	2	25	
	3	4	25	11	16	3	2	24	
	4	11	24	10	16	2	2	23	
	5	12	23	10	14	2	2	21	
	6	13	22	9	14	2	2	19	
	7	16	20	8	14	2	2	18	
	8	21	20	7	13	1	2	16	
	9	22	19	7	12	1	2	16	
13	1	12	22	20	5	7	11	10	
	2	15	21	16	5	7	10	9	
	3	16	20	16	4	7	9	9	
	4	17	17	15	4	7	8	9	
	5	24	13	13	3	7	8	8	
	6	25	12	10	3	7	7	8	
	7	26	10	9	2	7	5	8	
	8	29	8	7	2	7	5	8	
	9	30	4	7	2	7	4	8	
14	1	4	25	26	24	15	22	27	
	2	5	23	26	23	15	21	24	
	3	11	21	25	22	14	20	23	
	4	12	19	25	21	14	20	21	
	5	20	17	24	21	13	19	17	
	6	21	16	24	21	13	18	15	
	7	25	15	23	20	12	17	13	
	8	26	12	23	19	12	16	12	
	9	28	10	23	19	12	16	9	
15	1	2	24	18	22	30	28	27	
	2	4	23	17	18	29	27	24	
	3	7	21	17	17	29	25	23	
	4	8	18	17	14	28	25	20	
	5	11	16	17	12	28	24	20	
	6	15	14	16	11	28	23	18	
	7	18	11	16	9	28	23	15	
	8	19	9	16	4	27	21	14	
	9	26	8	16	4	27	21	12	
16	1	6	30	16	12	21	21	15	
	2	7	27	13	11	18	17	15	
	3	10	24	12	10	16	16	14	
	4	17	23	10	9	14	14	13	
	5	23	21	10	7	13	12	12	
	6	24	18	7	6	13	11	11	
	7	25	17	5	4	11	7	9	
	8	26	13	5	3	8	6	8	
	9	28	13	3	3	7	4	7	
17	1	8	28	10	24	17	22	29	
	2	14	24	9	22	16	19	27	
	3	15	21	9	22	13	18	27	
	4	16	21	8	20	13	14	27	
	5	22	18	8	19	11	12	26	
	6	23	14	8	18	10	11	25	
	7	24	14	7	17	8	9	25	
	8	25	11	7	16	7	8	24	
	9	30	8	6	15	5	6	24	
18	1	6	22	21	15	14	21	24	
	2	7	21	19	15	14	19	21	
	3	8	21	18	13	13	18	19	
	4	9	21	17	10	13	17	17	
	5	12	21	15	8	12	16	14	
	6	18	21	14	7	12	13	12	
	7	19	21	12	6	11	12	9	
	8	21	21	11	4	11	12	6	
	9	26	21	11	2	10	10	4	
19	1	2	9	21	26	24	15	20	
	2	5	9	21	25	24	15	18	
	3	10	9	21	21	22	14	15	
	4	11	9	21	20	21	14	15	
	5	12	9	20	16	20	13	12	
	6	18	9	20	10	18	12	11	
	7	19	9	20	7	18	12	8	
	8	22	9	20	6	16	12	8	
	9	30	9	20	2	15	11	5	
20	1	2	28	18	20	21	24	29	
	2	6	25	17	20	18	24	28	
	3	7	24	15	19	18	23	25	
	4	15	23	14	19	15	22	24	
	5	20	22	11	18	13	22	23	
	6	23	20	10	18	10	21	21	
	7	24	19	8	17	6	21	19	
	8	28	17	6	16	4	20	18	
	9	29	16	4	16	3	20	16	
21	1	3	24	6	23	27	24	9	
	2	4	22	6	23	25	23	8	
	3	5	21	6	21	24	22	8	
	4	6	19	6	20	20	21	7	
	5	8	16	6	20	17	19	7	
	6	16	15	5	19	15	18	7	
	7	20	15	5	18	10	17	6	
	8	24	12	5	17	9	17	6	
	9	26	11	5	17	6	15	6	
22	1	9	9	22	20	20	17	21	
	2	14	8	21	19	19	14	19	
	3	15	7	21	17	18	12	17	
	4	21	7	20	17	18	12	16	
	5	22	6	20	16	17	10	14	
	6	24	5	19	16	16	9	11	
	7	27	3	19	14	15	6	9	
	8	28	2	18	14	14	6	7	
	9	29	2	17	13	14	4	6	
23	1	4	23	17	19	18	19	17	
	2	9	23	17	18	17	17	17	
	3	10	21	15	18	16	16	17	
	4	11	19	15	18	16	16	16	
	5	12	18	14	17	15	15	16	
	6	13	17	13	17	14	15	16	
	7	14	15	12	17	14	14	15	
	8	15	14	11	17	14	13	15	
	9	16	11	9	17	13	12	15	
24	1	7	11	25	21	8	29	18	
	2	9	11	24	20	7	28	15	
	3	12	10	23	20	6	27	14	
	4	13	10	22	20	5	26	12	
	5	15	9	21	20	4	26	10	
	6	16	8	20	19	4	25	7	
	7	19	8	19	19	4	24	5	
	8	21	6	18	19	2	24	5	
	9	27	6	16	19	2	23	3	
25	1	1	24	5	22	23	22	8	
	2	2	23	5	18	22	22	8	
	3	8	22	4	17	22	22	8	
	4	10	21	4	15	22	22	8	
	5	12	20	4	14	21	22	8	
	6	16	20	3	12	21	22	8	
	7	21	18	3	11	21	22	8	
	8	25	18	2	11	21	22	8	
	9	29	17	2	9	21	22	8	
26	1	1	18	1	27	27	25	29	
	2	6	18	1	26	26	22	28	
	3	8	16	1	26	25	19	27	
	4	10	15	1	26	25	15	26	
	5	17	15	1	26	24	15	25	
	6	24	14	1	25	22	11	25	
	7	25	13	1	25	22	7	23	
	8	26	12	1	25	20	5	22	
	9	29	12	1	25	20	5	21	
27	1	3	25	18	13	22	20	21	
	2	5	23	17	12	22	19	20	
	3	7	23	16	11	22	19	20	
	4	11	22	16	11	22	17	19	
	5	17	21	15	9	22	17	18	
	6	20	20	15	9	22	16	17	
	7	23	19	14	8	22	14	16	
	8	25	18	13	6	22	13	16	
	9	30	16	13	6	22	13	15	
28	1	2	22	14	14	26	18	26	
	2	16	21	13	13	23	16	22	
	3	18	21	11	12	23	12	21	
	4	19	20	10	12	22	11	20	
	5	21	19	9	11	20	10	18	
	6	23	19	8	10	18	6	18	
	7	24	18	8	10	18	6	15	
	8	27	18	6	10	16	3	14	
	9	29	18	6	9	16	1	13	
29	1	6	21	25	11	3	8	24	
	2	7	19	25	11	2	7	23	
	3	10	15	23	9	2	6	21	
	4	11	13	23	9	2	5	20	
	5	12	12	22	8	2	5	18	
	6	16	10	21	7	2	3	17	
	7	18	8	20	6	2	2	17	
	8	22	6	18	5	2	1	15	
	9	30	3	18	3	2	1	14	
30	1	2	3	7	22	26	21	27	
	2	3	2	7	22	24	20	26	
	3	7	2	7	18	23	18	26	
	4	8	2	7	17	20	18	25	
	5	13	2	7	12	18	16	25	
	6	26	2	7	11	18	16	24	
	7	27	2	7	7	14	14	24	
	8	28	2	7	4	14	13	23	
	9	30	2	7	2	12	12	23	
31	1	1	17	16	11	9	13	20	
	2	2	17	14	11	9	13	18	
	3	7	14	13	9	9	13	15	
	4	11	12	12	8	8	13	13	
	5	13	9	10	8	7	13	11	
	6	19	9	9	7	7	13	10	
	7	22	6	8	6	7	13	7	
	8	23	5	5	6	6	13	5	
	9	24	3	5	5	6	13	3	
32	1	1	10	17	28	20	19	9	
	2	2	10	15	25	17	18	9	
	3	4	8	15	20	14	17	8	
	4	5	7	14	18	11	16	7	
	5	6	7	14	15	11	13	7	
	6	9	6	13	14	8	13	6	
	7	13	6	13	9	5	11	6	
	8	21	4	12	8	4	8	5	
	9	28	4	11	3	2	8	5	
33	1	2	12	25	16	22	25	26	
	2	9	11	25	16	20	24	25	
	3	11	9	23	15	18	22	21	
	4	14	9	22	13	16	22	21	
	5	15	8	22	13	15	21	17	
	6	16	8	20	10	13	19	16	
	7	21	6	20	10	11	16	12	
	8	22	5	19	8	11	16	9	
	9	27	5	18	8	9	15	9	
34	1	2	12	14	23	13	13	29	
	2	7	11	13	23	13	12	27	
	3	8	11	11	22	13	11	25	
	4	13	11	10	21	13	10	24	
	5	16	11	10	20	12	8	23	
	6	24	10	9	18	12	7	22	
	7	28	10	8	16	11	6	19	
	8	29	10	8	16	11	6	18	
	9	30	10	7	15	11	5	17	
35	1	1	30	21	26	19	24	28	
	2	5	28	17	23	19	23	26	
	3	10	28	16	23	18	22	22	
	4	12	27	15	21	16	22	18	
	5	15	25	13	20	16	21	15	
	6	20	24	13	20	14	19	14	
	7	23	23	10	18	13	18	10	
	8	24	22	8	17	13	18	10	
	9	25	22	8	17	12	16	7	
36	1	3	13	28	28	20	8	24	
	2	4	12	25	24	18	7	23	
	3	9	12	24	24	18	7	22	
	4	10	11	23	19	16	6	20	
	5	12	10	20	16	16	5	20	
	6	13	10	20	14	16	4	18	
	7	19	10	17	11	14	3	17	
	8	23	9	14	6	14	2	17	
	9	30	9	14	6	13	2	16	
37	1	4	14	30	25	30	14	13	
	2	6	14	26	24	25	12	13	
	3	14	14	21	20	23	11	10	
	4	15	14	18	19	22	10	9	
	5	19	14	15	13	19	10	9	
	6	22	14	14	13	16	10	6	
	7	26	14	9	9	15	9	5	
	8	27	14	5	5	13	7	3	
	9	28	14	5	5	10	7	3	
38	1	1	30	22	28	23	27	9	
	2	5	26	21	27	20	25	9	
	3	9	25	20	27	19	25	9	
	4	10	25	17	25	18	25	9	
	5	11	23	14	25	15	24	9	
	6	18	21	14	23	14	23	8	
	7	27	18	11	23	13	23	8	
	8	28	17	9	21	12	22	8	
	9	29	15	6	21	9	22	8	
39	1	3	25	26	18	12	27	21	
	2	5	23	24	16	11	25	20	
	3	12	22	21	15	10	25	20	
	4	17	20	19	14	9	24	20	
	5	19	16	17	14	9	24	20	
	6	26	14	15	12	8	24	20	
	7	27	14	12	11	8	23	20	
	8	28	11	11	10	8	23	20	
	9	30	9	10	10	7	22	20	
40	1	1	20	11	26	22	20	17	
	2	2	17	10	25	18	19	17	
	3	9	13	9	25	17	18	17	
	4	13	12	7	25	15	17	17	
	5	16	10	7	25	14	16	16	
	6	19	8	6	24	10	14	16	
	7	20	5	6	24	9	14	15	
	8	23	3	4	24	7	13	15	
	9	25	1	4	24	6	11	15	
41	1	1	29	12	18	4	17	25	
	2	13	28	12	15	4	16	23	
	3	19	27	12	13	4	16	22	
	4	20	27	12	11	4	14	19	
	5	23	27	12	11	4	13	15	
	6	27	26	11	8	4	11	15	
	7	28	26	11	6	4	10	10	
	8	29	25	11	5	4	7	9	
	9	30	25	11	3	4	7	6	
42	1	5	12	21	26	14	20	7	
	2	8	12	17	22	14	16	7	
	3	11	12	16	18	14	16	7	
	4	12	12	12	18	14	14	7	
	5	13	12	10	16	14	11	6	
	6	16	12	8	13	14	11	6	
	7	17	12	6	9	14	9	6	
	8	20	12	3	8	14	6	6	
	9	21	12	2	5	14	4	6	
43	1	1	16	25	12	29	12	27	
	2	2	15	24	11	29	12	26	
	3	3	15	23	10	29	12	22	
	4	4	15	22	8	29	12	20	
	5	16	14	22	7	28	12	14	
	6	20	13	21	6	28	12	10	
	7	21	13	21	5	28	12	7	
	8	25	12	20	4	28	12	6	
	9	26	11	20	2	28	12	2	
44	1	6	15	27	23	19	25	9	
	2	7	15	27	21	19	23	8	
	3	11	15	26	21	19	23	8	
	4	13	15	24	18	18	18	8	
	5	15	15	24	17	18	17	8	
	6	23	15	23	16	17	15	8	
	7	24	15	21	15	17	13	8	
	8	29	15	20	13	16	8	8	
	9	30	15	19	11	16	8	8	
45	1	2	27	11	16	9	29	24	
	2	3	25	11	15	9	29	22	
	3	11	23	11	13	8	29	21	
	4	13	21	11	11	8	29	19	
	5	14	20	11	9	8	28	17	
	6	15	18	11	8	7	28	17	
	7	16	15	11	7	7	28	15	
	8	28	15	11	4	6	28	13	
	9	29	13	11	3	6	28	11	
46	1	5	7	26	21	26	9	2	
	2	7	6	23	19	26	8	2	
	3	14	6	21	18	24	8	2	
	4	16	6	19	18	24	7	2	
	5	19	6	19	16	23	7	2	
	6	20	5	16	16	22	6	1	
	7	21	5	14	15	21	5	1	
	8	23	5	13	14	19	5	1	
	9	24	5	9	13	18	5	1	
47	1	2	4	20	26	11	20	15	
	2	4	4	20	26	11	18	13	
	3	9	4	18	22	10	18	13	
	4	11	4	17	22	10	18	13	
	5	12	4	16	20	9	17	12	
	6	14	4	14	18	9	16	12	
	7	15	4	13	15	9	15	11	
	8	16	4	12	13	8	14	11	
	9	23	4	11	11	8	14	10	
48	1	5	16	7	8	18	20	25	
	2	6	15	5	7	15	20	24	
	3	14	15	5	7	14	18	22	
	4	15	13	5	7	12	17	21	
	5	19	11	3	5	12	17	19	
	6	20	11	3	5	10	16	18	
	7	25	9	2	4	7	15	15	
	8	28	7	1	4	7	15	14	
	9	29	6	1	3	6	14	13	
49	1	7	20	29	16	24	12	26	
	2	12	16	26	15	23	11	26	
	3	13	15	23	15	23	11	25	
	4	17	14	22	13	20	9	24	
	5	18	12	18	12	19	8	24	
	6	23	11	14	10	18	6	24	
	7	27	10	14	8	15	6	23	
	8	28	8	9	7	14	4	22	
	9	29	6	8	7	13	3	22	
50	1	5	29	10	25	26	14	23	
	2	6	29	9	25	25	14	23	
	3	15	28	7	25	23	12	21	
	4	16	28	6	25	22	11	21	
	5	18	28	5	25	18	11	20	
	6	19	27	5	24	17	11	19	
	7	21	26	3	24	15	10	17	
	8	27	26	3	24	13	8	17	
	9	30	26	2	24	12	8	16	
51	1	5	23	16	4	22	27	14	
	2	6	20	15	4	18	25	13	
	3	11	19	15	4	17	21	12	
	4	14	18	13	4	17	21	11	
	5	20	17	13	4	15	17	11	
	6	21	16	12	4	13	15	10	
	7	22	14	10	4	11	11	10	
	8	27	13	9	4	11	11	10	
	9	28	10	9	4	8	8	9	
52	1	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2	N 3	N 4
	86	79	863	864	862	868

************************************************************************
