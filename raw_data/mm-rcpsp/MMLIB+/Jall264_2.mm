jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 4 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	7		2 3 4 6 7 8 15 
2	9	3		13 10 5 
3	9	4		17 12 11 10 
4	9	2		10 5 
5	9	3		17 14 9 
6	9	3		17 14 9 
7	9	3		18 17 14 
8	9	1		9 
9	9	3		22 18 16 
10	9	3		22 18 16 
11	9	2		18 14 
12	9	2		22 14 
13	9	4		29 22 21 16 
14	9	3		21 20 16 
15	9	3		29 21 16 
16	9	2		25 19 
17	9	6		34 26 25 24 23 21 
18	9	3		28 21 20 
19	9	7		34 30 28 27 26 24 23 
20	9	6		38 34 32 29 27 25 
21	9	5		38 32 31 30 27 
22	9	5		38 32 31 30 27 
23	9	6		44 40 38 37 33 31 
24	9	5		44 40 38 37 31 
25	9	4		47 40 31 30 
26	9	6		44 40 38 36 35 33 
27	9	5		44 37 36 35 33 
28	9	4		44 36 33 32 
29	9	3		44 40 31 
30	9	4		44 36 35 33 
31	9	3		39 36 35 
32	9	5		51 47 43 41 40 
33	9	3		43 41 39 
34	9	4		51 44 43 40 
35	9	3		51 43 41 
36	9	4		51 46 45 43 
37	9	3		51 47 42 
38	9	2		43 42 
39	9	4		51 48 46 45 
40	9	4		50 48 46 45 
41	9	1		42 
42	9	3		48 46 45 
43	9	2		50 48 
44	9	2		49 48 
45	9	1		49 
46	9	1		49 
47	9	1		48 
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
2	1	3	26	22	26	28	10	24	
	2	16	26	22	23	28	9	24	
	3	17	26	20	21	26	8	23	
	4	22	26	18	21	25	6	23	
	5	23	26	15	19	25	5	22	
	6	24	26	13	18	24	5	22	
	7	25	26	12	14	23	4	21	
	8	26	26	10	12	22	3	21	
	9	29	26	7	11	21	2	21	
3	1	3	27	20	24	14	16	21	
	2	7	25	20	23	13	15	21	
	3	8	23	20	23	12	15	20	
	4	9	22	20	23	9	15	17	
	5	20	17	20	22	8	15	16	
	6	23	16	20	21	7	14	15	
	7	26	14	20	20	7	14	14	
	8	27	11	20	19	5	14	11	
	9	28	10	20	19	3	14	10	
4	1	4	11	19	18	14	25	16	
	2	5	10	17	16	13	25	15	
	3	6	10	16	16	13	23	15	
	4	9	9	14	15	13	22	15	
	5	10	9	12	15	13	21	15	
	6	24	8	11	15	12	20	15	
	7	28	8	10	14	12	19	15	
	8	29	7	7	13	12	17	15	
	9	30	7	7	13	12	17	14	
5	1	2	22	18	5	24	15	20	
	2	3	21	17	5	23	14	20	
	3	7	19	16	5	22	13	20	
	4	9	15	14	5	21	13	20	
	5	10	13	10	5	21	12	20	
	6	11	13	8	5	21	12	20	
	7	13	10	6	5	20	11	20	
	8	15	7	4	5	20	11	20	
	9	20	7	3	5	19	10	20	
6	1	5	27	18	27	28	11	18	
	2	6	26	17	27	24	10	16	
	3	8	25	16	23	24	10	16	
	4	14	24	16	22	22	9	15	
	5	16	21	16	17	19	9	15	
	6	17	19	15	15	18	8	15	
	7	20	18	14	14	15	8	14	
	8	21	17	14	12	14	7	13	
	9	22	15	14	8	11	7	13	
7	1	1	11	24	30	7	25	23	
	2	11	10	23	30	7	22	21	
	3	12	10	21	30	5	19	21	
	4	13	9	20	30	5	19	19	
	5	19	9	20	30	4	17	15	
	6	20	9	19	30	4	14	14	
	7	24	8	18	30	2	11	11	
	8	26	8	17	30	2	9	9	
	9	27	8	16	30	1	9	7	
8	1	2	23	15	27	7	30	25	
	2	3	21	15	25	7	25	23	
	3	13	20	14	23	6	24	21	
	4	14	19	14	22	5	22	18	
	5	15	19	13	19	5	17	18	
	6	18	19	13	17	4	17	17	
	7	20	18	12	16	4	15	14	
	8	24	17	12	13	3	12	13	
	9	26	16	12	12	2	10	11	
9	1	3	22	28	13	23	6	27	
	2	14	20	27	12	22	5	26	
	3	15	16	27	11	22	5	24	
	4	19	15	26	11	20	4	24	
	5	20	13	26	9	20	4	22	
	6	21	12	26	9	19	3	22	
	7	26	10	26	7	18	3	21	
	8	28	7	25	6	16	2	20	
	9	29	7	25	6	16	2	19	
10	1	5	27	21	14	28	17	20	
	2	6	26	20	13	27	17	18	
	3	14	26	18	11	26	16	17	
	4	16	26	18	11	26	15	17	
	5	19	25	17	10	25	15	14	
	6	21	25	16	10	24	15	13	
	7	26	25	14	8	24	14	13	
	8	28	24	14	8	24	14	11	
	9	29	24	13	7	23	13	10	
11	1	1	18	21	29	21	13	12	
	2	2	16	20	25	19	12	12	
	3	3	14	19	23	19	12	12	
	4	4	11	19	21	18	12	12	
	5	7	9	18	20	18	11	11	
	6	8	6	17	17	18	11	11	
	7	19	6	16	15	17	10	10	
	8	24	3	15	13	16	10	10	
	9	30	2	15	10	16	10	10	
12	1	2	7	7	26	2	17	28	
	2	6	7	5	22	2	17	24	
	3	11	7	5	19	2	16	22	
	4	12	7	4	17	2	15	18	
	5	13	7	3	15	2	15	16	
	6	21	7	3	11	2	15	14	
	7	22	7	2	10	2	14	11	
	8	23	7	1	5	2	14	10	
	9	25	7	1	3	2	13	8	
13	1	1	21	8	12	25	29	26	
	2	9	19	8	11	22	29	24	
	3	10	17	8	10	22	28	24	
	4	11	15	8	9	18	26	22	
	5	12	15	8	6	17	25	22	
	6	13	13	8	6	15	24	21	
	7	16	11	8	5	11	23	20	
	8	21	9	8	4	8	22	18	
	9	23	9	8	3	6	22	18	
14	1	2	28	28	21	30	10	26	
	2	3	26	23	19	25	10	23	
	3	4	26	22	18	24	10	21	
	4	5	26	20	15	21	10	17	
	5	6	25	17	15	21	10	15	
	6	10	24	15	14	19	10	14	
	7	19	23	11	11	15	10	8	
	8	20	23	9	9	15	10	8	
	9	21	22	6	7	12	10	5	
15	1	1	15	13	29	12	20	24	
	2	2	14	13	26	12	20	23	
	3	3	13	13	23	11	19	23	
	4	5	11	13	19	10	19	22	
	5	6	10	13	18	9	18	22	
	6	8	9	13	15	9	17	21	
	7	9	9	13	10	8	17	20	
	8	11	7	13	8	8	16	20	
	9	27	6	13	6	7	15	19	
16	1	3	19	14	23	25	23	18	
	2	5	18	12	22	24	23	16	
	3	8	18	10	22	23	21	14	
	4	12	17	9	22	20	20	14	
	5	17	15	9	21	20	20	11	
	6	20	13	8	21	18	17	10	
	7	21	12	7	21	17	17	6	
	8	26	10	6	21	15	16	6	
	9	27	10	5	21	13	14	4	
17	1	1	16	25	6	21	13	24	
	2	19	15	24	5	21	11	23	
	3	21	13	24	5	18	11	22	
	4	23	11	23	5	15	10	21	
	5	25	8	22	5	15	9	21	
	6	26	8	22	4	12	9	20	
	7	27	7	22	4	11	8	19	
	8	28	5	21	4	9	7	17	
	9	29	3	21	4	7	7	16	
18	1	4	20	27	23	20	8	2	
	2	5	18	23	18	19	7	2	
	3	18	15	21	16	16	7	2	
	4	20	14	19	15	15	7	2	
	5	21	13	15	11	14	7	2	
	6	24	10	14	9	11	7	2	
	7	25	9	11	6	11	7	2	
	8	29	9	7	5	9	7	2	
	9	30	6	7	1	6	7	2	
19	1	4	8	23	16	26	17	8	
	2	5	7	23	16	24	15	6	
	3	7	5	22	15	19	14	5	
	4	8	5	22	13	16	11	5	
	5	11	4	22	13	12	9	4	
	6	13	4	21	12	10	6	4	
	7	18	2	20	10	6	5	2	
	8	21	2	20	9	6	4	1	
	9	27	1	20	9	3	2	1	
20	1	6	25	28	20	7	24	26	
	2	9	25	28	19	7	22	26	
	3	13	24	28	17	7	22	25	
	4	15	23	28	16	7	21	25	
	5	17	23	27	15	7	20	25	
	6	18	22	27	13	7	20	24	
	7	20	22	27	10	7	18	24	
	8	21	21	27	8	7	17	23	
	9	30	20	27	7	7	17	23	
21	1	7	13	27	11	18	26	30	
	2	8	11	25	11	15	23	29	
	3	11	10	24	11	15	20	29	
	4	13	9	21	11	14	19	29	
	5	14	7	19	11	13	16	29	
	6	15	6	16	11	12	14	28	
	7	19	4	14	11	11	12	28	
	8	28	4	9	11	10	12	28	
	9	29	1	7	11	8	10	28	
22	1	4	24	21	21	24	10	22	
	2	5	24	21	20	20	9	22	
	3	9	24	21	16	18	9	22	
	4	10	24	20	14	17	9	22	
	5	11	24	20	13	15	8	22	
	6	26	24	19	9	10	8	22	
	7	27	24	19	9	10	8	21	
	8	28	24	18	5	6	7	22	
	9	29	24	18	4	5	7	22	
23	1	1	14	3	14	13	29	14	
	2	5	13	3	12	13	24	13	
	3	6	13	3	11	12	24	11	
	4	8	12	3	11	11	20	11	
	5	15	12	3	8	10	19	9	
	6	16	11	3	7	10	15	8	
	7	21	10	3	6	9	14	5	
	8	25	9	3	5	8	10	4	
	9	29	9	3	5	8	8	3	
24	1	3	16	9	12	30	14	23	
	2	4	15	8	11	26	11	21	
	3	5	13	7	10	25	10	18	
	4	8	11	7	9	24	9	17	
	5	10	11	6	9	21	8	14	
	6	20	9	6	9	20	7	12	
	7	22	7	5	8	19	4	9	
	8	26	7	4	7	18	3	8	
	9	29	6	4	6	15	3	7	
25	1	2	13	27	21	9	17	2	
	2	3	13	25	20	7	17	2	
	3	10	13	23	17	7	17	2	
	4	13	13	22	16	7	17	2	
	5	14	13	22	14	6	17	2	
	6	17	13	20	12	5	17	2	
	7	20	13	18	10	4	17	2	
	8	23	13	17	7	4	17	2	
	9	27	13	16	6	3	17	2	
26	1	6	27	21	23	22	12	20	
	2	12	25	21	22	20	11	18	
	3	13	25	19	20	16	11	18	
	4	14	23	18	20	15	10	17	
	5	15	21	17	19	14	9	15	
	6	16	20	17	17	12	9	15	
	7	21	17	15	16	8	9	13	
	8	23	16	14	16	6	7	12	
	9	24	15	14	15	6	7	12	
27	1	2	8	13	18	13	26	20	
	2	4	7	12	16	12	23	19	
	3	7	7	12	14	11	21	19	
	4	9	7	11	11	9	20	17	
	5	10	6	10	9	7	18	16	
	6	20	6	10	8	6	17	15	
	7	21	5	9	7	5	15	14	
	8	26	5	8	5	2	13	14	
	9	27	5	8	3	1	12	12	
28	1	3	11	9	10	29	17	30	
	2	7	10	7	9	28	17	26	
	3	8	9	7	9	28	16	23	
	4	15	9	6	9	28	16	21	
	5	21	7	6	9	27	16	18	
	6	22	6	6	9	26	15	17	
	7	23	5	5	9	26	14	15	
	8	24	4	4	9	25	14	12	
	9	25	4	4	9	25	14	9	
29	1	3	22	25	16	23	25	21	
	2	12	21	23	14	23	25	21	
	3	17	21	23	13	18	21	21	
	4	21	21	22	13	15	16	21	
	5	23	20	21	11	13	13	21	
	6	25	20	21	10	10	11	20	
	7	26	20	20	9	9	8	20	
	8	29	20	19	8	7	5	20	
	9	30	20	19	8	2	3	20	
30	1	1	18	25	23	28	24	29	
	2	9	16	25	23	27	22	28	
	3	10	15	22	23	27	19	27	
	4	12	13	19	22	27	17	26	
	5	14	13	17	22	26	17	25	
	6	16	10	17	22	26	15	25	
	7	17	9	13	21	25	13	25	
	8	18	9	11	21	25	11	24	
	9	30	6	11	21	25	9	23	
31	1	7	24	24	18	10	15	24	
	2	11	23	22	17	8	15	24	
	3	12	23	22	17	7	13	21	
	4	15	23	21	17	6	12	18	
	5	16	23	21	16	6	12	16	
	6	21	23	20	15	4	11	13	
	7	23	23	19	15	4	11	13	
	8	25	23	18	15	3	9	9	
	9	28	23	18	14	2	9	8	
32	1	3	28	23	19	19	12	15	
	2	5	27	20	18	17	12	14	
	3	8	26	18	18	15	12	13	
	4	10	25	15	16	15	12	13	
	5	11	23	12	15	14	12	12	
	6	23	21	9	13	12	12	10	
	7	27	19	8	12	12	12	9	
	8	28	19	6	11	11	12	9	
	9	30	17	4	10	9	12	8	
33	1	5	24	25	24	13	23	17	
	2	9	22	22	23	13	22	15	
	3	10	20	21	21	13	21	15	
	4	11	18	19	21	13	19	14	
	5	12	18	19	19	13	19	14	
	6	13	15	17	17	13	17	13	
	7	27	14	16	14	13	16	13	
	8	28	12	14	14	13	15	11	
	9	30	11	13	11	13	15	11	
34	1	11	26	16	19	23	27	12	
	2	12	26	12	18	22	27	12	
	3	13	25	11	16	19	27	11	
	4	20	24	10	14	19	27	9	
	5	21	23	8	13	16	27	8	
	6	22	22	7	9	14	26	6	
	7	24	20	4	9	12	26	5	
	8	26	19	2	5	12	26	3	
	9	29	19	1	3	9	26	1	
35	1	4	30	23	18	25	7	27	
	2	6	28	21	17	23	7	27	
	3	9	28	20	17	17	6	26	
	4	12	26	19	16	17	5	24	
	5	13	26	19	15	12	5	24	
	6	19	24	18	15	10	4	23	
	7	21	23	17	15	6	4	23	
	8	24	23	16	13	3	3	21	
	9	29	22	15	13	1	3	21	
36	1	5	11	26	27	2	19	16	
	2	7	11	25	27	2	17	14	
	3	8	10	24	23	2	15	13	
	4	9	8	24	21	2	12	13	
	5	12	7	24	20	2	12	12	
	6	16	7	23	16	2	7	10	
	7	17	5	22	14	2	7	9	
	8	20	5	22	12	2	4	7	
	9	27	4	22	10	2	1	7	
37	1	2	15	14	23	11	25	25	
	2	3	14	13	21	10	25	23	
	3	6	13	13	19	9	21	18	
	4	8	12	13	16	9	21	17	
	5	20	11	12	13	8	19	14	
	6	21	10	12	12	8	15	12	
	7	25	9	11	9	7	12	8	
	8	26	9	11	7	7	11	8	
	9	28	8	11	5	7	9	4	
38	1	1	25	14	25	25	21	30	
	2	3	23	13	23	22	18	27	
	3	4	23	12	23	21	16	26	
	4	15	23	11	20	20	15	25	
	5	17	22	11	19	17	12	24	
	6	19	21	11	17	16	11	22	
	7	25	21	10	17	14	9	19	
	8	26	20	8	15	13	6	18	
	9	30	20	8	13	12	5	17	
39	1	3	24	13	13	29	12	28	
	2	4	24	13	12	25	12	26	
	3	6	23	13	11	23	11	24	
	4	8	22	13	8	19	10	24	
	5	10	22	12	8	17	8	22	
	6	13	21	12	6	11	8	21	
	7	14	21	12	5	11	7	19	
	8	15	21	12	3	5	6	17	
	9	21	20	12	2	4	5	17	
40	1	5	26	29	26	21	13	10	
	2	11	25	27	26	21	12	9	
	3	17	23	26	26	21	12	9	
	4	21	23	25	26	20	12	8	
	5	22	20	25	26	20	12	8	
	6	24	20	24	26	20	12	8	
	7	27	18	23	26	20	12	7	
	8	28	17	21	26	19	12	7	
	9	29	15	21	26	19	12	7	
41	1	1	25	27	29	21	3	30	
	2	5	24	26	29	21	3	29	
	3	15	24	22	28	20	3	29	
	4	16	23	22	26	19	3	28	
	5	18	22	18	26	17	2	27	
	6	25	22	17	25	16	2	27	
	7	27	20	16	23	15	2	26	
	8	28	19	14	23	14	1	26	
	9	30	19	10	22	14	1	26	
42	1	1	10	17	29	19	21	19	
	2	15	9	15	25	18	17	17	
	3	16	9	12	23	18	17	14	
	4	18	9	10	19	18	15	13	
	5	22	8	9	17	18	12	13	
	6	23	8	7	13	18	11	10	
	7	25	8	5	12	18	9	8	
	8	27	8	2	8	18	6	7	
	9	30	8	2	7	18	3	7	
43	1	3	26	7	13	29	22	27	
	2	11	25	6	12	27	20	27	
	3	13	24	6	11	25	20	26	
	4	14	22	6	10	24	20	24	
	5	17	22	5	10	23	19	24	
	6	20	20	5	8	21	19	23	
	7	25	20	4	7	20	18	22	
	8	26	18	4	6	19	17	20	
	9	28	18	4	6	19	17	19	
44	1	5	8	24	24	11	23	23	
	2	10	8	22	24	10	23	21	
	3	18	6	21	24	10	23	21	
	4	19	6	17	24	10	23	18	
	5	20	5	17	24	10	22	18	
	6	21	5	13	24	9	22	15	
	7	22	4	12	24	9	22	14	
	8	29	2	9	24	9	22	12	
	9	30	2	7	24	9	22	12	
45	1	5	28	14	11	15	25	20	
	2	7	28	14	8	13	21	20	
	3	9	27	13	8	11	21	20	
	4	16	27	12	6	11	18	20	
	5	17	26	11	5	9	14	20	
	6	18	26	11	4	8	12	20	
	7	19	25	11	3	8	11	20	
	8	20	24	9	2	6	7	20	
	9	22	24	9	2	6	5	20	
46	1	5	19	25	26	25	21	26	
	2	6	15	22	26	24	19	24	
	3	7	13	20	26	23	17	20	
	4	10	11	18	25	23	15	18	
	5	11	10	15	25	22	13	14	
	6	14	8	14	24	21	10	12	
	7	20	8	11	24	21	7	9	
	8	23	6	8	23	20	5	8	
	9	28	3	6	23	20	4	5	
47	1	9	22	11	24	22	4	21	
	2	10	22	10	22	21	4	20	
	3	17	22	10	20	20	4	18	
	4	19	22	10	18	19	4	18	
	5	21	22	10	15	18	4	17	
	6	22	22	10	15	16	4	16	
	7	23	22	10	12	16	4	15	
	8	26	22	10	10	15	4	14	
	9	29	22	10	10	14	4	13	
48	1	9	29	19	22	8	22	16	
	2	12	27	17	22	7	20	16	
	3	14	27	16	20	6	17	15	
	4	16	27	15	19	6	16	15	
	5	21	25	14	18	4	15	15	
	6	23	25	11	17	4	14	14	
	7	24	24	9	17	4	12	14	
	8	27	24	9	16	2	10	13	
	9	30	23	7	14	2	9	13	
49	1	3	13	30	20	15	28	25	
	2	5	12	29	18	13	27	25	
	3	8	12	29	18	12	26	25	
	4	12	12	28	16	11	26	25	
	5	13	11	28	13	9	25	25	
	6	15	11	28	11	8	23	25	
	7	16	10	27	10	5	22	25	
	8	21	10	27	7	4	21	25	
	9	25	10	27	6	4	21	25	
50	1	1	25	30	10	22	25	13	
	2	4	20	26	9	21	24	13	
	3	5	19	23	8	19	24	11	
	4	7	18	21	6	16	23	11	
	5	11	14	21	6	15	20	10	
	6	15	10	18	6	12	20	7	
	7	23	10	16	4	11	19	7	
	8	24	4	13	4	8	16	6	
	9	25	2	12	3	7	16	4	
51	1	1	10	19	18	14	28	26	
	2	3	9	16	16	13	28	25	
	3	8	8	15	16	12	28	25	
	4	9	7	13	12	12	28	25	
	5	12	7	10	11	10	28	25	
	6	16	6	8	8	9	27	24	
	7	23	6	5	7	8	27	24	
	8	25	5	4	4	7	27	24	
	9	30	5	1	4	5	27	24	
52	1	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2	N 3	N 4
	58	55	871	832	823	946

************************************************************************
