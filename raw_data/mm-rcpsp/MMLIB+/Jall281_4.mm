jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 4 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	9		2 3 4 5 7 8 10 15 17 
2	9	4		22 16 13 6 
3	9	6		31 26 23 22 19 11 
4	9	5		26 25 23 14 11 
5	9	5		31 29 23 22 9 
6	9	4		31 25 19 12 
7	9	6		51 26 20 19 18 16 
8	9	10		39 34 33 31 30 29 24 21 20 18 
9	9	9		51 39 34 33 30 24 21 20 18 
10	9	6		51 50 34 31 25 16 
11	9	5		51 34 20 18 16 
12	9	8		51 39 34 33 30 21 20 18 
13	9	11		51 49 37 34 31 30 28 26 24 23 21 
14	9	10		51 50 49 39 36 34 31 28 24 21 
15	9	9		51 49 39 36 34 31 24 23 21 
16	9	8		49 39 37 36 33 30 28 21 
17	9	3		50 48 19 
18	9	6		49 48 37 32 28 27 
19	9	5		39 36 34 27 24 
20	9	6		50 49 46 36 32 27 
21	9	4		48 46 32 27 
22	9	8		50 49 46 45 44 43 40 35 
23	9	7		47 46 45 44 43 38 35 
24	9	4		47 46 44 32 
25	9	3		44 39 32 
26	9	5		48 44 39 38 35 
27	9	7		47 45 44 43 42 41 38 
28	9	7		46 45 44 43 42 41 38 
29	9	5		49 48 45 38 35 
30	9	5		47 44 43 40 35 
31	9	5		45 44 42 41 38 
32	9	3		43 38 35 
33	9	5		46 43 42 41 40 
34	9	2		40 35 
35	9	2		42 41 
36	9	2		43 40 
37	9	2		46 41 
38	9	1		40 
39	9	1		45 
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
jobnr.	mode	dur	R1	R2	R3	R4	N1	N2	
------------------------------------------------------------------------
1	1	0	0	0	0	0	0	0	
2	1	3	19	21	18	19	22	29	
	2	4	19	21	16	19	18	28	
	3	5	19	21	15	15	17	28	
	4	16	19	21	15	13	14	27	
	5	17	19	20	13	11	10	27	
	6	22	18	20	11	8	8	26	
	7	23	18	20	11	8	5	26	
	8	25	18	19	10	3	3	26	
	9	29	18	19	8	2	1	25	
3	1	9	7	10	25	23	18	22	
	2	10	7	8	22	22	17	20	
	3	21	6	7	22	21	14	17	
	4	23	6	7	19	20	13	16	
	5	26	5	6	19	19	11	15	
	6	27	5	5	18	19	10	12	
	7	28	4	4	17	17	9	10	
	8	29	4	2	15	17	7	8	
	9	30	3	1	13	16	6	5	
4	1	5	5	13	28	24	26	28	
	2	6	5	11	27	24	24	28	
	3	18	5	10	21	22	24	26	
	4	19	5	9	18	20	24	24	
	5	20	5	8	15	20	23	22	
	6	23	5	8	15	18	23	19	
	7	27	5	7	12	16	22	17	
	8	29	5	5	6	15	22	16	
	9	30	5	4	5	14	21	15	
5	1	3	5	26	21	28	30	21	
	2	8	5	20	19	28	26	19	
	3	10	5	19	19	28	25	17	
	4	19	5	14	18	28	20	16	
	5	20	5	14	14	28	20	14	
	6	21	5	12	14	28	17	14	
	7	22	5	7	13	28	15	12	
	8	27	5	4	11	28	12	10	
	9	28	5	2	9	28	10	9	
6	1	6	20	15	3	16	16	17	
	2	12	19	15	3	16	15	15	
	3	18	19	14	3	16	15	14	
	4	19	19	12	3	15	13	14	
	5	22	18	12	3	15	12	13	
	6	23	18	11	3	15	11	13	
	7	25	18	9	3	15	10	11	
	8	26	18	8	3	14	9	11	
	9	28	18	8	3	14	9	10	
7	1	1	15	25	30	21	25	27	
	2	2	15	25	27	19	25	23	
	3	9	13	25	27	17	22	20	
	4	12	12	24	26	15	22	20	
	5	17	11	24	25	11	18	18	
	6	19	11	24	22	9	17	14	
	7	24	10	23	21	9	15	14	
	8	26	8	23	20	7	15	12	
	9	27	8	23	20	3	12	8	
8	1	3	22	25	27	14	25	26	
	2	8	21	22	25	11	25	25	
	3	9	21	20	25	11	24	25	
	4	10	21	17	22	9	24	25	
	5	14	21	15	22	7	23	25	
	6	17	21	13	20	7	22	25	
	7	19	21	10	19	5	21	25	
	8	28	21	8	17	3	21	25	
	9	29	21	6	16	1	20	25	
9	1	3	27	25	15	21	27	4	
	2	5	23	25	15	18	24	3	
	3	15	23	24	14	17	23	3	
	4	16	21	23	13	15	22	3	
	5	18	17	23	11	12	21	3	
	6	21	16	22	10	10	19	3	
	7	23	14	21	9	8	18	3	
	8	27	12	20	9	7	16	3	
	9	28	10	20	8	5	16	3	
10	1	1	14	15	18	26	27	20	
	2	5	12	14	17	23	24	18	
	3	10	11	13	15	20	19	16	
	4	12	11	13	12	18	17	16	
	5	20	9	12	9	18	16	14	
	6	21	8	12	7	15	11	12	
	7	25	7	12	6	11	8	10	
	8	26	7	11	5	11	5	10	
	9	29	6	11	3	8	3	7	
11	1	1	17	15	19	24	27	24	
	2	2	16	15	15	21	25	24	
	3	3	15	15	14	20	22	23	
	4	4	13	15	13	19	20	23	
	5	6	12	14	12	17	18	23	
	6	7	10	14	10	17	18	22	
	7	9	7	14	8	15	15	22	
	8	23	6	14	7	14	15	21	
	9	25	5	14	5	12	12	21	
12	1	1	18	23	26	22	17	23	
	2	3	17	22	26	22	17	22	
	3	4	17	21	24	20	17	20	
	4	5	17	20	23	19	17	16	
	5	11	17	20	21	18	16	12	
	6	19	16	19	20	17	16	10	
	7	28	16	19	19	17	16	9	
	8	29	16	18	19	15	15	3	
	9	30	16	18	18	15	15	3	
13	1	2	22	27	14	17	6	20	
	2	6	21	27	13	16	5	18	
	3	8	21	27	10	16	5	15	
	4	9	21	27	10	15	5	15	
	5	13	20	26	8	13	4	13	
	6	14	20	26	7	12	4	12	
	7	21	19	26	6	11	4	11	
	8	23	19	26	3	11	4	9	
	9	28	19	26	2	10	4	7	
14	1	1	24	27	22	28	6	22	
	2	2	23	25	20	27	6	21	
	3	3	23	21	18	27	5	20	
	4	8	23	19	17	27	5	19	
	5	14	23	15	16	26	3	18	
	6	16	23	13	14	26	3	17	
	7	17	23	10	14	26	3	17	
	8	24	23	4	13	25	1	15	
	9	26	23	3	12	25	1	15	
15	1	4	26	29	21	24	23	8	
	2	9	26	26	19	23	19	7	
	3	10	25	24	18	20	19	6	
	4	11	23	19	16	15	15	6	
	5	13	23	19	15	13	13	6	
	6	14	21	15	14	13	11	5	
	7	23	20	14	13	10	10	5	
	8	29	19	12	12	7	9	4	
	9	30	19	10	10	4	5	4	
16	1	8	25	17	20	16	23	18	
	2	9	22	17	17	14	20	17	
	3	13	21	17	17	14	18	14	
	4	16	20	17	16	11	16	12	
	5	17	18	17	14	11	12	12	
	6	18	17	17	14	9	10	8	
	7	23	17	17	12	6	10	7	
	8	24	15	17	11	5	8	6	
	9	30	14	17	11	5	6	3	
17	1	1	27	25	9	13	24	4	
	2	4	26	24	9	13	23	4	
	3	12	23	22	9	11	23	4	
	4	18	23	21	9	10	23	4	
	5	20	19	21	8	10	22	3	
	6	25	17	20	8	8	22	3	
	7	26	16	19	8	7	22	2	
	8	27	13	19	8	5	21	2	
	9	28	13	18	8	5	21	2	
18	1	2	15	24	25	28	26	19	
	2	5	15	22	22	27	25	16	
	3	6	14	21	21	27	24	15	
	4	8	14	18	17	27	24	13	
	5	9	14	15	17	27	23	13	
	6	15	13	13	12	27	22	11	
	7	16	13	11	11	27	22	10	
	8	17	12	11	9	27	21	8	
	9	25	12	8	5	27	21	7	
19	1	2	6	19	9	19	8	20	
	2	8	6	17	7	15	8	18	
	3	10	6	17	7	15	8	15	
	4	21	6	16	5	11	8	12	
	5	23	5	15	5	11	8	10	
	6	26	5	13	3	8	8	10	
	7	27	5	13	2	5	8	6	
	8	28	5	11	1	3	8	3	
	9	30	5	11	1	3	8	2	
20	1	5	16	28	22	25	8	28	
	2	6	15	27	21	24	7	27	
	3	7	14	26	18	20	6	26	
	4	14	13	24	16	19	6	24	
	5	17	13	22	14	16	5	23	
	6	18	12	21	13	14	5	23	
	7	19	11	19	11	11	5	21	
	8	20	11	18	10	10	4	20	
	9	22	10	18	8	7	4	20	
21	1	9	11	12	8	14	25	13	
	2	10	9	11	7	14	24	13	
	3	11	9	11	6	13	24	12	
	4	12	8	11	4	13	24	12	
	5	18	8	10	4	12	24	11	
	6	20	7	10	4	12	24	11	
	7	27	7	10	2	11	24	10	
	8	29	7	10	1	11	24	9	
	9	30	6	10	1	10	24	9	
22	1	2	15	17	5	10	25	10	
	2	7	13	17	5	9	22	9	
	3	10	12	17	5	9	18	9	
	4	13	10	17	5	9	16	9	
	5	14	7	17	4	9	13	9	
	6	18	7	17	4	9	10	8	
	7	19	6	17	3	9	8	8	
	8	20	3	17	3	9	4	8	
	9	22	3	17	3	9	3	8	
23	1	3	10	24	5	19	11	26	
	2	7	9	24	5	17	10	23	
	3	10	8	23	4	15	9	21	
	4	13	8	21	4	13	9	20	
	5	14	8	21	3	11	8	17	
	6	19	7	19	3	10	7	17	
	7	20	7	19	2	9	7	13	
	8	22	6	17	2	5	6	13	
	9	28	6	17	2	4	5	10	
24	1	3	29	12	11	23	22	18	
	2	4	29	11	9	23	18	17	
	3	5	29	9	9	23	17	16	
	4	6	29	9	9	23	15	16	
	5	7	28	6	7	22	11	16	
	6	21	28	6	7	22	8	15	
	7	22	28	4	7	22	6	15	
	8	24	28	3	5	22	3	14	
	9	29	28	2	5	22	1	14	
25	1	1	28	7	28	8	21	30	
	2	2	28	7	22	8	19	28	
	3	7	28	7	22	8	16	25	
	4	8	27	7	19	8	15	22	
	5	9	27	7	16	8	14	22	
	6	11	26	7	10	8	12	21	
	7	14	26	7	8	8	10	18	
	8	18	25	7	5	8	9	17	
	9	27	25	7	3	8	6	15	
26	1	4	28	21	9	24	29	26	
	2	5	25	20	9	22	29	25	
	3	11	23	18	8	20	29	24	
	4	13	21	16	8	18	29	22	
	5	16	19	15	7	18	29	22	
	6	17	18	14	7	16	28	20	
	7	24	14	11	6	15	28	19	
	8	27	12	8	6	13	28	19	
	9	30	9	7	6	10	28	18	
27	1	6	24	10	25	29	27	10	
	2	7	21	10	24	27	23	10	
	3	11	18	9	22	27	19	10	
	4	13	16	7	21	26	16	10	
	5	15	15	6	21	24	15	10	
	6	20	12	5	20	24	11	10	
	7	28	10	5	20	23	11	10	
	8	29	6	3	18	21	9	10	
	9	30	5	2	18	21	6	10	
28	1	2	28	28	18	26	13	19	
	2	7	27	27	17	25	13	19	
	3	8	27	26	16	22	13	19	
	4	9	27	26	15	20	13	19	
	5	11	26	24	15	19	12	19	
	6	15	25	23	14	18	12	19	
	7	18	25	23	14	17	11	19	
	8	22	25	21	12	13	11	19	
	9	24	24	20	12	12	11	19	
29	1	4	19	5	29	18	3	6	
	2	6	17	4	28	17	3	5	
	3	7	16	4	28	17	3	5	
	4	8	14	4	28	17	3	5	
	5	13	14	3	27	16	2	4	
	6	15	12	3	27	16	2	4	
	7	22	11	3	27	16	1	4	
	8	29	11	3	27	15	1	4	
	9	30	10	3	27	15	1	4	
30	1	2	28	29	19	7	24	21	
	2	5	27	28	16	6	22	19	
	3	14	27	28	14	5	22	15	
	4	15	26	28	13	5	22	14	
	5	18	26	27	11	4	21	12	
	6	19	25	27	10	3	21	10	
	7	23	25	27	9	3	20	7	
	8	25	23	27	6	2	19	7	
	9	30	23	27	6	2	19	5	
31	1	4	26	27	24	26	18	24	
	2	9	22	26	21	26	17	21	
	3	12	20	23	21	26	15	18	
	4	16	19	22	18	26	14	17	
	5	21	17	19	16	26	12	13	
	6	25	13	16	13	26	12	11	
	7	26	13	16	11	26	11	7	
	8	27	10	14	11	26	10	4	
	9	30	8	12	8	26	8	2	
32	1	4	24	16	7	26	22	19	
	2	5	23	16	5	25	20	18	
	3	7	23	16	5	24	18	17	
	4	9	23	16	5	23	16	16	
	5	13	23	16	4	23	14	13	
	6	15	23	15	3	22	13	13	
	7	16	23	15	2	21	11	11	
	8	18	23	15	2	21	10	10	
	9	23	23	15	1	20	6	8	
33	1	4	2	19	22	14	12	23	
	2	5	2	18	20	14	12	21	
	3	6	2	17	19	12	12	21	
	4	9	2	17	15	12	12	19	
	5	10	2	17	13	10	12	17	
	6	13	2	16	12	9	12	16	
	7	15	2	15	10	8	12	16	
	8	24	2	15	7	6	12	13	
	9	29	2	15	7	6	12	12	
34	1	2	3	16	29	7	14	26	
	2	3	2	15	29	5	11	23	
	3	5	2	15	29	5	11	22	
	4	6	2	15	29	5	9	19	
	5	12	2	15	29	4	8	18	
	6	13	1	15	29	3	7	16	
	7	14	1	15	29	3	4	11	
	8	21	1	15	29	3	4	10	
	9	23	1	15	29	2	1	7	
35	1	2	20	28	21	26	13	17	
	2	4	19	28	21	23	13	16	
	3	9	18	27	21	23	11	15	
	4	11	17	27	21	20	10	14	
	5	14	14	26	21	19	9	13	
	6	15	12	26	21	18	9	13	
	7	19	11	26	21	15	7	13	
	8	27	11	25	21	14	6	12	
	9	28	8	25	21	13	5	11	
36	1	1	11	22	27	20	19	18	
	2	9	11	22	24	19	19	18	
	3	16	9	22	24	19	17	16	
	4	20	9	21	21	18	16	13	
	5	22	8	21	18	16	16	12	
	6	23	7	20	14	15	15	10	
	7	24	6	20	14	14	14	9	
	8	28	4	19	10	13	12	7	
	9	30	4	19	8	13	12	6	
37	1	3	10	15	11	20	7	26	
	2	7	9	14	10	17	7	20	
	3	8	9	14	10	17	7	19	
	4	9	9	13	10	15	7	18	
	5	10	9	12	10	13	7	14	
	6	11	8	10	10	11	7	10	
	7	19	8	9	10	11	7	8	
	8	22	8	9	10	9	7	6	
	9	25	8	7	10	8	7	3	
38	1	4	14	17	19	20	19	26	
	2	6	13	17	18	18	18	25	
	3	7	13	17	18	18	18	24	
	4	12	12	17	16	18	18	24	
	5	17	10	16	15	16	17	22	
	6	18	10	16	14	16	17	22	
	7	21	9	16	13	15	17	21	
	8	23	8	16	12	14	17	20	
	9	25	7	16	11	14	17	20	
39	1	3	17	27	18	16	11	29	
	2	4	17	25	17	16	11	24	
	3	6	16	24	17	16	11	24	
	4	8	16	23	16	16	11	21	
	5	16	14	23	15	16	10	20	
	6	17	14	22	15	16	10	18	
	7	22	14	20	14	16	10	15	
	8	23	13	19	12	16	10	14	
	9	29	12	19	12	16	10	10	
40	1	5	29	25	29	3	25	22	
	2	8	29	22	29	3	24	22	
	3	10	29	21	28	3	24	22	
	4	12	29	19	28	3	22	22	
	5	14	28	19	26	3	22	22	
	6	15	28	18	26	3	21	22	
	7	17	28	15	25	3	20	22	
	8	19	27	13	24	3	18	22	
	9	23	27	12	24	3	17	22	
41	1	2	15	27	18	13	19	19	
	2	4	13	26	16	12	17	18	
	3	12	12	24	13	11	17	14	
	4	14	10	22	13	10	17	14	
	5	17	10	18	11	8	16	11	
	6	21	8	17	8	8	15	8	
	7	22	8	16	6	6	15	7	
	8	23	6	12	4	5	14	5	
	9	30	5	12	4	4	14	2	
42	1	3	25	28	21	20	19	19	
	2	12	24	24	21	20	16	17	
	3	13	24	23	19	18	15	16	
	4	16	23	21	17	16	13	14	
	5	19	22	15	16	13	10	14	
	6	20	22	13	13	12	10	12	
	7	22	21	13	13	12	6	11	
	8	27	20	9	10	9	5	10	
	9	28	20	5	9	8	3	8	
43	1	7	24	20	14	20	12	26	
	2	8	24	19	13	19	11	26	
	3	13	24	19	13	18	9	23	
	4	16	24	17	12	17	8	20	
	5	24	24	17	12	17	7	19	
	6	27	23	16	12	17	5	18	
	7	28	23	16	11	16	3	15	
	8	29	23	14	11	16	3	13	
	9	30	23	14	11	15	2	10	
44	1	2	24	25	6	23	23	20	
	2	7	23	24	5	23	21	19	
	3	10	22	24	5	21	21	18	
	4	14	19	24	5	21	16	17	
	5	16	18	23	4	19	14	16	
	6	20	16	23	4	18	12	16	
	7	23	14	23	4	17	9	15	
	8	28	13	22	4	15	7	14	
	9	30	10	22	4	14	5	13	
45	1	2	27	11	6	20	7	20	
	2	3	22	10	6	18	6	19	
	3	7	21	10	6	17	6	18	
	4	8	17	10	6	15	6	17	
	5	9	15	9	6	12	6	16	
	6	12	11	8	6	11	6	16	
	7	19	10	8	6	9	6	14	
	8	20	6	6	6	8	6	13	
	9	23	3	6	6	6	6	13	
46	1	2	20	13	26	8	20	17	
	2	4	18	13	24	8	19	16	
	3	6	17	13	23	8	17	16	
	4	9	16	13	21	7	16	15	
	5	19	16	13	17	7	13	14	
	6	22	15	12	15	6	11	14	
	7	23	13	12	14	5	8	14	
	8	24	13	12	13	5	8	13	
	9	28	11	12	11	5	5	13	
47	1	1	22	16	28	27	23	11	
	2	2	22	14	23	27	23	11	
	3	3	20	12	22	27	20	11	
	4	4	17	10	20	27	18	11	
	5	6	14	10	15	27	16	11	
	6	16	12	8	13	26	16	11	
	7	19	9	7	9	26	14	11	
	8	28	7	6	6	26	12	11	
	9	29	3	4	5	26	10	11	
48	1	4	11	6	25	16	16	30	
	2	6	10	6	25	15	13	29	
	3	8	9	6	25	13	12	29	
	4	11	7	6	25	13	12	28	
	5	12	7	6	25	10	10	28	
	6	19	7	5	24	9	9	27	
	7	20	5	5	24	9	8	26	
	8	22	5	5	24	6	6	26	
	9	27	4	5	24	6	5	26	
49	1	5	6	22	30	6	26	28	
	2	7	5	19	28	5	25	27	
	3	11	5	17	28	5	23	27	
	4	16	4	17	26	5	20	25	
	5	18	4	14	25	5	20	25	
	6	19	3	14	25	5	17	24	
	7	24	3	12	24	5	16	23	
	8	28	1	11	22	5	14	22	
	9	29	1	8	22	5	14	21	
50	1	8	21	8	11	28	22	29	
	2	9	19	8	10	24	21	28	
	3	11	16	8	10	24	19	27	
	4	19	16	8	10	22	17	26	
	5	21	13	8	10	18	16	25	
	6	25	12	8	9	17	15	24	
	7	27	11	8	9	14	15	22	
	8	28	10	8	9	14	13	21	
	9	29	9	8	9	10	11	21	
51	1	2	18	20	14	14	26	10	
	2	3	16	19	14	13	26	9	
	3	4	16	16	14	13	25	9	
	4	5	13	14	14	13	25	9	
	5	7	11	13	14	13	23	8	
	6	25	10	12	14	13	23	8	
	7	28	8	11	14	13	22	7	
	8	29	7	9	14	13	21	7	
	9	30	6	6	14	13	21	6	
52	1	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	R 3	R 4	N 1	N 2
	59	69	71	69	723	783

************************************************************************
