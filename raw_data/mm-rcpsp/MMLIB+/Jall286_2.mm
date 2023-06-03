jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 4 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	11		2 3 4 5 6 8 9 11 13 14 16 
2	9	5		23 21 12 10 7 
3	9	8		28 27 26 23 21 20 15 10 
4	9	6		37 26 23 21 18 10 
5	9	10		37 36 33 32 31 29 27 26 25 18 
6	9	10		37 36 33 31 28 25 24 22 21 19 
7	9	7		36 33 31 26 20 19 18 
8	9	8		37 33 31 28 26 24 20 19 
9	9	8		37 36 33 29 28 24 22 20 
10	9	8		51 36 33 32 31 25 24 19 
11	9	7		42 32 28 25 24 23 19 
12	9	4		33 27 19 17 
13	9	7		51 36 33 32 31 28 19 
14	9	6		51 36 33 28 24 19 
15	9	7		51 37 34 33 32 30 24 
16	9	10		51 42 40 38 36 35 34 32 30 28 
17	9	8		51 41 37 34 32 30 29 24 
18	9	6		51 42 34 30 28 24 
19	9	7		50 41 38 35 34 30 29 
20	9	6		51 44 41 34 32 30 
21	9	6		51 50 42 41 38 29 
22	9	5		51 38 35 32 30 
23	9	7		51 50 43 40 39 36 35 
24	9	5		46 40 39 38 35 
25	9	5		50 47 46 38 35 
26	9	5		50 43 40 39 35 
27	9	5		50 43 41 40 35 
28	9	6		50 48 44 43 41 39 
29	9	6		49 48 45 44 43 40 
30	9	5		49 47 46 43 39 
31	9	5		49 46 44 43 39 
32	9	4		50 48 47 39 
33	9	3		43 42 41 
34	9	4		47 46 45 43 
35	9	3		49 48 44 
36	9	3		47 46 44 
37	9	3		45 44 43 
38	9	2		45 43 
39	9	1		45 
40	9	1		47 
41	9	1		46 
42	9	1		44 
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
2	1	2	3	4	4	3	13	24	
	2	3	3	3	3	2	13	22	
	3	4	3	3	3	2	13	20	
	4	6	3	3	3	2	13	19	
	5	11	3	3	2	2	13	17	
	6	21	2	3	2	1	13	16	
	7	22	2	3	1	1	13	14	
	8	29	2	3	1	1	13	13	
	9	30	2	3	1	1	13	12	
3	1	1	3	5	4	4	18	13	
	2	7	2	4	4	4	16	11	
	3	8	2	4	4	4	14	11	
	4	12	2	4	4	3	11	9	
	5	13	2	3	3	3	9	9	
	6	17	1	3	3	3	8	7	
	7	20	1	3	3	3	4	7	
	8	23	1	3	2	2	4	5	
	9	24	1	3	2	2	2	5	
4	1	5	5	2	3	4	11	26	
	2	9	4	2	2	4	9	23	
	3	18	4	2	2	4	8	21	
	4	20	4	2	2	3	8	18	
	5	22	4	1	1	2	7	14	
	6	26	4	1	1	2	7	12	
	7	27	4	1	1	1	5	8	
	8	28	4	1	1	1	5	7	
	9	30	4	1	1	1	4	3	
5	1	3	2	4	4	3	14	26	
	2	7	2	4	4	3	12	25	
	3	8	2	3	3	3	12	21	
	4	10	2	3	3	3	12	20	
	5	11	1	2	3	2	10	17	
	6	15	1	2	2	2	10	17	
	7	18	1	1	2	2	9	14	
	8	21	1	1	1	1	8	13	
	9	28	1	1	1	1	8	11	
6	1	3	4	4	5	3	28	8	
	2	4	4	4	4	3	26	8	
	3	5	4	4	4	3	26	7	
	4	6	4	4	3	3	26	5	
	5	9	3	4	2	3	25	5	
	6	10	3	4	2	3	24	5	
	7	11	3	4	2	3	24	3	
	8	14	3	4	1	3	24	3	
	9	19	3	4	1	3	23	2	
7	1	2	3	3	3	3	28	20	
	2	3	2	3	3	3	24	20	
	3	6	2	3	3	3	21	20	
	4	8	2	3	3	3	20	19	
	5	14	2	3	3	2	16	18	
	6	18	1	3	3	2	13	18	
	7	22	1	3	3	1	12	17	
	8	23	1	3	3	1	9	17	
	9	24	1	3	3	1	4	17	
8	1	1	5	4	4	3	14	18	
	2	3	4	3	4	3	14	14	
	3	4	4	3	4	3	12	14	
	4	7	4	3	4	3	11	12	
	5	11	4	3	4	3	11	11	
	6	16	4	2	4	3	10	8	
	7	17	4	2	4	3	9	8	
	8	19	4	2	4	3	8	5	
	9	26	4	2	4	3	7	4	
9	1	1	4	5	5	5	28	17	
	2	6	4	5	4	4	28	16	
	3	7	4	5	4	4	28	15	
	4	8	3	5	4	4	27	15	
	5	12	3	5	3	4	27	14	
	6	14	3	5	3	3	27	13	
	7	19	2	5	3	3	26	13	
	8	22	2	5	3	3	26	12	
	9	29	2	5	3	3	26	11	
10	1	4	3	1	2	4	11	30	
	2	6	2	1	2	4	9	29	
	3	7	2	1	2	4	8	29	
	4	8	2	1	2	4	8	28	
	5	9	1	1	1	4	7	28	
	6	12	1	1	1	4	6	28	
	7	21	1	1	1	4	5	27	
	8	22	1	1	1	4	5	26	
	9	30	1	1	1	4	4	27	
11	1	7	3	5	1	3	18	12	
	2	9	3	4	1	2	18	11	
	3	10	3	4	1	2	17	11	
	4	11	3	3	1	2	15	9	
	5	12	3	2	1	1	14	8	
	6	22	3	2	1	1	14	7	
	7	24	3	1	1	1	12	7	
	8	26	3	1	1	1	11	6	
	9	28	3	1	1	1	11	5	
12	1	3	4	1	3	4	28	21	
	2	4	3	1	3	4	27	19	
	3	7	3	1	3	4	24	17	
	4	11	3	1	3	4	19	16	
	5	12	2	1	3	4	18	14	
	6	15	2	1	3	3	15	14	
	7	20	2	1	3	3	10	13	
	8	21	1	1	3	3	6	10	
	9	26	1	1	3	3	5	9	
13	1	7	5	3	3	4	23	16	
	2	17	4	3	3	4	21	16	
	3	18	4	3	3	4	20	13	
	4	21	4	3	2	4	18	11	
	5	24	3	3	2	4	15	11	
	6	25	3	3	2	3	11	10	
	7	26	2	3	1	3	9	8	
	8	27	2	3	1	3	7	5	
	9	28	2	3	1	3	7	4	
14	1	7	1	2	3	5	11	19	
	2	8	1	2	3	5	9	18	
	3	9	1	2	3	5	8	16	
	4	16	1	2	2	5	7	15	
	5	17	1	2	2	5	7	13	
	6	20	1	2	2	5	7	11	
	7	21	1	2	1	5	6	9	
	8	24	1	2	1	5	5	7	
	9	30	1	2	1	5	4	6	
15	1	8	3	3	2	4	6	22	
	2	9	3	2	2	3	6	22	
	3	10	3	2	2	3	5	22	
	4	18	3	2	2	3	5	21	
	5	22	2	1	2	3	4	21	
	6	24	2	1	2	3	4	20	
	7	25	1	1	2	3	3	19	
	8	29	1	1	2	3	3	18	
	9	30	1	1	2	3	3	17	
16	1	3	5	2	3	5	8	13	
	2	4	5	2	3	4	7	13	
	3	16	5	2	3	4	7	12	
	4	16	5	2	3	3	7	13	
	5	17	5	2	3	3	6	12	
	6	18	5	2	2	3	6	12	
	7	19	5	2	2	2	5	12	
	8	20	5	2	2	2	5	11	
	9	22	5	2	2	2	5	10	
17	1	1	4	1	3	4	30	24	
	2	3	4	1	3	3	29	22	
	3	5	4	1	3	3	28	21	
	4	7	4	1	3	3	28	19	
	5	11	4	1	3	2	26	18	
	6	16	4	1	3	2	26	17	
	7	26	4	1	3	2	25	14	
	8	27	4	1	3	2	24	13	
	9	28	4	1	3	2	24	12	
18	1	4	3	3	2	4	29	26	
	2	5	3	3	2	3	27	25	
	3	7	3	3	2	3	26	22	
	4	10	3	3	2	3	24	18	
	5	13	3	3	2	2	23	14	
	6	16	3	2	2	2	20	11	
	7	28	3	2	2	2	19	10	
	8	29	3	2	2	2	17	8	
	9	30	3	2	2	2	17	4	
19	1	2	2	5	4	3	18	20	
	2	5	2	4	4	3	17	17	
	3	6	2	4	4	3	16	17	
	4	7	2	3	4	3	16	13	
	5	10	2	3	4	3	15	13	
	6	14	2	3	3	2	15	11	
	7	18	2	2	3	2	14	9	
	8	21	2	2	3	2	14	7	
	9	26	2	2	3	2	13	6	
20	1	4	3	5	3	5	11	20	
	2	5	3	4	3	5	10	20	
	3	13	3	4	3	5	10	19	
	4	14	3	4	3	5	10	18	
	5	18	3	4	2	5	10	17	
	6	19	2	4	2	5	10	17	
	7	24	2	4	2	5	10	16	
	8	28	2	4	2	5	10	15	
	9	29	2	4	2	5	10	14	
21	1	1	4	4	3	3	28	18	
	2	3	4	3	3	3	26	18	
	3	11	4	3	3	3	24	18	
	4	13	3	3	3	3	23	18	
	5	15	3	3	3	3	22	18	
	6	17	3	3	2	3	21	18	
	7	24	2	3	2	3	20	18	
	8	25	2	3	2	3	19	18	
	9	26	2	3	2	3	18	18	
22	1	7	2	1	4	5	25	27	
	2	8	2	1	4	4	22	26	
	3	10	2	1	4	4	21	26	
	4	12	2	1	4	4	21	25	
	5	14	2	1	4	3	19	25	
	6	21	2	1	4	3	18	25	
	7	24	2	1	4	3	17	25	
	8	26	2	1	4	3	17	24	
	9	29	2	1	4	3	15	24	
23	1	1	1	3	1	5	20	22	
	2	2	1	3	1	5	20	21	
	3	7	1	3	1	5	20	20	
	4	8	1	3	1	5	19	21	
	5	16	1	3	1	5	19	20	
	6	17	1	2	1	5	19	20	
	7	22	1	2	1	5	18	19	
	8	26	1	2	1	5	18	18	
	9	29	1	2	1	5	18	17	
24	1	7	4	2	4	5	24	21	
	2	8	4	2	4	5	23	20	
	3	10	4	2	4	5	22	19	
	4	16	4	2	4	5	22	17	
	5	17	3	2	4	5	21	15	
	6	18	3	2	4	5	20	13	
	7	19	3	2	4	5	18	11	
	8	24	2	2	4	5	17	10	
	9	30	2	2	4	5	16	9	
25	1	1	3	1	5	2	18	30	
	2	4	3	1	5	1	15	29	
	3	5	3	1	5	1	14	27	
	4	7	2	1	5	1	13	26	
	5	12	2	1	5	1	13	25	
	6	16	2	1	5	1	11	25	
	7	21	2	1	5	1	9	24	
	8	22	1	1	5	1	8	23	
	9	28	1	1	5	1	8	22	
26	1	1	4	2	4	4	19	11	
	2	11	4	2	4	3	17	9	
	3	12	4	2	4	3	15	9	
	4	18	4	2	3	2	14	7	
	5	19	3	2	3	2	11	7	
	6	23	3	1	3	2	10	5	
	7	24	3	1	3	1	8	5	
	8	26	3	1	2	1	7	4	
	9	29	3	1	2	1	6	2	
27	1	1	4	4	4	5	20	29	
	2	2	3	4	4	4	18	28	
	3	3	3	4	4	4	16	27	
	4	5	3	4	4	4	15	27	
	5	10	3	4	3	3	14	26	
	6	12	3	3	3	3	12	26	
	7	16	3	3	3	3	11	25	
	8	18	3	3	3	3	7	25	
	9	22	3	3	3	3	6	25	
28	1	3	3	1	2	4	26	27	
	2	4	2	1	2	4	25	27	
	3	10	2	1	2	4	24	27	
	4	12	2	1	2	4	24	26	
	5	16	2	1	2	4	22	26	
	6	17	2	1	2	4	22	25	
	7	21	2	1	2	4	21	25	
	8	27	2	1	2	4	20	25	
	9	30	2	1	2	4	19	25	
29	1	2	3	3	5	5	16	21	
	2	9	3	2	4	4	13	21	
	3	13	3	2	4	4	11	19	
	4	16	3	2	4	3	9	19	
	5	18	3	2	4	3	9	17	
	6	19	3	2	4	3	7	15	
	7	22	3	2	4	3	5	14	
	8	24	3	2	4	2	2	12	
	9	27	3	2	4	2	1	12	
30	1	3	4	3	3	3	25	25	
	2	9	3	3	3	3	25	24	
	3	10	3	3	3	3	23	23	
	4	19	3	3	3	3	21	22	
	5	20	3	3	3	3	19	21	
	6	21	3	3	3	3	18	21	
	7	28	3	3	3	3	16	21	
	8	29	3	3	3	3	16	20	
	9	30	3	3	3	3	13	19	
31	1	4	3	4	3	5	22	22	
	2	9	2	4	3	5	21	20	
	3	10	2	4	3	5	20	18	
	4	12	2	4	3	5	19	14	
	5	15	2	4	3	5	18	12	
	6	16	1	4	2	5	16	12	
	7	25	1	4	2	5	15	8	
	8	26	1	4	2	5	14	7	
	9	29	1	4	2	5	14	6	
32	1	3	4	5	1	4	29	24	
	2	7	3	4	1	3	29	21	
	3	17	3	4	1	3	28	21	
	4	18	2	3	1	3	28	18	
	5	20	2	2	1	2	27	17	
	6	23	2	2	1	2	26	15	
	7	24	2	2	1	2	26	14	
	8	25	1	1	1	1	25	13	
	9	27	1	1	1	1	24	10	
33	1	3	3	3	3	4	18	30	
	2	4	3	3	2	3	16	27	
	3	7	3	3	2	3	16	21	
	4	14	3	3	2	3	16	20	
	5	15	2	2	2	2	15	16	
	6	20	2	2	2	2	14	15	
	7	23	2	2	2	1	14	11	
	8	24	1	2	2	1	13	7	
	9	27	1	2	2	1	13	6	
34	1	3	3	1	5	5	26	26	
	2	11	3	1	4	5	23	22	
	3	12	3	1	4	5	22	19	
	4	13	3	1	4	5	18	17	
	5	14	2	1	4	5	16	12	
	6	18	2	1	3	5	14	12	
	7	21	2	1	3	5	12	8	
	8	22	1	1	3	5	6	6	
	9	28	1	1	3	5	6	2	
35	1	3	2	5	2	4	28	21	
	2	4	2	4	2	3	26	21	
	3	5	2	4	2	3	21	20	
	4	7	2	4	2	3	20	20	
	5	9	2	3	2	3	16	19	
	6	10	2	3	2	3	14	19	
	7	12	2	2	2	3	11	18	
	8	18	2	2	2	3	9	18	
	9	28	2	2	2	3	5	18	
36	1	7	4	3	4	4	9	15	
	2	12	3	3	4	4	8	13	
	3	13	3	3	4	4	8	12	
	4	18	3	3	4	4	8	11	
	5	19	2	3	4	4	6	10	
	6	20	2	3	4	4	6	9	
	7	22	2	3	4	4	5	8	
	8	27	1	3	4	4	5	8	
	9	30	1	3	4	4	4	7	
37	1	1	3	4	5	4	29	23	
	2	2	2	4	4	4	28	22	
	3	4	2	3	4	4	26	19	
	4	10	2	3	4	4	23	19	
	5	16	1	3	3	4	22	16	
	6	18	1	2	3	4	20	16	
	7	22	1	2	3	4	19	13	
	8	26	1	1	3	4	15	12	
	9	30	1	1	3	4	15	11	
38	1	1	5	3	4	5	18	24	
	2	7	4	3	3	5	17	24	
	3	10	3	3	3	5	15	23	
	4	11	3	3	3	5	15	22	
	5	13	2	3	2	5	14	21	
	6	16	2	3	2	5	11	21	
	7	23	1	3	2	5	10	21	
	8	24	1	3	1	5	9	20	
	9	25	1	3	1	5	7	19	
39	1	2	4	3	3	5	29	15	
	2	3	3	2	2	4	28	15	
	3	4	3	2	2	4	27	14	
	4	6	3	2	2	4	27	13	
	5	8	2	2	2	4	26	13	
	6	17	2	2	1	4	26	13	
	7	20	2	2	1	4	25	12	
	8	22	2	2	1	4	25	11	
	9	29	2	2	1	4	25	10	
40	1	1	3	3	5	1	8	29	
	2	8	2	3	4	1	7	29	
	3	9	2	3	4	1	7	27	
	4	11	2	3	4	1	6	27	
	5	16	1	3	4	1	6	25	
	6	17	1	3	4	1	6	24	
	7	28	1	3	4	1	5	24	
	8	29	1	3	4	1	5	23	
	9	30	1	3	4	1	5	22	
41	1	5	1	2	3	5	11	23	
	2	8	1	2	3	4	11	22	
	3	9	1	2	3	4	11	21	
	4	10	1	2	2	4	11	22	
	5	13	1	1	2	3	11	21	
	6	16	1	1	2	3	10	21	
	7	22	1	1	2	2	10	21	
	8	28	1	1	1	2	10	21	
	9	29	1	1	1	2	10	20	
42	1	1	4	5	3	1	27	28	
	2	12	4	4	3	1	26	27	
	3	17	4	4	3	1	24	27	
	4	25	4	4	3	1	24	26	
	5	26	4	3	3	1	20	26	
	6	27	3	3	3	1	19	26	
	7	28	3	3	3	1	17	26	
	8	29	3	3	3	1	17	25	
	9	30	3	3	3	1	15	24	
43	1	6	5	4	5	1	10	17	
	2	7	4	4	5	1	9	16	
	3	8	4	4	5	1	7	15	
	4	12	4	4	5	1	6	14	
	5	13	3	3	5	1	5	14	
	6	18	3	3	5	1	4	13	
	7	19	3	3	5	1	4	12	
	8	21	3	3	5	1	2	11	
	9	22	3	3	5	1	2	10	
44	1	3	5	3	4	5	29	23	
	2	8	4	2	4	4	27	19	
	3	9	4	2	4	4	26	19	
	4	11	4	2	4	3	24	17	
	5	13	3	2	3	3	23	15	
	6	15	3	2	3	3	22	10	
	7	18	3	2	2	2	20	8	
	8	19	3	2	2	2	20	6	
	9	22	3	2	2	2	18	4	
45	1	7	3	4	3	4	7	5	
	2	8	3	3	3	4	6	4	
	3	9	3	3	3	4	6	3	
	4	13	3	3	3	3	5	4	
	5	14	3	2	2	3	5	4	
	6	16	3	2	2	2	4	4	
	7	21	3	2	2	2	3	4	
	8	22	3	1	1	1	3	4	
	9	28	3	1	1	1	3	3	
46	1	2	3	3	4	3	16	23	
	2	3	3	2	4	3	15	22	
	3	4	3	2	4	3	14	19	
	4	13	3	2	3	3	14	18	
	5	14	2	2	3	3	13	16	
	6	19	2	2	3	3	12	16	
	7	22	1	2	3	3	12	13	
	8	27	1	2	2	3	12	11	
	9	30	1	2	2	3	11	10	
47	1	7	1	4	1	3	28	26	
	2	8	1	4	1	3	28	25	
	3	10	1	4	1	3	27	25	
	4	14	1	4	1	2	27	25	
	5	15	1	3	1	2	25	24	
	6	16	1	3	1	2	25	23	
	7	17	1	3	1	1	24	22	
	8	18	1	2	1	1	23	22	
	9	19	1	2	1	1	23	21	
48	1	2	5	3	5	4	28	17	
	2	9	4	3	4	4	23	17	
	3	10	4	3	4	4	21	14	
	4	14	4	3	3	4	20	12	
	5	17	3	3	3	4	15	12	
	6	20	3	3	3	4	14	11	
	7	21	3	3	3	4	11	8	
	8	24	3	3	2	4	7	6	
	9	26	3	3	2	4	7	5	
49	1	3	5	3	3	3	24	30	
	2	4	4	3	3	3	24	27	
	3	5	4	3	3	3	24	21	
	4	12	4	3	3	3	24	20	
	5	13	3	3	3	3	24	17	
	6	14	3	3	3	3	23	12	
	7	22	2	3	3	3	23	10	
	8	23	2	3	3	3	23	9	
	9	25	2	3	3	3	23	6	
50	1	1	4	1	5	4	10	14	
	2	3	3	1	5	4	9	13	
	3	5	3	1	5	4	8	12	
	4	6	3	1	5	3	6	11	
	5	13	2	1	5	3	6	11	
	6	14	2	1	5	3	4	10	
	7	25	2	1	5	2	4	10	
	8	27	2	1	5	2	2	9	
	9	28	2	1	5	2	1	9	
51	1	1	3	5	5	4	23	2	
	2	2	3	4	5	4	21	2	
	3	10	3	4	5	4	21	1	
	4	11	3	4	5	4	17	2	
	5	17	3	3	5	4	16	2	
	6	18	3	3	5	4	14	2	
	7	22	3	3	5	4	11	2	
	8	24	3	3	5	4	9	2	
	9	29	3	3	5	4	8	2	
52	1	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	R 3	R 4	N 1	N 2
	31	31	30	35	661	704

************************************************************************
