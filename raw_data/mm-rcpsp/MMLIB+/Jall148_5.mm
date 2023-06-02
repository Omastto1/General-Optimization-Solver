jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 4 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	13		2 3 4 5 6 9 10 11 13 15 17 19 21 
2	6	5		18 16 14 12 8 
3	6	2		14 7 
4	6	8		34 29 26 25 24 22 20 16 
5	6	9		37 34 33 29 25 24 23 22 20 
6	6	7		34 29 28 27 26 25 16 
7	6	6		34 29 28 25 20 16 
8	6	9		37 36 34 28 27 25 24 23 22 
9	6	6		34 32 29 28 26 16 
10	6	5		34 32 29 20 16 
11	6	3		25 22 16 
12	6	6		37 36 32 28 24 23 
13	6	6		50 37 32 30 29 20 
14	6	6		39 37 36 33 24 23 
15	6	6		48 37 33 32 28 23 
16	6	7		51 48 39 37 36 33 23 
17	6	8		51 49 36 35 33 32 29 27 
18	6	7		46 37 36 35 32 29 28 
19	6	9		50 49 46 39 37 36 33 32 29 
20	6	5		51 49 36 35 27 
21	6	8		51 50 49 46 44 39 32 31 
22	6	7		50 49 48 45 35 32 30 
23	6	5		50 49 35 31 30 
24	6	7		51 50 48 46 45 44 31 
25	6	5		50 48 39 32 31 
26	6	6		50 48 42 39 36 35 
27	6	8		48 47 46 44 42 41 39 38 
28	6	8		51 49 44 43 42 41 40 39 
29	6	4		48 45 44 31 
30	6	5		47 46 43 42 38 
31	6	4		43 42 41 38 
32	6	3		47 41 38 
33	6	3		45 44 38 
34	6	3		49 46 41 
35	6	2		44 41 
36	6	2		43 40 
37	6	1		38 
38	6	1		40 
39	6	1		45 
40	6	1		52 
41	6	1		52 
42	6	1		52 
43	6	1		52 
44	6	1		52 
45	6	1		52 
46	6	1		52 
47	6	1		52 
48	6	1		52 
49	6	1		52 
50	6	1		52 
51	6	1		52 
52	1	0		
************************************************************************
REQUESTS/DURATIONS
jobnr.	mode	dur	R1	R2	N1	N2	N3	N4	
------------------------------------------------------------------------
1	1	0	0	0	0	0	0	0	
2	1	5	13	7	12	10	17	20	
	2	7	13	6	9	10	15	16	
	3	10	11	4	8	7	15	15	
	4	12	10	4	6	5	13	12	
	5	17	10	3	3	4	11	12	
	6	20	9	2	3	1	8	10	
3	1	3	16	8	20	7	16	14	
	2	4	13	8	19	7	16	12	
	3	10	9	8	19	6	16	10	
	4	13	9	7	18	5	16	9	
	5	16	7	7	17	5	16	7	
	6	20	4	6	17	4	16	7	
4	1	1	17	15	8	8	10	18	
	2	6	13	14	7	6	8	16	
	3	11	12	11	7	6	7	16	
	4	12	10	11	7	6	4	15	
	5	14	4	10	6	4	4	13	
	6	19	3	8	6	4	3	13	
5	1	1	15	2	16	5	12	12	
	2	7	14	2	16	3	11	8	
	3	11	14	2	16	3	10	8	
	4	15	13	1	16	2	7	5	
	5	17	12	1	15	1	7	3	
	6	19	11	1	15	1	5	3	
6	1	3	18	17	11	5	19	9	
	2	4	18	16	9	4	18	9	
	3	15	17	15	8	4	18	9	
	4	16	17	12	6	4	17	9	
	5	17	15	12	3	4	17	9	
	6	19	15	10	3	4	17	9	
7	1	8	19	14	12	19	9	11	
	2	13	19	14	12	16	7	9	
	3	14	19	13	10	10	6	7	
	4	15	19	13	6	9	5	4	
	5	18	19	12	6	5	4	4	
	6	19	19	11	3	4	3	1	
8	1	4	9	12	13	15	18	11	
	2	7	8	12	11	15	12	10	
	3	11	7	12	10	15	9	9	
	4	12	7	12	10	15	8	9	
	5	13	6	12	9	15	3	8	
	6	15	5	12	7	15	1	7	
9	1	5	5	19	19	17	18	14	
	2	11	5	17	18	16	18	10	
	3	12	3	11	15	14	18	9	
	4	14	3	9	14	14	17	9	
	5	15	1	8	10	12	17	7	
	6	19	1	3	8	10	16	4	
10	1	1	14	10	14	4	9	4	
	2	7	13	8	14	3	8	3	
	3	8	13	6	14	3	8	3	
	4	9	12	5	13	2	8	3	
	5	11	11	4	13	1	8	2	
	6	15	10	1	13	1	8	2	
11	1	1	18	2	19	18	6	11	
	2	5	17	2	17	17	6	8	
	3	9	15	2	17	16	6	6	
	4	16	14	2	16	15	6	5	
	5	18	14	2	14	14	6	3	
	6	20	13	2	14	13	6	1	
12	1	2	9	16	13	18	7	19	
	2	7	9	16	13	14	7	16	
	3	8	7	15	12	13	7	16	
	4	12	4	15	12	11	7	12	
	5	13	4	15	11	8	7	12	
	6	14	1	14	11	7	7	8	
13	1	4	9	13	14	15	4	6	
	2	9	9	12	13	14	4	5	
	3	10	9	12	12	14	4	5	
	4	11	9	10	12	13	3	5	
	5	12	9	10	10	13	2	5	
	6	20	9	9	10	13	2	5	
14	1	9	20	19	11	11	9	9	
	2	10	19	18	10	10	9	8	
	3	11	18	17	10	10	9	8	
	4	12	18	17	9	7	9	8	
	5	14	18	16	9	6	9	8	
	6	16	17	15	9	6	9	8	
15	1	1	8	10	12	10	10	15	
	2	2	8	9	12	8	10	14	
	3	3	8	6	11	6	10	14	
	4	15	7	4	10	6	10	13	
	5	16	7	4	9	5	10	12	
	6	17	6	1	9	3	10	12	
16	1	2	14	18	11	19	17	13	
	2	4	13	18	10	18	15	12	
	3	5	13	17	10	18	14	12	
	4	16	13	17	9	18	13	12	
	5	19	12	17	9	18	12	11	
	6	20	12	16	8	18	11	11	
17	1	2	7	7	16	17	12	7	
	2	3	7	7	13	17	11	7	
	3	4	6	7	11	16	11	7	
	4	14	6	7	9	16	11	7	
	5	15	4	7	8	15	10	6	
	6	17	4	7	7	15	9	6	
18	1	5	17	12	15	14	18	18	
	2	6	16	12	15	12	17	18	
	3	8	15	12	15	10	14	16	
	4	18	15	12	15	8	13	15	
	5	19	13	11	15	6	10	15	
	6	20	12	11	15	6	9	14	
19	1	1	10	4	11	12	14	18	
	2	2	10	3	10	11	13	15	
	3	3	10	3	10	11	13	11	
	4	10	10	2	10	10	13	11	
	5	11	10	2	10	10	13	7	
	6	12	10	2	10	10	13	4	
20	1	7	10	14	14	12	18	19	
	2	8	9	12	12	10	17	16	
	3	11	9	12	11	7	16	13	
	4	13	8	10	8	5	16	9	
	5	16	7	7	6	5	14	7	
	6	18	7	5	5	3	14	5	
21	1	1	17	14	7	13	8	18	
	2	2	16	14	6	13	7	14	
	3	15	13	12	6	11	5	10	
	4	17	13	12	6	11	3	10	
	5	18	9	11	5	10	3	6	
	6	19	8	9	5	9	1	4	
22	1	2	20	14	15	15	11	20	
	2	3	18	12	12	14	11	19	
	3	8	18	12	10	14	11	18	
	4	10	17	9	9	13	11	17	
	5	11	16	7	8	13	10	15	
	6	18	16	4	5	12	10	15	
23	1	8	11	9	2	17	19	17	
	2	9	10	9	1	13	15	15	
	3	10	10	9	1	11	15	15	
	4	11	9	9	1	9	12	12	
	5	18	9	9	1	5	10	11	
	6	20	8	9	1	2	9	7	
24	1	4	9	12	2	18	6	9	
	2	5	6	11	2	16	5	9	
	3	6	6	10	2	10	4	9	
	4	7	5	8	1	7	3	9	
	5	8	4	7	1	7	2	9	
	6	16	2	4	1	1	2	9	
25	1	4	9	17	11	13	18	4	
	2	8	7	16	11	11	18	3	
	3	9	7	15	9	11	15	3	
	4	14	7	14	8	8	14	2	
	5	15	5	13	7	7	14	2	
	6	18	5	13	7	5	11	2	
26	1	8	12	11	11	14	19	16	
	2	9	11	11	11	14	15	15	
	3	13	8	11	8	14	12	15	
	4	14	7	11	7	13	9	14	
	5	15	5	11	4	13	8	11	
	6	18	3	11	2	13	5	11	
27	1	1	19	15	16	9	12	15	
	2	7	17	14	15	6	11	14	
	3	8	16	14	15	6	11	11	
	4	18	15	14	15	4	11	7	
	5	19	14	14	15	3	10	5	
	6	20	11	14	15	2	9	3	
28	1	1	8	15	16	10	17	11	
	2	6	7	13	15	9	15	11	
	3	7	6	13	15	8	14	9	
	4	15	5	12	15	7	11	6	
	5	17	4	11	13	5	11	5	
	6	18	4	11	13	5	9	3	
29	1	2	17	20	14	13	14	9	
	2	3	16	17	14	10	10	7	
	3	4	16	17	14	10	10	6	
	4	8	15	15	13	6	6	4	
	5	15	13	14	13	5	4	3	
	6	16	12	13	13	2	4	3	
30	1	5	14	19	15	13	11	13	
	2	9	12	15	12	11	8	10	
	3	10	12	14	9	10	8	9	
	4	17	10	9	9	9	4	9	
	5	19	9	6	6	8	2	7	
	6	20	8	5	1	7	1	5	
31	1	2	20	10	6	19	14	12	
	2	4	17	8	4	16	11	12	
	3	6	14	7	4	16	10	11	
	4	9	13	4	3	13	8	10	
	5	16	9	4	2	11	8	7	
	6	17	9	3	2	10	7	6	
32	1	5	3	10	16	11	16	6	
	2	7	3	9	12	9	15	6	
	3	9	3	9	10	9	14	5	
	4	12	2	9	8	8	13	4	
	5	14	1	7	6	7	13	2	
	6	19	1	7	5	5	12	2	
33	1	7	14	14	16	7	16	8	
	2	10	12	14	12	6	14	6	
	3	12	10	14	12	5	13	5	
	4	13	7	14	9	5	12	3	
	5	19	4	13	6	4	9	2	
	6	20	4	13	5	4	9	1	
34	1	1	18	11	11	7	20	17	
	2	2	18	10	10	7	20	16	
	3	13	17	8	9	7	20	13	
	4	15	17	4	9	7	20	13	
	5	16	17	3	8	7	20	10	
	6	18	16	1	8	7	20	6	
35	1	7	14	17	14	11	17	18	
	2	15	14	15	13	11	13	18	
	3	17	14	13	13	11	12	17	
	4	18	14	10	12	10	8	16	
	5	19	13	4	12	10	7	15	
	6	20	13	2	12	10	4	15	
36	1	3	18	16	12	5	10	13	
	2	4	13	16	11	5	9	10	
	3	6	11	16	10	5	6	10	
	4	13	9	16	9	4	5	7	
	5	14	6	16	8	4	2	6	
	6	15	3	16	8	4	2	4	
37	1	5	12	15	18	16	9	7	
	2	14	12	13	13	15	8	7	
	3	15	10	13	9	15	8	7	
	4	17	9	11	7	13	7	7	
	5	19	7	11	3	13	7	7	
	6	20	6	10	3	12	7	7	
38	1	5	2	8	11	17	19	13	
	2	6	2	7	11	15	18	13	
	3	12	2	7	10	14	18	10	
	4	16	1	7	10	13	17	9	
	5	17	1	7	10	12	17	5	
	6	18	1	7	9	12	17	5	
39	1	4	16	17	19	20	12	15	
	2	11	15	17	16	18	12	14	
	3	15	11	17	12	16	10	14	
	4	18	11	17	9	13	10	14	
	5	19	7	17	9	12	8	14	
	6	20	6	17	6	8	6	14	
40	1	3	6	11	5	15	15	18	
	2	4	5	11	5	15	13	16	
	3	5	5	11	5	15	12	14	
	4	6	3	11	5	15	10	14	
	5	7	3	11	5	14	10	13	
	6	8	2	11	5	14	8	11	
41	1	4	17	6	14	17	20	11	
	2	8	14	5	12	17	18	11	
	3	9	12	5	8	17	17	11	
	4	12	11	5	7	17	16	11	
	5	13	8	3	6	17	14	11	
	6	14	6	3	3	17	14	11	
42	1	1	20	14	10	6	10	10	
	2	2	15	10	10	6	7	8	
	3	4	15	10	10	4	7	6	
	4	5	13	8	9	4	4	6	
	5	10	11	7	9	3	4	4	
	6	16	8	4	9	2	2	2	
43	1	2	6	16	13	10	16	14	
	2	4	5	16	10	9	14	14	
	3	10	4	16	9	9	12	14	
	4	11	4	15	9	9	12	14	
	5	12	4	15	7	8	11	14	
	6	17	3	15	6	8	9	14	
44	1	5	19	2	11	16	18	14	
	2	6	16	2	10	14	14	14	
	3	13	15	2	9	11	14	14	
	4	14	12	2	8	9	10	14	
	5	16	8	2	5	5	9	14	
	6	17	5	2	5	5	6	14	
45	1	10	13	10	9	17	7	11	
	2	13	12	8	8	13	7	9	
	3	15	12	7	8	10	5	9	
	4	17	10	4	8	10	5	8	
	5	18	9	4	8	7	4	6	
	6	19	8	2	8	5	3	5	
46	1	7	13	17	12	18	19	8	
	2	8	13	13	10	14	18	7	
	3	10	13	12	10	13	18	7	
	4	16	12	11	10	11	18	6	
	5	17	12	7	8	9	18	6	
	6	19	12	5	8	7	18	6	
47	1	10	4	11	12	12	10	9	
	2	10	3	10	11	11	10	10	
	3	11	3	10	11	11	10	9	
	4	12	2	9	11	10	9	9	
	5	17	1	7	11	9	8	9	
	6	20	1	7	11	8	8	9	
48	1	2	4	17	10	11	18	11	
	2	3	3	17	9	11	17	8	
	3	8	3	15	8	9	16	8	
	4	11	3	14	7	8	16	5	
	5	17	3	13	6	7	16	4	
	6	18	3	12	5	6	15	3	
49	1	4	13	15	11	18	20	15	
	2	6	13	13	10	18	19	12	
	3	8	13	13	10	16	18	11	
	4	11	13	12	8	15	18	10	
	5	13	13	10	8	14	18	7	
	6	17	13	8	7	14	17	6	
50	1	3	17	17	16	15	14	9	
	2	8	17	16	15	14	9	9	
	3	12	17	14	13	12	9	9	
	4	13	17	10	12	10	6	9	
	5	14	17	9	11	9	4	8	
	6	15	17	5	11	7	2	8	
51	1	3	14	9	17	13	17	9	
	2	4	11	9	15	12	17	8	
	3	5	11	9	13	9	17	7	
	4	15	8	9	9	7	17	5	
	5	17	7	9	7	6	17	5	
	6	19	6	9	7	4	17	4	
52	1	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2	N 3	N 4
	93	74	442	436	497	415

************************************************************************
