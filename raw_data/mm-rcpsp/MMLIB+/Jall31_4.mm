jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 4 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	13		2 3 4 5 6 7 8 9 11 12 13 15 17 
2	3	11		50 49 43 41 29 28 27 26 23 16 14 
3	3	8		49 47 42 33 32 24 18 10 
4	3	6		49 46 34 33 19 16 
5	3	11		50 49 46 45 44 43 42 40 28 27 22 
6	3	9		49 45 44 43 37 33 31 26 19 
7	3	8		51 50 47 46 45 28 21 20 
8	3	8		48 45 42 38 31 28 26 20 
9	3	5		50 45 39 30 18 
10	3	7		51 48 45 38 31 28 21 
11	3	7		50 45 42 38 30 24 20 
12	3	10		48 46 44 42 41 32 30 28 27 25 
13	3	9		49 42 41 40 38 34 32 29 27 
14	3	9		47 46 44 40 39 38 37 31 30 
15	3	7		46 42 39 38 30 27 24 
16	3	9		48 45 44 42 40 39 38 37 31 
17	3	7		42 41 40 39 38 30 24 
18	3	7		46 44 43 41 40 38 27 
19	3	6		47 42 41 39 38 36 
20	3	5		41 40 37 36 32 
21	3	5		43 40 39 37 30 
22	3	4		41 39 38 31 
23	3	4		46 39 36 32 
24	3	3		43 36 35 
25	3	3		37 36 35 
26	3	2		35 30 
27	3	2		37 35 
28	3	2		36 35 
29	3	2		44 35 
30	3	1		36 
31	3	1		35 
32	3	1		35 
33	3	1		36 
34	3	1		36 
35	3	1		52 
36	3	1		52 
37	3	1		52 
38	3	1		52 
39	3	1		52 
40	3	1		52 
41	3	1		52 
42	3	1		52 
43	3	1		52 
44	3	1		52 
45	3	1		52 
46	3	1		52 
47	3	1		52 
48	3	1		52 
49	3	1		52 
50	3	1		52 
51	3	1		52 
52	1	0		
************************************************************************
REQUESTS/DURATIONS
jobnr.	mode	dur	R1	R2	N1	N2	N3	N4	
------------------------------------------------------------------------
1	1	0	0	0	0	0	0	0	
2	1	5	8	8	4	9	5	5	
	2	9	7	5	3	8	5	4	
	3	10	7	4	2	7	3	1	
3	1	2	8	9	5	4	10	4	
	2	8	8	8	5	4	7	3	
	3	10	8	7	5	4	6	3	
4	1	3	4	8	7	6	9	7	
	2	6	4	8	5	6	8	7	
	3	10	4	7	4	6	7	5	
5	1	6	9	8	8	5	7	6	
	2	7	9	7	6	4	6	3	
	3	10	9	3	5	3	5	2	
6	1	2	7	3	6	9	7	9	
	2	3	6	3	4	9	7	8	
	3	7	6	2	4	9	6	8	
7	1	4	4	7	2	6	8	3	
	2	7	3	5	1	5	8	3	
	3	10	2	5	1	3	5	3	
8	1	1	6	6	10	3	9	8	
	2	2	5	6	6	3	6	7	
	3	7	4	6	5	3	3	7	
9	1	2	3	5	5	9	2	8	
	2	4	3	5	5	9	1	7	
	3	6	2	3	3	9	1	5	
10	1	3	6	5	10	8	4	8	
	2	7	5	5	6	7	4	6	
	3	8	5	3	4	7	4	6	
11	1	6	8	6	9	3	3	2	
	2	7	7	6	7	2	3	2	
	3	10	6	4	4	2	2	2	
12	1	2	4	6	2	8	1	1	
	2	5	4	4	2	6	1	1	
	3	10	4	2	1	5	1	1	
13	1	1	8	6	8	9	3	10	
	2	2	7	4	7	3	3	10	
	3	9	6	2	6	1	3	10	
14	1	4	7	7	8	9	9	6	
	2	5	7	5	7	9	4	3	
	3	8	1	4	7	7	4	2	
15	1	4	2	9	9	8	5	8	
	2	6	2	9	8	7	5	5	
	3	8	2	9	7	7	5	3	
16	1	6	6	10	6	3	8	10	
	2	8	3	6	4	3	5	6	
	3	9	2	6	2	2	3	5	
17	1	1	7	4	8	8	6	1	
	2	3	5	4	7	4	5	1	
	3	6	5	3	3	2	5	1	
18	1	6	7	7	4	6	5	4	
	2	9	5	4	4	3	5	2	
	3	10	4	3	2	2	4	2	
19	1	1	6	6	6	7	3	7	
	2	2	6	5	4	5	2	7	
	3	5	6	5	4	4	2	6	
20	1	1	9	6	3	9	7	8	
	2	4	7	3	2	7	4	7	
	3	5	6	2	2	6	4	6	
21	1	1	8	8	2	4	7	7	
	2	3	7	7	1	4	6	5	
	3	9	7	7	1	4	3	4	
22	1	2	1	7	6	9	8	2	
	2	7	1	6	5	3	6	2	
	3	8	1	6	3	3	5	1	
23	1	1	9	6	1	7	6	8	
	2	2	6	4	1	6	5	6	
	3	6	3	4	1	4	5	3	
24	1	1	9	7	1	6	3	9	
	2	6	8	7	1	5	3	9	
	3	9	8	7	1	2	3	9	
25	1	1	7	8	6	7	4	6	
	2	2	5	5	5	5	4	5	
	3	3	5	3	5	5	3	4	
26	1	7	8	5	8	9	10	6	
	2	9	5	5	6	8	7	5	
	3	10	3	5	6	8	6	3	
27	1	6	2	7	4	9	2	4	
	2	8	1	5	3	7	1	4	
	3	10	1	1	2	3	1	3	
28	1	2	6	10	10	8	7	8	
	2	5	3	9	9	8	6	6	
	3	9	1	9	8	8	5	5	
29	1	2	6	7	7	1	2	6	
	2	4	4	5	6	1	1	5	
	3	10	4	4	5	1	1	5	
30	1	1	1	10	2	5	8	6	
	2	5	1	8	1	3	6	5	
	3	6	1	8	1	2	5	5	
31	1	3	3	9	7	6	7	7	
	2	4	3	8	5	5	5	5	
	3	5	3	7	4	4	4	3	
32	1	1	8	5	2	6	8	8	
	2	4	8	3	1	5	7	6	
	3	8	8	3	1	3	7	4	
33	1	4	8	3	5	6	7	8	
	2	5	4	3	4	3	6	8	
	3	6	4	3	3	2	2	8	
34	1	3	7	5	5	5	5	7	
	2	8	4	5	5	5	4	5	
	3	9	1	5	5	4	2	3	
35	1	4	9	10	7	8	6	6	
	2	5	9	5	6	8	6	6	
	3	10	9	1	4	8	5	5	
36	1	7	9	10	7	8	7	2	
	2	8	7	9	7	8	6	2	
	3	9	4	9	5	8	5	2	
37	1	2	4	10	9	7	7	8	
	2	4	4	6	8	5	5	7	
	3	7	4	3	8	5	4	7	
38	1	3	6	8	3	3	7	6	
	2	4	6	5	3	3	7	3	
	3	5	6	4	2	2	7	1	
39	1	1	9	4	10	8	7	4	
	2	6	7	4	7	8	7	4	
	3	10	4	2	4	7	7	2	
40	1	3	5	9	5	7	5	7	
	2	6	4	9	4	6	5	6	
	3	8	4	8	3	3	5	6	
41	1	1	2	6	9	8	9	9	
	2	9	2	6	9	7	4	8	
	3	10	2	5	9	4	2	8	
42	1	1	5	6	8	5	3	6	
	2	8	5	3	8	4	3	6	
	3	10	3	2	6	4	2	6	
43	1	3	9	3	7	10	2	4	
	2	5	8	3	5	7	2	3	
	3	6	8	1	5	6	2	2	
44	1	3	8	8	10	9	3	10	
	2	4	4	7	9	7	2	8	
	3	9	3	5	9	6	2	8	
45	1	2	2	7	6	7	5	5	
	2	6	2	4	4	6	4	5	
	3	9	2	3	3	4	4	4	
46	1	1	9	5	8	4	5	8	
	2	6	6	5	5	3	4	6	
	3	10	5	4	5	1	2	3	
47	1	2	5	7	9	9	5	9	
	2	8	5	6	7	4	4	5	
	3	10	4	4	4	4	3	4	
48	1	8	9	5	7	7	7	9	
	2	9	5	4	6	7	5	7	
	3	10	5	4	4	3	2	7	
49	1	1	4	7	3	9	4	10	
	2	3	2	6	3	8	2	9	
	3	5	1	4	3	7	2	7	
50	1	2	9	7	10	9	6	7	
	2	4	7	4	7	6	4	7	
	3	10	6	3	6	3	4	7	
51	1	5	9	6	6	8	9	5	
	2	6	7	5	4	7	9	5	
	3	7	5	5	4	5	9	1	
52	1	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2	N 3	N 4
	50	49	228	252	217	244

************************************************************************
