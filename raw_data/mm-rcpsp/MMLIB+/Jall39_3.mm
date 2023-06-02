jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 4 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	10		2 3 4 5 6 7 8 10 11 19 
2	3	7		24 22 17 16 14 13 9 
3	3	7		24 22 21 16 15 14 12 
4	3	6		30 28 27 24 17 12 
5	3	4		24 18 16 13 
6	3	4		25 24 15 13 
7	3	4		28 24 16 13 
8	3	3		27 21 13 
9	3	10		36 33 32 31 30 28 27 26 25 23 
10	3	11		36 33 32 31 30 28 27 26 24 23 22 
11	3	5		34 32 26 23 16 
12	3	1		13 
13	3	6		51 34 33 29 26 20 
14	3	9		51 37 36 33 31 30 29 26 23 
15	3	10		51 49 48 36 35 34 33 30 29 27 
16	3	8		51 38 36 33 31 30 29 27 
17	3	8		51 49 36 33 32 31 25 23 
18	3	7		47 37 36 34 31 28 26 
19	3	8		51 50 48 40 36 35 29 26 
20	3	7		50 49 38 36 32 31 23 
21	3	9		49 48 45 38 36 35 33 30 29 
22	3	8		50 48 47 46 45 43 40 39 
23	3	7		47 46 45 44 43 40 39 
24	3	6		47 46 45 44 39 35 
25	3	4		46 45 34 29 
26	3	6		49 46 45 44 42 38 
27	3	6		50 45 43 40 39 37 
28	3	6		51 49 45 44 42 39 
29	3	5		47 44 43 42 41 
30	3	5		50 47 46 43 41 
31	3	3		48 44 35 
32	3	3		44 41 35 
33	3	3		44 43 42 
34	3	3		44 43 41 
35	3	2		43 42 
36	3	2		46 41 
37	3	2		42 41 
38	3	1		39 
39	3	1		41 
40	3	1		42 
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
2	1	3	5	6	6	9	6	4	
	2	4	3	4	5	6	6	4	
	3	5	3	3	2	5	6	4	
3	1	3	1	5	5	7	7	6	
	2	5	1	5	3	7	5	4	
	3	8	1	3	1	7	5	3	
4	1	2	9	9	3	7	9	9	
	2	3	7	9	3	4	6	8	
	3	7	6	8	3	4	5	8	
5	1	1	5	5	10	8	7	2	
	2	4	5	5	4	8	4	3	
	3	5	5	5	4	8	4	2	
6	1	3	7	10	4	10	6	8	
	2	6	5	9	3	7	4	6	
	3	7	4	8	3	4	4	6	
7	1	5	10	4	8	9	6	3	
	2	6	6	4	7	9	5	2	
	3	10	4	4	6	9	5	2	
8	1	6	7	2	8	9	7	4	
	2	7	6	1	7	8	4	3	
	3	8	5	1	7	8	3	2	
9	1	3	5	8	9	9	9	10	
	2	5	5	7	9	9	6	5	
	3	8	5	7	9	9	4	4	
10	1	3	5	3	8	5	3	8	
	2	5	3	2	7	2	2	6	
	3	10	2	1	3	2	2	6	
11	1	1	3	9	9	9	9	8	
	2	2	3	8	8	8	6	8	
	3	3	3	8	8	6	6	7	
12	1	1	3	9	7	8	8	10	
	2	9	3	8	6	7	6	7	
	3	10	3	8	5	6	5	5	
13	1	5	8	7	7	6	3	5	
	2	6	8	5	7	6	2	5	
	3	7	8	3	4	5	1	5	
14	1	4	8	5	4	9	8	3	
	2	6	8	5	4	4	7	3	
	3	9	8	2	3	2	6	2	
15	1	1	7	7	8	9	10	6	
	2	8	4	4	7	8	7	4	
	3	9	4	4	6	6	6	2	
16	1	4	10	8	8	6	9	7	
	2	5	8	7	8	6	7	6	
	3	7	7	5	7	6	5	4	
17	1	4	8	4	8	2	8	6	
	2	6	7	3	6	1	6	4	
	3	8	6	3	6	1	3	3	
18	1	3	4	1	8	4	10	9	
	2	9	3	1	6	3	8	8	
	3	10	3	1	5	3	8	8	
19	1	2	5	10	9	1	7	5	
	2	7	2	9	7	1	6	3	
	3	10	2	9	7	1	6	1	
20	1	5	5	7	9	5	3	6	
	2	9	3	4	8	5	2	4	
	3	10	3	2	8	3	2	2	
21	1	1	4	8	7	8	3	6	
	2	2	2	4	6	7	3	4	
	3	9	1	3	5	7	3	2	
22	1	4	4	10	8	5	1	7	
	2	6	4	7	7	4	1	4	
	3	10	4	4	7	4	1	2	
23	1	4	10	8	3	8	2	6	
	2	5	8	5	2	7	2	4	
	3	9	8	4	2	7	2	4	
24	1	5	4	6	8	7	6	7	
	2	8	4	5	5	6	6	6	
	3	10	3	5	2	5	6	6	
25	1	5	8	7	6	10	7	8	
	2	6	7	6	6	8	7	7	
	3	9	6	2	5	5	6	6	
26	1	2	8	5	8	6	9	2	
	2	3	7	5	8	6	8	2	
	3	6	7	5	8	6	7	2	
27	1	1	9	4	5	6	7	1	
	2	3	6	2	3	3	4	1	
	3	9	2	2	1	2	3	1	
28	1	5	7	9	7	10	9	5	
	2	8	4	7	6	8	8	3	
	3	10	3	5	5	6	6	2	
29	1	6	6	9	6	7	8	9	
	2	7	6	6	3	6	8	8	
	3	8	6	5	3	3	8	6	
30	1	1	8	2	9	6	10	8	
	2	6	8	1	9	6	8	5	
	3	7	7	1	8	3	6	4	
31	1	4	7	2	6	8	6	4	
	2	7	6	1	4	7	4	4	
	3	8	6	1	2	7	4	3	
32	1	1	8	6	6	8	9	7	
	2	3	8	5	5	5	8	7	
	3	10	8	5	4	5	8	5	
33	1	7	4	2	7	7	6	4	
	2	8	2	2	4	6	5	3	
	3	9	2	2	2	3	5	2	
34	1	2	9	5	5	6	2	5	
	2	3	9	4	3	6	1	3	
	3	4	7	2	2	4	1	3	
35	1	2	9	9	5	10	9	9	
	2	6	5	7	5	7	9	3	
	3	8	3	7	5	7	8	3	
36	1	4	7	9	6	7	4	5	
	2	5	5	8	5	6	3	4	
	3	7	5	8	5	6	1	2	
37	1	5	7	3	4	9	8	8	
	2	8	5	3	3	9	4	8	
	3	9	4	1	2	7	3	5	
38	1	4	8	9	6	6	8	8	
	2	6	7	7	5	6	7	4	
	3	10	6	3	4	2	7	3	
39	1	1	6	9	9	2	9	6	
	2	2	4	7	5	2	5	4	
	3	9	4	7	2	2	5	4	
40	1	3	4	3	9	6	6	7	
	2	7	3	3	6	6	5	6	
	3	8	3	3	3	6	4	4	
41	1	6	4	6	9	7	9	3	
	2	7	3	5	8	3	8	1	
	3	9	2	3	7	3	8	1	
42	1	5	10	8	6	6	7	6	
	2	7	9	7	5	6	6	4	
	3	8	9	7	5	6	4	4	
43	1	2	9	9	5	8	7	8	
	2	6	6	8	5	5	5	6	
	3	7	3	7	5	1	3	2	
44	1	3	10	9	5	9	9	7	
	2	7	8	9	5	6	9	4	
	3	10	6	9	5	5	8	3	
45	1	1	9	5	5	3	8	6	
	2	2	9	4	3	2	4	6	
	3	4	9	3	1	1	2	6	
46	1	3	7	2	6	3	10	8	
	2	4	5	2	5	3	9	6	
	3	5	3	2	4	3	9	3	
47	1	1	1	1	9	10	1	4	
	2	2	1	1	7	8	1	3	
	3	4	1	1	6	6	1	2	
48	1	4	6	1	4	7	6	10	
	2	6	6	1	3	6	3	7	
	3	10	6	1	3	6	3	6	
49	1	4	8	4	5	7	8	7	
	2	5	5	3	3	6	5	6	
	3	9	3	1	2	5	4	2	
50	1	6	2	4	3	4	8	4	
	2	8	2	4	2	2	5	4	
	3	9	1	2	2	2	4	4	
51	1	4	7	9	6	9	6	8	
	2	9	6	9	6	4	5	7	
	3	10	4	9	6	3	4	5	
52	1	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2	N 3	N 4
	22	23	303	318	314	280

************************************************************************
