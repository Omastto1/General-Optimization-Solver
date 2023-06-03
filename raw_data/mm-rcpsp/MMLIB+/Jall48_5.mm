jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 4 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	6		2 3 4 5 7 12 
2	3	6		14 13 11 9 8 6 
3	3	5		18 14 11 10 8 
4	3	6		22 20 17 16 15 13 
5	3	5		26 22 16 15 14 
6	3	3		20 18 10 
7	3	6		26 21 20 18 17 16 
8	3	5		26 22 21 20 17 
9	3	4		26 25 17 15 
10	3	3		17 16 15 
11	3	3		26 17 15 
12	3	2		17 15 
13	3	6		27 26 25 24 23 21 
14	3	2		21 17 
15	3	4		27 24 23 21 
16	3	4		27 25 24 23 
17	3	2		23 19 
18	3	3		27 24 23 
19	3	3		36 27 24 
20	3	3		25 24 23 
21	3	7		36 34 33 32 31 29 28 
22	3	1		23 
23	3	5		36 34 32 31 28 
24	3	5		34 32 31 29 28 
25	3	5		36 34 32 31 28 
26	3	5		36 34 31 29 28 
27	3	5		34 33 31 30 29 
28	3	5		41 40 39 38 30 
29	3	5		42 41 39 37 35 
30	3	4		44 42 37 35 
31	3	3		45 43 39 
32	3	3		46 43 40 
33	3	3		46 44 38 
34	3	2		41 40 
35	3	5		49 48 46 45 43 
36	3	4		49 46 45 44 
37	3	4		49 48 45 43 
38	3	4		49 48 45 43 
39	3	3		49 46 44 
40	3	3		49 48 42 
41	3	3		51 49 48 
42	3	2		47 45 
43	3	2		50 47 
44	3	2		51 48 
45	3	2		51 50 
46	3	1		47 
47	3	1		51 
48	3	1		50 
49	3	1		50 
50	3	1		52 
51	3	1		52 
52	1	0		
************************************************************************
REQUESTS/DURATIONS
jobnr.	mode	dur	R1	R2	N1	N2	N3	N4	
------------------------------------------------------------------------
1	1	0	0	0	0	0	0	0	
2	1	1	8	9	4	6	8	6	
	2	3	6	9	4	6	5	5	
	3	7	4	8	2	4	4	5	
3	1	4	6	2	3	8	3	8	
	2	7	4	2	2	7	3	8	
	3	8	3	2	2	7	1	7	
4	1	6	9	3	4	6	6	6	
	2	9	9	2	4	5	5	5	
	3	10	9	1	3	5	5	2	
5	1	3	5	9	6	8	4	8	
	2	7	5	9	3	5	4	7	
	3	10	5	9	3	4	4	7	
6	1	1	7	5	7	1	2	9	
	2	4	7	5	6	1	2	7	
	3	5	6	2	2	1	2	4	
7	1	7	9	9	8	6	9	7	
	2	8	8	9	8	5	8	4	
	3	10	4	8	8	3	7	3	
8	1	5	9	7	5	9	4	4	
	2	6	5	6	3	9	3	4	
	3	8	4	6	2	9	3	4	
9	1	1	6	8	10	7	8	1	
	2	2	5	6	8	6	7	1	
	3	9	4	5	6	5	5	1	
10	1	2	8	5	1	7	10	7	
	2	6	7	5	1	4	8	3	
	3	10	7	5	1	1	8	2	
11	1	4	6	7	5	7	2	8	
	2	6	4	5	4	7	1	4	
	3	9	2	3	1	5	1	4	
12	1	1	4	5	4	9	5	7	
	2	2	3	3	3	9	4	5	
	3	5	2	2	3	9	4	5	
13	1	4	7	7	8	9	1	9	
	2	7	6	6	7	7	1	8	
	3	10	6	3	6	6	1	8	
14	1	2	9	9	3	6	7	9	
	2	4	7	8	2	5	6	9	
	3	8	6	8	2	3	6	9	
15	1	2	4	4	7	10	2	10	
	2	3	3	4	5	8	1	9	
	3	7	3	3	5	5	1	9	
16	1	2	8	8	8	10	8	7	
	2	7	8	6	7	8	6	6	
	3	10	8	5	7	6	5	6	
17	1	6	7	8	5	5	7	5	
	2	7	5	7	5	3	4	2	
	3	10	5	3	5	3	4	2	
18	1	1	6	4	9	7	8	10	
	2	4	6	3	7	4	7	9	
	3	8	5	2	7	3	5	9	
19	1	1	10	7	7	5	5	2	
	2	5	10	7	7	5	4	2	
	3	9	10	6	7	4	2	2	
20	1	6	9	6	4	8	5	8	
	2	7	7	6	2	6	5	5	
	3	8	4	2	1	4	5	4	
21	1	3	7	5	6	9	3	7	
	2	4	6	5	6	7	1	5	
	3	8	6	5	3	7	1	3	
22	1	1	7	9	6	1	6	3	
	2	4	6	3	6	1	4	2	
	3	10	2	1	4	1	2	2	
23	1	5	9	4	3	8	2	2	
	2	7	6	2	1	6	2	2	
	3	10	6	1	1	6	2	2	
24	1	3	9	9	6	3	4	8	
	2	5	7	8	5	3	4	4	
	3	9	6	6	4	1	4	2	
25	1	5	3	7	5	8	2	4	
	2	6	3	6	5	4	2	3	
	3	9	1	4	5	2	2	2	
26	1	2	7	6	9	9	10	7	
	2	4	5	5	7	8	9	6	
	3	10	5	5	3	7	8	6	
27	1	2	7	5	5	7	3	3	
	2	3	6	5	5	7	1	2	
	3	8	3	5	5	7	1	2	
28	1	2	9	8	3	8	6	7	
	2	4	8	6	2	8	6	5	
	3	5	7	1	1	8	4	5	
29	1	6	10	8	5	4	1	4	
	2	7	9	8	4	4	1	4	
	3	10	9	6	3	3	1	4	
30	1	1	5	9	6	7	2	9	
	2	3	3	6	5	6	2	4	
	3	8	1	6	4	6	2	1	
31	1	7	6	8	5	2	8	5	
	2	8	6	8	4	2	7	5	
	3	10	6	8	3	2	7	5	
32	1	5	10	8	4	8	9	8	
	2	6	7	4	3	5	9	7	
	3	8	4	4	2	3	8	7	
33	1	2	9	7	4	5	5	5	
	2	3	7	5	2	4	3	4	
	3	8	7	5	1	4	2	3	
34	1	1	6	5	8	9	8	8	
	2	4	5	4	8	6	7	8	
	3	5	4	3	6	6	7	8	
35	1	4	8	4	9	7	7	6	
	2	5	5	4	5	7	6	5	
	3	6	4	2	4	5	5	2	
36	1	2	6	5	2	3	1	5	
	2	6	6	4	2	2	1	5	
	3	10	5	3	2	2	1	5	
37	1	1	8	10	6	5	7	8	
	2	2	6	10	6	3	6	5	
	3	7	4	10	5	2	4	2	
38	1	1	10	10	4	3	6	5	
	2	2	10	8	2	3	6	3	
	3	5	10	7	2	3	5	3	
39	1	2	5	7	7	10	9	10	
	2	6	5	5	7	9	8	9	
	3	8	1	4	6	9	7	9	
40	1	2	8	2	3	5	8	8	
	2	9	7	2	3	3	8	5	
	3	10	6	1	1	2	7	3	
41	1	2	9	10	5	5	6	10	
	2	8	5	10	4	2	4	6	
	3	9	4	10	4	1	3	6	
42	1	6	5	7	7	4	7	5	
	2	7	2	7	6	3	4	4	
	3	8	1	7	6	3	2	4	
43	1	2	7	9	4	6	7	8	
	2	6	5	9	3	4	7	6	
	3	9	3	9	3	3	7	6	
44	1	6	4	3	8	9	3	9	
	2	9	3	3	8	7	3	9	
	3	10	3	3	8	7	3	8	
45	1	2	10	2	10	5	10	9	
	2	5	8	2	10	3	9	7	
	3	6	5	2	10	3	7	7	
46	1	5	1	7	9	8	10	7	
	2	8	1	5	8	4	9	4	
	3	9	1	4	5	4	7	3	
47	1	3	8	6	4	1	7	9	
	2	4	5	6	2	1	5	5	
	3	9	2	3	2	1	3	4	
48	1	1	6	8	3	4	9	3	
	2	6	4	5	2	4	6	1	
	3	8	4	3	1	4	4	1	
49	1	4	3	8	10	6	10	8	
	2	6	3	7	9	5	10	8	
	3	8	3	4	8	4	10	8	
50	1	5	7	6	8	10	8	9	
	2	8	5	5	6	8	7	9	
	3	10	4	4	3	7	7	9	
51	1	3	6	4	8	3	5	8	
	2	5	6	4	6	1	4	6	
	3	9	5	4	4	1	4	6	
52	1	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2	N 3	N 4
	21	19	265	289	272	311

************************************************************************
