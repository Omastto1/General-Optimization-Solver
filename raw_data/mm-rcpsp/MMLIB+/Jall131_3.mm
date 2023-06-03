jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	7		2 3 4 5 6 7 8 
2	6	5		16 15 11 10 9 
3	6	7		19 18 16 13 12 11 10 
4	6	4		18 13 11 10 
5	6	5		22 19 18 17 12 
6	6	5		22 18 17 16 12 
7	6	5		19 18 17 16 12 
8	6	4		19 18 14 11 
9	6	4		22 19 18 12 
10	6	5		28 27 23 22 17 
11	6	4		27 23 22 17 
12	6	3		23 20 14 
13	6	3		23 20 14 
14	6	6		29 28 27 26 25 21 
15	6	6		29 28 27 26 25 21 
16	6	6		29 27 26 25 24 23 
17	6	3		29 21 20 
18	6	2		29 20 
19	6	4		29 27 26 23 
20	6	5		32 31 26 25 24 
21	6	3		32 31 24 
22	6	4		32 31 30 25 
23	6	5		36 34 32 31 30 
24	6	4		36 35 34 30 
25	6	4		44 35 34 33 
26	6	4		44 38 36 33 
27	6	3		38 33 32 
28	6	2		35 30 
29	6	3		38 35 33 
30	6	2		44 33 
31	6	2		38 33 
32	6	4		47 44 40 37 
33	6	4		47 41 40 39 
34	6	3		43 41 40 
35	6	2		42 37 
36	6	2		46 37 
37	6	2		43 41 
38	6	2		46 41 
39	6	4		51 50 49 43 
40	6	3		51 45 42 
41	6	4		51 50 49 45 
42	6	2		49 46 
43	6	1		45 
44	6	1		45 
45	6	1		48 
46	6	1		48 
47	6	1		50 
48	6	1		52 
49	6	1		52 
50	6	1		52 
51	6	1		52 
52	1	0		
************************************************************************
REQUESTS/DURATIONS
jobnr.	mode	dur	R1	R2	N1	N2	
------------------------------------------------------------------------
1	1	0	0	0	0	0	
2	1	5	12	16	19	16	
	2	9	12	16	17	14	
	3	12	12	12	16	12	
	4	13	12	7	16	10	
	5	16	12	5	14	8	
	6	17	12	1	14	5	
3	1	5	3	16	16	11	
	2	6	3	15	15	10	
	3	8	3	15	13	9	
	4	10	3	15	10	8	
	5	11	3	15	6	6	
	6	17	3	15	5	4	
4	1	2	19	15	12	10	
	2	4	17	13	10	10	
	3	6	16	9	9	10	
	4	17	15	8	9	10	
	5	18	14	6	7	10	
	6	19	13	3	5	10	
5	1	2	17	19	6	10	
	2	4	15	19	5	7	
	3	10	11	19	3	7	
	4	13	10	19	2	5	
	5	16	9	19	2	4	
	6	20	6	19	1	1	
6	1	6	11	3	16	15	
	2	8	8	2	13	14	
	3	10	8	2	10	14	
	4	14	7	1	8	12	
	5	17	3	1	3	11	
	6	18	3	1	1	9	
7	1	9	16	16	17	14	
	2	10	14	14	17	13	
	3	14	14	14	17	12	
	4	18	13	11	17	12	
	5	19	12	10	16	10	
	6	20	10	8	16	9	
8	1	3	5	13	3	10	
	2	7	4	12	2	10	
	3	8	4	12	2	9	
	4	11	4	12	2	8	
	5	12	4	11	1	10	
	6	14	4	10	1	10	
9	1	7	13	15	13	15	
	2	10	13	14	11	14	
	3	13	11	13	11	12	
	4	16	9	11	7	10	
	5	17	8	10	7	7	
	6	18	7	10	4	6	
10	1	5	6	8	14	8	
	2	7	5	7	11	7	
	3	14	5	7	9	7	
	4	16	4	6	7	7	
	5	17	3	6	5	7	
	6	20	3	5	1	7	
11	1	9	4	4	18	9	
	2	11	3	3	13	8	
	3	12	3	3	11	8	
	4	14	3	2	8	8	
	5	15	2	1	5	8	
	6	17	2	1	3	8	
12	1	6	11	12	18	18	
	2	8	11	12	17	17	
	3	9	11	12	15	16	
	4	12	11	11	11	15	
	5	16	11	11	9	14	
	6	20	11	11	9	13	
13	1	3	19	13	15	15	
	2	4	15	13	14	14	
	3	6	11	12	13	14	
	4	14	7	12	12	14	
	5	15	6	11	11	14	
	6	17	1	11	11	14	
14	1	2	17	6	12	20	
	2	9	14	6	12	17	
	3	10	13	5	10	15	
	4	16	10	4	9	15	
	5	18	8	3	9	13	
	6	20	6	2	7	10	
15	1	2	16	17	18	9	
	2	6	16	17	17	8	
	3	7	16	17	15	7	
	4	15	16	17	12	6	
	5	16	16	17	10	4	
	6	17	16	17	8	3	
16	1	5	11	17	6	18	
	2	10	11	16	6	15	
	3	11	10	13	6	14	
	4	12	9	12	6	13	
	5	17	8	9	6	11	
	6	19	6	8	6	9	
17	1	5	6	19	18	2	
	2	7	6	19	16	2	
	3	9	5	18	16	2	
	4	10	5	18	13	2	
	5	11	5	16	12	2	
	6	13	4	16	11	2	
18	1	6	18	6	11	15	
	2	7	16	6	11	14	
	3	8	13	4	11	13	
	4	9	10	4	11	12	
	5	10	7	3	11	10	
	6	14	3	1	11	9	
19	1	4	12	8	18	11	
	2	10	10	7	18	9	
	3	12	10	7	18	8	
	4	13	10	6	18	6	
	5	17	8	6	18	6	
	6	18	8	6	18	4	
20	1	1	16	17	20	9	
	2	3	11	13	18	8	
	3	4	9	11	16	8	
	4	7	6	10	16	7	
	5	14	4	7	14	6	
	6	20	3	4	12	6	
21	1	4	14	14	11	9	
	2	6	12	11	9	8	
	3	8	10	10	7	7	
	4	15	8	8	5	4	
	5	16	5	7	3	3	
	6	20	5	6	1	2	
22	1	1	17	5	4	14	
	2	3	15	4	4	13	
	3	7	13	3	4	13	
	4	12	12	3	4	11	
	5	13	12	2	3	10	
	6	16	11	2	3	9	
23	1	1	12	7	11	20	
	2	3	10	6	9	18	
	3	7	9	6	9	15	
	4	8	8	6	6	14	
	5	12	8	4	6	14	
	6	16	6	4	4	12	
24	1	6	8	10	14	15	
	2	8	7	9	13	15	
	3	9	7	9	11	12	
	4	11	6	8	10	11	
	5	14	4	8	7	8	
	6	17	4	8	6	7	
25	1	1	11	6	12	3	
	2	2	10	5	10	3	
	3	3	8	5	8	3	
	4	13	6	5	7	3	
	5	14	4	5	5	3	
	6	16	2	5	4	3	
26	1	7	14	18	17	13	
	2	8	13	17	15	10	
	3	11	12	17	12	9	
	4	16	11	17	8	7	
	5	18	9	17	7	5	
	6	20	7	17	5	4	
27	1	2	12	10	10	7	
	2	9	9	9	8	7	
	3	10	7	7	7	6	
	4	16	5	7	5	4	
	5	18	3	5	4	4	
	6	20	1	4	2	3	
28	1	6	19	14	18	8	
	2	9	18	12	14	6	
	3	11	18	9	12	6	
	4	15	17	7	11	5	
	5	17	17	7	7	3	
	6	20	16	3	6	3	
29	1	2	18	3	18	16	
	2	10	15	3	16	16	
	3	11	14	3	15	13	
	4	12	11	2	13	12	
	5	14	11	2	8	12	
	6	17	9	2	8	9	
30	1	4	12	15	15	16	
	2	5	11	11	15	15	
	3	13	11	11	15	14	
	4	15	11	6	15	13	
	5	16	11	6	15	12	
	6	17	11	2	15	12	
31	1	10	11	12	17	8	
	2	11	10	12	15	7	
	3	12	9	12	15	7	
	4	14	8	12	13	6	
	5	19	7	11	10	6	
	6	20	6	11	10	6	
32	1	2	19	4	4	18	
	2	11	16	4	4	15	
	3	12	15	3	4	13	
	4	17	13	2	3	10	
	5	19	9	1	3	7	
	6	20	7	1	3	4	
33	1	1	14	14	16	12	
	2	5	12	13	12	10	
	3	8	10	12	11	10	
	4	11	7	10	9	8	
	5	12	7	6	6	8	
	6	19	3	6	3	7	
34	1	5	9	17	17	15	
	2	6	8	15	17	13	
	3	11	8	13	14	13	
	4	15	8	11	11	13	
	5	17	8	7	10	12	
	6	20	8	2	6	11	
35	1	4	17	14	12	8	
	2	6	16	14	11	7	
	3	7	14	13	10	7	
	4	8	13	13	9	6	
	5	14	11	12	9	5	
	6	19	10	12	7	5	
36	1	3	12	18	15	8	
	2	4	12	16	14	7	
	3	6	12	11	11	7	
	4	10	12	9	8	5	
	5	16	11	8	6	4	
	6	20	11	5	5	3	
37	1	7	12	17	8	16	
	2	8	11	16	8	16	
	3	13	9	15	7	14	
	4	15	9	14	7	13	
	5	16	8	12	6	12	
	6	18	7	12	6	12	
38	1	2	18	13	7	8	
	2	6	17	11	6	8	
	3	7	12	7	5	8	
	4	12	9	7	5	8	
	5	15	5	5	5	8	
	6	20	5	3	4	8	
39	1	3	20	10	15	14	
	2	4	18	7	15	13	
	3	5	16	6	14	13	
	4	6	16	4	13	11	
	5	7	14	3	11	9	
	6	12	14	1	10	8	
40	1	2	13	17	11	8	
	2	3	13	16	11	8	
	3	10	11	15	9	7	
	4	13	10	14	7	7	
	5	16	10	14	5	6	
	6	19	9	12	3	6	
41	1	1	11	19	10	17	
	2	3	11	15	9	15	
	3	9	10	14	8	13	
	4	12	10	12	6	13	
	5	14	9	11	5	12	
	6	17	9	8	5	11	
42	1	1	8	18	15	14	
	2	6	8	14	11	13	
	3	14	8	11	10	13	
	4	18	8	10	7	12	
	5	19	7	7	5	12	
	6	20	7	5	5	12	
43	1	3	14	12	12	17	
	2	8	11	9	12	16	
	3	10	9	9	12	16	
	4	12	6	6	11	16	
	5	14	4	5	11	15	
	6	15	3	3	10	15	
44	1	3	15	18	13	13	
	2	8	13	15	11	12	
	3	14	12	14	10	11	
	4	16	12	14	9	11	
	5	18	11	13	5	10	
	6	20	10	11	5	10	
45	1	1	4	19	13	20	
	2	11	3	17	9	18	
	3	15	3	14	8	18	
	4	16	2	9	6	18	
	5	17	1	7	5	16	
	6	18	1	4	4	16	
46	1	1	20	6	15	18	
	2	9	18	6	14	18	
	3	10	18	6	12	16	
	4	11	18	6	11	16	
	5	12	16	6	11	14	
	6	13	16	6	9	14	
47	1	5	18	17	11	17	
	2	7	18	16	8	17	
	3	15	18	15	8	13	
	4	17	18	15	6	12	
	5	18	18	15	4	9	
	6	20	18	14	3	7	
48	1	2	6	11	16	18	
	2	5	6	10	15	16	
	3	10	6	9	15	13	
	4	14	6	9	15	7	
	5	19	5	7	14	4	
	6	20	5	6	13	2	
49	1	3	12	13	17	20	
	2	7	9	12	16	19	
	3	8	8	12	14	19	
	4	16	6	12	14	19	
	5	17	5	11	12	19	
	6	20	4	10	9	19	
50	1	7	13	9	1	14	
	2	13	9	8	1	13	
	3	17	7	8	1	13	
	4	18	6	8	1	13	
	5	19	5	7	1	12	
	6	20	3	7	1	12	
51	1	4	20	7	17	18	
	2	5	17	7	15	18	
	3	14	13	6	15	16	
	4	18	10	4	14	15	
	5	19	8	3	13	15	
	6	20	6	2	13	14	
52	1	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2
	58	58	497	531

************************************************************************
