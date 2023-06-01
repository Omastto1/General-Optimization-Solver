jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 4 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	4		2 5 6 8 
2	6	2		7 3 
3	6	1		4 
4	6	5		18 13 12 10 9 
5	6	2		14 9 
6	6	4		23 20 18 12 
7	6	3		17 14 11 
8	6	3		18 17 13 
9	6	2		17 11 
10	6	4		20 17 15 14 
11	6	5		23 21 20 19 15 
12	6	3		19 17 15 
13	6	4		23 20 19 15 
14	6	3		27 22 19 
15	6	2		22 16 
16	6	6		33 30 29 27 26 24 
17	6	4		27 26 25 21 
18	6	4		34 31 25 22 
19	6	5		33 31 30 28 24 
20	6	4		33 30 29 26 
21	6	7		42 36 33 32 31 30 29 
22	6	3		33 29 26 
23	6	5		42 34 33 30 28 
24	6	7		42 40 39 38 35 34 32 
25	6	6		42 41 40 36 33 29 
26	6	2		42 28 
27	6	6		46 42 40 36 35 31 
28	6	5		40 39 38 36 32 
29	6	5		46 39 38 37 35 
30	6	3		41 40 38 
31	6	4		45 39 38 37 
32	6	3		43 41 37 
33	6	3		46 39 37 
34	6	2		45 36 
35	6	4		51 45 44 43 
36	6	1		37 
37	6	5		51 50 49 47 44 
38	6	5		51 50 49 47 44 
39	6	3		51 48 43 
40	6	2		44 43 
41	6	2		46 45 
42	6	1		43 
43	6	3		50 49 47 
44	6	1		48 
45	6	1		47 
46	6	1		48 
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
2	1	2	8	16	9	19	19	14	
	2	9	7	15	9	17	18	13	
	3	10	6	13	9	14	16	11	
	4	15	5	12	9	10	16	8	
	5	17	2	12	9	8	14	8	
	6	18	2	11	9	5	14	5	
3	1	4	3	18	10	10	17	9	
	2	6	2	17	10	10	15	8	
	3	9	2	17	9	10	15	7	
	4	14	1	16	7	9	12	5	
	5	19	1	16	6	9	12	4	
	6	20	1	16	6	9	11	3	
4	1	5	19	9	10	16	17	15	
	2	6	14	9	10	15	12	12	
	3	7	12	9	10	15	12	11	
	4	8	9	9	10	15	8	9	
	5	19	7	9	10	14	6	8	
	6	20	2	9	10	14	3	6	
5	1	4	12	12	18	19	13	4	
	2	13	11	10	16	17	13	4	
	3	15	11	9	15	16	13	4	
	4	17	11	8	14	14	13	4	
	5	19	11	6	13	12	13	4	
	6	20	11	6	12	10	13	4	
6	1	2	14	19	20	16	8	12	
	2	11	14	17	19	15	6	11	
	3	12	14	14	19	13	6	11	
	4	16	14	12	18	12	5	9	
	5	19	13	12	18	12	4	9	
	6	20	13	10	17	11	4	8	
7	1	3	9	10	16	4	9	20	
	2	4	9	8	15	4	8	19	
	3	6	7	7	15	3	7	19	
	4	7	6	6	15	3	6	19	
	5	11	5	4	14	3	6	19	
	6	12	5	3	14	2	4	19	
8	1	1	19	17	8	15	18	8	
	2	11	18	15	6	13	17	8	
	3	12	18	14	5	10	17	8	
	4	13	18	12	4	9	17	7	
	5	17	18	11	2	6	15	7	
	6	19	18	9	1	6	15	7	
9	1	1	16	14	13	17	15	12	
	2	3	15	12	12	12	15	11	
	3	12	14	11	11	10	14	9	
	4	14	11	10	6	7	14	8	
	5	18	11	7	4	4	14	6	
	6	20	9	6	1	3	13	5	
10	1	4	17	6	13	10	10	19	
	2	11	17	6	11	9	9	17	
	3	12	15	4	9	7	8	17	
	4	13	15	4	8	6	8	15	
	5	19	13	2	7	5	7	14	
	6	20	12	2	5	4	6	14	
11	1	6	11	12	8	12	9	13	
	2	10	11	12	7	10	8	12	
	3	11	10	12	7	9	8	12	
	4	13	8	12	6	6	7	12	
	5	14	8	11	4	3	7	12	
	6	15	6	11	3	3	7	12	
12	1	3	16	16	11	15	17	17	
	2	6	15	13	10	15	14	14	
	3	10	12	11	10	14	13	13	
	4	16	12	8	10	14	11	10	
	5	17	10	8	9	14	9	9	
	6	20	9	6	9	13	8	6	
13	1	6	6	10	19	7	16	14	
	2	8	6	10	19	7	16	13	
	3	10	6	9	19	5	16	12	
	4	11	5	9	19	4	15	7	
	5	15	5	9	19	3	15	6	
	6	19	5	8	19	3	14	4	
14	1	5	11	16	17	11	15	16	
	2	6	11	16	13	11	13	15	
	3	7	9	10	12	10	9	13	
	4	10	8	7	9	8	7	12	
	5	14	7	5	7	7	6	11	
	6	15	6	3	5	4	3	11	
15	1	1	19	8	12	7	19	17	
	2	4	16	7	9	5	15	16	
	3	5	15	5	8	4	14	11	
	4	7	12	4	7	4	10	9	
	5	13	12	3	5	2	10	8	
	6	17	10	2	4	2	8	4	
16	1	1	18	13	16	19	16	5	
	2	10	15	13	14	17	16	5	
	3	15	14	12	11	15	15	5	
	4	16	12	12	8	15	13	5	
	5	17	11	11	4	11	13	5	
	6	19	10	11	3	11	12	5	
17	1	1	16	11	14	7	14	16	
	2	2	13	9	10	5	14	15	
	3	4	12	6	9	4	12	13	
	4	15	12	6	9	3	11	8	
	5	16	11	2	6	2	9	6	
	6	20	10	2	4	1	8	3	
18	1	2	6	11	19	6	8	7	
	2	3	5	11	18	6	7	6	
	3	13	5	11	16	5	5	6	
	4	14	3	11	13	5	5	5	
	5	16	3	11	12	5	4	4	
	6	17	1	11	11	4	3	3	
19	1	1	2	5	18	13	12	15	
	2	8	2	4	17	12	11	13	
	3	12	2	4	16	12	9	11	
	4	15	2	3	16	12	7	10	
	5	16	2	2	16	12	7	10	
	6	18	2	2	15	12	5	8	
20	1	2	19	8	11	11	13	9	
	2	7	18	8	10	9	11	7	
	3	10	17	8	9	8	11	6	
	4	16	16	8	6	5	7	5	
	5	17	16	8	6	4	5	4	
	6	20	14	8	5	2	5	2	
21	1	2	16	12	12	17	9	10	
	2	6	15	11	11	16	8	9	
	3	10	15	11	11	13	6	8	
	4	12	14	10	11	9	5	6	
	5	13	14	10	10	7	3	6	
	6	16	14	9	10	7	1	5	
22	1	1	20	14	18	16	13	8	
	2	11	17	12	14	16	13	7	
	3	12	16	10	14	13	12	6	
	4	13	14	6	10	10	11	6	
	5	15	13	5	10	10	10	3	
	6	17	12	3	8	6	9	3	
23	1	5	10	14	14	15	17	4	
	2	6	9	13	14	14	17	4	
	3	10	8	13	13	13	15	4	
	4	13	8	13	12	13	11	4	
	5	16	6	13	10	11	11	4	
	6	19	6	13	10	11	7	4	
24	1	8	8	20	13	14	10	11	
	2	9	8	17	13	14	9	9	
	3	10	6	17	12	10	8	9	
	4	18	6	15	9	8	8	5	
	5	19	5	14	8	7	6	3	
	6	20	3	13	7	6	6	1	
25	1	2	6	13	3	6	18	20	
	2	3	4	10	3	5	17	17	
	3	5	4	9	3	4	17	15	
	4	11	3	7	2	4	17	14	
	5	14	1	6	2	3	17	12	
	6	19	1	5	2	2	17	10	
26	1	3	17	9	11	20	11	3	
	2	5	17	9	8	19	9	3	
	3	8	15	7	8	19	7	3	
	4	13	11	7	6	19	6	3	
	5	16	10	6	4	18	4	2	
	6	18	8	5	2	18	3	2	
27	1	3	17	13	20	16	14	6	
	2	5	13	13	17	16	12	5	
	3	7	13	12	16	12	9	5	
	4	9	10	11	14	10	8	5	
	5	13	9	9	14	9	7	5	
	6	18	8	9	13	6	3	5	
28	1	1	13	12	16	7	19	12	
	2	11	12	12	14	6	17	10	
	3	12	11	12	12	5	17	7	
	4	13	9	11	10	5	16	5	
	5	14	8	11	10	5	14	3	
	6	16	7	11	8	4	13	1	
29	1	5	20	8	5	13	10	12	
	2	7	15	6	5	13	9	11	
	3	11	14	6	4	11	7	9	
	4	14	12	4	4	10	5	7	
	5	15	10	3	4	9	4	5	
	6	16	6	3	3	9	4	4	
30	1	1	14	16	17	11	5	15	
	2	3	13	15	15	9	4	15	
	3	4	13	15	13	8	4	11	
	4	5	13	15	8	5	3	11	
	5	10	13	15	5	3	2	7	
	6	19	13	15	5	1	2	6	
31	1	4	6	17	12	10	18	10	
	2	6	6	17	12	7	15	9	
	3	10	6	13	11	6	15	9	
	4	11	5	13	11	6	13	9	
	5	17	5	9	11	3	10	8	
	6	20	4	9	10	3	9	8	
32	1	1	14	16	6	10	16	12	
	2	2	13	14	5	10	15	11	
	3	3	12	9	5	7	12	9	
	4	4	11	9	4	4	11	7	
	5	7	9	7	4	4	10	5	
	6	18	8	3	3	2	8	2	
33	1	2	17	16	16	9	11	6	
	2	5	16	15	15	8	10	6	
	3	6	16	15	12	8	9	6	
	4	7	16	14	9	5	8	6	
	5	15	16	14	7	4	7	6	
	6	18	16	13	5	3	5	6	
34	1	2	12	8	11	12	18	13	
	2	10	10	8	9	12	17	12	
	3	11	8	8	8	10	17	11	
	4	16	4	8	8	9	16	11	
	5	17	2	8	6	9	16	10	
	6	18	1	8	5	8	15	9	
35	1	2	10	9	9	14	13	15	
	2	4	9	8	8	13	10	13	
	3	13	7	8	8	10	9	13	
	4	16	6	8	8	9	6	12	
	5	17	6	8	7	6	3	11	
	6	18	4	8	7	3	3	10	
36	1	1	16	15	14	10	12	9	
	2	2	14	13	14	8	12	9	
	3	5	14	12	14	8	10	8	
	4	7	11	11	14	6	10	8	
	5	13	9	10	13	6	9	7	
	6	20	9	10	13	4	8	7	
37	1	1	18	15	17	12	16	9	
	2	3	14	14	16	10	15	8	
	3	12	12	13	16	10	13	7	
	4	13	8	13	15	9	13	7	
	5	18	7	13	15	8	12	6	
	6	19	5	12	14	7	11	5	
38	1	1	8	15	10	8	19	15	
	2	5	7	11	9	7	18	14	
	3	8	7	11	8	6	17	11	
	4	12	7	9	8	6	14	9	
	5	18	6	5	7	5	12	5	
	6	19	6	3	7	4	12	2	
39	1	3	18	19	13	13	14	5	
	2	11	17	19	11	10	12	4	
	3	13	17	19	10	9	10	4	
	4	14	16	19	8	7	7	3	
	5	15	16	19	7	4	4	2	
	6	16	15	19	5	2	2	2	
40	1	2	19	16	11	9	8	15	
	2	5	18	15	11	9	8	14	
	3	6	18	11	11	8	5	12	
	4	14	18	8	11	6	4	11	
	5	16	18	7	11	5	4	10	
	6	20	18	5	11	4	1	10	
41	1	1	18	18	9	10	15	12	
	2	8	18	18	9	9	15	12	
	3	11	18	18	7	8	14	9	
	4	12	18	18	6	8	14	7	
	5	18	17	18	5	7	12	5	
	6	19	17	18	4	6	12	4	
42	1	5	14	15	19	9	18	10	
	2	10	13	15	17	9	14	8	
	3	13	10	14	16	9	14	7	
	4	14	7	13	16	9	11	6	
	5	15	6	11	14	9	10	4	
	6	20	3	11	14	9	6	1	
43	1	2	18	20	17	17	18	4	
	2	6	15	19	14	14	14	4	
	3	7	14	19	14	13	12	4	
	4	13	11	18	13	10	11	4	
	5	14	9	18	10	9	10	3	
	6	20	9	17	8	7	7	3	
44	1	7	11	8	12	17	13	13	
	2	12	10	7	11	16	13	10	
	3	14	8	7	11	15	12	9	
	4	15	6	7	9	15	11	9	
	5	19	3	7	8	14	10	6	
	6	20	2	7	8	14	9	6	
45	1	2	5	14	10	10	17	15	
	2	3	4	13	10	10	16	12	
	3	7	3	13	10	8	11	10	
	4	11	2	12	10	8	7	8	
	5	17	1	12	10	6	5	5	
	6	18	1	12	10	6	2	2	
46	1	5	12	16	2	17	18	5	
	2	9	10	12	2	17	18	5	
	3	12	9	10	2	15	18	5	
	4	14	6	9	2	13	17	5	
	5	18	4	7	1	12	17	5	
	6	20	1	5	1	11	16	5	
47	1	3	15	10	7	14	18	14	
	2	4	13	8	7	14	16	12	
	3	11	11	8	5	12	15	9	
	4	15	11	6	5	12	14	7	
	5	16	8	4	3	11	11	6	
	6	18	8	3	3	9	10	3	
48	1	4	11	2	12	12	12	11	
	2	5	11	2	11	10	12	11	
	3	11	11	2	9	10	12	11	
	4	12	11	2	8	8	11	11	
	5	15	11	2	7	8	11	11	
	6	20	11	2	5	7	10	11	
49	1	6	9	18	12	16	20	18	
	2	8	8	17	10	16	15	15	
	3	9	8	14	9	16	14	12	
	4	11	6	13	8	15	12	10	
	5	12	6	9	4	15	8	7	
	6	13	4	8	3	15	7	4	
50	1	2	10	9	16	9	15	11	
	2	4	8	8	15	9	13	9	
	3	10	6	8	15	9	10	8	
	4	11	5	6	15	8	6	8	
	5	14	3	6	15	8	4	6	
	6	17	2	5	15	8	2	5	
51	1	5	9	15	13	13	17	17	
	2	7	9	14	13	10	16	13	
	3	10	8	14	12	8	16	12	
	4	12	6	12	12	6	16	7	
	5	13	6	11	11	5	16	7	
	6	16	5	10	11	4	16	3	
52	1	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2	N 3	N 4
	57	59	575	546	635	506

************************************************************************
