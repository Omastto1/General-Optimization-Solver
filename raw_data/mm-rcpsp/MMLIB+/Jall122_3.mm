jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	10		2 3 4 5 8 11 12 13 16 18 
2	6	4		14 10 7 6 
3	6	6		24 23 22 20 19 9 
4	6	6		37 21 20 19 17 7 
5	6	9		37 25 24 22 21 20 19 17 15 
6	6	8		37 36 32 28 24 23 22 17 
7	6	7		36 34 31 28 25 24 15 
8	6	9		37 36 34 32 31 30 29 26 22 
9	6	7		37 32 29 27 26 25 21 
10	6	7		32 31 29 28 27 26 19 
11	6	7		37 34 29 28 27 26 21 
12	6	8		50 37 36 34 30 29 25 22 
13	6	6		37 36 34 32 30 22 
14	6	7		49 40 37 34 29 27 26 
15	6	5		32 30 29 27 26 
16	6	8		51 41 37 35 34 33 32 30 
17	6	7		51 50 41 34 30 29 27 
18	6	5		51 50 34 28 24 
19	6	8		51 50 49 46 40 36 34 33 
20	6	7		51 49 47 40 39 33 28 
21	6	8		50 48 47 46 44 41 36 30 
22	6	6		51 49 40 39 35 27 
23	6	6		48 46 35 34 31 30 
24	6	5		49 47 45 40 29 
25	6	5		49 46 40 39 33 
26	6	7		51 50 47 46 43 41 39 
27	6	4		47 46 45 33 
28	6	3		46 41 35 
29	6	3		48 39 35 
30	6	4		49 40 39 38 
31	6	4		47 44 40 38 
32	6	5		50 49 46 43 42 
33	6	4		48 44 43 38 
34	6	4		47 45 44 43 
35	6	3		44 43 38 
36	6	1		39 
37	6	1		38 
38	6	1		42 
39	6	1		42 
40	6	1		43 
41	6	1		45 
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
jobnr.	mode	dur	R1	R2	N1	N2	
------------------------------------------------------------------------
1	1	0	0	0	0	0	
2	1	1	9	4	8	16	
	2	3	9	4	5	14	
	3	6	9	4	5	13	
	4	11	9	4	3	8	
	5	13	9	4	3	6	
	6	16	9	4	2	3	
3	1	7	15	11	19	15	
	2	11	15	9	18	12	
	3	16	15	9	18	9	
	4	17	15	8	18	7	
	5	19	15	6	17	7	
	6	20	15	6	17	4	
4	1	11	14	11	19	14	
	2	16	11	7	19	14	
	3	17	9	6	18	13	
	4	18	9	4	18	13	
	5	19	6	3	17	12	
	6	20	5	2	16	11	
5	1	1	11	13	13	12	
	2	3	10	12	12	11	
	3	4	10	11	12	10	
	4	6	10	6	11	10	
	5	7	8	4	11	8	
	6	15	8	3	10	8	
6	1	3	8	13	16	14	
	2	12	8	11	13	12	
	3	13	7	9	13	12	
	4	15	5	8	11	12	
	5	18	5	7	7	10	
	6	19	4	7	6	10	
7	1	6	19	5	15	17	
	2	9	15	5	13	15	
	3	11	14	5	12	13	
	4	13	10	5	11	11	
	5	16	7	5	9	9	
	6	20	7	5	7	9	
8	1	2	6	14	3	7	
	2	3	6	13	2	7	
	3	6	5	13	2	6	
	4	7	5	12	2	4	
	5	12	5	12	2	3	
	6	13	4	12	2	2	
9	1	2	6	11	19	5	
	2	3	5	10	13	4	
	3	4	5	10	10	4	
	4	12	5	10	7	3	
	5	17	4	9	5	3	
	6	20	4	9	1	3	
10	1	8	13	16	8	4	
	2	14	12	14	7	4	
	3	17	11	13	6	4	
	4	18	11	12	6	4	
	5	19	10	10	5	4	
	6	20	9	9	4	4	
11	1	1	7	14	18	15	
	2	9	6	13	16	14	
	3	10	6	12	13	14	
	4	13	6	11	10	14	
	5	17	4	7	8	12	
	6	19	4	6	8	12	
12	1	8	19	15	12	2	
	2	11	15	14	11	2	
	3	12	15	14	7	2	
	4	13	12	13	6	2	
	5	15	11	13	5	1	
	6	17	10	12	1	1	
13	1	4	14	17	13	18	
	2	10	13	16	12	18	
	3	11	11	15	12	15	
	4	12	10	14	12	15	
	5	13	9	13	11	12	
	6	18	7	13	11	10	
14	1	1	11	17	11	7	
	2	3	11	16	11	6	
	3	12	9	16	10	4	
	4	13	6	15	8	3	
	5	18	5	15	8	2	
	6	20	3	15	6	2	
15	1	5	19	18	9	16	
	2	6	18	17	8	15	
	3	7	18	17	7	15	
	4	14	18	17	6	15	
	5	18	18	16	4	15	
	6	20	18	16	4	14	
16	1	1	15	9	12	17	
	2	2	11	9	10	15	
	3	4	11	9	9	15	
	4	17	8	9	8	13	
	5	18	7	9	5	12	
	6	20	6	9	5	10	
17	1	4	17	15	14	16	
	2	6	17	13	9	16	
	3	7	17	12	7	16	
	4	12	16	12	5	15	
	5	17	16	9	5	15	
	6	18	16	9	1	14	
18	1	3	15	15	16	10	
	2	5	15	13	15	8	
	3	6	14	12	14	6	
	4	7	13	11	13	6	
	5	8	12	10	11	4	
	6	19	11	10	10	2	
19	1	3	1	17	17	19	
	2	8	1	16	15	16	
	3	11	1	16	14	13	
	4	12	1	16	13	12	
	5	13	1	16	10	8	
	6	17	1	16	9	7	
20	1	9	11	2	17	18	
	2	10	9	1	16	17	
	3	11	9	1	16	16	
	4	12	5	1	15	17	
	5	18	5	1	15	16	
	6	19	2	1	15	17	
21	1	4	12	16	14	15	
	2	9	11	15	13	13	
	3	10	10	13	13	11	
	4	12	9	10	12	11	
	5	17	8	8	12	10	
	6	18	8	8	12	9	
22	1	3	19	6	8	9	
	2	6	19	6	7	8	
	3	13	19	5	7	8	
	4	17	19	4	6	7	
	5	19	19	3	5	6	
	6	20	19	2	5	6	
23	1	11	9	8	9	11	
	2	14	8	8	8	10	
	3	17	7	8	7	8	
	4	18	7	8	7	5	
	5	19	6	8	5	2	
	6	20	5	8	5	2	
24	1	6	11	13	11	8	
	2	12	9	12	11	5	
	3	15	7	12	11	4	
	4	16	7	10	11	4	
	5	19	5	9	11	2	
	6	20	3	8	11	1	
25	1	1	3	18	17	10	
	2	4	3	16	17	10	
	3	5	3	16	17	9	
	4	6	2	15	17	7	
	5	9	2	12	17	6	
	6	14	2	11	17	5	
26	1	1	9	6	12	18	
	2	2	9	6	10	15	
	3	3	6	5	8	12	
	4	12	4	5	8	10	
	5	18	3	5	4	7	
	6	19	1	4	3	2	
27	1	1	5	1	9	8	
	2	2	5	1	8	7	
	3	7	5	1	7	7	
	4	11	5	1	6	6	
	5	13	5	1	6	5	
	6	20	5	1	4	5	
28	1	2	18	12	14	8	
	2	3	17	11	12	6	
	3	5	17	10	12	6	
	4	8	16	10	11	6	
	5	13	15	10	10	5	
	6	14	14	9	10	4	
29	1	4	5	15	18	8	
	2	7	4	13	18	7	
	3	16	4	11	18	5	
	4	18	3	11	18	4	
	5	19	2	8	17	4	
	6	20	2	8	17	3	
30	1	6	20	19	17	9	
	2	12	16	17	14	8	
	3	13	13	14	14	8	
	4	15	12	13	12	7	
	5	18	10	13	10	7	
	6	19	6	10	8	7	
31	1	5	9	14	16	3	
	2	7	9	10	14	2	
	3	11	9	9	12	2	
	4	16	9	7	11	2	
	5	18	9	4	10	1	
	6	19	9	4	9	1	
32	1	1	7	13	15	8	
	2	7	7	13	13	7	
	3	9	7	13	13	6	
	4	10	7	12	13	5	
	5	11	7	12	11	4	
	6	15	7	12	11	3	
33	1	11	17	13	13	12	
	2	12	17	13	13	11	
	3	16	16	13	8	10	
	4	17	14	12	7	10	
	5	18	12	11	3	9	
	6	20	11	11	3	9	
34	1	4	6	17	5	7	
	2	8	5	13	4	7	
	3	12	3	12	4	7	
	4	13	3	10	2	6	
	5	14	1	8	1	6	
	6	17	1	4	1	5	
35	1	5	20	14	15	19	
	2	9	17	11	15	18	
	3	11	16	9	12	16	
	4	16	15	6	10	15	
	5	18	14	4	9	13	
	6	20	14	1	5	12	
36	1	2	12	16	19	11	
	2	3	10	15	18	11	
	3	12	10	14	18	9	
	4	17	9	14	17	9	
	5	18	8	12	16	6	
	6	19	7	12	16	5	
37	1	2	4	16	10	8	
	2	3	4	15	10	8	
	3	4	4	15	9	8	
	4	10	4	15	8	8	
	5	12	4	15	7	7	
	6	17	4	15	6	7	
38	1	1	11	10	5	19	
	2	8	9	9	4	17	
	3	10	7	7	3	13	
	4	11	7	4	3	11	
	5	18	5	2	2	9	
	6	19	2	2	1	6	
39	1	1	17	13	10	8	
	2	3	16	12	10	7	
	3	5	15	12	10	7	
	4	12	15	12	10	4	
	5	13	13	12	10	3	
	6	18	13	12	10	2	
40	1	4	19	11	16	12	
	2	7	19	10	12	11	
	3	11	19	10	11	9	
	4	13	19	9	9	9	
	5	15	19	9	8	7	
	6	18	19	9	7	4	
41	1	2	12	17	3	16	
	2	10	12	14	3	14	
	3	13	8	14	3	12	
	4	14	8	12	2	11	
	5	17	5	10	1	9	
	6	18	3	9	1	7	
42	1	1	18	19	11	12	
	2	4	15	18	11	12	
	3	8	12	18	11	12	
	4	15	8	17	11	12	
	5	16	5	17	11	12	
	6	17	3	16	11	12	
43	1	3	13	14	7	16	
	2	7	10	10	5	15	
	3	11	8	8	5	15	
	4	14	7	7	4	15	
	5	17	5	3	4	15	
	6	19	4	1	3	15	
44	1	4	16	13	15	19	
	2	7	13	11	13	15	
	3	9	12	11	11	14	
	4	13	7	10	8	9	
	5	18	5	8	8	6	
	6	19	2	5	6	3	
45	1	4	20	13	4	18	
	2	11	20	13	3	17	
	3	13	20	13	3	16	
	4	14	20	13	3	15	
	5	16	20	13	3	12	
	6	19	20	13	3	11	
46	1	3	18	15	12	13	
	2	8	17	13	10	12	
	3	9	15	11	10	12	
	4	10	15	8	6	12	
	5	19	14	5	4	12	
	6	20	12	3	3	12	
47	1	4	17	16	17	14	
	2	7	15	14	14	12	
	3	16	13	10	14	9	
	4	17	13	6	11	8	
	5	18	11	5	5	5	
	6	20	9	2	2	3	
48	1	2	8	13	17	16	
	2	4	8	10	15	14	
	3	7	7	9	15	11	
	4	11	7	5	14	9	
	5	19	7	5	13	8	
	6	20	6	2	12	4	
49	1	2	9	10	6	18	
	2	6	9	10	6	17	
	3	15	8	10	6	17	
	4	16	6	9	6	17	
	5	17	5	8	6	17	
	6	18	5	8	6	16	
50	1	3	18	10	10	8	
	2	5	16	9	10	8	
	3	14	12	7	10	8	
	4	16	9	6	10	8	
	5	18	7	6	10	8	
	6	20	2	4	10	8	
51	1	1	17	10	15	4	
	2	3	15	8	13	4	
	3	12	12	8	12	3	
	4	13	9	6	8	3	
	5	14	5	6	6	3	
	6	18	3	5	4	2	
52	1	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2
	75	78	493	472

************************************************************************
