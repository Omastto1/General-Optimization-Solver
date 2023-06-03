jobs  (incl. supersource/sink ):	102
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 4 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	9		2 3 4 5 6 8 10 11 13 
2	3	8		24 23 22 18 15 14 12 9 
3	3	7		25 24 22 20 19 16 7 
4	3	5		25 24 22 12 9 
5	3	8		30 27 25 24 23 18 16 12 
6	3	9		30 28 26 25 24 22 20 17 15 
7	3	7		30 29 28 23 21 17 15 
8	3	5		20 17 16 15 14 
9	3	9		38 33 31 30 28 27 26 20 16 
10	3	6		34 28 26 23 17 15 
11	3	8		38 37 26 25 24 23 22 20 
12	3	4		34 29 28 17 
13	3	7		40 38 37 32 27 25 22 
14	3	6		34 31 28 27 26 25 
15	3	6		40 37 36 33 32 27 
16	3	8		48 41 40 39 37 36 34 32 
17	3	7		48 39 38 37 33 32 31 
18	3	7		48 41 40 39 36 33 26 
19	3	5		41 39 38 34 23 
20	3	7		48 43 40 39 36 35 34 
21	3	6		48 40 34 33 31 26 
22	3	6		48 46 44 41 35 31 
23	3	7		58 53 48 47 46 40 35 
24	3	7		58 48 47 46 44 43 35 
25	3	5		48 47 43 35 33 
26	3	8		58 52 47 46 45 44 43 42 
27	3	6		53 48 45 44 41 39 
28	3	5		53 47 46 36 35 
29	3	7		58 48 47 46 45 43 41 
30	3	9		59 58 54 52 48 47 45 44 42 
31	3	6		58 50 47 45 43 42 
32	3	3		47 44 35 
33	3	6		59 58 54 52 44 42 
34	3	9		59 58 57 55 54 53 49 47 46 
35	3	6		59 54 52 50 45 42 
36	3	5		59 54 45 44 42 
37	3	4		58 49 46 43 
38	3	5		59 53 52 49 44 
39	3	7		61 59 58 57 54 49 47 
40	3	3		54 49 44 
41	3	7		61 59 57 55 52 51 49 
42	3	6		64 61 57 55 51 49 
43	3	6		59 57 55 54 53 51 
44	3	5		64 61 57 56 50 
45	3	4		69 61 55 49 
46	3	7		71 69 64 63 62 61 56 
47	3	3		64 60 51 
48	3	7		75 71 69 66 64 63 61 
49	3	5		71 63 62 60 56 
50	3	1		51 
51	3	6		71 69 68 66 63 62 
52	3	4		71 63 62 56 
53	3	5		69 66 64 63 62 
54	3	5		74 71 70 65 62 
55	3	2		62 56 
56	3	7		75 74 73 70 68 67 66 
57	3	7		79 76 75 73 71 68 67 
58	3	7		79 76 74 73 72 70 68 
59	3	2		76 62 
60	3	5		79 76 75 73 67 
61	3	4		74 73 72 65 
62	3	4		79 75 73 67 
63	3	8		88 84 82 77 76 74 72 70 
64	3	2		76 67 
65	3	6		85 84 80 78 76 68 
66	3	9		100 88 87 84 83 82 79 76 72 
67	3	8		101 88 84 83 82 81 77 72 
68	3	10		101 99 98 90 88 86 83 82 81 77 
69	3	10		100 98 91 90 88 87 86 85 82 80 
70	3	8		100 98 90 87 86 85 83 80 
71	3	7		100 99 98 90 88 85 81 
72	3	6		98 91 90 85 80 78 
73	3	5		99 91 90 84 78 
74	3	5		101 100 99 87 78 
75	3	5		99 98 87 86 85 
76	3	7		101 97 95 94 93 90 89 
77	3	6		100 96 94 93 91 89 
78	3	4		97 96 89 86 
79	3	4		98 96 89 86 
80	3	5		99 97 94 93 92 
81	3	3		96 93 87 
82	3	4		97 95 93 92 
83	3	4		96 94 93 92 
84	3	4		98 95 93 92 
85	3	3		96 94 89 
86	3	2		94 93 
87	3	2		95 92 
88	3	2		97 92 
89	3	1		92 
90	3	1		92 
91	3	1		95 
92	3	1		102 
93	3	1		102 
94	3	1		102 
95	3	1		102 
96	3	1		102 
97	3	1		102 
98	3	1		102 
99	3	1		102 
100	3	1		102 
101	3	1		102 
102	1	0		
************************************************************************
REQUESTS/DURATIONS
jobnr.	mode	dur	R1	R2	N1	N2	N3	N4	
------------------------------------------------------------------------
1	1	0	0	0	0	0	0	0	
2	1	3	8	4	9	10	5	7	
	2	6	8	2	5	7	5	7	
	3	8	8	1	4	5	4	7	
3	1	2	5	3	8	2	6	2	
	2	6	4	3	8	2	6	2	
	3	7	4	3	8	2	6	1	
4	1	4	8	9	7	10	8	6	
	2	5	4	8	6	9	6	5	
	3	8	2	8	5	9	3	4	
5	1	3	6	9	8	10	4	9	
	2	5	6	8	8	8	4	5	
	3	10	5	8	5	6	4	4	
6	1	1	9	6	8	3	8	8	
	2	4	8	4	6	2	8	6	
	3	7	6	4	6	2	7	2	
7	1	1	4	5	5	9	10	9	
	2	5	2	5	3	6	7	9	
	3	9	2	4	2	3	3	9	
8	1	4	6	7	10	5	10	9	
	2	5	6	5	9	4	9	9	
	3	6	3	2	8	3	9	9	
9	1	2	7	8	7	5	9	10	
	2	3	5	8	6	3	5	5	
	3	9	4	8	5	1	3	2	
10	1	1	6	9	2	5	3	6	
	2	2	5	9	2	4	3	6	
	3	10	5	9	2	2	3	4	
11	1	2	6	4	8	8	10	9	
	2	3	5	2	8	8	10	7	
	3	4	5	2	8	8	10	4	
12	1	3	6	6	10	2	7	7	
	2	8	4	4	6	2	5	7	
	3	9	3	2	4	1	5	6	
13	1	5	7	8	7	8	8	7	
	2	6	4	5	4	8	6	5	
	3	7	2	3	3	8	6	1	
14	1	5	10	5	10	9	6	5	
	2	6	7	4	9	9	5	3	
	3	7	7	3	9	8	2	3	
15	1	3	10	6	10	9	7	6	
	2	4	6	5	8	7	6	6	
	3	8	3	4	8	3	5	5	
16	1	6	4	9	8	8	10	3	
	2	8	4	4	8	7	8	3	
	3	10	3	4	8	5	4	3	
17	1	3	9	9	9	9	2	7	
	2	7	8	7	7	9	2	6	
	3	10	8	6	6	7	2	6	
18	1	1	6	6	10	5	7	5	
	2	7	6	4	6	4	5	4	
	3	8	6	2	4	4	3	2	
19	1	6	6	9	9	2	2	4	
	2	9	4	7	7	2	1	4	
	3	10	4	7	5	2	1	3	
20	1	1	7	8	8	7	6	5	
	2	3	7	5	5	5	6	5	
	3	6	7	2	5	2	3	5	
21	1	2	7	5	8	1	5	9	
	2	7	5	4	7	1	5	7	
	3	8	2	4	5	1	5	7	
22	1	3	8	6	5	9	4	5	
	2	5	7	5	4	9	4	5	
	3	8	1	5	3	9	2	3	
23	1	5	5	7	7	6	1	3	
	2	6	4	5	6	2	1	3	
	3	8	4	4	5	2	1	3	
24	1	2	6	6	5	5	9	7	
	2	3	4	6	5	5	9	5	
	3	10	2	2	5	4	9	3	
25	1	1	8	10	8	8	6	7	
	2	4	4	9	6	7	6	4	
	3	5	4	9	4	7	4	1	
26	1	5	9	6	6	10	8	5	
	2	6	8	5	6	8	7	4	
	3	7	8	5	6	7	6	4	
27	1	5	6	9	3	9	4	4	
	2	9	3	9	3	9	4	3	
	3	10	2	9	3	9	3	1	
28	1	2	6	9	7	2	7	3	
	2	4	5	4	5	1	6	3	
	3	8	5	3	4	1	5	1	
29	1	3	5	6	9	5	7	9	
	2	6	3	5	9	4	7	8	
	3	10	3	4	9	3	3	7	
30	1	4	6	3	4	3	8	8	
	2	6	5	3	3	3	7	6	
	3	9	4	2	3	2	5	6	
31	1	5	10	9	9	2	7	6	
	2	7	7	8	6	2	6	5	
	3	8	4	8	4	2	6	3	
32	1	1	6	7	9	6	7	7	
	2	8	6	6	8	5	5	6	
	3	10	4	6	7	5	3	5	
33	1	4	6	6	3	9	8	8	
	2	5	4	5	2	9	5	6	
	3	9	2	5	1	9	3	3	
34	1	4	6	9	6	6	8	6	
	2	6	6	8	5	6	8	5	
	3	10	3	8	5	4	8	3	
35	1	4	8	6	7	7	6	6	
	2	7	5	4	6	4	6	4	
	3	8	3	3	5	4	4	3	
36	1	6	6	7	7	5	8	5	
	2	7	5	7	6	5	5	4	
	3	8	2	7	6	3	2	4	
37	1	3	6	10	7	2	5	6	
	2	6	4	10	6	1	5	4	
	3	8	2	10	5	1	4	1	
38	1	1	7	9	7	7	8	9	
	2	5	6	8	5	5	8	5	
	3	7	5	7	4	4	6	4	
39	1	7	8	6	7	8	5	7	
	2	9	7	5	4	8	4	7	
	3	10	7	5	2	8	4	7	
40	1	1	3	8	9	4	7	10	
	2	2	3	7	6	3	6	9	
	3	10	2	7	5	3	6	9	
41	1	1	5	9	5	1	4	6	
	2	3	3	6	3	1	3	6	
	3	8	3	4	2	1	1	6	
42	1	5	5	7	3	8	2	3	
	2	6	5	6	3	4	2	2	
	3	10	5	5	2	2	2	2	
43	1	7	4	6	6	10	5	10	
	2	9	2	5	4	7	5	8	
	3	10	1	3	2	5	4	8	
44	1	1	9	7	2	6	4	9	
	2	4	9	5	2	5	2	8	
	3	5	9	4	2	5	1	7	
45	1	3	6	5	6	8	6	4	
	2	4	5	2	6	8	5	4	
	3	5	2	1	5	6	4	4	
46	1	5	6	8	1	6	8	3	
	2	6	6	5	1	3	7	2	
	3	10	6	2	1	3	6	2	
47	1	1	6	5	8	2	5	8	
	2	3	4	4	6	1	5	6	
	3	4	4	1	6	1	5	4	
48	1	1	6	7	8	5	3	9	
	2	2	4	5	6	3	3	7	
	3	6	4	5	5	3	1	5	
49	1	1	8	6	2	10	8	7	
	2	2	8	5	2	8	6	7	
	3	8	7	3	2	6	3	2	
50	1	4	9	4	1	4	9	5	
	2	8	9	3	1	3	5	4	
	3	10	9	2	1	3	4	3	
51	1	1	2	8	9	6	2	8	
	2	6	1	8	5	4	2	6	
	3	7	1	7	2	3	2	4	
52	1	1	2	9	7	3	5	5	
	2	6	2	8	5	2	2	4	
	3	7	2	7	2	2	2	2	
53	1	4	9	7	6	5	5	8	
	2	5	9	5	5	3	4	8	
	3	9	8	2	5	1	4	8	
54	1	5	5	10	4	8	7	5	
	2	6	4	6	3	8	7	3	
	3	7	2	6	3	7	7	1	
55	1	1	9	8	5	6	3	2	
	2	3	9	6	4	5	2	2	
	3	6	9	4	3	5	2	2	
56	1	3	10	6	8	5	3	9	
	2	5	8	5	8	3	3	7	
	3	9	6	3	8	3	3	3	
57	1	4	5	3	4	10	8	5	
	2	8	5	3	4	8	4	4	
	3	9	5	2	4	4	2	2	
58	1	2	6	6	10	9	9	8	
	2	6	4	3	6	9	8	6	
	3	8	3	3	5	9	8	6	
59	1	5	5	10	8	7	7	7	
	2	7	4	8	6	5	7	5	
	3	10	4	4	5	2	7	4	
60	1	1	10	8	7	6	7	9	
	2	2	8	8	3	5	7	6	
	3	6	8	8	2	4	7	4	
61	1	1	8	6	8	9	6	4	
	2	4	6	5	4	9	4	3	
	3	6	6	5	3	9	3	2	
62	1	3	9	6	3	7	9	2	
	2	6	5	4	2	4	5	1	
	3	7	3	4	1	3	4	1	
63	1	3	7	5	4	6	5	4	
	2	7	4	5	3	4	4	4	
	3	8	4	4	3	3	2	4	
64	1	5	10	3	7	6	5	8	
	2	8	8	2	6	5	3	7	
	3	10	8	2	6	1	3	5	
65	1	5	4	10	5	7	3	7	
	2	6	3	8	2	6	3	5	
	3	7	2	7	2	6	3	3	
66	1	4	10	8	9	6	8	7	
	2	7	4	8	8	4	7	4	
	3	10	3	5	6	4	6	2	
67	1	5	2	9	6	8	4	2	
	2	6	1	6	6	6	4	2	
	3	10	1	4	6	5	4	2	
68	1	4	3	10	10	3	10	6	
	2	5	2	10	4	3	9	3	
	3	7	2	10	3	3	8	2	
69	1	7	5	7	5	6	6	5	
	2	8	5	6	2	5	4	4	
	3	10	3	6	2	4	4	4	
70	1	1	8	8	6	1	2	3	
	2	2	6	5	4	1	1	3	
	3	4	5	1	4	1	1	3	
71	1	7	9	9	5	7	7	5	
	2	8	7	8	5	3	7	5	
	3	9	7	8	1	2	6	3	
72	1	1	9	4	1	6	6	4	
	2	2	6	4	1	4	6	3	
	3	10	5	3	1	3	6	3	
73	1	2	9	8	8	10	7	9	
	2	5	6	7	6	9	7	7	
	3	7	6	5	6	9	7	6	
74	1	2	9	5	1	3	9	7	
	2	3	6	4	1	2	8	6	
	3	4	5	3	1	1	7	5	
75	1	5	8	6	8	9	4	9	
	2	8	8	5	6	7	3	5	
	3	9	6	5	5	5	2	4	
76	1	6	9	5	10	7	7	5	
	2	7	9	4	8	6	7	4	
	3	10	8	4	8	6	6	4	
77	1	3	8	7	6	3	7	6	
	2	4	6	5	5	3	6	5	
	3	8	5	5	2	3	6	2	
78	1	1	8	3	9	9	10	7	
	2	2	7	3	5	8	9	6	
	3	4	3	3	1	7	9	5	
79	1	3	8	6	7	3	7	1	
	2	7	8	5	6	3	6	1	
	3	10	7	4	5	3	6	1	
80	1	5	10	7	6	3	8	10	
	2	8	8	4	5	2	6	9	
	3	10	8	3	4	2	6	8	
81	1	4	6	8	3	5	6	7	
	2	5	4	7	3	3	3	7	
	3	9	4	6	3	2	3	2	
82	1	5	9	2	7	6	6	5	
	2	7	8	2	6	5	3	5	
	3	8	8	2	5	3	2	5	
83	1	5	3	7	9	2	9	10	
	2	8	2	6	7	2	7	8	
	3	9	2	5	4	2	7	5	
84	1	7	9	3	7	3	8	4	
	2	8	9	2	4	2	8	4	
	3	9	9	2	2	1	8	3	
85	1	1	4	8	6	10	6	6	
	2	7	3	8	3	9	4	5	
	3	10	2	8	1	9	2	4	
86	1	6	4	5	7	4	6	3	
	2	7	3	5	7	4	5	2	
	3	8	2	5	7	3	3	1	
87	1	2	2	10	5	7	6	3	
	2	3	2	6	5	7	2	2	
	3	5	2	5	5	6	1	2	
88	1	3	7	8	9	6	4	1	
	2	6	6	5	7	5	2	1	
	3	10	6	3	7	1	1	1	
89	1	2	7	9	9	6	9	6	
	2	7	7	8	8	5	8	5	
	3	9	5	8	8	5	4	5	
90	1	2	5	8	5	8	9	9	
	2	3	5	8	3	7	7	8	
	3	10	5	8	1	6	4	8	
91	1	4	6	8	8	8	4	7	
	2	6	6	5	7	5	4	6	
	3	7	5	2	7	4	4	5	
92	1	3	7	2	7	9	1	8	
	2	4	6	2	5	7	1	8	
	3	5	5	2	3	5	1	8	
93	1	5	7	10	10	5	5	6	
	2	6	6	7	8	4	3	5	
	3	7	6	4	8	2	3	5	
94	1	1	8	8	3	5	8	9	
	2	2	4	7	3	3	7	6	
	3	7	4	4	3	3	7	6	
95	1	2	8	8	5	7	6	6	
	2	3	7	8	3	5	5	4	
	3	4	5	8	3	1	4	2	
96	1	5	4	7	6	9	6	6	
	2	6	4	7	4	7	4	6	
	3	10	3	4	3	4	4	6	
97	1	1	3	10	6	2	6	7	
	2	3	3	8	5	1	4	6	
	3	4	3	8	5	1	4	5	
98	1	4	7	6	6	5	9	4	
	2	6	6	5	5	4	9	3	
	3	9	6	5	4	4	7	3	
99	1	2	8	9	7	4	9	8	
	2	5	6	7	5	2	6	5	
	3	6	4	6	4	2	6	1	
100	1	3	7	4	6	8	7	9	
	2	6	7	3	4	7	6	7	
	3	7	7	2	3	7	4	5	
101	1	2	6	7	2	7	7	7	
	2	5	6	4	1	4	7	5	
	3	8	6	2	1	3	3	2	
102	1	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2	N 3	N 4
	24	25	478	451	479	450

************************************************************************
