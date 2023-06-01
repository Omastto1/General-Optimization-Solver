jobs  (incl. supersource/sink ):	102
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	7		2 3 4 5 6 10 11 
2	3	9		23 20 19 17 16 15 14 13 12 
3	3	4		20 17 9 7 
4	3	5		22 20 13 12 7 
5	3	5		29 21 19 15 8 
6	3	5		31 25 23 20 12 
7	3	9		33 31 28 25 24 23 21 19 18 
8	3	5		31 24 23 20 13 
9	3	4		29 24 23 13 
10	3	6		33 31 28 24 19 18 
11	3	5		33 32 26 21 15 
12	3	6		33 32 28 24 21 18 
13	3	4		33 32 28 18 
14	3	7		36 34 32 31 30 28 26 
15	3	5		30 28 27 25 24 
16	3	5		40 33 28 24 21 
17	3	4		34 26 24 22 
18	3	5		39 36 30 27 26 
19	3	11		47 44 43 41 40 38 37 36 35 34 32 
20	3	9		44 43 41 40 39 38 36 32 28 
21	3	9		47 44 43 41 38 37 36 35 34 
22	3	7		44 43 41 40 39 36 28 
23	3	8		47 43 41 40 38 36 35 34 
24	3	9		53 47 44 42 41 39 37 36 35 
25	3	7		43 41 40 39 38 36 35 
26	3	6		44 43 41 40 38 37 
27	3	5		43 42 41 40 35 
28	3	6		53 47 45 42 37 35 
29	3	4		43 42 41 33 
30	3	7		61 57 49 48 47 44 40 
31	3	6		57 53 48 45 41 39 
32	3	6		53 52 50 49 45 42 
33	3	6		57 51 49 48 44 39 
34	3	5		53 51 49 45 39 
35	3	10		61 59 55 54 52 51 50 49 48 46 
36	3	10		62 61 60 59 57 55 52 51 48 45 
37	3	8		61 59 57 55 50 49 48 46 
38	3	4		62 54 50 42 
39	3	8		62 61 60 59 55 52 50 46 
40	3	7		66 62 59 55 53 52 45 
41	3	5		61 59 55 49 46 
42	3	6		61 59 55 51 48 46 
43	3	6		59 57 55 54 52 46 
44	3	5		66 59 55 54 46 
45	3	2		54 46 
46	3	5		78 67 63 58 56 
47	3	7		81 74 68 67 64 63 59 
48	3	6		78 68 66 64 63 58 
49	3	6		70 68 64 63 62 60 
50	3	7		78 76 72 71 68 67 58 
51	3	6		78 76 72 68 67 58 
52	3	4		78 77 63 58 
53	3	7		81 77 75 74 73 70 63 
54	3	6		81 75 73 70 67 63 
55	3	9		81 78 76 74 73 71 69 68 65 
56	3	4		76 74 68 64 
57	3	6		81 77 72 71 66 65 
58	3	7		81 75 74 73 70 69 65 
59	3	7		78 77 76 73 70 69 65 
60	3	5		78 77 71 66 65 
61	3	11		88 86 84 83 82 81 76 74 73 71 70 
62	3	10		86 83 82 80 79 76 74 73 71 69 
63	3	6		79 76 72 71 69 65 
64	3	8		90 86 80 79 75 73 72 69 
65	3	9		100 90 88 87 86 84 83 82 80 
66	3	8		90 87 86 85 83 79 76 73 
67	3	7		101 90 88 83 80 79 77 
68	3	8		101 100 98 90 87 83 79 77 
69	3	6		100 99 88 87 85 84 
70	3	7		100 99 98 95 87 85 79 
71	3	9		101 98 96 95 94 93 92 90 89 
72	3	6		99 98 96 95 89 85 
73	3	9		101 100 99 98 96 94 93 92 91 
74	3	5		101 99 97 95 85 
75	3	3		98 95 82 
76	3	8		99 98 96 95 94 93 92 91 
77	3	4		97 96 95 85 
78	3	4		100 96 95 85 
79	3	6		96 94 93 92 91 89 
80	3	6		99 98 96 95 91 89 
81	3	3		99 97 85 
82	3	2		99 85 
83	3	5		95 94 93 92 91 
84	3	4		98 95 91 89 
85	3	4		94 93 92 91 
86	3	4		99 95 92 91 
87	3	3		93 92 91 
88	3	2		98 96 
89	3	1		97 
90	3	1		91 
91	3	1		102 
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
jobnr.	mode	dur	R1	R2	N1	N2	
------------------------------------------------------------------------
1	1	0	0	0	0	0	
2	1	1	5	0	4	7	
	2	5	0	2	3	7	
	3	9	5	0	2	4	
3	1	1	0	4	8	10	
	2	4	0	2	8	8	
	3	5	1	0	8	8	
4	1	1	0	2	9	5	
	2	5	4	0	6	5	
	3	9	4	0	3	4	
5	1	1	5	0	7	8	
	2	2	0	2	5	6	
	3	8	5	0	5	4	
6	1	3	0	4	4	9	
	2	5	0	2	3	5	
	3	8	1	0	2	1	
7	1	6	3	0	9	2	
	2	7	0	3	7	2	
	3	9	2	0	7	2	
8	1	4	0	4	4	9	
	2	7	0	3	2	8	
	3	9	1	0	2	6	
9	1	5	2	0	4	6	
	2	9	0	4	2	5	
	3	10	2	0	1	4	
10	1	1	0	2	9	5	
	2	5	4	0	7	4	
	3	10	4	0	7	1	
11	1	4	3	0	8	5	
	2	5	2	0	8	5	
	3	10	2	0	7	4	
12	1	3	5	0	4	8	
	2	5	4	0	3	8	
	3	10	0	2	3	6	
13	1	1	0	1	1	9	
	2	2	4	0	1	6	
	3	10	4	0	1	5	
14	1	2	0	3	6	7	
	2	6	1	0	5	6	
	3	9	0	2	4	4	
15	1	4	4	0	6	3	
	2	7	0	3	6	3	
	3	8	2	0	6	3	
16	1	4	2	0	1	5	
	2	4	0	3	1	3	
	3	10	2	0	1	2	
17	1	4	0	3	8	4	
	2	7	0	2	7	3	
	3	7	2	0	6	2	
18	1	1	3	0	7	10	
	2	4	2	0	6	10	
	3	5	2	0	6	9	
19	1	6	0	4	8	10	
	2	8	0	3	7	10	
	3	10	0	2	7	10	
20	1	2	3	0	2	8	
	2	6	2	0	2	6	
	3	9	1	0	2	5	
21	1	7	1	0	6	7	
	2	8	0	2	6	7	
	3	9	0	2	6	3	
22	1	1	0	5	9	5	
	2	6	0	4	9	5	
	3	8	0	4	8	5	
23	1	1	4	0	6	7	
	2	9	3	0	3	5	
	3	10	3	0	2	4	
24	1	5	0	3	1	6	
	2	6	0	3	1	5	
	3	7	0	3	1	3	
25	1	3	0	4	6	8	
	2	8	4	0	5	6	
	3	9	0	2	5	5	
26	1	1	0	3	4	10	
	2	1	3	0	3	6	
	3	10	0	3	1	5	
27	1	1	0	4	10	3	
	2	5	0	4	8	1	
	3	6	3	0	8	1	
28	1	1	0	5	9	7	
	2	3	0	5	7	3	
	3	4	2	0	3	3	
29	1	1	0	3	9	6	
	2	2	0	3	9	5	
	3	8	0	3	9	4	
30	1	1	4	0	3	3	
	2	1	0	1	3	3	
	3	6	4	0	3	2	
31	1	2	3	0	10	5	
	2	8	0	3	6	2	
	3	10	2	0	3	2	
32	1	2	0	5	7	10	
	2	7	0	3	5	8	
	3	9	4	0	3	7	
33	1	3	0	5	10	4	
	2	6	0	4	10	4	
	3	8	2	0	10	4	
34	1	4	3	0	4	9	
	2	8	0	3	3	6	
	3	10	0	1	2	2	
35	1	1	0	3	7	10	
	2	7	4	0	5	5	
	3	10	3	0	4	3	
36	1	6	4	0	8	9	
	2	7	0	1	6	9	
	3	10	3	0	6	9	
37	1	2	2	0	7	7	
	2	7	1	0	7	5	
	3	10	1	0	7	2	
38	1	7	5	0	9	7	
	2	9	3	0	9	7	
	3	10	3	0	9	6	
39	1	3	0	5	4	3	
	2	3	3	0	4	3	
	3	6	3	0	2	3	
40	1	4	0	4	9	3	
	2	6	0	3	8	2	
	3	9	4	0	6	2	
41	1	7	0	3	7	6	
	2	8	3	0	5	4	
	3	8	0	2	5	3	
42	1	2	3	0	6	3	
	2	5	0	4	4	3	
	3	9	3	0	3	2	
43	1	3	3	0	8	5	
	2	6	3	0	5	3	
	3	8	3	0	3	2	
44	1	5	3	0	8	8	
	2	7	0	5	6	6	
	3	10	0	5	6	2	
45	1	5	0	3	9	3	
	2	6	3	0	7	2	
	3	7	2	0	7	2	
46	1	2	0	2	8	8	
	2	5	0	1	8	7	
	3	9	2	0	8	6	
47	1	3	3	0	4	8	
	2	6	2	0	3	8	
	3	10	0	3	3	8	
48	1	2	3	0	9	8	
	2	9	0	3	8	8	
	3	10	1	0	7	8	
49	1	2	0	5	8	4	
	2	6	1	0	7	3	
	3	8	1	0	4	3	
50	1	1	0	4	3	9	
	2	6	1	0	3	6	
	3	7	0	2	2	2	
51	1	6	0	5	8	9	
	2	8	2	0	7	8	
	3	9	2	0	5	8	
52	1	2	3	0	9	9	
	2	5	0	2	6	8	
	3	7	0	2	3	6	
53	1	7	4	0	7	10	
	2	7	0	1	7	4	
	3	8	2	0	7	2	
54	1	5	5	0	4	7	
	2	6	0	3	4	6	
	3	10	4	0	4	2	
55	1	2	0	2	9	9	
	2	3	1	0	8	7	
	3	4	1	0	8	3	
56	1	1	0	2	5	4	
	2	8	2	0	4	3	
	3	10	2	0	3	2	
57	1	5	0	1	4	5	
	2	9	2	0	4	5	
	3	10	2	0	4	3	
58	1	3	3	0	3	8	
	2	3	0	4	2	8	
	3	5	0	4	2	7	
59	1	2	2	0	8	7	
	2	3	1	0	7	5	
	3	6	1	0	7	4	
60	1	2	0	3	8	5	
	2	5	0	2	8	4	
	3	6	0	2	8	1	
61	1	1	3	0	7	9	
	2	1	0	2	7	7	
	3	8	0	2	7	4	
62	1	3	4	0	8	7	
	2	4	0	4	8	6	
	3	4	1	0	4	5	
63	1	3	4	0	3	7	
	2	6	4	0	3	6	
	3	8	4	0	3	5	
64	1	3	5	0	6	8	
	2	6	0	4	6	7	
	3	8	3	0	6	6	
65	1	1	0	3	8	5	
	2	2	3	0	6	5	
	3	5	0	1	4	3	
66	1	5	4	0	10	1	
	2	6	2	0	6	2	
	3	8	2	0	6	1	
67	1	5	0	4	7	5	
	2	8	0	4	7	4	
	3	9	1	0	6	3	
68	1	2	0	5	9	8	
	2	6	0	5	8	8	
	3	10	1	0	8	6	
69	1	1	0	3	7	4	
	2	2	0	3	5	3	
	3	3	0	2	5	1	
70	1	2	5	0	9	8	
	2	4	0	4	7	5	
	3	9	0	3	7	3	
71	1	6	4	0	7	6	
	2	10	3	0	5	5	
	3	10	0	2	5	1	
72	1	7	0	5	7	8	
	2	8	0	3	6	3	
	3	9	0	3	1	3	
73	1	4	4	0	5	9	
	2	7	0	3	3	6	
	3	8	0	3	2	6	
74	1	1	0	2	8	8	
	2	8	3	0	5	4	
	3	8	0	1	3	2	
75	1	5	0	3	5	5	
	2	5	4	0	4	3	
	3	9	0	3	2	3	
76	1	5	0	4	7	6	
	2	6	0	4	6	6	
	3	10	0	3	6	6	
77	1	2	2	0	6	7	
	2	5	0	3	5	7	
	3	6	0	1	5	5	
78	1	4	0	4	7	3	
	2	9	0	2	6	3	
	3	10	0	2	6	2	
79	1	2	2	0	6	6	
	2	3	0	3	6	6	
	3	4	0	3	5	6	
80	1	1	4	0	9	5	
	2	5	0	4	8	4	
	3	8	0	4	8	3	
81	1	5	0	4	9	3	
	2	6	3	0	7	1	
	3	7	0	3	6	1	
82	1	5	0	4	4	9	
	2	6	3	0	3	8	
	3	10	0	1	3	6	
83	1	2	0	4	4	5	
	2	7	2	0	2	4	
	3	9	0	4	1	4	
84	1	3	2	0	6	10	
	2	5	0	4	5	10	
	3	9	1	0	3	10	
85	1	2	2	0	7	7	
	2	7	0	3	7	6	
	3	10	1	0	7	4	
86	1	6	0	5	6	7	
	2	8	4	0	4	6	
	3	8	0	5	2	4	
87	1	1	0	5	6	2	
	2	5	0	5	5	2	
	3	6	4	0	5	2	
88	1	8	0	2	5	5	
	2	9	3	0	4	4	
	3	10	0	2	4	1	
89	1	4	0	1	2	10	
	2	5	3	0	2	8	
	3	6	3	0	2	7	
90	1	1	4	0	5	4	
	2	6	0	3	5	3	
	3	7	0	1	4	3	
91	1	2	0	5	7	4	
	2	2	1	0	7	4	
	3	9	0	4	6	3	
92	1	2	5	0	8	9	
	2	3	4	0	6	6	
	3	6	0	1	6	6	
93	1	7	2	0	7	8	
	2	7	0	1	5	3	
	3	8	0	1	1	3	
94	1	3	0	4	2	8	
	2	9	0	3	1	8	
	3	9	3	0	1	8	
95	1	1	4	0	7	6	
	2	7	3	0	3	4	
	3	7	0	2	3	4	
96	1	6	3	0	6	5	
	2	7	3	0	6	4	
	3	8	2	0	3	2	
97	1	2	3	0	7	6	
	2	6	3	0	6	4	
	3	10	0	4	5	2	
98	1	1	0	2	5	2	
	2	3	0	1	5	2	
	3	7	0	1	5	1	
99	1	3	0	4	4	9	
	2	6	4	0	4	8	
	3	8	0	2	4	7	
100	1	4	0	4	9	9	
	2	5	0	3	5	7	
	3	7	0	3	4	5	
101	1	3	0	4	7	7	
	2	7	0	4	6	6	
	3	8	0	2	4	6	
102	1	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2
	29	36	502	466

************************************************************************
