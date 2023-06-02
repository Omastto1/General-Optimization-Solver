jobs  (incl. supersource/sink ):	102
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 4 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	7		2 3 6 7 8 10 13 
2	3	4		11 9 5 4 
3	3	6		24 19 16 15 12 9 
4	3	5		24 19 15 14 12 
5	3	6		26 22 19 17 16 15 
6	3	7		35 23 22 20 19 18 15 
7	3	6		32 29 27 25 17 14 
8	3	5		27 25 24 20 14 
9	3	5		32 27 25 20 14 
10	3	5		33 32 27 16 14 
11	3	4		32 29 25 14 
12	3	9		38 32 30 27 26 25 23 21 17 
13	3	7		45 30 25 24 23 22 17 
14	3	7		35 34 30 23 22 21 18 
15	3	9		38 37 34 32 30 29 27 25 21 
16	3	3		38 23 21 
17	3	6		42 37 35 33 31 28 
18	3	4		42 39 36 26 
19	3	4		45 38 31 28 
20	3	4		45 37 34 28 
21	3	4		51 45 42 28 
22	3	4		51 48 38 28 
23	3	6		62 58 42 41 40 31 
24	3	3		41 32 29 
25	3	8		51 49 46 42 41 40 39 36 
26	3	9		58 51 50 46 45 43 41 40 37 
27	3	9		57 51 49 48 47 46 44 39 36 
28	3	7		49 47 46 41 40 39 36 
29	3	8		57 51 49 47 46 44 39 36 
30	3	7		57 51 47 46 44 39 36 
31	3	8		57 51 49 48 47 46 44 39 
32	3	10		62 61 58 57 53 50 49 48 46 44 
33	3	7		62 61 56 49 47 41 39 
34	3	7		62 61 58 53 49 41 40 
35	3	4		51 46 40 39 
36	3	7		63 62 61 58 53 50 43 
37	3	5		61 57 49 48 44 
38	3	6		63 62 53 52 50 46 
39	3	3		53 50 43 
40	3	4		64 59 55 44 
41	3	6		75 63 60 59 57 54 
42	3	3		55 53 47 
43	3	4		75 64 60 52 
44	3	6		71 67 63 60 56 54 
45	3	4		64 63 55 53 
46	3	7		75 70 67 64 60 59 54 
47	3	6		75 71 67 60 59 54 
48	3	5		67 64 63 56 54 
49	3	2		75 52 
50	3	4		72 64 59 55 
51	3	8		78 72 70 69 66 65 63 61 
52	3	5		71 70 67 59 54 
53	3	4		72 71 68 59 
54	3	9		85 78 77 76 74 73 72 69 68 
55	3	7		85 78 77 74 70 67 66 
56	3	6		78 77 75 70 69 65 
57	3	6		85 73 71 70 67 64 
58	3	6		85 75 74 72 68 65 
59	3	6		85 78 77 74 69 65 
60	3	7		85 81 79 77 76 74 66 
61	3	7		89 84 76 75 74 71 68 
62	3	8		91 89 87 85 81 74 73 72 
63	3	10		91 89 84 82 81 80 79 77 76 74 
64	3	10		89 87 86 84 83 82 81 80 79 77 
65	3	4		89 83 76 73 
66	3	3		89 84 68 
67	3	8		94 91 89 84 83 82 81 76 
68	3	8		93 91 88 87 86 83 82 80 
69	3	7		91 89 87 84 83 81 79 
70	3	5		94 84 83 79 76 
71	3	5		91 87 86 79 77 
72	3	8		101 100 99 93 88 84 82 80 
73	3	5		101 99 84 80 79 
74	3	6		98 93 90 88 86 83 
75	3	3		100 82 81 
76	3	5		100 98 88 87 86 
77	3	7		101 99 98 96 93 90 88 
78	3	5		100 99 97 96 90 
79	3	4		100 98 93 88 
80	3	4		97 96 94 90 
81	3	4		101 99 96 88 
82	3	4		98 97 96 90 
83	3	4		100 99 97 92 
84	3	3		97 96 90 
85	3	3		100 99 90 
86	3	3		101 96 92 
87	3	2		96 90 
88	3	2		97 92 
89	3	2		96 92 
90	3	1		92 
91	3	1		92 
92	3	1		95 
93	3	1		95 
94	3	1		98 
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
2	1	1	8	7	1	10	9	8	
	2	3	5	7	1	9	8	5	
	3	10	2	7	1	9	6	2	
3	1	4	6	4	8	7	2	7	
	2	8	4	3	5	5	2	6	
	3	9	4	3	5	2	2	6	
4	1	3	4	4	4	3	9	6	
	2	8	3	2	3	2	5	3	
	3	9	3	2	1	2	3	2	
5	1	3	1	6	5	6	4	8	
	2	6	1	6	4	4	2	4	
	3	10	1	3	4	2	2	4	
6	1	1	3	4	5	7	7	6	
	2	2	2	3	4	6	6	6	
	3	9	2	2	3	5	6	5	
7	1	4	5	5	2	7	5	4	
	2	6	4	5	2	5	2	4	
	3	7	3	5	2	5	1	2	
8	1	5	4	8	7	8	8	2	
	2	6	3	8	4	5	8	2	
	3	7	2	8	3	5	6	2	
9	1	2	9	5	6	10	5	8	
	2	8	9	4	6	4	3	7	
	3	9	9	4	6	3	3	6	
10	1	5	5	10	3	6	4	5	
	2	6	4	8	2	4	4	3	
	3	10	4	6	2	4	4	2	
11	1	2	10	8	10	3	8	7	
	2	6	8	8	9	3	5	6	
	3	8	5	8	9	2	3	6	
12	1	1	9	5	6	7	1	6	
	2	5	9	5	5	5	1	5	
	3	6	7	3	4	4	1	5	
13	1	3	8	5	8	3	7	6	
	2	5	6	5	6	2	6	5	
	3	10	4	3	3	2	4	4	
14	1	2	9	9	9	9	7	5	
	2	5	8	9	4	7	5	3	
	3	9	8	9	4	2	5	3	
15	1	3	6	9	6	8	4	8	
	2	6	5	8	5	6	2	6	
	3	10	5	8	5	4	2	6	
16	1	1	7	7	10	7	10	3	
	2	6	7	5	8	6	9	2	
	3	10	5	1	8	6	9	2	
17	1	1	5	6	5	7	7	8	
	2	8	4	6	4	4	4	5	
	3	9	4	2	3	2	4	5	
18	1	2	7	4	8	7	9	4	
	2	6	6	2	5	7	6	3	
	3	8	5	2	1	7	5	3	
19	1	1	4	8	5	10	10	5	
	2	8	3	7	5	5	8	4	
	3	9	3	4	5	3	8	2	
20	1	8	7	9	8	4	8	7	
	2	9	5	9	6	2	7	6	
	3	10	5	9	6	2	6	5	
21	1	7	5	9	9	9	8	8	
	2	8	3	7	7	7	6	5	
	3	9	2	2	6	7	4	2	
22	1	1	4	4	5	9	3	10	
	2	4	3	3	4	8	3	5	
	3	10	3	3	3	8	1	3	
23	1	4	3	3	9	7	7	7	
	2	5	3	3	8	6	7	6	
	3	8	2	3	7	3	5	6	
24	1	4	5	9	4	10	10	6	
	2	5	5	9	3	6	6	5	
	3	6	3	9	3	4	3	4	
25	1	2	5	8	9	9	9	8	
	2	4	3	4	7	8	8	6	
	3	8	3	2	6	8	6	6	
26	1	6	2	3	4	9	9	8	
	2	7	2	3	3	5	9	7	
	3	8	2	3	1	3	9	7	
27	1	5	5	6	6	9	9	8	
	2	6	4	5	5	9	9	7	
	3	9	4	3	5	9	9	6	
28	1	5	9	8	8	5	8	3	
	2	6	9	7	8	5	4	2	
	3	10	9	7	8	3	4	1	
29	1	5	6	8	10	5	7	5	
	2	9	5	7	9	3	6	5	
	3	10	5	7	8	1	6	1	
30	1	3	6	5	4	9	2	6	
	2	4	6	4	3	8	2	5	
	3	5	6	4	2	8	2	5	
31	1	2	6	4	5	4	5	2	
	2	6	5	4	2	4	4	2	
	3	7	3	3	1	4	3	2	
32	1	3	7	7	3	8	7	10	
	2	4	6	5	3	7	6	7	
	3	10	5	4	3	1	3	2	
33	1	1	7	6	8	6	3	8	
	2	4	4	3	7	4	3	6	
	3	8	2	2	7	2	1	6	
34	1	1	6	7	9	4	8	8	
	2	5	6	7	6	3	5	4	
	3	9	6	7	4	2	3	3	
35	1	1	9	10	6	6	9	9	
	2	6	9	9	5	4	4	7	
	3	7	9	7	3	1	3	7	
36	1	7	8	5	9	3	10	6	
	2	8	5	5	6	2	5	4	
	3	9	4	5	6	2	4	1	
37	1	6	6	7	4	10	2	7	
	2	7	5	7	3	7	1	5	
	3	8	2	3	3	4	1	4	
38	1	1	8	7	8	7	7	9	
	2	6	4	5	7	5	6	8	
	3	7	2	4	5	5	6	8	
39	1	4	9	8	6	3	3	7	
	2	6	7	7	6	3	2	7	
	3	8	7	7	6	3	2	6	
40	1	3	2	3	8	9	7	7	
	2	8	2	2	6	7	6	5	
	3	9	1	1	5	7	5	5	
41	1	2	9	6	3	2	7	9	
	2	4	8	4	2	2	6	8	
	3	8	8	4	2	2	2	8	
42	1	7	9	10	8	7	5	9	
	2	8	8	9	7	7	4	8	
	3	9	7	9	3	5	3	8	
43	1	1	10	3	8	6	5	3	
	2	2	9	3	5	5	4	3	
	3	7	9	3	4	2	2	3	
44	1	1	7	7	8	4	6	4	
	2	2	5	7	7	4	5	3	
	3	6	1	7	7	4	5	2	
45	1	1	9	4	9	4	6	3	
	2	2	8	3	8	4	6	3	
	3	7	6	2	7	4	6	1	
46	1	1	6	7	4	6	7	5	
	2	3	5	4	4	3	6	4	
	3	10	4	3	3	1	5	4	
47	1	1	8	5	5	9	2	10	
	2	2	5	5	3	7	2	9	
	3	6	4	5	3	4	2	8	
48	1	2	2	7	5	8	8	8	
	2	6	1	6	5	5	8	7	
	3	7	1	6	5	4	7	5	
49	1	2	8	6	4	4	7	6	
	2	4	5	6	3	4	6	5	
	3	9	3	3	2	4	5	5	
50	1	3	6	4	9	9	7	6	
	2	5	5	4	5	9	3	6	
	3	10	5	4	3	9	2	6	
51	1	6	3	4	9	6	5	9	
	2	7	3	3	8	5	5	8	
	3	9	3	1	7	5	4	6	
52	1	3	6	8	1	10	5	9	
	2	8	3	7	1	7	4	9	
	3	10	2	7	1	5	2	9	
53	1	1	1	7	8	5	10	9	
	2	3	1	5	8	5	6	8	
	3	10	1	5	6	5	3	6	
54	1	3	9	9	3	7	5	4	
	2	4	9	5	2	4	5	3	
	3	7	9	3	1	3	3	3	
55	1	6	7	9	10	7	1	10	
	2	7	7	6	8	4	1	6	
	3	9	5	5	8	2	1	3	
56	1	1	9	7	3	10	7	7	
	2	2	7	5	3	9	6	7	
	3	9	1	5	2	8	5	7	
57	1	1	7	7	7	3	8	5	
	2	3	3	7	7	1	6	5	
	3	4	2	6	5	1	5	4	
58	1	3	6	5	7	5	2	8	
	2	5	6	5	7	4	1	7	
	3	7	4	4	7	4	1	4	
59	1	2	6	5	7	8	8	5	
	2	3	4	4	4	5	7	3	
	3	8	4	3	1	4	7	3	
60	1	2	10	9	8	9	1	4	
	2	4	10	5	8	7	1	4	
	3	7	10	4	8	7	1	4	
61	1	6	1	7	7	9	8	3	
	2	7	1	6	6	8	8	2	
	3	10	1	6	2	7	8	2	
62	1	7	7	5	5	8	6	1	
	2	8	6	3	5	5	6	1	
	3	9	6	3	3	5	6	1	
63	1	2	9	8	3	6	4	8	
	2	4	6	3	3	4	4	8	
	3	10	5	3	3	3	3	7	
64	1	3	3	1	7	8	7	8	
	2	4	2	1	7	7	4	7	
	3	9	2	1	3	5	3	3	
65	1	4	9	4	4	5	6	5	
	2	5	8	4	4	3	6	5	
	3	7	7	4	4	3	4	2	
66	1	3	9	1	7	2	6	10	
	2	8	6	1	5	2	6	9	
	3	9	2	1	2	2	4	8	
67	1	2	8	9	7	4	7	7	
	2	4	5	9	7	3	6	6	
	3	9	4	9	6	3	6	6	
68	1	2	5	7	3	7	7	10	
	2	6	3	6	2	7	6	7	
	3	7	3	4	2	6	4	6	
69	1	4	8	9	7	3	3	5	
	2	5	6	4	6	2	2	4	
	3	9	2	3	6	1	1	1	
70	1	7	8	3	3	8	10	8	
	2	8	7	3	3	5	7	8	
	3	10	5	3	3	5	6	7	
71	1	1	8	5	3	10	3	8	
	2	3	7	5	2	5	3	7	
	3	9	7	5	1	3	3	6	
72	1	6	7	7	8	5	2	6	
	2	8	7	6	8	2	1	3	
	3	9	7	4	6	2	1	2	
73	1	2	3	6	9	6	7	7	
	2	5	3	4	3	5	5	6	
	3	9	2	4	3	3	3	3	
74	1	3	5	10	3	5	7	8	
	2	8	3	7	2	2	7	6	
	3	10	3	4	2	1	6	4	
75	1	5	9	8	7	5	8	2	
	2	6	8	5	3	5	7	2	
	3	8	7	4	3	3	5	2	
76	1	4	3	6	4	7	7	7	
	2	8	2	5	3	7	6	7	
	3	10	2	5	3	5	6	7	
77	1	1	7	4	5	1	3	3	
	2	2	6	4	5	1	3	3	
	3	3	3	2	5	1	1	3	
78	1	5	9	1	9	9	8	5	
	2	6	8	1	6	7	6	4	
	3	10	8	1	5	7	3	4	
79	1	5	7	2	7	7	8	8	
	2	6	7	2	7	5	7	6	
	3	8	5	1	6	3	7	6	
80	1	6	10	10	6	7	1	9	
	2	7	10	10	6	5	1	9	
	3	8	10	10	5	4	1	9	
81	1	2	4	4	9	8	6	8	
	2	4	3	3	9	5	5	6	
	3	9	3	3	9	4	5	5	
82	1	5	10	9	6	8	7	7	
	2	7	7	9	5	7	4	3	
	3	8	7	9	3	7	3	2	
83	1	3	5	7	10	7	6	8	
	2	7	4	7	6	4	5	6	
	3	10	4	4	2	3	3	4	
84	1	3	10	10	1	5	6	8	
	2	5	9	9	1	5	5	4	
	3	7	8	9	1	5	4	1	
85	1	1	10	9	6	10	1	7	
	2	2	10	7	5	5	1	4	
	3	8	10	7	3	1	1	2	
86	1	2	6	9	8	9	1	1	
	2	4	5	7	8	7	1	1	
	3	7	5	4	7	2	1	1	
87	1	6	8	5	7	5	4	8	
	2	9	8	2	3	2	3	8	
	3	10	6	2	2	1	2	8	
88	1	2	1	8	4	2	7	4	
	2	9	1	7	4	1	4	2	
	3	10	1	4	4	1	3	1	
89	1	1	3	9	5	6	5	9	
	2	4	2	5	2	5	5	8	
	3	6	1	2	1	4	5	7	
90	1	1	8	7	4	10	9	2	
	2	2	6	5	3	9	6	2	
	3	8	6	2	2	8	5	2	
91	1	6	6	5	5	10	3	2	
	2	7	6	4	5	8	3	1	
	3	9	4	3	5	7	2	1	
92	1	3	5	7	6	7	6	9	
	2	5	4	6	5	5	4	7	
	3	10	3	4	5	5	2	7	
93	1	2	1	6	6	9	4	8	
	2	3	1	6	6	6	2	7	
	3	10	1	6	5	2	2	3	
94	1	1	8	7	8	5	9	7	
	2	2	7	7	8	3	9	4	
	3	9	6	7	8	1	9	2	
95	1	1	3	5	7	5	8	7	
	2	2	3	4	7	4	7	3	
	3	3	3	3	7	3	7	2	
96	1	2	6	9	4	8	8	9	
	2	6	6	7	3	8	7	7	
	3	9	5	6	2	8	2	4	
97	1	8	8	6	6	4	9	1	
	2	9	6	6	3	4	7	1	
	3	10	3	6	1	4	7	1	
98	1	3	8	3	4	7	5	8	
	2	8	8	3	4	4	3	4	
	3	9	8	2	2	4	1	2	
99	1	3	8	9	5	5	6	6	
	2	4	7	9	4	5	6	4	
	3	7	5	8	4	4	6	2	
100	1	3	2	3	2	4	7	5	
	2	5	2	3	1	4	3	4	
	3	10	1	1	1	2	3	3	
101	1	3	7	6	10	9	7	9	
	2	4	6	6	8	7	6	9	
	3	10	4	6	7	6	1	9	
102	1	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2	N 3	N 4
	29	23	457	460	442	472

************************************************************************
