jobs  (incl. supersource/sink ):	102
RESOURCES
- renewable                 : 4 R
- nonrenewable              : 4 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	6		2 3 4 6 8 16 
2	9	8		15 14 13 12 11 10 9 7 
3	9	4		15 11 9 5 
4	9	5		31 19 18 15 11 
5	9	3		31 17 10 
6	9	3		31 11 10 
7	9	8		31 29 28 24 23 20 19 17 
8	9	7		31 28 24 23 20 17 13 
9	9	6		31 29 28 23 19 17 
10	9	6		28 24 23 22 19 18 
11	9	10		35 34 29 28 27 25 24 23 22 21 
12	9	3		24 19 17 
13	9	6		35 29 27 25 22 21 
14	9	6		35 34 29 25 22 21 
15	9	5		35 34 29 22 21 
16	9	6		43 38 36 34 31 21 
17	9	5		37 34 27 25 22 
18	9	8		45 38 35 34 33 30 29 27 
19	9	3		38 35 21 
20	9	6		45 43 38 33 30 27 
21	9	5		39 33 32 30 26 
22	9	9		45 43 42 41 40 39 38 32 30 
23	9	9		45 43 42 41 40 39 38 33 30 
24	9	4		43 39 36 26 
25	9	6		45 42 41 39 33 30 
26	9	10		50 49 47 46 45 44 42 41 40 37 
27	9	6		44 42 41 40 39 36 
28	9	7		49 48 47 46 42 39 38 
29	9	5		44 43 40 37 36 
30	9	7		53 52 50 49 47 46 44 
31	9	7		58 57 55 48 42 41 39 
32	9	7		58 57 55 53 50 49 48 
33	9	4		49 48 46 44 
34	9	7		58 55 54 53 52 50 48 
35	9	6		61 57 53 51 49 47 
36	9	6		66 57 56 52 49 48 
37	9	6		65 58 57 55 52 48 
38	9	8		65 61 58 57 56 54 52 51 
39	9	6		63 61 56 53 52 50 
40	9	6		65 60 58 55 54 53 
41	9	7		66 65 61 60 56 54 52 
42	9	6		65 61 59 56 54 51 
43	9	6		66 61 60 58 54 53 
44	9	5		62 61 57 55 51 
45	9	8		82 76 66 63 62 61 60 53 
46	9	5		61 60 56 55 54 
47	9	6		75 67 62 60 58 55 
48	9	5		70 67 62 61 51 
49	9	5		82 76 65 59 54 
50	9	8		77 76 71 70 67 65 62 60 
51	9	7		82 77 76 75 71 63 60 
52	9	8		82 77 76 74 72 69 68 59 
53	9	6		77 74 69 67 64 59 
54	9	9		79 78 77 75 69 68 67 64 62 
55	9	8		82 78 76 74 70 69 68 66 
56	9	11		91 84 83 82 76 75 74 72 71 70 69 
57	9	9		84 83 81 78 77 76 75 71 67 
58	9	7		90 83 82 76 69 68 63 
59	9	8		91 84 83 81 78 75 71 70 
60	9	5		78 74 69 68 64 
61	9	10		101 91 84 81 79 78 77 75 74 73 
62	9	7		91 90 84 83 81 73 72 
63	9	7		91 85 84 79 78 73 72 
64	9	6		91 90 84 83 73 72 
65	9	7		100 86 83 80 79 78 75 
66	9	5		83 81 77 73 72 
67	9	4		91 90 73 72 
68	9	6		101 91 88 84 81 80 
69	9	3		101 87 73 
70	9	3		101 90 73 
71	9	2		79 73 
72	9	6		101 98 88 87 86 80 
73	9	5		100 98 88 86 80 
74	9	5		100 98 88 86 80 
75	9	5		97 94 90 87 85 
76	9	9		101 100 98 97 96 95 94 93 92 
77	9	8		100 99 98 97 96 94 93 89 
78	9	7		99 98 97 95 94 93 89 
79	9	6		98 97 95 94 93 88 
80	9	7		99 97 96 95 94 93 92 
81	9	4		99 94 93 87 
82	9	3		96 95 86 
83	9	5		97 96 94 93 92 
84	9	4		100 95 93 89 
85	9	3		95 93 92 
86	9	2		93 89 
87	9	2		96 89 
88	9	1		89 
89	9	1		92 
90	9	1		92 
91	9	1		98 
92	9	1		102 
93	9	1		102 
94	9	1		102 
95	9	1		102 
96	9	1		102 
97	9	1		102 
98	9	1		102 
99	9	1		102 
100	9	1		102 
101	9	1		102 
102	1	0		
************************************************************************
REQUESTS/DURATIONS
jobnr.	mode	dur	R1	R2	R3	R4	N1	N2	N3	N4	
------------------------------------------------------------------------
1	1	0	0	0	0	0	0	0	0	0	
2	1	6	2	5	4	4	24	29	24	25	
	2	7	2	4	4	3	20	28	23	24	
	3	8	2	4	4	3	19	27	22	24	
	4	12	2	4	4	3	18	27	21	23	
	5	15	1	3	4	3	13	25	20	21	
	6	17	1	3	4	3	13	25	19	20	
	7	26	1	3	4	3	10	25	18	18	
	8	28	1	3	4	3	6	23	18	17	
	9	29	1	3	4	3	5	23	17	15	
3	1	6	2	2	4	4	17	26	24	13	
	2	8	2	1	3	4	15	24	24	12	
	3	9	2	1	3	4	15	24	18	11	
	4	12	2	1	3	3	11	23	17	10	
	5	13	1	1	3	3	10	22	13	7	
	6	24	1	1	3	3	9	22	13	6	
	7	27	1	1	3	3	8	22	9	4	
	8	28	1	1	3	2	5	20	4	3	
	9	30	1	1	3	2	4	20	4	1	
4	1	1	4	2	2	5	27	25	27	27	
	2	2	3	2	1	4	25	23	27	27	
	3	5	3	2	1	4	23	21	25	25	
	4	11	3	2	1	4	20	17	23	25	
	5	18	3	2	1	3	17	17	22	24	
	6	19	2	2	1	3	14	14	21	23	
	7	23	2	2	1	2	11	14	19	22	
	8	29	2	2	1	2	8	12	18	22	
	9	30	2	2	1	2	3	10	16	21	
5	1	5	2	4	4	5	27	21	15	11	
	2	8	1	4	4	4	26	20	14	11	
	3	9	1	4	4	4	26	20	13	11	
	4	11	1	4	4	3	26	20	12	11	
	5	16	1	4	4	3	25	20	12	11	
	6	20	1	3	3	3	25	19	10	11	
	7	21	1	3	3	2	25	19	9	11	
	8	22	1	3	3	2	25	19	8	11	
	9	25	1	3	3	2	25	19	7	11	
6	1	1	4	4	4	2	6	16	24	9	
	2	6	4	4	4	1	5	15	24	8	
	3	16	4	4	4	1	5	15	23	7	
	4	17	4	4	4	1	5	12	21	6	
	5	18	4	4	4	1	5	12	20	6	
	6	23	3	4	4	1	5	10	20	5	
	7	24	3	4	4	1	5	7	18	5	
	8	25	3	4	4	1	5	6	17	4	
	9	27	3	4	4	1	5	5	16	3	
7	1	8	4	3	4	3	17	14	25	8	
	2	10	3	3	4	3	17	12	23	8	
	3	11	3	3	4	3	16	10	23	7	
	4	12	2	3	4	3	14	10	22	5	
	5	13	2	2	3	3	11	7	22	5	
	6	16	2	2	3	3	10	6	22	4	
	7	21	2	1	3	3	9	6	21	3	
	8	23	1	1	2	3	8	5	20	3	
	9	25	1	1	2	3	7	3	20	2	
8	1	3	4	4	4	4	21	9	16	19	
	2	10	3	4	4	4	16	8	15	18	
	3	11	3	4	4	4	15	7	14	14	
	4	13	3	3	4	4	14	5	14	14	
	5	14	3	3	4	4	10	5	13	10	
	6	22	3	3	4	3	7	4	12	9	
	7	26	3	3	4	3	6	3	12	6	
	8	27	3	2	4	3	4	1	12	6	
	9	29	3	2	4	3	1	1	11	4	
9	1	1	5	4	3	1	26	28	27	22	
	2	2	4	4	3	1	23	23	24	22	
	3	5	4	4	3	1	22	20	19	22	
	4	17	4	4	2	1	21	19	18	22	
	5	23	4	3	2	1	20	16	15	22	
	6	24	4	3	2	1	18	13	12	22	
	7	26	4	3	1	1	17	11	10	22	
	8	28	4	3	1	1	16	5	5	22	
	9	29	4	3	1	1	16	3	3	22	
10	1	1	3	3	2	4	27	21	26	15	
	2	2	3	3	1	4	26	20	25	14	
	3	3	3	3	1	4	26	20	25	12	
	4	17	3	3	1	3	26	19	24	11	
	5	18	3	2	1	3	26	18	24	10	
	6	20	3	2	1	2	25	18	24	8	
	7	22	3	2	1	2	25	17	24	8	
	8	29	3	1	1	1	25	15	23	7	
	9	30	3	1	1	1	25	15	23	5	
11	1	5	3	4	4	4	25	21	23	26	
	2	6	3	3	3	4	24	20	22	26	
	3	7	3	3	3	4	24	20	21	26	
	4	8	2	3	3	4	24	19	19	26	
	5	17	2	2	2	3	23	19	18	26	
	6	21	2	2	2	3	23	18	17	26	
	7	22	2	2	1	2	23	18	17	26	
	8	23	1	2	1	2	23	18	15	26	
	9	26	1	2	1	2	23	17	15	26	
12	1	2	5	4	3	3	28	26	17	19	
	2	6	4	4	2	2	25	23	17	17	
	3	7	4	4	2	2	23	22	17	17	
	4	8	3	4	2	2	18	20	17	15	
	5	14	3	4	2	2	16	18	17	15	
	6	23	3	4	1	2	11	17	17	14	
	7	26	3	4	1	2	11	16	17	13	
	8	27	2	4	1	2	5	14	17	13	
	9	29	2	4	1	2	4	13	17	12	
13	1	3	4	3	3	4	25	20	28	26	
	2	8	4	3	3	4	24	19	26	25	
	3	9	4	3	3	4	24	17	23	22	
	4	14	4	3	3	4	24	14	19	21	
	5	15	4	2	3	3	23	10	17	20	
	6	16	4	2	2	3	22	8	15	17	
	7	22	4	1	2	3	21	5	10	14	
	8	23	4	1	2	3	20	4	9	13	
	9	24	4	1	2	3	20	1	6	11	
14	1	3	4	2	4	2	28	3	26	24	
	2	5	3	2	4	1	28	3	26	23	
	3	8	3	2	4	1	28	3	26	22	
	4	11	3	2	4	1	28	3	25	22	
	5	17	3	2	4	1	28	2	25	22	
	6	22	2	1	4	1	28	2	25	21	
	7	23	2	1	4	1	28	1	24	21	
	8	29	2	1	4	1	28	1	24	20	
	9	30	2	1	4	1	28	1	24	19	
15	1	2	3	4	3	2	15	22	5	12	
	2	6	3	3	2	2	15	21	4	11	
	3	8	3	3	2	2	14	18	4	10	
	4	15	3	2	2	2	13	17	4	9	
	5	20	2	2	1	2	12	15	4	9	
	6	24	2	2	1	1	11	14	4	9	
	7	26	2	1	1	1	10	11	4	8	
	8	27	1	1	1	1	9	9	4	8	
	9	30	1	1	1	1	8	8	4	7	
16	1	9	3	3	3	3	17	20	14	29	
	2	12	2	3	2	3	17	16	13	25	
	3	13	2	3	2	3	15	16	11	22	
	4	15	2	3	2	3	14	12	10	17	
	5	16	1	3	2	3	14	10	10	14	
	6	18	1	3	2	3	14	9	9	12	
	7	22	1	3	2	3	13	7	8	10	
	8	23	1	3	2	3	12	5	7	7	
	9	26	1	3	2	3	11	3	5	4	
17	1	8	4	5	4	4	19	29	23	13	
	2	13	4	5	3	4	17	25	22	11	
	3	14	4	5	3	4	15	24	22	11	
	4	15	3	5	2	4	12	23	21	10	
	5	16	3	5	2	4	12	22	21	10	
	6	17	3	5	2	4	11	21	21	9	
	7	28	2	5	2	4	8	19	20	8	
	8	29	2	5	1	4	5	16	20	8	
	9	30	2	5	1	4	4	16	20	7	
18	1	1	2	3	2	3	18	13	24	17	
	2	10	1	3	2	3	18	12	24	16	
	3	16	1	3	2	3	18	10	23	15	
	4	17	1	3	2	3	17	9	22	13	
	5	18	1	3	2	3	17	8	20	13	
	6	22	1	3	1	3	17	6	19	10	
	7	26	1	3	1	3	17	4	18	10	
	8	27	1	3	1	3	16	4	17	9	
	9	28	1	3	1	3	16	2	16	8	
19	1	15	3	4	3	4	18	15	23	7	
	2	16	3	3	3	4	18	13	23	7	
	3	17	3	3	3	3	16	13	22	7	
	4	18	3	3	3	3	15	13	20	7	
	5	20	3	3	3	3	13	12	20	7	
	6	23	3	3	3	2	13	12	19	7	
	7	28	3	3	3	2	12	11	18	7	
	8	29	3	3	3	1	11	10	17	7	
	9	30	3	3	3	1	9	10	17	7	
20	1	2	1	4	4	2	21	28	29	12	
	2	10	1	4	3	1	20	28	27	11	
	3	13	1	4	3	1	19	28	25	10	
	4	16	1	4	2	1	18	28	23	8	
	5	18	1	3	2	1	15	27	22	8	
	6	19	1	3	2	1	14	27	20	6	
	7	24	1	3	1	1	14	27	18	5	
	8	25	1	3	1	1	11	27	17	4	
	9	28	1	3	1	1	11	27	15	4	
21	1	6	4	4	4	3	19	10	22	29	
	2	7	3	3	4	3	18	8	22	28	
	3	16	3	3	4	3	16	8	20	25	
	4	18	3	2	4	3	15	8	20	24	
	5	19	3	2	4	2	14	7	19	22	
	6	20	3	2	4	2	12	6	17	20	
	7	21	3	1	4	2	10	6	17	18	
	8	22	3	1	4	1	9	5	16	16	
	9	28	3	1	4	1	8	5	15	12	
22	1	7	5	3	2	2	17	4	22	12	
	2	8	5	3	2	2	17	4	19	10	
	3	13	5	3	2	2	17	4	18	10	
	4	14	5	3	2	2	17	4	17	9	
	5	16	5	3	2	2	17	4	17	8	
	6	17	5	3	1	2	16	3	15	9	
	7	19	5	3	1	2	16	3	15	8	
	8	24	5	3	1	2	16	3	14	8	
	9	29	5	3	1	2	16	3	12	7	
23	1	1	1	4	5	3	23	8	22	14	
	2	11	1	4	4	3	22	8	18	13	
	3	12	1	4	4	3	22	8	16	13	
	4	15	1	3	4	2	22	8	15	13	
	5	16	1	3	3	2	21	7	13	13	
	6	17	1	2	3	2	21	7	12	13	
	7	19	1	2	3	2	21	6	8	13	
	8	28	1	1	3	1	20	6	8	13	
	9	29	1	1	3	1	20	6	6	13	
24	1	6	3	5	4	4	21	18	22	19	
	2	11	3	4	3	3	20	17	21	19	
	3	14	3	4	3	3	18	17	20	18	
	4	16	3	4	3	2	17	17	19	18	
	5	17	2	4	3	2	16	16	19	18	
	6	18	2	4	3	2	15	16	19	17	
	7	22	2	4	3	1	14	15	18	17	
	8	26	2	4	3	1	12	14	18	16	
	9	30	2	4	3	1	11	14	17	16	
25	1	2	5	5	4	3	21	24	23	19	
	2	5	4	5	3	2	19	23	21	18	
	3	6	4	5	3	2	18	23	20	17	
	4	12	4	5	2	2	17	23	19	15	
	5	16	4	5	2	2	17	22	17	14	
	6	17	4	5	2	2	16	22	15	14	
	7	20	4	5	1	2	14	22	14	12	
	8	23	4	5	1	2	13	21	11	11	
	9	28	4	5	1	2	12	21	10	11	
26	1	3	4	4	5	5	18	26	18	12	
	2	4	3	3	4	4	15	25	18	12	
	3	5	3	3	4	4	15	20	18	11	
	4	10	2	3	4	3	13	17	17	9	
	5	14	2	3	4	3	12	14	17	9	
	6	17	2	3	4	2	10	12	17	9	
	7	20	2	3	4	2	7	10	17	8	
	8	24	1	3	4	1	7	8	16	6	
	9	28	1	3	4	1	5	4	16	6	
27	1	7	3	3	4	3	28	27	27	24	
	2	11	2	2	4	3	27	26	26	21	
	3	12	2	2	4	3	24	26	25	21	
	4	13	2	2	3	3	21	25	24	18	
	5	14	2	2	3	3	18	25	24	18	
	6	16	2	1	3	3	15	25	23	15	
	7	21	2	1	2	3	12	24	21	15	
	8	24	2	1	2	3	12	24	20	12	
	9	25	2	1	2	3	8	24	20	10	
28	1	1	4	4	1	4	17	23	12	26	
	2	4	3	4	1	4	14	19	12	24	
	3	6	3	4	1	4	13	17	12	23	
	4	11	3	4	1	4	13	15	12	17	
	5	12	2	3	1	4	10	12	12	16	
	6	13	2	3	1	4	9	10	12	11	
	7	14	1	3	1	4	9	7	12	11	
	8	23	1	2	1	4	7	6	12	8	
	9	26	1	2	1	4	6	3	12	3	
29	1	3	2	3	2	3	27	19	21	18	
	2	6	2	3	2	3	26	19	20	16	
	3	7	2	3	2	3	25	17	19	16	
	4	18	2	3	2	3	22	16	17	15	
	5	19	2	3	2	3	20	15	15	15	
	6	23	2	3	1	3	18	15	15	15	
	7	26	2	3	1	3	17	13	14	14	
	8	27	2	3	1	3	15	13	12	14	
	9	30	2	3	1	3	12	12	12	13	
30	1	2	4	1	5	2	16	20	27	23	
	2	3	4	1	4	2	16	17	26	21	
	3	11	4	1	4	2	14	17	26	19	
	4	13	4	1	4	2	13	16	26	16	
	5	15	3	1	4	2	13	15	25	15	
	6	16	3	1	4	2	11	13	25	12	
	7	18	3	1	4	2	11	12	25	8	
	8	21	3	1	4	2	10	12	25	7	
	9	24	3	1	4	2	9	11	25	4	
31	1	8	5	3	4	3	15	3	25	20	
	2	10	4	3	3	3	14	3	21	18	
	3	11	3	3	3	3	13	3	19	17	
	4	14	3	3	3	3	12	3	18	17	
	5	22	3	2	3	3	12	3	16	15	
	6	23	2	2	3	3	12	3	13	13	
	7	25	1	2	3	3	11	3	13	13	
	8	26	1	1	3	3	10	3	9	11	
	9	30	1	1	3	3	9	3	9	10	
32	1	4	5	5	4	3	12	26	20	24	
	2	6	4	4	4	2	11	25	19	21	
	3	11	4	4	4	2	10	25	19	18	
	4	12	4	4	4	2	8	25	18	18	
	5	13	3	4	4	2	7	25	18	14	
	6	14	3	4	4	1	5	25	17	12	
	7	16	2	4	4	1	5	25	17	11	
	8	24	2	4	4	1	3	25	15	8	
	9	25	2	4	4	1	2	25	15	7	
33	1	4	5	2	1	3	22	28	21	14	
	2	7	5	2	1	3	22	25	20	13	
	3	9	5	2	1	3	22	25	20	12	
	4	10	5	2	1	3	22	23	18	12	
	5	14	5	1	1	3	22	20	17	11	
	6	15	5	1	1	2	22	20	14	11	
	7	16	5	1	1	2	22	17	14	11	
	8	23	5	1	1	2	22	17	11	9	
	9	25	5	1	1	2	22	15	10	9	
34	1	1	1	1	5	1	29	16	16	17	
	2	8	1	1	5	1	29	16	16	15	
	3	15	1	1	5	1	28	15	16	15	
	4	19	1	1	5	1	28	14	16	13	
	5	21	1	1	5	1	26	14	16	13	
	6	22	1	1	5	1	26	14	16	11	
	7	23	1	1	5	1	26	13	16	10	
	8	24	1	1	5	1	25	13	16	10	
	9	30	1	1	5	1	24	12	16	9	
35	1	2	4	3	2	4	10	29	21	22	
	2	5	4	3	2	4	8	26	19	20	
	3	6	4	3	2	4	8	25	17	18	
	4	12	4	3	2	4	6	23	16	17	
	5	14	3	3	1	4	6	22	14	15	
	6	15	3	3	1	3	5	21	11	13	
	7	17	2	3	1	3	4	19	10	11	
	8	28	2	3	1	3	2	17	10	10	
	9	29	2	3	1	3	2	17	7	9	
36	1	2	3	1	5	1	23	24	12	11	
	2	3	3	1	4	1	23	22	9	10	
	3	11	3	1	4	1	20	21	9	10	
	4	12	3	1	4	1	20	18	7	9	
	5	15	2	1	3	1	18	17	6	8	
	6	16	2	1	3	1	17	14	5	7	
	7	17	2	1	3	1	15	11	5	5	
	8	21	1	1	2	1	13	9	4	5	
	9	25	1	1	2	1	12	9	3	3	
37	1	3	4	2	4	4	11	19	16	14	
	2	5	3	2	4	4	10	18	14	13	
	3	6	3	2	4	4	8	18	14	11	
	4	7	3	2	4	3	7	18	13	10	
	5	17	3	2	3	3	7	18	12	9	
	6	26	3	2	3	3	7	18	11	8	
	7	27	3	2	3	3	5	18	9	8	
	8	28	3	2	2	2	5	18	9	6	
	9	29	3	2	2	2	4	18	8	5	
38	1	1	2	3	2	5	3	24	4	9	
	2	2	1	3	2	4	3	23	4	7	
	3	3	1	3	2	4	3	20	4	7	
	4	4	1	3	2	3	3	19	4	7	
	5	5	1	3	2	2	2	16	4	6	
	6	13	1	3	1	2	2	14	4	5	
	7	16	1	3	1	2	1	11	4	5	
	8	23	1	3	1	1	1	11	4	3	
	9	27	1	3	1	1	1	9	4	3	
39	1	1	3	2	2	2	24	30	30	16	
	2	2	3	2	2	1	24	28	26	15	
	3	6	3	2	2	1	24	28	25	15	
	4	8	3	2	2	1	24	27	23	14	
	5	15	3	1	2	1	24	27	20	12	
	6	21	3	1	2	1	23	26	18	11	
	7	22	3	1	2	1	23	25	14	9	
	8	23	3	1	2	1	23	25	14	8	
	9	24	3	1	2	1	23	24	11	8	
40	1	2	3	5	4	3	26	8	11	11	
	2	4	3	4	3	3	22	7	11	11	
	3	6	3	4	3	3	20	7	10	9	
	4	15	3	4	3	2	19	7	8	9	
	5	20	3	3	2	2	17	6	8	8	
	6	21	3	3	2	2	15	6	7	8	
	7	22	3	3	2	1	12	5	6	6	
	8	24	3	3	2	1	12	5	5	5	
	9	25	3	3	2	1	8	5	3	5	
41	1	2	5	5	3	4	10	21	19	20	
	2	3	4	5	3	4	9	21	19	17	
	3	6	4	5	3	4	9	19	19	16	
	4	7	4	5	3	4	8	19	18	14	
	5	16	3	5	3	3	7	18	17	14	
	6	21	3	5	3	3	6	17	17	12	
	7	22	3	5	3	2	5	16	17	11	
	8	25	3	5	3	2	4	15	16	10	
	9	30	3	5	3	2	3	15	16	8	
42	1	2	3	1	1	4	14	25	17	27	
	2	3	3	1	1	3	11	24	16	27	
	3	4	3	1	1	3	10	24	15	27	
	4	10	3	1	1	2	9	24	13	26	
	5	11	3	1	1	2	7	24	12	25	
	6	12	3	1	1	2	5	24	12	25	
	7	20	3	1	1	1	5	24	11	25	
	8	22	3	1	1	1	3	24	10	24	
	9	25	3	1	1	1	2	24	8	24	
43	1	2	3	5	4	2	26	14	27	25	
	2	4	3	4	3	2	26	13	27	24	
	3	9	3	4	3	2	26	12	25	22	
	4	10	3	3	3	2	26	12	21	21	
	5	12	2	3	2	2	26	11	19	21	
	6	21	2	3	2	2	26	11	18	19	
	7	26	2	2	2	2	26	10	17	18	
	8	27	2	2	1	2	26	9	13	17	
	9	30	2	2	1	2	26	9	12	16	
44	1	1	2	4	3	2	25	29	10	5	
	2	3	2	4	3	1	23	27	9	5	
	3	4	2	4	3	1	20	26	8	5	
	4	7	2	4	3	1	18	26	6	5	
	5	9	2	3	3	1	14	23	6	4	
	6	12	1	3	3	1	12	22	6	4	
	7	14	1	2	3	1	10	21	5	3	
	8	15	1	2	3	1	9	20	3	3	
	9	18	1	2	3	1	7	20	3	3	
45	1	2	4	4	3	5	1	29	27	15	
	2	5	3	4	2	4	1	23	27	14	
	3	14	3	4	2	4	1	23	27	13	
	4	17	2	4	2	4	1	17	27	12	
	5	18	2	4	2	4	1	16	26	10	
	6	21	2	3	2	4	1	13	26	9	
	7	22	2	3	2	4	1	10	25	8	
	8	24	1	3	2	4	1	5	25	7	
	9	29	1	3	2	4	1	3	25	6	
46	1	1	1	1	5	2	22	21	21	27	
	2	2	1	1	4	2	21	17	21	26	
	3	5	1	1	3	2	19	17	20	26	
	4	9	1	1	3	2	18	14	19	25	
	5	15	1	1	2	1	17	12	19	24	
	6	16	1	1	2	1	16	12	19	24	
	7	18	1	1	2	1	15	10	18	23	
	8	23	1	1	1	1	15	9	17	23	
	9	30	1	1	1	1	13	6	17	23	
47	1	4	5	3	1	4	25	29	16	18	
	2	5	5	3	1	3	23	29	14	17	
	3	6	5	3	1	3	23	27	13	17	
	4	9	5	3	1	2	21	27	11	16	
	5	15	5	2	1	2	18	26	9	16	
	6	19	5	2	1	2	17	25	8	15	
	7	24	5	2	1	2	17	23	6	15	
	8	25	5	1	1	1	15	23	5	14	
	9	29	5	1	1	1	13	22	3	14	
48	1	6	5	2	1	2	18	12	15	10	
	2	9	4	1	1	1	18	12	14	10	
	3	11	4	1	1	1	15	12	12	10	
	4	12	3	1	1	1	13	11	12	10	
	5	20	3	1	1	1	10	11	9	10	
	6	21	3	1	1	1	8	10	8	10	
	7	22	3	1	1	1	8	9	8	10	
	8	25	2	1	1	1	5	9	6	10	
	9	27	2	1	1	1	4	9	5	10	
49	1	1	2	2	5	2	11	13	27	19	
	2	5	1	2	4	1	9	12	26	18	
	3	9	1	2	4	1	8	12	24	17	
	4	14	1	2	3	1	8	12	24	16	
	5	18	1	2	3	1	7	11	22	16	
	6	21	1	2	3	1	7	11	22	15	
	7	22	1	2	2	1	5	11	20	15	
	8	23	1	2	2	1	4	10	19	15	
	9	30	1	2	2	1	4	10	18	14	
50	1	1	1	3	4	5	21	28	7	28	
	2	3	1	3	4	4	18	26	7	26	
	3	7	1	3	4	4	17	26	7	25	
	4	8	1	3	4	4	14	25	7	22	
	5	13	1	2	4	4	14	24	6	21	
	6	17	1	2	4	4	13	23	6	19	
	7	20	1	2	4	4	10	22	6	17	
	8	22	1	2	4	4	8	21	6	15	
	9	23	1	2	4	4	8	20	6	13	
51	1	3	2	2	4	4	24	21	27	20	
	2	12	2	1	4	4	23	18	26	18	
	3	13	2	1	4	4	20	18	26	17	
	4	14	2	1	4	4	17	14	25	16	
	5	21	2	1	4	4	15	14	25	15	
	6	23	2	1	3	4	13	12	25	15	
	7	26	2	1	3	4	11	9	25	13	
	8	29	2	1	3	4	9	6	24	13	
	9	30	2	1	3	4	8	5	24	11	
52	1	4	4	4	4	5	23	22	15	18	
	2	5	4	4	4	4	23	19	14	17	
	3	7	3	3	4	4	21	19	13	17	
	4	8	3	3	4	4	20	14	13	16	
	5	9	2	2	4	3	19	14	11	16	
	6	10	2	2	3	3	18	10	11	16	
	7	15	2	2	3	2	18	10	10	16	
	8	16	1	1	3	2	17	6	8	15	
	9	28	1	1	3	2	16	5	8	15	
53	1	4	4	2	2	5	20	20	22	26	
	2	10	4	2	2	4	19	19	20	26	
	3	13	3	2	2	4	18	17	20	26	
	4	14	3	2	2	3	15	16	19	26	
	5	15	3	2	2	3	15	16	17	26	
	6	21	2	1	2	2	14	16	17	26	
	7	23	2	1	2	2	11	15	16	26	
	8	24	1	1	2	1	9	13	14	26	
	9	25	1	1	2	1	8	13	13	26	
54	1	9	5	2	4	4	16	28	14	20	
	2	10	4	2	4	4	12	26	13	17	
	3	12	4	2	4	4	12	21	13	16	
	4	16	3	2	4	4	10	18	12	15	
	5	19	3	2	3	4	8	18	12	14	
	6	20	3	1	3	4	7	15	11	12	
	7	26	3	1	2	4	6	12	10	11	
	8	29	2	1	2	4	4	10	10	9	
	9	30	2	1	2	4	2	7	9	8	
55	1	5	5	2	4	5	29	24	19	29	
	2	6	5	1	3	4	28	20	17	28	
	3	13	5	1	3	3	28	19	17	27	
	4	14	5	1	3	3	27	16	15	27	
	5	16	5	1	3	3	27	15	15	26	
	6	17	5	1	3	2	27	14	15	26	
	7	18	5	1	3	1	26	12	13	26	
	8	25	5	1	3	1	26	11	13	25	
	9	30	5	1	3	1	26	8	12	25	
56	1	2	3	1	3	3	25	22	6	12	
	2	4	3	1	3	2	23	20	6	12	
	3	5	3	1	3	2	19	19	6	12	
	4	6	3	1	3	2	19	19	6	11	
	5	7	3	1	2	2	16	18	5	11	
	6	18	3	1	2	1	15	18	5	11	
	7	22	3	1	2	1	12	17	5	11	
	8	23	3	1	2	1	11	15	5	11	
	9	24	3	1	2	1	10	15	5	11	
57	1	2	5	4	3	3	23	22	25	15	
	2	7	5	4	2	2	19	21	24	14	
	3	14	5	4	2	2	18	18	24	14	
	4	16	5	4	2	2	17	18	23	13	
	5	17	5	4	2	2	15	15	23	13	
	6	18	5	3	2	2	12	14	22	13	
	7	21	5	3	2	2	10	12	21	13	
	8	22	5	3	2	2	8	12	21	12	
	9	23	5	3	2	2	7	11	21	12	
58	1	4	2	2	4	4	23	17	25	24	
	2	6	2	2	3	4	18	16	24	22	
	3	14	2	2	3	4	17	16	22	20	
	4	15	2	2	3	3	15	14	19	18	
	5	24	2	2	3	3	12	13	18	17	
	6	27	2	2	2	3	8	12	17	16	
	7	28	2	2	2	3	7	11	15	15	
	8	29	2	2	2	2	3	11	11	13	
	9	30	2	2	2	2	3	9	10	11	
59	1	4	4	5	4	4	11	10	25	9	
	2	5	4	4	3	4	11	10	20	8	
	3	6	4	4	3	4	9	8	19	7	
	4	8	4	4	3	4	8	8	16	6	
	5	9	4	4	3	4	8	6	13	6	
	6	10	4	3	3	4	7	5	10	5	
	7	13	4	3	3	4	5	4	9	4	
	8	14	4	3	3	4	4	3	6	2	
	9	23	4	3	3	4	4	3	4	2	
60	1	15	1	3	4	4	23	19	19	26	
	2	16	1	3	3	4	20	18	16	24	
	3	17	1	3	3	4	20	18	16	20	
	4	22	1	3	3	3	19	17	15	16	
	5	23	1	3	2	3	17	16	12	15	
	6	24	1	3	2	2	15	16	12	11	
	7	25	1	3	1	2	13	15	11	10	
	8	28	1	3	1	1	12	15	9	6	
	9	29	1	3	1	1	11	15	9	3	
61	1	1	1	4	2	3	16	30	15	26	
	2	9	1	4	2	3	15	27	14	25	
	3	19	1	3	2	3	13	24	13	21	
	4	20	1	3	2	3	12	21	12	21	
	5	23	1	3	2	3	11	17	11	17	
	6	24	1	2	2	3	11	15	10	17	
	7	25	1	2	2	3	9	13	9	13	
	8	26	1	1	2	3	9	9	8	11	
	9	29	1	1	2	3	7	6	8	10	
62	1	1	2	4	3	3	29	6	18	22	
	2	14	2	3	3	3	29	5	16	20	
	3	16	2	3	3	3	28	5	16	20	
	4	19	2	3	3	3	28	5	12	16	
	5	20	1	2	3	3	26	4	12	14	
	6	24	1	2	3	3	26	4	8	13	
	7	25	1	2	3	3	26	3	6	9	
	8	26	1	2	3	3	25	2	6	6	
	9	27	1	2	3	3	24	2	3	4	
63	1	5	5	4	4	2	13	19	12	28	
	2	13	5	4	3	2	13	18	11	24	
	3	14	5	4	3	2	13	17	11	21	
	4	15	5	3	2	2	12	17	9	19	
	5	16	5	3	2	2	12	15	9	16	
	6	22	5	3	2	1	12	15	7	13	
	7	23	5	3	1	1	11	15	7	12	
	8	24	5	2	1	1	11	13	5	7	
	9	28	5	2	1	1	11	13	4	6	
64	1	2	5	4	2	4	8	26	30	8	
	2	5	5	4	1	3	7	26	28	7	
	3	8	5	4	1	3	7	26	27	7	
	4	9	5	3	1	3	6	25	26	6	
	5	13	5	3	1	3	6	24	25	4	
	6	14	5	2	1	2	5	24	23	4	
	7	20	5	2	1	2	5	24	22	2	
	8	23	5	1	1	2	4	23	22	1	
	9	29	5	1	1	2	3	23	20	1	
65	1	1	4	5	2	3	23	20	28	28	
	2	7	3	4	2	3	22	19	25	26	
	3	8	3	4	2	3	21	16	25	25	
	4	13	3	4	2	3	21	15	24	25	
	5	14	3	4	2	3	20	13	23	24	
	6	15	3	3	1	3	20	12	22	21	
	7	21	3	3	1	3	19	10	21	21	
	8	22	3	3	1	3	19	7	19	20	
	9	23	3	3	1	3	19	5	18	18	
66	1	12	5	2	5	4	20	23	20	26	
	2	16	4	2	4	4	16	21	20	25	
	3	21	4	2	4	4	16	20	20	25	
	4	22	4	2	3	4	12	18	20	25	
	5	23	4	2	3	4	11	15	20	24	
	6	26	4	2	3	4	9	15	20	24	
	7	27	4	2	2	4	6	13	20	23	
	8	28	4	2	2	4	4	11	20	22	
	9	29	4	2	2	4	4	8	20	22	
67	1	12	5	3	4	2	22	21	21	8	
	2	13	4	3	4	2	22	21	19	8	
	3	18	4	3	4	2	17	20	17	8	
	4	20	4	2	4	2	15	20	17	7	
	5	21	3	2	4	2	15	20	14	6	
	6	22	3	2	4	2	12	19	13	6	
	7	26	3	2	4	2	8	18	12	6	
	8	27	3	1	4	2	6	18	11	5	
	9	29	3	1	4	2	3	18	11	5	
68	1	2	2	3	4	4	28	27	11	20	
	2	6	1	3	3	4	27	26	10	17	
	3	7	1	3	3	4	26	26	10	16	
	4	8	1	3	3	4	24	26	10	14	
	5	13	1	2	3	4	23	26	10	12	
	6	16	1	2	2	3	22	26	10	12	
	7	17	1	2	2	3	21	26	10	10	
	8	25	1	2	2	3	20	26	10	8	
	9	27	1	2	2	3	20	26	10	7	
69	1	3	1	5	1	3	26	29	12	23	
	2	12	1	5	1	2	21	28	12	22	
	3	14	1	5	1	2	18	26	10	20	
	4	16	1	5	1	2	18	25	9	18	
	5	20	1	5	1	2	13	24	8	17	
	6	22	1	5	1	1	12	23	7	14	
	7	23	1	5	1	1	8	20	5	13	
	8	24	1	5	1	1	5	19	5	11	
	9	25	1	5	1	1	2	19	3	8	
70	1	4	4	4	2	3	24	18	10	23	
	2	6	3	3	2	3	23	14	8	21	
	3	10	3	3	2	3	23	13	8	21	
	4	14	3	3	2	3	23	10	6	19	
	5	16	2	3	1	3	23	9	6	19	
	6	24	2	2	1	2	23	7	4	19	
	7	25	2	2	1	2	23	6	3	17	
	8	26	1	2	1	2	23	2	2	17	
	9	27	1	2	1	2	23	1	2	16	
71	1	1	3	4	5	4	13	20	30	15	
	2	10	2	4	4	3	13	18	27	13	
	3	11	2	4	4	3	11	18	25	13	
	4	12	2	4	4	3	10	17	23	12	
	5	13	2	4	4	3	10	17	20	11	
	6	14	2	4	4	2	9	17	17	11	
	7	23	2	4	4	2	9	16	16	10	
	8	28	2	4	4	2	7	16	14	9	
	9	30	2	4	4	2	7	15	12	9	
72	1	6	4	5	4	4	9	25	17	21	
	2	7	4	4	4	4	8	23	17	19	
	3	8	4	4	4	4	8	22	15	18	
	4	10	4	4	4	4	8	22	15	17	
	5	22	4	4	4	4	8	21	13	17	
	6	27	4	3	4	4	8	19	13	17	
	7	28	4	3	4	4	8	17	11	16	
	8	29	4	3	4	4	8	17	11	14	
	9	30	4	3	4	4	8	15	10	14	
73	1	1	3	4	4	4	12	15	10	29	
	2	8	3	4	3	4	12	14	10	28	
	3	17	3	4	3	3	12	12	10	26	
	4	18	3	4	3	3	12	11	10	24	
	5	24	3	4	3	2	12	11	9	24	
	6	26	2	4	2	2	12	10	9	21	
	7	28	2	4	2	1	12	9	8	20	
	8	29	2	4	2	1	12	9	8	18	
	9	30	2	4	2	1	12	8	8	18	
74	1	5	4	2	5	4	27	23	25	19	
	2	12	4	1	4	3	27	21	25	16	
	3	13	4	1	4	3	27	21	23	13	
	4	15	3	1	4	3	26	20	22	11	
	5	16	3	1	3	3	26	20	22	10	
	6	17	3	1	3	2	26	19	20	8	
	7	18	2	1	3	2	26	19	19	5	
	8	21	2	1	3	2	25	18	18	5	
	9	29	2	1	3	2	25	18	18	3	
75	1	1	5	4	1	4	21	9	26	21	
	2	4	4	3	1	4	20	8	25	20	
	3	9	4	3	1	4	20	8	21	20	
	4	13	4	3	1	4	20	8	21	19	
	5	18	3	2	1	3	19	7	19	20	
	6	24	3	2	1	3	18	7	14	19	
	7	27	3	1	1	2	18	7	13	19	
	8	29	2	1	1	2	17	7	11	19	
	9	30	2	1	1	2	16	7	9	19	
76	1	3	1	5	3	4	17	24	3	26	
	2	8	1	4	2	4	17	22	2	23	
	3	9	1	4	2	4	15	22	2	20	
	4	18	1	4	2	4	15	21	2	17	
	5	19	1	3	2	3	14	20	2	14	
	6	20	1	3	1	3	13	18	2	9	
	7	21	1	2	1	3	12	17	2	6	
	8	24	1	2	1	3	11	17	2	4	
	9	27	1	2	1	3	11	16	2	2	
77	1	6	4	3	5	4	19	19	17	22	
	2	9	3	3	4	3	19	17	15	20	
	3	10	3	3	4	3	19	17	14	16	
	4	14	3	3	3	3	19	16	11	14	
	5	20	2	3	3	3	19	16	11	12	
	6	25	2	3	2	3	19	16	7	9	
	7	28	2	3	1	3	19	15	6	7	
	8	29	1	3	1	3	19	14	3	6	
	9	30	1	3	1	3	19	14	2	5	
78	1	3	2	4	3	5	5	9	24	12	
	2	7	2	3	2	4	4	9	24	12	
	3	9	2	3	2	4	4	9	24	11	
	4	11	2	3	2	4	4	9	24	8	
	5	18	2	2	1	4	3	9	23	8	
	6	19	2	2	1	4	3	9	23	7	
	7	20	2	2	1	4	2	9	23	5	
	8	22	2	1	1	4	2	9	23	3	
	9	25	2	1	1	4	2	9	23	2	
79	1	2	3	2	5	2	13	28	29	12	
	2	3	3	2	4	2	10	26	28	12	
	3	7	3	2	4	2	10	25	27	10	
	4	13	3	2	4	2	8	24	26	9	
	5	14	3	2	3	2	8	22	25	9	
	6	19	3	2	3	2	6	22	23	7	
	7	23	3	2	2	2	5	21	23	6	
	8	24	3	2	2	2	4	20	21	5	
	9	25	3	2	2	2	3	19	21	5	
80	1	6	4	4	4	1	22	24	28	12	
	2	12	4	4	4	1	21	23	26	12	
	3	13	4	4	4	1	19	23	24	11	
	4	15	4	4	4	1	17	23	22	8	
	5	19	4	4	3	1	13	21	17	7	
	6	20	4	4	3	1	11	21	15	6	
	7	22	4	4	3	1	11	20	12	5	
	8	23	4	4	3	1	7	19	10	4	
	9	24	4	4	3	1	7	19	9	3	
81	1	13	3	4	3	2	17	26	25	27	
	2	14	3	3	3	2	17	23	21	25	
	3	20	3	3	3	2	17	22	20	24	
	4	21	3	3	3	2	17	18	17	20	
	5	22	3	3	2	1	17	17	14	18	
	6	23	3	2	2	1	17	13	10	13	
	7	24	3	2	1	1	17	12	9	9	
	8	25	3	2	1	1	17	7	6	9	
	9	26	3	2	1	1	17	5	4	4	
82	1	7	4	3	3	4	17	16	23	11	
	2	8	3	3	3	4	15	13	22	9	
	3	9	3	3	3	4	14	13	21	9	
	4	14	3	3	2	4	12	12	21	8	
	5	15	2	3	2	4	10	10	20	7	
	6	22	2	3	2	4	8	9	20	7	
	7	23	2	3	2	4	7	7	19	7	
	8	24	2	3	1	4	5	5	19	5	
	9	26	2	3	1	4	3	5	18	5	
83	1	3	5	1	5	4	28	12	21	17	
	2	6	4	1	5	3	25	10	19	15	
	3	13	4	1	5	3	24	9	19	13	
	4	15	3	1	5	3	20	9	17	12	
	5	23	3	1	5	3	18	7	17	9	
	6	26	3	1	5	2	14	7	15	8	
	7	27	2	1	5	2	13	6	14	5	
	8	29	2	1	5	2	11	5	14	5	
	9	30	2	1	5	2	6	3	13	2	
84	1	2	3	5	3	2	28	22	29	14	
	2	6	3	4	3	2	28	22	29	13	
	3	9	3	4	3	2	28	21	28	13	
	4	12	3	3	3	2	28	19	27	13	
	5	14	3	3	3	2	27	19	27	12	
	6	15	3	2	3	2	27	17	26	12	
	7	16	3	1	3	2	27	17	26	12	
	8	22	3	1	3	2	27	16	25	12	
	9	26	3	1	3	2	27	15	25	12	
85	1	2	5	4	5	4	26	19	21	11	
	2	4	4	4	4	3	24	17	20	10	
	3	7	3	4	4	3	22	16	17	10	
	4	10	3	4	4	3	18	15	17	10	
	5	11	2	4	4	3	17	14	14	10	
	6	16	2	4	4	3	15	14	13	10	
	7	19	1	4	4	3	10	12	11	10	
	8	24	1	4	4	3	10	11	11	10	
	9	29	1	4	4	3	8	10	10	10	
86	1	4	4	4	1	4	28	13	29	28	
	2	5	4	3	1	3	24	12	28	28	
	3	7	4	3	1	3	23	10	25	27	
	4	8	4	2	1	3	19	9	23	26	
	5	9	4	2	1	3	16	8	21	25	
	6	10	3	2	1	2	12	7	21	25	
	7	11	3	1	1	2	12	5	20	24	
	8	17	3	1	1	2	9	2	16	23	
	9	24	3	1	1	2	6	1	15	23	
87	1	9	5	4	4	5	27	16	12	9	
	2	10	4	4	4	4	26	15	11	8	
	3	12	3	4	4	4	23	15	10	8	
	4	13	3	4	4	3	21	12	8	8	
	5	20	3	4	3	3	19	11	8	7	
	6	23	2	3	3	3	15	11	6	7	
	7	25	1	3	3	3	14	10	6	6	
	8	26	1	3	2	2	10	8	5	6	
	9	29	1	3	2	2	9	7	4	6	
88	1	1	3	2	4	4	27	21	2	28	
	2	2	3	2	4	4	26	19	2	23	
	3	14	3	2	4	4	24	16	2	23	
	4	15	3	2	4	4	23	13	2	19	
	5	16	2	1	4	4	21	13	2	18	
	6	22	2	1	4	4	18	11	2	16	
	7	23	2	1	4	4	18	8	2	12	
	8	27	2	1	4	4	15	5	2	10	
	9	29	2	1	4	4	14	3	2	8	
89	1	2	2	3	2	3	28	18	5	3	
	2	4	2	2	2	3	27	16	4	3	
	3	14	2	2	2	3	26	16	3	3	
	4	15	2	2	2	3	24	15	3	3	
	5	18	2	2	2	3	24	14	2	3	
	6	19	2	1	2	3	21	14	2	3	
	7	24	2	1	2	3	21	13	1	3	
	8	26	2	1	2	3	20	13	1	3	
	9	27	2	1	2	3	19	12	1	3	
90	1	5	2	4	4	5	28	13	10	28	
	2	7	2	4	4	4	27	13	10	27	
	3	10	2	4	4	4	24	13	9	27	
	4	14	2	4	4	4	23	13	8	26	
	5	15	1	3	3	3	22	13	8	25	
	6	18	1	3	3	3	19	13	7	22	
	7	26	1	2	3	3	18	13	7	21	
	8	27	1	2	2	3	16	13	7	21	
	9	30	1	2	2	3	14	13	6	19	
91	1	2	4	4	4	1	17	16	17	27	
	2	7	4	4	4	1	16	15	16	26	
	3	8	4	4	4	1	16	15	16	25	
	4	9	4	4	4	1	15	14	16	25	
	5	10	4	4	4	1	15	14	15	24	
	6	14	4	4	4	1	15	14	14	24	
	7	15	4	4	4	1	15	14	14	23	
	8	21	4	4	4	1	14	13	13	23	
	9	26	4	4	4	1	14	13	13	22	
92	1	1	4	5	3	2	26	17	21	23	
	2	2	4	4	3	2	22	17	20	23	
	3	3	4	4	3	2	22	15	19	22	
	4	11	4	4	3	2	16	14	19	22	
	5	12	4	4	3	1	14	14	18	20	
	6	14	4	4	3	1	11	12	18	20	
	7	15	4	4	3	1	8	12	17	19	
	8	24	4	4	3	1	7	11	17	19	
	9	30	4	4	3	1	4	10	17	18	
93	1	7	5	2	3	2	19	16	29	26	
	2	8	4	2	3	2	18	15	27	25	
	3	9	4	2	3	2	16	13	23	25	
	4	10	4	2	3	2	16	12	21	25	
	5	11	3	2	2	1	13	12	18	24	
	6	19	3	2	2	1	11	10	18	24	
	7	20	3	2	2	1	9	10	14	24	
	8	22	3	2	2	1	9	9	14	24	
	9	29	3	2	2	1	6	8	10	24	
94	1	5	4	4	4	4	11	19	28	19	
	2	6	3	3	3	3	11	19	28	18	
	3	14	3	3	3	3	11	19	26	17	
	4	17	3	3	3	3	10	19	23	14	
	5	21	3	2	3	3	10	18	22	13	
	6	22	3	2	3	3	10	18	22	12	
	7	23	3	2	3	3	10	18	20	11	
	8	24	3	2	3	3	9	18	17	9	
	9	30	3	2	3	3	9	18	17	8	
95	1	4	5	1	1	3	19	29	18	30	
	2	7	4	1	1	3	19	26	18	26	
	3	8	4	1	1	3	19	22	17	25	
	4	11	4	1	1	3	18	19	15	24	
	5	15	4	1	1	3	17	16	14	22	
	6	16	4	1	1	3	17	11	13	21	
	7	20	4	1	1	3	16	11	12	19	
	8	21	4	1	1	3	16	6	9	17	
	9	30	4	1	1	3	16	5	9	15	
96	1	6	3	4	1	5	28	20	8	10	
	2	7	3	3	1	4	26	20	8	8	
	3	8	3	3	1	4	25	18	7	8	
	4	9	3	3	1	4	25	18	7	7	
	5	10	3	2	1	4	23	17	6	6	
	6	15	2	2	1	3	23	17	6	4	
	7	18	2	1	1	3	22	15	6	3	
	8	27	2	1	1	3	20	15	5	3	
	9	28	2	1	1	3	19	14	5	2	
97	1	3	5	3	3	4	21	15	17	20	
	2	7	4	2	3	3	20	13	15	18	
	3	8	4	2	3	3	17	12	15	16	
	4	9	4	2	3	3	17	10	15	15	
	5	11	4	2	2	2	13	9	13	13	
	6	12	4	1	2	2	11	9	13	12	
	7	19	4	1	1	2	10	8	13	10	
	8	29	4	1	1	2	8	7	11	8	
	9	30	4	1	1	2	6	5	11	6	
98	1	1	5	4	4	5	8	25	24	16	
	2	2	4	3	4	4	7	24	22	15	
	3	4	4	3	4	4	7	24	20	14	
	4	5	4	3	4	3	6	23	18	11	
	5	7	4	3	4	3	6	23	18	10	
	6	18	3	3	4	3	6	23	16	8	
	7	21	3	3	4	3	5	23	15	8	
	8	22	3	3	4	2	5	22	13	5	
	9	27	3	3	4	2	4	22	11	5	
99	1	2	2	5	2	4	24	29	22	20	
	2	10	2	5	1	4	22	28	22	19	
	3	20	2	5	1	4	22	28	18	19	
	4	21	2	5	1	4	20	26	18	19	
	5	22	2	5	1	4	20	26	14	18	
	6	23	2	5	1	4	20	24	13	18	
	7	24	2	5	1	4	19	24	11	18	
	8	27	2	5	1	4	18	23	10	18	
	9	28	2	5	1	4	17	22	8	18	
100	1	1	5	3	4	1	24	19	25	3	
	2	9	5	3	4	1	23	17	22	3	
	3	10	5	3	3	1	22	15	19	3	
	4	13	5	3	3	1	21	14	16	3	
	5	14	5	3	3	1	20	12	12	2	
	6	15	5	3	2	1	20	7	12	2	
	7	16	5	3	1	1	20	5	7	2	
	8	19	5	3	1	1	18	3	3	2	
	9	24	5	3	1	1	18	3	3	1	
101	1	1	1	3	3	3	6	17	28	11	
	2	6	1	3	3	3	5	17	25	10	
	3	15	1	3	3	3	5	15	23	10	
	4	16	1	3	3	3	5	12	21	10	
	5	19	1	3	2	2	4	11	18	9	
	6	21	1	3	2	2	4	9	17	9	
	7	25	1	3	2	2	3	5	13	9	
	8	26	1	3	1	2	3	3	12	9	
	9	27	1	3	1	2	3	3	9	9	
102	1	0	0	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	R 3	R 4	N 1	N 2	N 3	N 4
	27	26	26	27	1771	1789	1781	1649

************************************************************************
