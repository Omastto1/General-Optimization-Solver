jobs  (incl. supersource/sink ):	102
RESOURCES
- renewable                 : 4 R
- nonrenewable              : 4 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	10		2 3 4 5 6 7 10 11 13 14 
2	9	7		23 22 20 18 17 16 12 
3	9	5		26 25 24 9 8 
4	9	4		25 24 22 8 
5	9	6		25 24 23 20 18 17 
6	9	2		25 8 
7	9	7		27 26 25 24 23 21 20 
8	9	4		21 20 19 18 
9	9	5		35 27 22 21 20 
10	9	3		21 20 19 
11	9	4		35 25 21 20 
12	9	3		35 25 15 
13	9	3		29 28 18 
14	9	5		35 33 29 27 25 
15	9	4		33 29 27 21 
16	9	4		31 30 28 26 
17	9	3		35 29 21 
18	9	6		51 35 33 32 30 27 
19	9	3		51 27 23 
20	9	6		43 39 36 33 29 28 
21	9	8		51 46 43 38 36 32 31 30 
22	9	8		51 49 46 38 37 36 31 30 
23	9	7		49 46 43 36 32 31 30 
24	9	8		51 50 49 46 43 36 31 30 
25	9	12		51 49 47 46 45 43 42 41 40 39 34 32 
26	9	9		49 43 42 40 38 36 35 34 32 
27	9	8		49 43 42 38 37 36 34 31 
28	9	6		51 49 47 46 38 32 
29	9	7		51 49 47 42 40 37 34 
30	9	3		42 40 34 
31	9	7		59 56 54 45 44 41 39 
32	9	4		56 54 48 37 
33	9	5		58 56 50 48 42 
34	9	8		59 58 56 54 53 52 48 44 
35	9	6		59 56 52 48 45 44 
36	9	5		59 56 47 44 41 
37	9	6		58 57 53 52 50 44 
38	9	3		48 45 41 
39	9	7		63 62 60 58 57 53 52 
40	9	8		66 61 59 58 57 56 55 53 
41	9	6		62 60 57 55 53 52 
42	9	6		62 60 57 54 53 52 
43	9	3		60 53 48 
44	9	7		72 69 63 62 61 60 55 
45	9	6		63 62 61 58 57 53 
46	9	5		69 60 58 57 55 
47	9	4		62 61 55 54 
48	9	8		72 69 67 66 63 62 61 57 
49	9	6		72 69 63 62 61 55 
50	9	5		66 63 62 59 55 
51	9	11		79 78 75 72 71 69 68 67 66 65 61 
52	9	9		79 73 71 69 67 66 65 64 61 
53	9	11		88 79 78 74 73 72 71 70 67 65 64 
54	9	9		79 74 73 72 70 69 66 65 64 
55	9	9		88 78 74 73 71 70 67 65 64 
56	9	8		80 75 73 72 71 70 67 62 
57	9	7		88 78 74 71 70 65 64 
58	9	9		88 87 85 79 78 74 72 70 64 
59	9	7		88 87 78 73 72 70 65 
60	9	7		84 83 79 78 71 67 66 
61	9	11		90 88 87 86 85 83 81 80 77 74 70 
62	9	8		88 87 86 85 81 79 74 68 
63	9	5		101 79 76 75 65 
64	9	6		101 84 80 76 75 68 
65	9	7		98 90 85 84 83 81 77 
66	9	7		99 90 85 82 81 80 76 
67	9	5		101 90 82 77 76 
68	9	6		98 96 90 83 82 77 
69	9	10		100 98 97 95 92 90 89 88 87 84 
70	9	5		100 99 84 82 76 
71	9	7		99 98 95 93 90 87 82 
72	9	4		100 90 83 76 
73	9	7		99 98 95 93 86 83 82 
74	9	7		101 99 98 95 92 89 84 
75	9	6		99 96 95 93 86 82 
76	9	7		98 97 96 95 93 92 89 
77	9	7		100 99 97 95 93 92 89 
78	9	7		101 98 97 96 95 92 89 
79	9	6		98 96 95 93 92 91 
80	9	5		97 95 93 92 91 
81	9	4		100 95 93 89 
82	9	3		97 92 89 
83	9	2		97 92 
84	9	2		96 93 
85	9	2		92 91 
86	9	2		92 91 
87	9	2		96 94 
88	9	2		96 94 
89	9	1		91 
90	9	1		91 
91	9	1		94 
92	9	1		94 
93	9	1		94 
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
2	1	3	4	2	4	1	23	18	29	26	
	2	11	4	2	3	1	20	17	26	23	
	3	12	4	2	3	1	18	15	25	23	
	4	16	4	2	3	1	16	14	24	19	
	5	17	4	2	3	1	14	12	21	19	
	6	22	4	2	3	1	12	10	20	17	
	7	23	4	2	3	1	10	8	20	14	
	8	24	4	2	3	1	9	4	17	11	
	9	25	4	2	3	1	7	4	16	11	
3	1	6	4	3	2	2	9	21	24	24	
	2	9	3	2	2	1	8	19	19	23	
	3	13	3	2	2	1	8	18	17	21	
	4	18	2	2	2	1	7	18	14	21	
	5	23	2	1	2	1	5	16	11	20	
	6	25	2	1	2	1	5	15	8	18	
	7	26	1	1	2	1	4	14	8	18	
	8	27	1	1	2	1	3	12	3	16	
	9	28	1	1	2	1	1	11	3	16	
4	1	2	1	3	3	4	26	17	21	28	
	2	11	1	2	3	4	25	14	20	22	
	3	14	1	2	3	4	25	13	19	20	
	4	15	1	2	3	3	25	11	19	19	
	5	17	1	1	3	3	25	9	17	16	
	6	21	1	1	3	3	24	8	15	10	
	7	22	1	1	3	2	24	6	14	7	
	8	24	1	1	3	2	24	5	13	6	
	9	25	1	1	3	2	24	2	12	4	
5	1	2	3	5	4	4	1	17	19	26	
	2	3	3	4	3	4	1	16	18	24	
	3	12	3	4	3	4	1	13	16	23	
	4	13	3	4	3	4	1	12	15	22	
	5	16	3	3	3	3	1	10	15	19	
	6	19	3	3	3	3	1	10	14	19	
	7	23	3	3	3	3	1	7	14	17	
	8	24	3	3	3	3	1	6	13	16	
	9	30	3	3	3	3	1	5	12	13	
6	1	2	3	1	4	2	21	20	22	27	
	2	6	3	1	4	2	18	19	20	26	
	3	12	3	1	4	2	17	19	19	24	
	4	13	3	1	4	2	15	19	18	22	
	5	14	3	1	3	2	13	19	17	21	
	6	15	2	1	3	1	10	19	16	20	
	7	17	2	1	3	1	8	19	15	18	
	8	18	2	1	3	1	4	19	14	15	
	9	26	2	1	3	1	3	19	12	14	
7	1	2	4	5	3	3	25	27	30	25	
	2	4	3	4	2	3	22	27	27	23	
	3	9	3	4	2	3	21	26	26	19	
	4	11	3	4	2	3	20	24	23	18	
	5	15	2	4	1	3	17	24	22	16	
	6	20	2	3	1	3	15	23	18	13	
	7	22	2	3	1	3	12	21	17	13	
	8	26	2	3	1	3	9	20	14	10	
	9	29	2	3	1	3	9	20	13	7	
8	1	4	2	3	4	1	27	13	7	10	
	2	5	1	3	4	1	26	13	7	8	
	3	7	1	3	4	1	25	13	6	8	
	4	14	1	2	3	1	24	12	5	7	
	5	15	1	2	3	1	23	12	5	5	
	6	19	1	2	3	1	21	12	4	5	
	7	20	1	2	2	1	21	12	4	4	
	8	21	1	1	2	1	20	11	3	2	
	9	26	1	1	2	1	19	11	2	2	
9	1	2	4	4	4	5	24	17	16	26	
	2	3	4	4	4	4	21	17	15	23	
	3	9	4	4	4	4	20	17	14	23	
	4	12	4	3	4	4	18	17	13	21	
	5	13	3	3	3	3	15	17	13	20	
	6	15	3	3	3	3	15	17	11	19	
	7	21	2	2	2	2	11	17	9	18	
	8	28	2	2	2	2	11	17	9	17	
	9	29	2	2	2	2	8	17	7	16	
10	1	2	1	5	2	3	23	25	30	25	
	2	3	1	4	1	2	21	24	30	23	
	3	6	1	4	1	2	21	24	30	22	
	4	7	1	3	1	2	19	23	30	22	
	5	13	1	3	1	1	18	23	30	20	
	6	18	1	3	1	1	17	23	30	19	
	7	21	1	3	1	1	16	22	30	19	
	8	22	1	2	1	1	16	21	30	17	
	9	24	1	2	1	1	15	21	30	17	
11	1	5	2	2	5	3	14	29	26	28	
	2	9	2	2	4	3	14	29	25	25	
	3	10	2	2	3	3	12	29	21	24	
	4	11	2	2	3	3	12	28	20	21	
	5	15	1	2	2	3	10	27	19	20	
	6	20	1	1	2	3	9	27	17	20	
	7	22	1	1	2	3	8	27	14	18	
	8	23	1	1	1	3	8	26	12	15	
	9	27	1	1	1	3	6	26	12	13	
12	1	2	1	3	5	3	28	24	11	22	
	2	15	1	3	4	3	27	23	10	20	
	3	16	1	3	4	3	24	23	9	18	
	4	18	1	3	4	3	23	23	8	16	
	5	19	1	3	3	2	19	23	7	15	
	6	20	1	2	3	2	18	23	6	14	
	7	23	1	2	3	2	16	23	5	13	
	8	29	1	2	3	1	13	23	5	11	
	9	30	1	2	3	1	12	23	4	9	
13	1	14	5	5	1	4	26	27	22	3	
	2	20	4	4	1	3	25	22	21	3	
	3	22	4	4	1	3	25	21	19	3	
	4	23	4	4	1	3	25	20	17	3	
	5	24	3	4	1	2	24	17	14	2	
	6	25	3	4	1	2	24	15	13	2	
	7	26	3	4	1	2	24	13	10	1	
	8	27	2	4	1	2	24	10	7	1	
	9	30	2	4	1	2	24	8	7	1	
14	1	6	5	5	5	1	7	21	28	21	
	2	11	5	4	4	1	6	21	28	17	
	3	15	5	4	4	1	6	21	26	16	
	4	17	5	3	3	1	5	21	26	15	
	5	18	5	3	3	1	5	21	25	11	
	6	19	5	3	3	1	4	21	24	11	
	7	21	5	2	2	1	4	21	24	8	
	8	25	5	2	2	1	2	21	23	7	
	9	27	5	2	2	1	2	21	22	5	
15	1	1	4	2	3	2	26	25	4	23	
	2	2	4	2	3	1	26	21	4	20	
	3	3	4	2	3	1	24	18	3	17	
	4	4	4	2	3	1	23	17	3	14	
	5	5	4	2	2	1	23	15	2	14	
	6	8	4	2	2	1	23	12	2	10	
	7	9	4	2	1	1	22	7	1	9	
	8	28	4	2	1	1	20	5	1	7	
	9	29	4	2	1	1	20	5	1	6	
16	1	10	1	2	3	4	11	29	12	11	
	2	11	1	2	2	4	10	26	12	11	
	3	12	1	2	2	4	8	24	12	10	
	4	13	1	2	2	4	7	23	12	10	
	5	14	1	2	2	4	7	21	12	9	
	6	15	1	2	1	4	7	21	12	8	
	7	18	1	2	1	4	5	20	12	7	
	8	19	1	2	1	4	4	17	12	7	
	9	28	1	2	1	4	4	17	12	6	
17	1	3	3	4	4	5	14	20	30	25	
	2	5	3	4	4	4	11	18	29	22	
	3	7	3	4	4	4	10	15	27	21	
	4	13	3	4	3	4	8	15	27	18	
	5	20	3	4	3	4	7	11	25	16	
	6	25	3	4	3	4	6	10	25	14	
	7	26	3	4	3	4	6	7	23	12	
	8	27	3	4	2	4	4	6	22	10	
	9	30	3	4	2	4	3	5	22	8	
18	1	3	3	3	1	1	25	29	11	23	
	2	8	2	2	1	1	24	23	10	21	
	3	9	2	2	1	1	24	22	9	17	
	4	13	2	2	1	1	24	19	9	15	
	5	18	2	2	1	1	23	15	8	13	
	6	20	2	2	1	1	22	14	7	10	
	7	24	2	2	1	1	22	11	7	8	
	8	26	2	2	1	1	21	9	6	5	
	9	27	2	2	1	1	20	5	6	3	
19	1	1	4	3	4	5	18	18	12	27	
	2	3	3	2	4	5	17	16	12	27	
	3	5	3	2	4	5	17	15	11	25	
	4	10	3	2	4	5	15	15	10	24	
	5	11	3	2	3	5	15	14	9	24	
	6	23	3	2	3	5	14	12	8	23	
	7	24	3	2	3	5	13	12	7	21	
	8	26	3	2	2	5	11	10	5	21	
	9	27	3	2	2	5	11	9	5	19	
20	1	7	3	4	3	4	20	28	6	25	
	2	15	3	4	3	4	20	26	5	24	
	3	18	3	4	3	4	18	22	5	24	
	4	19	3	3	3	4	17	22	4	24	
	5	23	3	3	3	3	17	17	4	24	
	6	27	3	2	3	3	17	17	3	24	
	7	28	3	1	3	3	15	14	3	24	
	8	29	3	1	3	2	14	12	3	24	
	9	30	3	1	3	2	14	11	2	24	
21	1	3	4	4	5	3	12	28	13	28	
	2	4	3	4	4	3	12	24	12	26	
	3	5	3	4	4	3	12	24	11	25	
	4	6	3	4	4	3	12	22	9	23	
	5	7	3	3	4	3	12	18	8	22	
	6	20	3	3	4	3	12	16	7	21	
	7	21	3	3	4	3	12	14	6	21	
	8	22	3	3	4	3	12	12	5	20	
	9	29	3	3	4	3	12	10	4	19	
22	1	1	2	4	1	4	16	15	6	26	
	2	2	2	4	1	3	16	15	6	23	
	3	4	2	4	1	3	16	15	6	20	
	4	9	2	4	1	3	16	15	5	17	
	5	10	2	4	1	3	15	14	5	15	
	6	17	2	4	1	2	15	14	4	13	
	7	19	2	4	1	2	15	14	4	7	
	8	26	2	4	1	2	15	14	3	4	
	9	30	2	4	1	2	15	14	3	3	
23	1	3	4	4	1	4	16	22	22	6	
	2	8	3	4	1	3	13	22	21	5	
	3	10	3	4	1	3	13	22	19	4	
	4	11	3	4	1	3	12	22	18	4	
	5	14	3	4	1	3	10	21	15	3	
	6	15	3	4	1	2	8	21	15	3	
	7	16	3	4	1	2	6	21	13	2	
	8	17	3	4	1	2	3	21	12	2	
	9	20	3	4	1	2	2	21	10	1	
24	1	1	4	4	3	4	27	28	18	15	
	2	4	4	4	3	3	27	27	16	14	
	3	6	4	4	3	3	27	24	14	14	
	4	7	4	4	3	3	27	23	13	12	
	5	8	4	4	3	3	27	21	13	11	
	6	16	4	3	2	2	27	20	12	9	
	7	19	4	3	2	2	27	19	11	8	
	8	20	4	3	2	2	27	15	9	7	
	9	21	4	3	2	2	27	14	8	6	
25	1	1	5	2	4	2	13	28	18	23	
	2	3	4	1	4	1	11	28	17	22	
	3	10	4	1	4	1	11	27	17	20	
	4	11	4	1	3	1	10	27	16	17	
	5	12	4	1	3	1	9	26	16	17	
	6	14	4	1	3	1	7	26	16	15	
	7	20	4	1	2	1	6	25	16	12	
	8	27	4	1	2	1	5	25	15	11	
	9	28	4	1	2	1	5	25	15	9	
26	1	1	2	4	3	5	27	19	5	18	
	2	3	2	4	3	5	24	16	5	17	
	3	4	2	4	3	5	23	16	4	17	
	4	7	2	4	3	5	22	14	4	16	
	5	8	2	3	2	5	22	14	3	15	
	6	9	1	3	2	5	21	13	2	15	
	7	15	1	2	2	5	19	11	2	15	
	8	16	1	2	1	5	18	11	1	14	
	9	26	1	2	1	5	18	9	1	13	
27	1	7	4	2	2	4	30	30	21	16	
	2	8	3	2	1	3	29	26	21	16	
	3	9	3	2	1	3	29	26	20	15	
	4	10	2	2	1	3	28	24	19	15	
	5	13	2	2	1	3	28	23	18	15	
	6	21	2	2	1	2	28	22	18	14	
	7	24	1	2	1	2	27	20	17	13	
	8	27	1	2	1	2	27	17	17	13	
	9	28	1	2	1	2	27	16	16	13	
28	1	7	2	5	3	2	16	25	12	22	
	2	9	2	4	3	2	16	25	10	22	
	3	14	2	4	3	2	13	22	9	22	
	4	18	2	4	3	2	12	21	8	21	
	5	22	2	4	3	2	10	17	7	21	
	6	23	2	4	3	2	8	17	6	20	
	7	24	2	4	3	2	8	14	5	20	
	8	29	2	4	3	2	7	12	3	19	
	9	30	2	4	3	2	5	10	3	19	
29	1	4	4	4	4	5	22	23	5	22	
	2	9	4	4	4	4	19	23	5	19	
	3	10	3	4	4	3	17	21	5	17	
	4	22	3	4	4	3	14	21	5	17	
	5	23	2	4	4	3	12	18	5	13	
	6	26	2	3	4	2	11	18	5	9	
	7	27	1	3	4	1	8	17	5	9	
	8	28	1	3	4	1	6	16	5	6	
	9	30	1	3	4	1	6	15	5	3	
30	1	2	2	5	4	2	24	28	27	28	
	2	4	1	4	3	1	21	24	26	24	
	3	13	1	4	3	1	19	21	22	23	
	4	14	1	3	2	1	16	19	19	19	
	5	17	1	2	2	1	15	19	17	17	
	6	18	1	2	2	1	11	16	16	13	
	7	26	1	1	1	1	9	13	12	8	
	8	28	1	1	1	1	7	11	10	4	
	9	30	1	1	1	1	6	7	6	2	
31	1	3	3	2	3	3	21	19	24	27	
	2	9	3	2	3	3	19	19	23	27	
	3	10	3	2	3	3	19	18	22	22	
	4	11	3	2	3	2	16	18	22	20	
	5	17	3	2	3	2	13	18	21	16	
	6	18	3	2	3	2	12	17	21	13	
	7	23	3	2	3	1	11	17	20	9	
	8	27	3	2	3	1	8	16	19	8	
	9	29	3	2	3	1	7	16	19	3	
32	1	4	2	3	2	4	6	17	24	27	
	2	6	1	2	2	3	6	16	23	27	
	3	12	1	2	2	3	6	15	22	27	
	4	13	1	2	2	3	6	13	22	27	
	5	21	1	1	2	3	6	11	20	28	
	6	22	1	1	2	3	6	11	20	27	
	7	23	1	1	2	3	6	10	19	27	
	8	24	1	1	2	3	6	8	19	27	
	9	27	1	1	2	3	6	7	18	27	
33	1	7	5	5	3	4	25	11	24	11	
	2	9	5	4	2	4	25	10	24	10	
	3	14	5	4	2	4	25	8	24	10	
	4	19	5	4	2	4	24	8	24	8	
	5	20	5	4	1	3	24	7	24	8	
	6	21	5	3	1	3	23	5	24	8	
	7	22	5	3	1	3	23	5	24	7	
	8	27	5	3	1	3	22	4	24	6	
	9	30	5	3	1	3	22	2	24	5	
34	1	8	2	2	2	4	26	18	14	16	
	2	16	1	2	2	3	24	16	13	15	
	3	22	1	2	2	3	22	14	13	14	
	4	23	1	2	2	3	19	13	13	12	
	5	24	1	2	2	2	18	12	12	12	
	6	25	1	2	2	2	13	11	12	12	
	7	26	1	2	2	2	12	10	11	10	
	8	27	1	2	2	2	9	8	10	9	
	9	28	1	2	2	2	8	7	10	9	
35	1	14	1	3	5	4	26	8	21	28	
	2	15	1	2	4	4	24	8	20	23	
	3	17	1	2	4	4	22	8	19	20	
	4	19	1	2	4	4	21	8	19	17	
	5	20	1	2	4	3	19	8	17	16	
	6	23	1	2	4	3	16	8	16	11	
	7	24	1	2	4	3	14	8	13	9	
	8	29	1	2	4	3	12	8	12	7	
	9	30	1	2	4	3	12	8	12	3	
36	1	11	5	4	4	2	30	25	8	20	
	2	13	5	4	4	2	29	23	6	19	
	3	15	5	4	4	2	29	22	6	19	
	4	21	5	4	4	2	28	21	5	19	
	5	23	5	4	3	2	28	20	5	19	
	6	25	5	4	3	2	28	18	4	19	
	7	27	5	4	3	2	27	16	4	19	
	8	28	5	4	3	2	27	15	4	19	
	9	29	5	4	3	2	27	15	3	19	
37	1	2	4	4	4	4	19	18	16	13	
	2	3	3	3	4	3	19	17	15	13	
	3	5	3	3	4	3	18	17	15	13	
	4	6	3	3	4	3	18	16	15	13	
	5	7	3	3	4	3	17	16	14	13	
	6	18	3	3	4	2	17	15	14	13	
	7	21	3	3	4	2	16	15	14	13	
	8	22	3	3	4	2	16	14	14	13	
	9	27	3	3	4	2	16	14	14	12	
38	1	1	4	5	4	4	21	20	23	3	
	2	9	3	4	3	3	20	18	23	3	
	3	19	3	4	3	3	20	16	18	3	
	4	21	3	4	3	3	18	13	16	3	
	5	23	2	4	2	3	18	12	15	2	
	6	24	2	4	2	3	16	9	13	2	
	7	27	2	4	1	3	16	5	10	2	
	8	29	2	4	1	3	14	4	8	2	
	9	30	2	4	1	3	14	2	3	2	
39	1	4	3	2	5	5	25	10	7	19	
	2	10	2	2	4	4	24	8	5	17	
	3	18	2	2	3	4	20	8	5	16	
	4	19	2	2	3	4	18	8	4	15	
	5	24	2	2	3	4	16	7	4	14	
	6	26	2	2	2	4	13	6	4	13	
	7	27	2	2	1	4	10	6	3	13	
	8	28	2	2	1	4	8	5	2	11	
	9	29	2	2	1	4	8	5	2	10	
40	1	5	4	3	4	4	27	6	22	23	
	2	13	4	2	3	4	26	5	21	23	
	3	14	4	2	3	4	25	5	19	23	
	4	15	4	2	2	3	25	4	19	22	
	5	18	4	2	2	3	24	3	18	21	
	6	19	4	2	2	2	23	3	17	21	
	7	20	4	2	1	1	22	2	16	20	
	8	21	4	2	1	1	21	2	16	20	
	9	22	4	2	1	1	19	1	15	20	
41	1	6	4	2	4	4	20	16	28	14	
	2	7	4	1	4	3	18	15	25	13	
	3	9	4	1	4	3	17	15	23	12	
	4	10	4	1	4	3	15	14	19	11	
	5	19	4	1	4	3	12	13	17	11	
	6	20	4	1	3	2	11	10	15	10	
	7	23	4	1	3	2	9	10	14	10	
	8	24	4	1	3	2	8	8	11	8	
	9	28	4	1	3	2	7	7	9	8	
42	1	1	4	4	3	3	26	21	25	23	
	2	3	4	4	3	3	24	20	24	23	
	3	6	4	4	3	3	23	20	23	21	
	4	9	4	3	3	3	21	20	23	20	
	5	13	3	3	2	2	21	20	22	19	
	6	16	3	2	2	2	20	20	22	17	
	7	24	2	1	2	2	18	20	21	15	
	8	25	2	1	2	2	17	20	21	15	
	9	27	2	1	2	2	15	20	20	13	
43	1	1	4	2	3	4	25	23	17	15	
	2	4	4	1	3	4	25	23	16	14	
	3	10	4	1	3	4	22	22	14	12	
	4	11	4	1	3	4	21	22	13	11	
	5	13	3	1	2	3	19	20	11	9	
	6	15	3	1	2	3	19	20	7	8	
	7	16	3	1	2	3	16	20	6	7	
	8	17	3	1	2	3	15	19	5	5	
	9	24	3	1	2	3	14	18	1	5	
44	1	4	4	4	3	2	19	22	20	23	
	2	8	4	3	3	2	17	19	19	23	
	3	10	3	3	3	2	17	17	19	18	
	4	14	3	3	3	2	16	13	19	16	
	5	16	3	3	2	1	15	10	18	16	
	6	20	2	2	2	1	15	8	18	12	
	7	21	2	2	1	1	14	7	18	11	
	8	22	1	2	1	1	14	5	18	8	
	9	26	1	2	1	1	13	2	18	5	
45	1	7	4	4	4	5	21	30	26	20	
	2	10	4	3	4	4	21	29	22	18	
	3	21	3	3	4	4	21	27	19	17	
	4	22	3	3	4	4	21	26	17	16	
	5	23	3	3	3	3	21	26	13	15	
	6	24	2	3	3	3	21	25	11	13	
	7	25	2	3	3	3	21	24	7	12	
	8	27	1	3	3	3	21	24	5	9	
	9	28	1	3	3	3	21	23	2	8	
46	1	3	2	3	3	3	13	7	11	24	
	2	6	2	2	3	3	13	6	11	23	
	3	7	2	2	3	3	13	6	10	22	
	4	8	2	2	3	3	13	6	10	21	
	5	12	2	2	3	2	13	5	9	21	
	6	17	2	2	3	2	13	4	8	21	
	7	18	2	2	3	2	13	3	8	19	
	8	26	2	2	3	2	13	2	7	18	
	9	28	2	2	3	2	13	2	7	17	
47	1	7	4	1	1	1	11	24	16	7	
	2	9	4	1	1	1	10	22	16	6	
	3	10	4	1	1	1	10	20	16	6	
	4	21	4	1	1	1	8	20	16	6	
	5	22	4	1	1	1	8	18	15	6	
	6	25	4	1	1	1	8	17	15	5	
	7	26	4	1	1	1	6	15	15	5	
	8	29	4	1	1	1	6	14	14	5	
	9	30	4	1	1	1	5	14	14	5	
48	1	12	5	3	2	4	26	23	12	25	
	2	16	4	2	2	3	25	23	9	21	
	3	19	4	2	2	3	24	22	8	18	
	4	21	3	2	2	3	22	22	7	14	
	5	22	3	1	2	3	21	21	7	13	
	6	23	3	1	2	3	19	21	5	11	
	7	24	3	1	2	3	16	21	4	7	
	8	25	2	1	2	3	16	20	3	6	
	9	26	2	1	2	3	13	20	3	2	
49	1	1	5	2	4	4	11	18	13	23	
	2	4	4	2	3	4	11	17	11	23	
	3	5	4	2	3	4	11	15	11	23	
	4	7	4	2	3	4	11	14	9	23	
	5	18	4	2	3	4	11	12	9	23	
	6	23	4	2	2	4	11	11	7	23	
	7	25	4	2	2	4	11	9	5	23	
	8	27	4	2	2	4	11	7	5	23	
	9	28	4	2	2	4	11	5	4	23	
50	1	8	4	4	3	3	30	30	20	29	
	2	9	4	4	3	2	26	23	19	26	
	3	15	4	4	3	2	22	23	19	23	
	4	17	4	4	3	2	22	20	18	23	
	5	20	3	4	3	1	18	16	16	21	
	6	22	3	3	2	1	16	12	16	18	
	7	27	2	3	2	1	12	8	14	15	
	8	28	2	3	2	1	11	7	14	13	
	9	29	2	3	2	1	8	4	13	11	
51	1	9	3	4	2	2	23	17	26	23	
	2	10	3	4	2	1	23	17	24	23	
	3	11	3	4	2	1	23	17	23	21	
	4	13	3	4	2	1	22	17	22	21	
	5	20	2	3	2	1	22	16	20	19	
	6	21	2	3	2	1	22	16	19	18	
	7	22	2	3	2	1	22	16	16	17	
	8	23	2	2	2	1	21	15	15	17	
	9	29	2	2	2	1	21	15	14	16	
52	1	7	3	2	5	5	26	29	24	15	
	2	8	3	1	4	5	23	27	22	13	
	3	20	3	1	4	5	23	26	20	12	
	4	21	3	1	4	5	22	26	18	10	
	5	25	2	1	4	5	21	24	18	8	
	6	26	2	1	4	5	19	24	15	8	
	7	27	2	1	4	5	18	22	15	5	
	8	29	2	1	4	5	17	21	12	5	
	9	30	2	1	4	5	16	21	12	4	
53	1	8	3	2	5	2	16	7	26	23	
	2	11	3	1	4	2	15	7	25	21	
	3	14	3	1	4	2	14	5	20	19	
	4	15	3	1	4	2	14	4	19	19	
	5	16	3	1	3	2	14	4	16	17	
	6	17	2	1	3	2	13	4	13	16	
	7	25	2	1	2	2	12	2	12	13	
	8	27	2	1	2	2	12	1	8	11	
	9	30	2	1	2	2	12	1	8	10	
54	1	4	3	3	2	4	12	18	23	30	
	2	7	2	3	2	4	12	18	22	29	
	3	10	2	3	2	4	11	16	20	28	
	4	11	2	3	2	4	10	16	19	28	
	5	12	2	3	2	4	9	15	17	27	
	6	13	2	3	2	4	8	14	17	27	
	7	23	2	3	2	4	8	12	15	26	
	8	27	2	3	2	4	7	12	14	25	
	9	28	2	3	2	4	6	11	13	25	
55	1	1	5	3	2	4	24	11	28	22	
	2	2	4	3	2	4	23	10	27	19	
	3	10	4	3	2	4	21	9	26	17	
	4	13	3	3	2	4	18	8	26	16	
	5	14	3	2	2	4	14	8	24	12	
	6	16	3	2	2	3	11	6	24	12	
	7	23	2	2	2	3	9	5	22	10	
	8	27	2	2	2	3	6	5	21	6	
	9	30	2	2	2	3	6	4	20	6	
56	1	3	1	1	4	4	21	24	16	24	
	2	5	1	1	3	4	20	19	16	23	
	3	6	1	1	3	4	17	17	15	23	
	4	7	1	1	3	4	17	17	13	23	
	5	12	1	1	3	4	15	15	13	22	
	6	15	1	1	3	3	12	12	11	22	
	7	22	1	1	3	3	10	11	10	22	
	8	24	1	1	3	3	10	7	9	22	
	9	25	1	1	3	3	8	7	8	22	
57	1	8	4	5	4	3	20	30	18	22	
	2	9	3	4	4	3	18	28	17	21	
	3	13	3	4	3	3	18	28	16	19	
	4	14	3	4	3	3	17	28	12	19	
	5	19	2	4	3	3	15	27	11	16	
	6	20	2	3	2	3	14	27	8	16	
	7	21	2	3	2	3	12	26	7	14	
	8	26	2	3	1	3	11	26	5	11	
	9	27	2	3	1	3	10	25	3	11	
58	1	2	5	3	3	2	12	25	15	14	
	2	3	4	3	2	2	11	23	13	13	
	3	11	4	3	2	2	10	23	12	11	
	4	19	4	3	2	2	10	21	12	10	
	5	20	3	3	2	2	10	18	11	9	
	6	21	3	3	2	2	9	14	9	9	
	7	22	2	3	2	2	9	14	9	8	
	8	23	2	3	2	2	8	11	7	6	
	9	24	2	3	2	2	8	9	7	5	
59	1	2	3	4	2	4	22	28	5	24	
	2	4	2	4	2	4	20	24	5	23	
	3	12	2	4	2	4	19	24	5	23	
	4	22	2	4	2	4	18	21	5	21	
	5	23	2	4	2	3	18	19	5	20	
	6	24	2	4	2	3	17	16	5	19	
	7	26	2	4	2	3	16	15	5	18	
	8	28	2	4	2	3	16	13	5	15	
	9	30	2	4	2	3	15	12	5	15	
60	1	3	4	3	2	3	28	12	18	23	
	2	6	4	3	1	2	23	12	17	22	
	3	7	4	3	1	2	23	12	15	22	
	4	19	4	3	1	2	19	12	13	22	
	5	20	4	2	1	1	14	12	13	21	
	6	21	4	2	1	1	12	11	11	21	
	7	22	4	2	1	1	11	11	9	21	
	8	23	4	2	1	1	7	11	8	21	
	9	26	4	2	1	1	4	11	7	21	
61	1	5	1	5	2	4	29	18	29	20	
	2	6	1	5	2	3	29	18	26	20	
	3	7	1	5	2	3	27	18	26	20	
	4	12	1	5	2	3	27	18	25	19	
	5	23	1	5	2	2	26	18	22	18	
	6	25	1	5	2	2	26	18	22	17	
	7	26	1	5	2	1	24	18	20	18	
	8	28	1	5	2	1	24	18	19	17	
	9	29	1	5	2	1	23	18	18	17	
62	1	2	3	1	3	2	11	26	19	24	
	2	5	3	1	2	2	11	25	17	23	
	3	11	3	1	2	2	10	24	17	23	
	4	13	2	1	2	2	10	24	16	22	
	5	14	2	1	2	2	9	24	16	22	
	6	20	2	1	1	2	8	23	15	22	
	7	21	1	1	1	2	8	23	15	22	
	8	29	1	1	1	2	8	22	14	21	
	9	30	1	1	1	2	7	22	14	21	
63	1	1	5	4	2	5	8	16	17	7	
	2	2	4	4	2	4	7	16	16	6	
	3	6	4	4	2	4	7	14	16	6	
	4	10	4	3	2	4	5	13	16	6	
	5	11	4	3	2	3	5	12	15	5	
	6	12	4	2	2	3	3	11	15	5	
	7	19	4	1	2	2	2	11	15	4	
	8	20	4	1	2	2	1	9	15	3	
	9	27	4	1	2	2	1	9	15	2	
64	1	9	4	4	4	4	8	28	21	20	
	2	11	3	4	3	4	7	27	20	19	
	3	14	3	4	3	4	6	26	20	18	
	4	15	2	4	3	3	5	26	18	16	
	5	18	2	4	3	3	4	25	17	15	
	6	19	2	4	3	2	4	25	17	14	
	7	24	2	4	3	2	3	24	15	14	
	8	26	1	4	3	1	1	24	14	12	
	9	30	1	4	3	1	1	24	13	11	
65	1	5	4	2	4	4	7	26	23	17	
	2	8	3	2	4	4	7	26	23	17	
	3	9	3	2	4	4	6	26	21	17	
	4	12	3	2	4	4	6	26	20	16	
	5	14	3	2	3	4	4	26	20	16	
	6	16	3	2	3	3	4	26	18	16	
	7	17	3	2	3	3	4	26	17	16	
	8	18	3	2	3	3	3	26	15	15	
	9	20	3	2	3	3	2	26	15	15	
66	1	4	4	4	1	4	16	17	19	21	
	2	11	3	4	1	4	15	15	18	20	
	3	12	3	3	1	4	14	15	16	20	
	4	13	3	3	1	4	13	13	14	20	
	5	15	2	2	1	4	11	13	13	19	
	6	18	2	2	1	4	11	11	12	19	
	7	19	2	2	1	4	10	10	10	19	
	8	20	2	1	1	4	8	10	9	19	
	9	23	2	1	1	4	8	9	8	19	
67	1	3	4	3	4	4	24	26	9	20	
	2	5	4	3	4	4	24	25	9	20	
	3	7	4	3	4	4	24	24	9	19	
	4	8	4	3	4	4	23	23	9	18	
	5	9	4	3	4	4	23	21	9	18	
	6	11	4	3	4	4	23	21	9	17	
	7	13	4	3	4	4	22	20	9	17	
	8	17	4	3	4	4	22	18	9	16	
	9	18	4	3	4	4	22	17	9	16	
68	1	1	5	3	5	1	26	24	29	20	
	2	9	4	3	4	1	25	20	28	17	
	3	10	4	3	4	1	24	17	26	15	
	4	11	4	3	4	1	23	15	25	13	
	5	14	3	2	3	1	21	13	25	12	
	6	25	3	2	3	1	21	10	25	12	
	7	28	3	1	3	1	19	8	23	9	
	8	29	2	1	2	1	19	4	22	7	
	9	30	2	1	2	1	17	3	22	7	
69	1	3	3	4	4	4	17	29	28	3	
	2	9	3	4	4	4	16	29	25	3	
	3	11	3	4	4	4	16	27	23	3	
	4	12	3	4	4	4	15	27	22	3	
	5	18	3	3	4	3	15	25	22	3	
	6	23	3	3	4	3	15	25	20	2	
	7	25	3	3	4	3	15	23	17	2	
	8	27	3	3	4	3	14	22	17	2	
	9	28	3	3	4	3	14	21	16	2	
70	1	2	1	4	5	4	24	13	28	10	
	2	4	1	3	4	4	24	13	26	9	
	3	13	1	3	4	4	24	12	23	8	
	4	22	1	3	4	4	24	11	23	7	
	5	23	1	3	4	4	24	10	20	7	
	6	24	1	3	4	4	24	10	17	6	
	7	25	1	3	4	4	24	9	14	6	
	8	26	1	3	4	4	24	8	13	6	
	9	27	1	3	4	4	24	8	11	5	
71	1	4	4	3	4	4	10	24	26	18	
	2	8	3	3	4	4	9	23	25	16	
	3	10	3	3	4	4	8	23	24	14	
	4	12	3	3	4	4	8	23	22	12	
	5	13	3	3	3	3	8	23	20	10	
	6	18	3	3	3	3	7	23	20	9	
	7	19	3	3	3	3	6	23	18	7	
	8	20	3	3	3	2	6	23	16	5	
	9	26	3	3	3	2	6	23	15	5	
72	1	6	4	4	4	4	25	19	7	24	
	2	15	3	4	4	4	21	17	6	21	
	3	17	3	4	4	4	20	15	6	19	
	4	21	3	4	4	4	16	15	6	19	
	5	22	3	3	4	4	15	12	6	15	
	6	24	3	3	4	4	11	10	6	14	
	7	26	3	3	4	4	6	9	6	13	
	8	28	3	3	4	4	6	8	6	10	
	9	30	3	3	4	4	2	7	6	9	
73	1	8	3	4	4	4	7	19	2	11	
	2	12	3	4	4	4	7	18	1	11	
	3	13	3	3	3	3	7	18	1	11	
	4	14	3	3	3	3	7	18	1	10	
	5	19	3	2	2	2	7	18	1	11	
	6	20	3	2	2	2	7	17	1	11	
	7	22	3	1	2	2	7	17	1	11	
	8	26	3	1	1	1	7	17	1	11	
	9	27	3	1	1	1	7	17	1	10	
74	1	1	3	3	3	5	28	17	29	25	
	2	2	3	3	3	4	27	16	23	22	
	3	3	3	3	3	4	22	14	20	21	
	4	11	3	3	3	4	18	13	19	19	
	5	15	3	3	3	4	17	13	15	15	
	6	21	3	3	3	3	14	11	14	14	
	7	24	3	3	3	3	9	9	11	10	
	8	25	3	3	3	3	7	7	8	8	
	9	28	3	3	3	3	3	6	3	7	
75	1	5	5	3	3	5	9	5	15	25	
	2	6	5	2	2	4	9	4	14	25	
	3	7	5	2	2	4	9	4	12	25	
	4	15	5	2	2	3	9	4	11	25	
	5	19	5	2	2	3	9	3	11	25	
	6	21	5	2	2	2	9	3	9	24	
	7	22	5	2	2	1	9	3	8	24	
	8	23	5	2	2	1	9	2	6	24	
	9	24	5	2	2	1	9	2	5	24	
76	1	8	1	4	4	4	12	17	20	18	
	2	9	1	4	4	4	11	17	18	16	
	3	13	1	4	4	4	10	17	17	14	
	4	18	1	3	4	4	10	17	13	13	
	5	19	1	3	4	3	9	17	12	13	
	6	22	1	3	4	3	9	17	9	10	
	7	23	1	2	4	3	8	17	6	9	
	8	24	1	2	4	3	8	17	4	8	
	9	25	1	2	4	3	7	17	2	7	
77	1	2	2	3	4	5	20	24	21	26	
	2	3	2	3	4	4	18	20	20	22	
	3	5	2	3	4	4	17	20	19	21	
	4	8	2	3	3	4	16	16	18	19	
	5	13	1	3	3	4	16	14	17	18	
	6	15	1	3	3	4	14	11	16	18	
	7	17	1	3	2	4	14	6	16	16	
	8	22	1	3	2	4	13	4	15	14	
	9	29	1	3	2	4	12	3	14	13	
78	1	1	2	2	3	3	11	18	27	22	
	2	3	1	1	3	3	11	14	23	19	
	3	7	1	1	3	3	10	14	22	19	
	4	11	1	1	3	3	8	11	18	17	
	5	13	1	1	2	2	8	9	17	16	
	6	19	1	1	2	2	7	6	12	15	
	7	20	1	1	2	1	6	4	11	14	
	8	26	1	1	2	1	5	4	8	13	
	9	29	1	1	2	1	5	1	5	12	
79	1	4	5	5	2	4	21	29	15	18	
	2	5	4	4	2	4	21	28	15	16	
	3	10	4	4	2	4	21	27	13	15	
	4	13	4	4	2	4	20	27	13	14	
	5	14	3	4	2	4	19	27	12	12	
	6	21	3	4	2	4	19	26	11	12	
	7	23	3	4	2	4	18	25	9	11	
	8	29	3	4	2	4	18	25	8	10	
	9	30	3	4	2	4	18	25	8	8	
80	1	2	3	3	3	4	16	13	16	19	
	2	4	2	3	3	4	16	13	16	17	
	3	8	2	3	3	4	15	13	15	15	
	4	9	2	3	3	3	14	13	15	13	
	5	10	2	3	2	3	14	13	14	12	
	6	11	1	3	2	2	13	13	14	10	
	7	23	1	3	2	2	13	13	13	6	
	8	24	1	3	2	1	12	13	13	4	
	9	25	1	3	2	1	12	13	13	3	
81	1	4	4	3	3	5	16	16	17	22	
	2	5	4	3	2	4	15	16	16	21	
	3	7	4	3	2	4	13	15	16	21	
	4	14	4	3	2	4	11	14	15	20	
	5	15	4	3	2	3	9	11	15	20	
	6	16	4	3	1	3	6	11	15	19	
	7	17	4	3	1	3	4	9	15	19	
	8	22	4	3	1	3	3	8	14	18	
	9	28	4	3	1	3	2	8	14	18	
82	1	17	5	5	3	3	11	19	27	11	
	2	18	4	4	2	3	11	18	27	10	
	3	19	4	4	2	3	11	15	23	10	
	4	20	4	4	2	3	11	12	22	10	
	5	21	4	4	1	3	11	11	19	9	
	6	22	3	4	1	3	11	9	18	8	
	7	24	3	4	1	3	11	5	16	8	
	8	26	3	4	1	3	11	4	13	7	
	9	30	3	4	1	3	11	3	13	7	
83	1	1	4	2	3	3	2	18	29	14	
	2	4	3	1	3	3	2	16	24	12	
	3	5	3	1	3	3	2	13	22	12	
	4	11	3	1	3	3	2	13	22	11	
	5	17	3	1	2	3	2	10	20	8	
	6	22	3	1	2	3	2	8	17	7	
	7	23	3	1	2	3	2	8	14	7	
	8	25	3	1	1	3	2	5	13	6	
	9	28	3	1	1	3	2	4	11	5	
84	1	3	3	5	4	5	27	17	18	13	
	2	9	3	4	4	4	26	16	18	11	
	3	20	3	4	3	4	26	15	16	9	
	4	23	3	4	3	4	26	14	16	8	
	5	25	2	4	3	3	26	13	15	7	
	6	26	2	4	2	3	26	13	13	6	
	7	28	1	4	1	2	26	12	13	3	
	8	29	1	4	1	2	26	11	11	2	
	9	30	1	4	1	2	26	11	10	1	
85	1	1	3	5	4	1	19	30	9	16	
	2	5	2	5	4	1	19	28	8	15	
	3	7	2	5	4	1	18	28	8	15	
	4	11	2	5	3	1	16	27	7	15	
	5	12	2	5	2	1	15	27	7	15	
	6	13	2	5	2	1	15	27	6	15	
	7	14	2	5	1	1	13	26	5	15	
	8	28	2	5	1	1	13	26	5	14	
	9	29	2	5	1	1	12	25	5	15	
86	1	2	3	5	3	1	22	23	23	21	
	2	4	3	4	2	1	22	21	21	20	
	3	9	3	4	2	1	22	20	21	19	
	4	14	3	3	2	1	22	18	21	18	
	5	20	3	3	1	1	21	16	20	17	
	6	22	3	3	1	1	21	12	20	15	
	7	23	3	3	1	1	21	11	19	14	
	8	24	3	2	1	1	20	9	18	14	
	9	25	3	2	1	1	20	7	18	13	
87	1	2	3	3	4	3	11	27	28	13	
	2	5	2	3	3	3	10	26	27	13	
	3	7	2	3	3	3	10	25	26	13	
	4	8	2	3	3	3	10	25	25	13	
	5	13	2	3	3	3	9	25	23	13	
	6	14	2	3	2	2	9	24	22	12	
	7	16	2	3	2	2	8	24	22	12	
	8	23	2	3	2	2	8	23	20	12	
	9	24	2	3	2	2	8	23	20	11	
88	1	2	4	1	4	4	29	23	29	29	
	2	3	4	1	3	4	28	20	25	27	
	3	8	4	1	3	4	28	20	23	27	
	4	9	4	1	3	4	27	16	18	26	
	5	15	3	1	2	4	27	12	14	26	
	6	17	3	1	2	3	26	10	12	25	
	7	26	3	1	1	3	25	9	11	25	
	8	28	3	1	1	3	25	6	5	23	
	9	30	3	1	1	3	25	4	3	23	
89	1	1	4	5	4	2	22	19	13	17	
	2	6	4	4	3	2	19	16	13	15	
	3	9	4	4	3	2	18	16	12	12	
	4	10	4	4	3	2	18	14	11	10	
	5	12	4	4	3	2	16	12	9	9	
	6	14	3	3	3	2	16	12	9	8	
	7	20	3	3	3	2	15	10	7	6	
	8	21	3	3	3	2	13	10	6	2	
	9	26	3	3	3	2	12	8	6	2	
90	1	5	3	1	2	3	29	27	21	26	
	2	6	3	1	2	2	27	25	19	24	
	3	11	3	1	2	2	22	24	16	21	
	4	12	3	1	2	2	19	21	13	20	
	5	13	3	1	2	1	17	21	11	17	
	6	20	3	1	2	1	16	19	10	17	
	7	25	3	1	2	1	12	17	7	14	
	8	26	3	1	2	1	12	15	4	13	
	9	29	3	1	2	1	8	14	3	11	
91	1	1	4	4	2	2	25	20	21	21	
	2	3	3	3	2	1	25	18	19	19	
	3	7	3	3	2	1	24	18	19	18	
	4	8	2	2	2	1	22	14	18	17	
	5	11	2	2	2	1	19	13	18	14	
	6	22	2	2	2	1	18	13	17	14	
	7	23	1	1	2	1	18	9	17	13	
	8	24	1	1	2	1	16	9	16	11	
	9	28	1	1	2	1	14	7	15	9	
92	1	1	4	3	4	4	25	18	26	26	
	2	4	3	3	4	3	23	16	25	24	
	3	8	3	3	4	3	23	15	23	22	
	4	14	3	3	4	3	23	15	23	21	
	5	15	2	3	3	3	22	14	20	20	
	6	16	2	2	3	3	21	12	19	18	
	7	23	2	2	3	3	21	12	18	16	
	8	26	1	2	3	3	21	10	16	16	
	9	30	1	2	3	3	20	10	16	14	
93	1	5	4	3	4	5	12	27	12	11	
	2	6	3	2	3	5	12	26	11	8	
	3	8	3	2	3	5	12	25	11	7	
	4	11	3	2	3	5	12	23	11	6	
	5	12	2	2	2	5	12	23	11	6	
	6	20	2	1	2	5	12	22	11	4	
	7	21	2	1	2	5	12	21	11	3	
	8	26	2	1	2	5	12	21	11	2	
	9	28	2	1	2	5	12	20	11	1	
94	1	2	2	5	2	1	16	17	26	12	
	2	3	2	4	2	1	16	17	24	12	
	3	4	2	4	2	1	14	17	22	11	
	4	5	2	3	2	1	13	17	20	11	
	5	21	2	3	1	1	13	16	20	10	
	6	24	2	3	1	1	11	16	18	9	
	7	25	2	2	1	1	11	16	16	9	
	8	26	2	2	1	1	9	15	14	8	
	9	29	2	2	1	1	9	15	13	8	
95	1	2	5	5	2	4	19	10	28	23	
	2	3	4	4	2	4	16	9	26	23	
	3	5	3	4	2	4	16	9	24	23	
	4	7	3	4	2	4	14	9	20	23	
	5	8	2	4	2	4	13	9	17	23	
	6	11	2	4	2	4	12	8	16	23	
	7	14	2	4	2	4	12	8	14	23	
	8	24	1	4	2	4	10	8	8	23	
	9	25	1	4	2	4	10	8	7	23	
96	1	1	4	4	4	3	15	6	9	8	
	2	6	3	4	4	3	13	4	7	7	
	3	8	3	4	4	3	12	4	6	7	
	4	14	2	4	4	3	12	3	5	7	
	5	17	2	4	4	3	10	3	4	7	
	6	18	2	3	4	3	9	2	4	7	
	7	22	1	3	4	3	9	2	3	7	
	8	23	1	3	4	3	7	1	2	7	
	9	27	1	3	4	3	7	1	1	7	
97	1	2	4	2	1	3	27	26	17	9	
	2	3	3	2	1	3	24	23	16	8	
	3	7	3	2	1	3	23	23	14	7	
	4	16	3	2	1	2	21	19	13	7	
	5	19	3	1	1	2	18	17	12	6	
	6	21	2	1	1	2	16	13	11	6	
	7	25	2	1	1	2	13	9	11	5	
	8	27	2	1	1	1	11	7	9	5	
	9	28	2	1	1	1	10	4	8	5	
98	1	7	3	5	4	4	26	26	21	21	
	2	8	3	4	3	4	26	24	19	18	
	3	9	3	4	3	4	26	21	18	18	
	4	15	3	3	3	4	26	17	16	16	
	5	17	3	3	3	3	25	14	15	15	
	6	21	3	3	3	3	25	11	14	15	
	7	24	3	2	3	2	24	7	14	13	
	8	25	3	2	3	2	24	3	13	12	
	9	27	3	2	3	2	24	1	11	12	
99	1	1	5	5	1	5	20	26	12	19	
	2	10	4	4	1	4	19	25	11	18	
	3	12	3	4	1	4	19	25	9	16	
	4	13	3	4	1	3	19	23	8	14	
	5	14	3	3	1	3	18	22	8	13	
	6	20	2	3	1	3	18	22	7	12	
	7	23	2	3	1	3	18	20	6	10	
	8	29	1	3	1	2	18	19	6	9	
	9	30	1	3	1	2	18	19	5	7	
100	1	1	3	2	4	4	10	4	27	11	
	2	4	3	2	4	3	8	4	24	9	
	3	6	3	2	4	3	7	4	21	8	
	4	9	2	2	3	2	6	4	20	8	
	5	14	2	2	3	2	5	4	18	5	
	6	15	2	1	3	2	4	4	12	4	
	7	18	2	1	2	1	4	4	10	4	
	8	27	1	1	2	1	2	4	7	2	
	9	29	1	1	2	1	2	4	5	2	
101	1	1	3	2	1	2	25	19	26	21	
	2	6	3	2	1	2	24	18	23	20	
	3	8	3	2	1	2	24	18	23	19	
	4	9	3	2	1	2	22	17	20	19	
	5	13	3	2	1	2	22	16	18	19	
	6	14	3	2	1	2	21	16	14	19	
	7	15	3	2	1	2	21	15	11	18	
	8	25	3	2	1	2	19	14	9	18	
	9	28	3	2	1	2	19	13	6	17	
102	1	0	0	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	R 3	R 4	N 1	N 2	N 3	N 4
	22	21	22	23	1551	1616	1454	1514

************************************************************************
