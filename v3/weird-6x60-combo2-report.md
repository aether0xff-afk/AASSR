# APASSR Trace Report

## Summary

- episodes: 6
- successes: 0
- solved events: 0
- distinct action signatures: 235
- distinct action chains: 360
- distinct response transitions: 33

## No Solved Event Yet

No `solved_delta > 0` event was found in this run. The sections below summarize the most unusual traces observed so far.

## High-Novelty Steps

| ep | step | template | status | new_kv | solved | novelty | reward | action |
|---:|---:|---|---:|---:|---:|---:|---:|---|
| 0 | 0 | HTTP_GET_PATH | 200 | 27 | 0 | 0.600 | 2.300 | GET / |
| 0 | 5 | HTTP_GET_PATH | 200 | 2 | 0 | 0.600 | 2.300 | GET /robots.txt |
| 0 | 10 | HTTP_GET_API | 200 | 24 | 0 | 0.600 | 2.300 | GET /api/Feedbacks |
| 0 | 18 | HTTP_QUERY_PROBE | 200 | 1 | 0 | 0.600 | 2.300 | PROBE /api/Recycles?background=1 |
| 0 | 19 | HTTP_POST_PROBE | 201 | 14 | 0 | 0.600 | 2.300 | POST_PROBE /api/Users background=1 |
| 0 | 37 | HTTP_GET_PATH | 200 | 5 | 0 | 0.600 | 2.300 | GET chunk-VS3A3LTT.js |
| 0 | 38 | HTTP_QUERY_PROBE | 200 | 455 | 0 | 0.600 | 2.300 | PROBE /api/Challenges/?background=1 |
| 1 | 33 | HTTP_QUERY_PROBE | 200 | 3 | 0 | 0.600 | 2.300 | PROBE /api/Quantitys?background=0 |
| 1 | 34 | HTTP_POST_PROBE | 201 | 4 | 0 | 0.600 | 2.300 | POST_PROBE /api/SecurityAnswers background=0 |
| 4 | 54 | HTTP_GET_API | 200 | 1 | 0 | 0.600 | 2.300 | GET /rest/user/whoami?current= |
| 5 | 48 | HTTP_POST_COMBO | 201 | 3 | 0 | 0.600 | 2.300 | POST_COMBO /api/SecurityAnswers fields=palette,popup,background,text,button |
| 0 | 3 | NMAP_SCAN_HOST | 0 | 2 | 0 | 0.600 | 2.200 | NMAP 127.0.0.1:3000 |
| 0 | 1 | HTTP_HEAD_PATH | 200 | 0 | 0 | 0.600 | 2.100 | HEAD / |
| 0 | 2 | HTTP_OPTIONS_PATH | 204 | 0 | 0 | 0.600 | 2.100 | OPTIONS / |
| 0 | 7 | HTTP_GET_PATH | 200 | 0 | 0 | 0.600 | 2.100 | GET assets/public/favicon_js.ico |
| 0 | 12 | HTTP_QUERY_PROBE | 200 | 0 | 0 | 0.600 | 2.100 | PROBE /api/Feedbacks?UserId=1 |
| 1 | 36 | HTTP_GET_API | 200 | 0 | 0 | 0.600 | 2.100 | GET /api/Hints?background=0 |
| 1 | 44 | HTTP_POST_PROBE | 200 | 0 | 0 | 0.600 | 2.100 | POST_PROBE /rest/chat background=0 |
| 0 | 39 | HTTP_QUERY_PROBE | 500 | 1 | 0 | 0.600 | 2.000 | PROBE /rest/continue-code-findIt/apply/?background=1 |
| 1 | 27 | HTTP_GET_API | 401 | 1 | 0 | 0.600 | 2.000 | GET /api/Cards?background=0 |

## Action Mix

- HTTP_GET_PATH: 90
- HTTP_OPTIONS_PATH: 66
- HTTP_HEAD_PATH: 64
- HTTP_QUERY_PROBE: 49
- HTTP_GET_API: 37
- HTTP_POST_PROBE: 24
- HTTP_POST_COMBO: 15
- NMAP_SCAN_HOST: 9
- WEB_FINGERPRINT: 6
