# APASSR Trace Report

## Summary

- episodes: 6
- successes: 0
- solved events: 0
- distinct action signatures: 214
- distinct action chains: 360
- distinct response transitions: 30

## No Solved Event Yet

No `solved_delta > 0` event was found in this run. The sections below summarize the most unusual traces observed so far.

## High-Novelty Steps

| ep | step | template | status | new_kv | solved | novelty | reward | action |
|---:|---:|---|---:|---:|---:|---:|---:|---|
| 0 | 0 | HTTP_GET_PATH | 200 | 27 | 0 | 0.600 | 2.300 | GET / |
| 0 | 5 | HTTP_GET_PATH | 200 | 2 | 0 | 0.600 | 2.300 | GET /robots.txt |
| 0 | 10 | HTTP_GET_API | 200 | 24 | 0 | 0.600 | 2.300 | GET /api/Feedbacks |
| 0 | 17 | HTTP_QUERY_PROBE | 200 | 1 | 0 | 0.600 | 2.300 | PROBE /api/Recycles?background=1 |
| 0 | 18 | HTTP_POST_PROBE | 201 | 13 | 0 | 0.600 | 2.300 | POST_PROBE /api/Users background=1 |
| 0 | 35 | HTTP_GET_PATH | 200 | 5 | 0 | 0.600 | 2.300 | GET chunk-VS3A3LTT.js |
| 0 | 37 | HTTP_QUERY_PROBE | 200 | 454 | 0 | 0.600 | 2.300 | PROBE /api/Challenges/?content=1 |
| 1 | 31 | HTTP_POST_PROBE | 201 | 4 | 0 | 0.600 | 2.300 | POST_PROBE /api/SecurityAnswers content=0 |
| 1 | 33 | HTTP_GET_API | 200 | 1 | 0 | 0.600 | 2.300 | GET /api/SecurityQuestions?content=0 |
| 5 | 47 | HTTP_QUERY_PROBE | 200 | 3 | 0 | 0.600 | 2.300 | PROBE /api/Quantitys?theme=1 |
| 5 | 57 | HTTP_POST_PROBE | 201 | 1 | 0 | 0.600 | 2.300 | POST_PROBE /api/Users theme=1 |
| 0 | 3 | NMAP_SCAN_HOST | 0 | 2 | 0 | 0.600 | 2.200 | NMAP 127.0.0.1:3000 |
| 0 | 1 | HTTP_HEAD_PATH | 200 | 0 | 0 | 0.600 | 2.100 | HEAD / |
| 0 | 2 | HTTP_OPTIONS_PATH | 204 | 0 | 0 | 0.600 | 2.100 | OPTIONS / |
| 0 | 7 | HTTP_GET_PATH | 200 | 0 | 0 | 0.600 | 2.100 | GET assets/public/favicon_js.ico |
| 0 | 12 | HTTP_QUERY_PROBE | 200 | 0 | 0 | 0.600 | 2.100 | PROBE /api/Feedbacks?comment=1 |
| 0 | 46 | HTTP_GET_API | 200 | 0 | 0 | 0.600 | 2.100 | GET /api/Challenges/?background=1 |
| 1 | 41 | HTTP_POST_PROBE | 200 | 0 | 0 | 0.600 | 2.100 | POST_PROBE /rest/chat content=0 |
| 0 | 30 | HTTP_QUERY_PROBE | 500 | 1 | 0 | 0.600 | 2.000 | PROBE /rest/user/reset-password?background=1 |
| 3 | 31 | HTTP_POST_PROBE | 400 | 1 | 0 | 0.600 | 2.000 | POST_PROBE /rest/memories position=1 |

## Action Mix

- HTTP_GET_PATH: 99
- HTTP_OPTIONS_PATH: 74
- HTTP_HEAD_PATH: 70
- HTTP_QUERY_PROBE: 45
- HTTP_POST_PROBE: 32
- HTTP_GET_API: 25
- NMAP_SCAN_HOST: 9
- WEB_FINGERPRINT: 6
