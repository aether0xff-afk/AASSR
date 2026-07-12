# APASSR Trace Report

## Summary

- episodes: 10
- successes: 0
- solved events: 0
- distinct action signatures: 321
- distinct action chains: 797
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
| 0 | 64 | HTTP_GET_API | 200 | 6 | 0 | 0.600 | 2.300 | GET /api/Deliverys?background=1 |
| 0 | 74 | HTTP_GET_API | 200 | 1 | 0 | 0.600 | 2.300 | GET /api/SecurityQuestions?background=1 |
| 1 | 36 | HTTP_POST_COMBO | 201 | 10 | 0 | 0.600 | 2.300 | POST_COMBO /api/Users fields=palette,popup,background,text,button |
| 2 | 64 | HTTP_POST_PROBE | 201 | 1 | 0 | 0.600 | 2.300 | POST_PROBE /api/Users palette=0 |
| 3 | 23 | HTTP_QUERY_PROBE | 200 | 3 | 0 | 0.600 | 2.300 | PROBE /rest/captcha?popup=1 |
| 0 | 3 | NMAP_SCAN_HOST | 0 | 2 | 0 | 0.600 | 2.200 | NMAP 127.0.0.1:3000 |
| 0 | 1 | HTTP_HEAD_PATH | 200 | 0 | 0 | 0.600 | 2.100 | HEAD / |
| 0 | 2 | HTTP_OPTIONS_PATH | 204 | 0 | 0 | 0.600 | 2.100 | OPTIONS / |
| 0 | 7 | HTTP_GET_PATH | 200 | 0 | 0 | 0.600 | 2.100 | GET assets/public/favicon_js.ico |
| 0 | 12 | HTTP_QUERY_PROBE | 200 | 0 | 0 | 0.600 | 2.100 | PROBE /api/Feedbacks?UserId=1 |
| 3 | 40 | HTTP_POST_PROBE | 200 | 0 | 0 | 0.600 | 2.100 | POST_PROBE /rest/chat palette=1 |
| 4 | 41 | HTTP_POST_COMBO | 200 | 0 | 0 | 0.600 | 2.100 | POST_COMBO /rest/chat fields=palette,popup,background,text,button |
| 0 | 39 | HTTP_QUERY_PROBE | 500 | 1 | 0 | 0.600 | 2.000 | PROBE /rest/continue-code-findIt/apply/?background=1 |

## Action Mix

- HTTP_GET_PATH: 241
- HTTP_OPTIONS_PATH: 142
- HTTP_HEAD_PATH: 138
- HTTP_GET_API: 103
- HTTP_QUERY_PROBE: 90
- HTTP_POST_PROBE: 38
- HTTP_POST_COMBO: 30
- NMAP_SCAN_HOST: 10
- WEB_FINGERPRINT: 8
