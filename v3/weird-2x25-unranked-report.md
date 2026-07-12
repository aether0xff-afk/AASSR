# APASSR Trace Report

## Summary

- episodes: 2
- successes: 0
- solved events: 0
- distinct action signatures: 47
- distinct action chains: 50
- distinct response transitions: 22

## No Solved Event Yet

No `solved_delta > 0` event was found in this run. The sections below summarize the most unusual traces observed so far.

## High-Novelty Steps

| ep | step | template | status | new_kv | solved | novelty | reward | action |
|---:|---:|---|---:|---:|---:|---:|---:|---|
| 0 | 0 | HTTP_GET_PATH | 200 | 27 | 0 | 0.600 | 2.300 | GET / |
| 0 | 5 | HTTP_GET_PATH | 200 | 2 | 0 | 0.600 | 2.300 | GET /robots.txt |
| 0 | 10 | HTTP_GET_API | 200 | 24 | 0 | 0.600 | 2.300 | GET /api/Feedbacks |
| 0 | 18 | HTTP_QUERY_PROBE | 200 | 1 | 0 | 0.600 | 2.300 | PROBE /api/Recycles?above=1 |
| 0 | 19 | HTTP_POST_PROBE | 201 | 14 | 0 | 0.600 | 2.300 | POST_PROBE /api/Users above=1 |
| 1 | 13 | HTTP_GET_API | 200 | 1 | 0 | 0.600 | 2.300 | GET /rest/continue-code |
| 1 | 22 | HTTP_GET_PATH | 200 | 3 | 0 | 0.600 | 2.300 | GET chunk-VS3A3LTT.js |
| 0 | 3 | NMAP_SCAN_HOST | 0 | 2 | 0 | 0.600 | 2.200 | NMAP 127.0.0.1:3000 |
| 0 | 1 | HTTP_HEAD_PATH | 200 | 0 | 0 | 0.600 | 2.100 | HEAD / |
| 0 | 2 | HTTP_OPTIONS_PATH | 204 | 0 | 0 | 0.600 | 2.100 | OPTIONS / |
| 0 | 7 | HTTP_GET_PATH | 200 | 0 | 0 | 0.600 | 2.100 | GET assets/public/favicon_js.ico |
| 0 | 12 | HTTP_QUERY_PROBE | 200 | 0 | 0 | 0.600 | 2.100 | PROBE /api/Feedbacks?UserId=1 |
| 1 | 15 | HTTP_GET_API | 200 | 0 | 0 | 0.600 | 2.100 | GET /rest/continue-code-findIt |
| 1 | 19 | HTTP_QUERY_PROBE | 500 | 1 | 0 | 0.600 | 2.000 | PROBE /rest/continue-code/apply/?ChallengeDependencies=1 |
| 0 | 11 | HTTP_POST_COMBO | 500 | 0 | 0 | 0.600 | 1.800 | POST_COMBO /api/Feedbacks fields=palette,popup,background,text,button,theme,position,co... |
| 0 | 14 | HTTP_POST_PROBE | 500 | 0 | 0 | 0.600 | 1.800 | POST_PROBE /api/Feedbacks UserId=1 |
| 0 | 17 | HTTP_GET_API | 401 | 0 | 0 | 0.600 | 1.800 | GET /api/Addresss?current= |
| 0 | 20 | HTTP_QUERY_PROBE | 401 | 0 | 0 | 0.600 | 1.800 | PROBE /api/Users/27?above=1 |
| 0 | 21 | HTTP_POST_COMBO | 401 | 0 | 0 | 0.600 | 1.800 | POST_COMBO /rest/deluxe-membership fields=palette,popup,background,text,button,theme,po... |
| 0 | 24 | HTTP_POST_PROBE | 401 | 0 | 0 | 0.600 | 1.800 | POST_PROBE /rest/user/authentication-details/ above=1 |

## Action Mix

- HTTP_GET_API: 10
- HTTP_GET_PATH: 9
- HTTP_HEAD_PATH: 7
- HTTP_OPTIONS_PATH: 6
- HTTP_QUERY_PROBE: 6
- HTTP_POST_PROBE: 5
- HTTP_POST_COMBO: 3
- NMAP_SCAN_HOST: 2
- WEB_FINGERPRINT: 2
