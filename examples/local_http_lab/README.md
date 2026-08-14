# Local HTTP 실제 I/O 실험

이 예시는 기존 in-process pentest simulator 대신 **실제 localhost 소켓/HTTP 요청**으로 새 Core-Plugin 경계를 확인하기 위한 작은 smoke target이다.

중요: 이 서비스 자체는 단순한 통제 환경이다. 연구의 최종 벤치마크라고 주장하지 않는다. `LocalHttpPlugin`은 같은 loopback origin의 다른 로컬 웹 서비스에도 연결할 수 있다.

## 1. 로컬 서비스 실행

PowerShell/터미널 1:

```bash
python examples/local_http_lab/server.py --port 8765
```

## 2. AASSR 실행

터미널 2:

```bash
python scripts/run_local_http_core.py \
  --base-url http://127.0.0.1:8765 \
  --episodes 64 \
  --max-steps 32 \
  --device cpu \
  --output runs/local_http_core.json
```

CUDA를 사용할 수 있으면 `--device cuda:0`으로 바꿀 수 있다.

## 환경의 외부 보상

서비스는 실제 HTTP 응답 헤더로 외부 신호를 전달한다.

- 성공 응답: `X-AASSR-Reward: 1`, `X-AASSR-Terminated: 1`
- terminal dead end: `X-AASSR-Reward: -1`, `X-AASSR-Terminated: 1`
- 그 외: `0`

Plugin은 이 신호를 그대로 전달할 뿐 추가 보상을 만들지 않는다.

## 안전 경계

`LocalHttpPlugin`은 `localhost`, `127.0.0.1`, `::1`만 허용하고 configured origin 밖 요청을 거부한다. 이 예시는 외부 시스템을 대상으로 하지 않는다.
