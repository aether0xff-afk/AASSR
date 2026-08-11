# AASSR Wiki Source

이 디렉터리는 AASSR GitHub Wiki에 사용할 Markdown source다.

현재 문서는 **`agent/imagination-gate-ablation`의 current-generation runtime**을 기준으로 작성되어 있으며, 과거 v0.4/초기 v2 구현은 current architecture와 분리해서 다룬다.

## Pages

1. [Home](Home.md)
2. [AASSR in 5 Minutes](AASSR-in-5-Minutes.md)
3. [Core Architecture](Core-Architecture.md)
4. [ASEQ](ASEQ.md)
5. [Experiments](Experiments.md)
6. [Development History](Development-History.md)
7. [Current Status](Current-Status.md)
8. [Reproduction](Reproduction.md)
9. [Glossary](Glossary.md)

GitHub Wiki로 publish할 때는 같은 파일 이름을 사용하면 된다. `_Sidebar.md`는 Wiki의 navigation sidebar로 바로 사용할 수 있다.

## Documentation rule

위키에서 다음 세 상태를 구분한다.

- 🟢 **Active** — current runtime에서 실제 사용 중
- 🟡 **Experimental** — 구현되어 있으나 성능/최종 설계 검증 중
- ⚪ **Historical / Pending** — 과거 재현용 또는 아직 실행 전

구현 여부와 연구적 성능 주장을 같은 것으로 취급하지 않는다.

## Executable source of truth

현재 component contract는 다음 파일을 우선한다.

```text
src/aassr_v2/current_manifest.py
```

상세 current-generation 설명:

```text
docs/aassr_current_generation.md
```
