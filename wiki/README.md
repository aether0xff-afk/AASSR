# AASSR Wiki Source

이 디렉터리는 AASSR GitHub Wiki의 **version-controlled canonical Markdown source**다.

> 이 `README.md` 자체는 source-directory 설명서이므로 실제 GitHub Wiki에는 publish하지 않는다. `.github/workflows/sync-github-wiki.yml`이 `wiki/README.md`를 제외한 나머지 `wiki/*.md`를 `${repository}.wiki.git`으로 동기화한다.

## Source of truth hierarchy

```text
Executable architecture contract
→ src/aassr_v2/current_manifest.py

Research wiki source
→ wiki/*.md

Published GitHub Wiki
→ AASSR.wiki.git
```

현재 문서는 **`main`의 current-generation runtime**을 기준으로 작성한다.

특정 연구 브랜치에서 실험 중인 구현은 `main`에 merge되기 전까지 current Wiki architecture로 승격하지 않는다.

과거 v0.4/초기 Prophecy/Imagination/effect-composition 계열은 [Development History](Development-History.md) 또는 별도 historical diagnostic 페이지에 분리한다.

---

## Main entry pages

1. [Home](Home.md)
2. [AASSR in 5 Minutes](AASSR-in-5-Minutes.md)
3. [Concept Index](Concept-Index.md)
4. [Research Questions](Research-Questions.md)
5. [Evidence Matrix](Evidence-Matrix.md)
6. [Research Architecture](Research-Architecture.md)
7. [Experiments](Experiments.md)
8. [Current Status](Current-Status.md)
9. [Reproduction](Reproduction.md)
10. [Glossary](Glossary.md)

Historical evidence:

- [Historical Imagination Diagnostic — 2026-08-11](Historical-Imagination-Diagnostic-2026-08-11.md)
- [Development History](Development-History.md)

---

## Documentation rules

### 1. Current architecture와 historical evidence를 섞지 않는다

```text
Current
= main의 manifest/code contract

Historical
= 과거 commit/checkpoint에서 얻은 mechanism/failure evidence
```

### 2. 구현과 성능을 구분한다

```text
implemented
!= regression-validated
!= mechanism evidence
!= reduced performance evidence
!= multi-seed performance evidence
!= final blind evidence
```

### 3. 중요한 전문용어는 내부 링크를 건다

짧은 정의는 [Glossary](Glossary.md), 긴 일반 개념은 [Concept Index](Concept-Index.md), AASSR-specific 구현은 해당 Core Mechanism 페이지로 연결한다.

### 4. 숫자에는 evidence class를 붙인다

예:

```text
24/24 → 0/24 stalled
= ASEQ mechanism diagnostic

4/20 vs 4/20, 86 interventions
= 2026-08-11 historical root-cause diagnostic
```

### 5. Claim은 Evidence Matrix와 맞아야 한다

각 연구 질문의 H1/H0, metrics, 현재 claim boundary는 [Evidence Matrix](Evidence-Matrix.md)에서 관리한다.

---

## Status legend

- 🟢 **Active** — current runtime contract
- 🟡 **Experimental** — current 구현이지만 performance claim 검증 중
- 🔵 **Evidence** — 특정 controlled diagnostic에서 mechanism evidence 존재
- ⚪ **Pending** — current performance/final evidence 전
- 🕰️ **Historical** — 과거 architecture/checkpoint evidence

---

## Publishing

`main`의 `wiki/**`가 바뀌면 `Sync GitHub Wiki` GitHub Actions workflow가 자동으로 published Wiki를 갱신한다.

특수 페이지:

- `_Sidebar.md` — Wiki navigation sidebar
- `_Footer.md` — 공통 footer navigation
- `Home.md` — Wiki landing page

`README.md`는 publish 대상에서 제외한다.
