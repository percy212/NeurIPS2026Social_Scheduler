# Video Generation API

이미지 여러 장 + 음성 + 안무(choreography)를 받아 동영상(mp4)을 생성해 돌려주는 API 서버.
한 서버에서 여러 GPU 를 사용하며, GPU 당 worker 프로세스 1개에 모델을 상주시키고 요청을 분배한다.

> 현재 `generation.py` 는 **DUMMY 구현**이다. 실제 모델로 교체하는 방법은
> [`GENERATION_SPEC.md`](./GENERATION_SPEC.md) 를 참고.

---

## 1. 파일 구성

| 파일 | 역할 |
|---|---|
| `main.py` | FastAPI 서버. 업로드 수신, 입력 검증, 스케줄러 호출, 결과 mp4 반환 (`/generate`, `/health`, `/queue`) |
| `scheduler.py` | `GPUScheduler`. GPU 별 worker 프로세스 관리, 작업 분배, 재시도, NVML 기반 health check |
| `generation.py` | 모델 계층. 에러 코드 / 요청·결과 타입 / 모델 옵션 선언 / `VideoGenerationModel` (현재 더미) |
| `GENERATION_SPEC.md` | `generation.py` 를 새로 작성할 때 지켜야 할 규약 문서 |

의존 방향: `main.py → scheduler.py → generation.py`, 그리고 `main.py → generation.py` (타입·옵션 검증용).

## 2. 아키텍처

```
FastAPI(asyncio) ──generate()──> GPUScheduler ──task_q[gpu]──> worker process (모델 상주)
                                      ^   |                          |
                                      |   └─ NVML health check(5분)   |
                                      └────── result_q (공유) <───────┘
```

- **API 프로세스**는 GPU 를 쓰지 않는다. `generation.py` 를 import 하지만 모델은 로드하지 않는다.
- **worker 프로세스**는 GPU 1장당 1개이며 `spawn` 방식으로 기동된다. 시작 시 `CUDA_VISIBLE_DEVICES=<gpu_id>` 를 설정한 뒤 모델을 한 번 로드하고, 이후 작업을 하나씩 처리한다. 따라서 worker 안에서는 항상 `cuda:0` 이다.
- 업로드 파일은 요청별 임시 디렉터리에 저장되고 worker 에는 **파일 경로만** 전달된다 (프로세스 간 pickle).
- 결과 mp4 는 응답이 전송된 뒤 임시 디렉터리째 삭제된다.

요청 처리 순서 (`POST /generate`):

1. 정의되지 않은 인자 수집 + 모델 옵션 값 검증 (`resolve_options`)
2. 가벼운 입력 검증 (이미지 수, 안무, fps) — GPU 를 쓰기 전에 걸러냄
3. 업로드 파일을 디스크에 저장 (크기 제한 확인)
4. `scheduler.generate()` — GPU 대기 → 배정 → 실행 → 실패 시 정책에 따라 재시도
5. 성공하면 mp4 응답, 실패하면 JSON 에러 응답

---

## 3. 빠른 시작

```bash
pip install fastapi uvicorn python-multipart nvidia-ml-py torch

# 반드시 --workers 1 : GPU 스케줄러가 이 프로세스 안에서 worker 들을 관리한다
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

서버 시작 시 모든 GPU worker 가 모델 로딩을 마칠 때까지(최대 600초) 기다린 뒤 요청을 받기 시작한다.

```bash
curl -i -X POST http://localhost:8000/generate \
  -F "images=@a.jpg" -F "images=@b.jpg" -F "audio=@voice.wav" \
  -F "choreography=dance_a" -F "fps=25" \
  -F "guidance_scale=5.0" -F "foo=bar" \
  -o result.mp4
```

참고
- 현재 더미 모델은 `torch` 로 CUDA 텐서를 직접 할당하므로 **GPU 가 없는 환경에서는 `load()` 가 실패**한다. `DISABLE_NVML=1` 은 NVML 점검만 끄며 CUDA 를 대신해 주지 않는다.
- `NUM_GPUS` 는 실제 GPU 수 이하여야 한다. GPU index 는 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 기준이라 NVML 의 index 와 일치한다.

---

## 4. API

### `POST /generate`

`multipart/form-data`

| 필드 | 타입 | 필수 | 기본값 | 제약 |
|---|---|---|---|---|
| `images` | file[] | O | | 1~10장, 장당 20MB 이하 |
| `audio` | file | O | | 50MB 이하 |
| `choreography` | str | O | | `dance_a`, `dance_b`, `dance_c`, `kpop_01`, `kpop_02` (`generation.SUPPORTED_CHOREOGRAPHIES`) |
| `fps` | int | | 25 | 1~60 |
| `resolution` | int | | 512 | **main.py 에서는 검증하지 않음** → 모델 쪽에서 검증해야 함 |
| `seed` | int | | 없음 | |
| *모델 옵션* | str | | 모델 기본값 | `generation.MODEL_OPTIONS` 에 선언된 것 (아래) |

현재(더미) 모델 옵션

| 옵션 | 타입 | 기본값 | 범위 |
|---|---|---|---|
| `guidance_scale` | float | 3.0 | 0.0 ~ 20.0 |
| `num_inference_steps` | int | 30 | 1 ~ 100 |
| `background` | str | `original` | `original` / `blur` / `studio` |

**성공 (200)** — `video/mp4` 본문 (`result.mp4`) + 선택적 헤더

| 헤더 | 설명 |
|---|---|
| `X-Ignored-Args` | 무시된 인자 JSON. 예: `[{"name":"foo","reason":"unsupported_argument"}]` (최대 20개, 이름은 64자까지) |
| `X-Ignored-Args-Count` | 무시된 인자의 전체 개수 |
| `X-Warnings` | 모델이 보고한 경고 문자열 JSON 배열 (최대 10개) |

**실패** — JSON

```json
{
  "detail": {
    "error_code": 2001,
    "error_name": "CUDA_OOM",
    "message": "...",
    "attempts": 3,
    "ignored_args": [{"name": "foo", "reason": "unsupported_argument"}]
  }
}
```

| 에러 코드 범위 / 코드 | HTTP |
|---|---|
| 1xxx (입력 문제) | 422 |
| `QUEUE_FULL`, `NO_GPU_AVAILABLE` | 503 |
| `TIMEOUT` | 504 |
| 그 외 (2xxx, 3xxx, `WORKER_CRASHED`, `UNKNOWN`) | 500 |

> 필수 필드 누락·타입 오류처럼 FastAPI 가 직접 거절하는 경우는 위 형식이 아니라 FastAPI 기본 422 형식(`{"detail": [...]}`)으로 응답한다.

#### 정의되지 않은 인자 처리

정의되지 않은 인자는 **에러가 아니라 무시**하고 응답으로 알려준다.

| 상황 | reason |
|---|---|
| `CORE_FIELDS` 도 `MODEL_OPTIONS` 도 아닌 form 필드 | `unsupported_argument` |
| 선언되지 않은 파일 필드 | `unsupported_file_field` |
| query 파라미터 (전부) | `unsupported_query_param` |

반대로 **정의된 옵션인데 값이 잘못된 경우**(`guidance_scale=abc`, 범위 밖, 허용되지 않는 choice)는 `INVALID_INPUT` (422) 으로 실패한다. 이때도 `detail.ignored_args` 에 무시된 인자를 함께 담아 준다.

### `GET /health`

스케줄러/GPU 상태. **GPU 가 전부 죽어 있어도 항상 200** 이므로, 상태 판단은 본문으로 해야 한다.

```json
{
  "waiting": 0,
  "gpus": [
    {"gpu": 0, "state": "idle", "reason": null, "free_vram_mb": 31204, "jobs_done": 12, "restarts": 0}
  ]
}
```

### `GET /queue`

현재 대기 중인 요청(FIFO 순서)과 GPU 에서 실행 중인 요청 목록. 파일 경로 등은 노출하지 않는다.

```json
{
  "max_queue": 32,
  "queue_full": false,
  "waiting_count": 1,
  "running_count": 1,
  "waiting": [
    {"position": 1, "job_id": "a1b2c3d4e5f6", "choreography": "dance_a", "num_images": 2,
     "attempts": 0, "waited_sec": 12.3, "total_sec": 12.3}
  ],
  "running": [
    {"job_id": "...", "gpu": 0, "choreography": "dance_b", "num_images": 3,
     "attempt": 1, "running_sec": 4.2, "timeout_in_sec": 25.8, "total_sec": 9.0}
  ]
}
```

- `attempts > 0` 인 대기 항목은 실패 후 재시도를 위해 다시 줄을 선 요청이다.
- `timeout_in_sec` 은 실행 중인 요청이 `JOB_TIMEOUT` 에 걸리기까지 남은 시간이다.

---

## 5. 설정

### 환경변수 (`main.py`)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `NUM_GPUS` | 4 | 사용할 GPU 수 (index 0 ~ N-1) |
| `MAX_QUEUE` | 32 | GPU 를 기다리는 요청의 최대 수. 초과하면 즉시 503 `QUEUE_FULL` |
| `JOB_TIMEOUT` | 30 | **1회 시도**의 최대 실행 시간(초). 큐 대기 시간은 포함하지 않음 |
| `LOAD_VRAM_MB` | 5000 | 모델 로드에 필요한 free VRAM. worker spawn 직전에 확인 |
| `INFER_VRAM_MB` | 5000 | 추론에 필요한 free VRAM(모델 로드 후 기준). 작업 배정 직전에 확인 |
| `HEALTH_CHECK_INTERVAL` | 300 | GPU health check 주기(초) |
| `DISABLE_NVML` | 0 | `1` 이면 NVML 점검을 끄고 항상 정상으로 취급 (개발용) |
| `WORK_ROOT` | 시스템 임시 디렉터리 | 업로드/결과 파일을 두는 작업 디렉터리 |

> `JOB_TIMEOUT=30`, `LOAD_VRAM_MB=INFER_VRAM_MB=5000` 은 **더미 모델 기준 값**이다. 실제 모델에서는 반드시 실측해서 바꿔야 한다 (산정 방법은 `GENERATION_SPEC.md` §7).

### 스케줄러 인자 (환경변수로 노출되지 않음)

`main.py` 의 `GPUScheduler(...)` 호출에 인자를 추가해서 바꾼다.

| 인자 | 기본값 | 설명 |
|---|---|---|
| `max_total_attempts` | 4 | 한 요청의 총 시도 횟수 상한 (에러 코드별 재시도 한도와 별개) |
| `max_restarts` | 5 | worker 연속 재시작 한도. 초과하면 해당 GPU 는 DEAD (작업이 한 번 성공하면 0 으로 초기화) |
| `startup_timeout` | 600 | 서버 시작 시 모든 worker 의 모델 로딩을 기다리는 최대 시간(초) |

---

## 6. 스케줄러 동작

### GPU(worker) 상태

| 상태 | 의미 |
|---|---|
| `starting` | worker 기동 / 모델 로딩 중 |
| `idle` | 모델 로드 완료, 작업을 받을 수 있음 |
| `busy` | `reason` = `job`(작업 중) · `low_vram_infer`(추론용 VRAM 부족, 모델은 로드됨) · `low_vram_load`(로드용 VRAM 부족, worker 미기동) |
| `dead` | GPU 오류(NVML 조회 실패) 또는 worker 재시작 한도 초과 |

### VRAM 기준 2개

- **`LOAD_VRAM_MB`** — worker 를 띄우기 *직전*에 확인. 부족하면 worker 를 띄우지 않고 `busy(low_vram_load)` 로 대기한다 (OOM 을 내지 않기 위함).
- **`INFER_VRAM_MB`** — 작업을 배정하기 *직전*에 확인. 부족하면 그 GPU 는 `busy(low_vram_infer)` 로 두고 다른 GPU 를 고른다. 이 경우는 시도 횟수를 소모하지 않는다.

### Health check (`HEALTH_CHECK_INTERVAL` 마다 전체 GPU 점검)

| 현재 상태 | 조건 | 전이 |
|---|---|---|
| 아무 상태 | NVML 오류 | `dead` (worker kill, 진행 중 작업은 `WORKER_CRASHED` 로 재시도) |
| `dead` | NVML 정상 복구 | 로드용 VRAM 확인 후 worker 재기동 → 로딩 완료 시 `idle` (재시작 한도 초과로 DEAD 된 경우도 다음 점검 때 복구 시도) |
| `idle` | free < `INFER_VRAM_MB` | `busy(low_vram_infer)` |
| `busy(low_vram_infer)` | free 회복 | `idle` |
| `busy(low_vram_load)` | free ≥ `LOAD_VRAM_MB` | worker spawn |

작업 중(`busy/job`)인 GPU 는 VRAM 이 낮은 게 정상이므로 VRAM 판정에서 제외한다.
별도로 1초마다 worker 프로세스 생존을 확인해, 죽었으면 진행 중 작업을 `WORKER_CRASHED` 로 처리하고 재시작한다.

### GPU 선택

`idle` 인 GPU 중 `jobs_done` 이 가장 적은 GPU 를 고른다. 모든 GPU 가 `dead` 면 `NO_GPU_AVAILABLE`(503). 큐 대기 시간에는 타임아웃이 없으므로, 클라이언트 쪽에서 적절한 요청 타임아웃을 두는 것이 좋다.

### 에러 코드와 재시도 정책

| 코드 | 이름 | 최대 재시도 | 다른 GPU | worker 재시작 | 백오프 |
|---|---|---|---|---|---|
| 1001 | `INVALID_INPUT` | 0 | | | |
| 1002 | `IMAGE_DECODE_FAILED` | 0 | | | |
| 1003 | `AUDIO_DECODE_FAILED` | 0 | | | |
| 1004 | `FACE_NOT_DETECTED` | 0 | | | |
| 1005 | `UNSUPPORTED_CHOREOGRAPHY` | 0 | | | |
| 2001 | `CUDA_OOM` | 2 | O | | 1.0s |
| 2002 | `CUDA_RUNTIME_ERROR` | 2 | O | O | |
| 2003 | `MODEL_INFERENCE_FAILED` | 1 | | | 0.5s |
| 3001 | `IO_ERROR` | 2 | | | 1.0s |
| 3002 | `ENCODING_FAILED` | 1 | | | 0.5s |
| 4001 | `TIMEOUT` (scheduler) | 1 | O | O | |
| 4002 | `WORKER_CRASHED` (scheduler) | 2 | O | O | |
| 4003 | `QUEUE_FULL` (scheduler) | — 요청 접수 단계에서 즉시 실패 | | | |
| 4004 | `NO_GPU_AVAILABLE` (scheduler) | — | | | |
| 9999 | `UNKNOWN` (모델이 `GenerationError` 외 예외를 던진 경우) | 1 | O | | |

- 재시도 한도는 **에러 코드별**로 센다. 이와 별개로 총 시도 횟수는 `max_total_attempts`(4) 를 넘지 않는다.
- "다른 GPU" 는 직전 GPU 를 피한다. 살아있는 GPU 가 전부 제외 대상이면 제외 조건을 무시한다.
- 4xxx 는 scheduler 가 직접 발생시키는 코드다. 모델(`generation.py`)은 던지지 않는다.
- 타임아웃(`JOB_TIMEOUT`) 이 나면 CUDA 작업은 중단할 수 없으므로 worker 를 죽이고 재시작한다. 요청이 취소(`CancelledError`)된 경우에도 같은 이유로 worker 를 재시작한다.
- `generation.py` 수정 후에 에러코드에 맞춰 `RETRY_POLICIES` **수정 필요**

---

## 7. 더미 모델로 동작 확인하기

더미 `generate()` 는 `extra` 로 들어온 `_simulate` 값에 따라 에러를 흉내 낸다. 스케줄러의 재시도/재시작 동작을 확인하는 용도다.

```bash
curl -i -X POST http://localhost:8000/generate \
  -F "images=@a.jpg" -F "audio=@voice.wav" -F "choreography=dance_a" \
  -F "_simulate=CUDA_OOM" -F "_alloc_mb=30000" -o /dev/null
```

| `_simulate` | 추가 인자 | 기대 동작 |
|---|---|---|
| (없음) | | 약 5초 후 `DUMMY_MP4_CONTENT` 가 담긴 200 응답 |
| `IMAGE_DECODE_FAILED` / `AUDIO_DECODE_FAILED` / `FACE_NOT_DETECTED` | | 즉시 422, `attempts=1` |
| `CUDA_OOM` | `_alloc_mb` (기본 30000) | 실제로 VRAM 을 할당. GPU 용량이 부족하면 OOM → 다른 GPU 로 재시도, 용량이 충분하면 성공 |
| `CUDA_RUNTIME_ERROR` | | worker 재시작 + 다른 GPU 재시도, 최종 500 (`attempts=3`) |
| `MODEL_INFERENCE_FAILED` | | 같은 GPU 1회 재시도, 최종 500 (`attempts=2`) |
| `IO_ERROR` | | 같은 GPU 2회 재시도, 최종 500 (`attempts=3`) |
| `ENCODING_FAILED` | | 1회 재시도, 최종 500 (`attempts=2`) |
| `WORKER_CRASH` | | worker 프로세스가 `sys.exit(-1)` → `WORKER_CRASHED` 로 감지, 재시작 + 재시도, 최종 500 (`attempts=3`) |
| `TIMEOUT` | `_sleep_sec` (기본 35) | `JOB_TIMEOUT`(30) 초과 → 재시작 + 1회 재시도, 최종 504 (`attempts=2`) |

실행 중에 큐와 GPU 상태를 같이 보면 동작을 이해하기 쉽다.

```bash
watch -n 1 'curl -s localhost:8000/queue; echo; curl -s localhost:8000/health'
```

> `_simulate`, `_alloc_mb`, `_sleep_sec` 는 더미 전용 훅이다. 실제 모델에서는 **반드시 제거**해야 한다 (외부에서 프로세스 종료·대량 할당·sleep 을 유발할 수 있음).

---

## 8. 실제 모델로 교체하기

[`GENERATION_SPEC.md`](./GENERATION_SPEC.md) 에 `generation.py` 가 노출해야 하는 이름, 각 타입/메서드의 규약, 에러 매핑, 금지 사항, 골격 코드, 통합 전 체크리스트가 정리되어 있다. 요약:

1. `scheduler.py`, `main.py` 는 수정하지 않는다. `generation.py` 안에서만 바꾼다.
2. 모듈 최상단에서 `torch` 등 무거운 import 를 하지 않는다 (API 프로세스도 import 한다).
3. `VideoGenerationModel` 의 `load()` / `generate()` / `clear_cache()` 와 `MODEL_OPTIONS`, `SUPPORTED_CHOREOGRAPHIES` 를 실제 내용으로 채운다.
4. 실패는 `GenerationError(ErrorCode.X, message)` 로 알린다. 코드에 따라 재시도 방식이 달라진다.
5. `JOB_TIMEOUT`, `LOAD_VRAM_MB`, `INFER_VRAM_MB` 를 실측 값으로 설정한다.

---

## 9. 운영 시 주의

- **`--workers 1` 필수.** 스케줄러가 API 프로세스 안에서 worker 들을 관리한다. uvicorn worker 를 늘리면 GPU 마다 모델이 중복 로드되고 GPU 상태가 공유되지 않는다.
- **인증이 없다.** `/generate`, `/health`, `/queue` 모두 열려 있으므로 외부에 노출한다면 앞단(리버스 프록시 등)에서 접근을 제어할 것.
- **큐는 메모리에만 있다.** 서버를 재시작하면 대기/실행 중이던 요청은 사라진다.
- **디스크 사용량.** 업로드 파일과 결과 mp4 가 `WORK_ROOT` 에 요청 단위로 임시 저장된다. 요청 1건당 최대 (이미지 10 × 20MB + 오디오 50MB + 결과 mp4) 이다. `QUEUE_FULL` 판정은 업로드를 디스크에 저장한 *뒤* 스케줄러에서 이뤄지므로, 요청이 몰리면 `MAX_QUEUE` 로 계산한 것보다 더 많은 임시 파일이 잠시 쌓일 수 있다. 여유 공간을 넉넉히 잡을 것.
- **`/health` 는 항상 200** 이다. 로드밸런서 health check 로 쓰려면 본문의 GPU 상태를 해석하는 래퍼가 필요하다.