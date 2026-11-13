# `generation.py` 작성 가이드 (통합 규약)

현재 `generation.py` 는 더미다. 이 문서는 **실제 모델로 `generation.py` 를 새로 작성할 때**
`main.py` / `scheduler.py` 를 한 줄도 고치지 않고 매끄럽게 통합되도록 지켜야 할 규약을 정리한다.

- 코드에서 직접 확인되는 규약은 "**반드시**"로 표시했다.
- 코드가 강제하지는 않지만 안정적인 운영을 위해 권장하는 것은 "권장"으로 표시했다.

---

## 0. 한 장 요약 (체크리스트)

- [ ] `generation.py` 가 §1 의 이름을 **전부, 같은 이름으로** export 한다.
- [ ] 모듈 최상단에서 `torch`, `diffusers`, `cv2` 등 무거운 라이브러리를 import 하지 않는다 (함수 안에서 import).
- [ ] 모듈 import 시 CUDA 초기화 / 모델 로드 / 파일 읽기 같은 부작용이 없다.
- [ ] `resolve_options()` 는 순수 함수이며 GPU 를 쓰지 않는다.
- [ ] `VideoGenerationModel.load()` 는 **worker 프로세스당 1회**, `generate()` 는 **한 번에 하나씩** 호출된다.
- [ ] 실패는 `GenerationError(ErrorCode.X, message)` 로 알린다. 에러 코드에 따라 재시도/재시작 방식이 달라진다 (§5).
- [ ] `generate()` 는 성공 시 `req.output_path` 에 **유효한 mp4** 를 쓰고 `GenerationResult` 를 반환한다.
- [ ] 실패 후에도 같은 프로세스에서 다음 `generate()` 가 정상 동작한다 (재진입 안전).
- [ ] 더미의 테스트 훅(`_simulate`, `_alloc_mb`, `_sleep_sec`) 을 제거했다.
- [ ] `JOB_TIMEOUT`, `LOAD_VRAM_MB`, `INFER_VRAM_MB` 를 실측해서 설정했다 (§7).
- [ ] §10 의 스모크 테스트를 통과했다.

---

## 1. 반드시 export 해야 하는 이름

| 이름 | 종류 | 사용처 | 용도 |
|---|---|---|---|
| `ErrorCode` | `IntEnum` | main, scheduler | 에러 분류, HTTP 상태 매핑, 재시도 정책의 키 |
| `GenerationError` | `Exception` | main, scheduler(worker) | 모델 → 스케줄러로 에러 전달 |
| `OptionSpec` | dataclass | generation 내부 | 모델 옵션 선언 |
| `MODEL_OPTIONS` | `dict[str, OptionSpec]` | main | 정의된 옵션 / 정의되지 않은 인자 구분 |
| `ResolvedOptions` | dataclass | main, generation | `resolve_options` 의 반환 타입 (`.values`, `.ignored`) |
| `resolve_options(extra)` | 함수 | main(사전 검증), generation(재호출) | 모델 옵션 파싱·검증 |
| `SUPPORTED_CHOREOGRAPHIES` | `set[str]` | main | 안무 검증 / API 문서 |
| `GenerationRequest` | dataclass | main(생성), scheduler(pickle 전달) | 요청 |
| `GenerationResult` | dataclass | scheduler(worker), main | 결과 |
| `VideoGenerationModel` | class | scheduler worker 프로세스 | 모델 |

`main.py` 가 하는 import:

```python
from generation import (MODEL_OPTIONS, SUPPORTED_CHOREOGRAPHIES, ErrorCode,
                        GenerationError, GenerationRequest, resolve_options)
```

`scheduler.py` 가 하는 import:

```python
from generation import (ErrorCode, GenerationError, GenerationRequest, GenerationResult)
# worker 프로세스 안에서 (CUDA_VISIBLE_DEVICES 설정 후):
from generation import VideoGenerationModel
```

> 더미의 `GenerationError` / `OptionSpec` / `ResolvedOptions` / `resolve_options` / `GenerationRequest` / `GenerationResult` 부분은 규약 그대로이므로 **복사해서 유지**하고, `ErrorCode`(§3.1, 수정 시 알려줄 것), `MODEL_OPTIONS`, `SUPPORTED_CHOREOGRAPHIES`, `VideoGenerationModel` 만 실제 내용으로 바꾸는 것이 가장 안전하다.

---

## 2. 프로세스 모델과 import 규칙

```
API 프로세스 (uvicorn)                 worker 프로세스 (GPU 당 1개, spawn)
 └ import main → import generation      └ import scheduler → import generation   (모듈 import)
   (타입/옵션 검증만 사용,                 └ CUDA_VISIBLE_DEVICES=<gpu_id> 설정
    GPU 사용 안 함)                         └ VideoGenerationModel("cuda:0").load()
                                            └ 루프: generate(req) → clear_cache()
```

### 2.1 import 규칙 (반드시)

- `generation.py` 는 **API 프로세스에서도 import** 된다. 따라서 모듈 최상단에서 `torch` 등 무거운 import 를 하지 않는다. 필요한 곳(`load()`, `generate()`, `clear_cache()`) 안에서 import 한다.
- worker 는 `CUDA_VISIBLE_DEVICES` 를 설정하기 *전에* `scheduler` → `generation` 모듈을 import 한다 (spawn 된 프로세스가 `_worker_main` 을 찾으려고 `scheduler` 를 import 하기 때문). 모듈 import 시점에 CUDA 를 초기화하는 코드(`torch.cuda.set_device`, 모듈 레벨 GPU 텐서, `torch.cuda.is_available()` 이후 디바이스 선택 등)가 있으면 GPU 매핑이 깨질 수 있다.
- 모듈 import 시 모델 로딩, 대용량 파일 읽기, 네트워크 호출, `logging.basicConfig()` 호출 등 부작용을 일으키지 않는다.

> 더미는 파일 최상단에서 `import torch` 를 한다. 더미라서 동작하지만, 자기 docstring 의 지침("무거운 import 는 load() 안에서")과도 어긋나므로 실제 구현에서는 하지 말 것.

### 2.2 디바이스 (반드시)

- worker 는 GPU 1장만 보이도록 `CUDA_VISIBLE_DEVICES=<gpu_id>` 를 설정하므로 모델은 항상 `VideoGenerationModel(device="cuda:0")` 로 생성된다.
- `cuda:1` 같은 인덱스를 하드코딩하거나 멀티 GPU 병렬(DataParallel 등)을 쓰지 않는다. 항상 `self.device` 를 사용한다.

---

## 3. 타입 상세

### 3.1 `ErrorCode` / `GenerationError`

```python
class GenerationError(Exception):
    def __init__(self, code: ErrorCode, message: str = ""):
        super().__init__(f"[{code.name}({code.value})] {message}")
        self.code = code
        self.message = message
```

- `.code` 와 `.message` 속성이 **반드시** 있어야 한다. worker 가 `e.code`, `e.message` 를 읽어 스케줄러로 보낸다.
- 아래 멤버는 현재의 dummy 에러 코드 인데, 실제 상황에 맞게 수정이 필요할 것으로 보임. `main.py` / `scheduler.py` 가 이름으로 참조하므로 **수정 시에 꼭 알려줄 것**.

| 범위 | 코드 | 참조처 |
|---|---|---|
| 1xxx 입력 | `INVALID_INPUT`(1001), `IMAGE_DECODE_FAILED`(1002), `AUDIO_DECODE_FAILED`(1003), `FACE_NOT_DETECTED`(1004), `UNSUPPORTED_CHOREOGRAPHY`(1005) | main (`INVALID_INPUT`, `UNSUPPORTED_CHOREOGRAPHY`), HTTP 422 매핑 |
| 2xxx GPU/추론 | `CUDA_OOM`(2001), `CUDA_RUNTIME_ERROR`(2002), `MODEL_INFERENCE_FAILED`(2003) | scheduler 재시도 정책 |
| 3xxx I/O | `IO_ERROR`(3001), `ENCODING_FAILED`(3002) | scheduler 재시도 정책 |
| 4xxx scheduler 전용 | `TIMEOUT`(4001), `WORKER_CRASHED`(4002), `QUEUE_FULL`(4003), `NO_GPU_AVAILABLE`(4004) | scheduler / main. **모델은 던지지 않는다** |
| 기타 | `UNKNOWN`(9999) | scheduler (예상치 못한 예외의 기본값) |

에러 코드 가이드:
- 번호 범위의 의미를 지킨다. 1xxx 는 HTTP 422 로 매핑된다 (`main._http_status`).
- `scheduler.RETRY_POLICIES` 에 등록하지 않으면 **재시도 없음**(`DEFAULT_POLICY`)으로 처리된다. 재시도가 필요하면 `scheduler.py` 도 함께 수정해야 한다.
- 2xxx/3xxx 는 HTTP 500 으로 매핑된다.

### 3.2 `OptionSpec` / `MODEL_OPTIONS` / `resolve_options`

```python
@dataclass(frozen=True)
class OptionSpec:
    type: type                         # float / int / str ...
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    choices: Optional[tuple] = None
    description: str = ""

@dataclass
class ResolvedOptions:
    values: dict                                       # 기본값이 채워진 최종 옵션
    ignored: list[dict] = field(default_factory=list)  # [{"name": ..., "reason": ...}]

def resolve_options(extra: Optional[dict]) -> ResolvedOptions: ...
```

`resolve_options` 규약 (반드시):

| 입력 | 동작 |
|---|---|
| `extra` 가 `None` / `{}` | 모든 옵션의 기본값을 담은 `ResolvedOptions` 반환 |
| `MODEL_OPTIONS` 에 **없는** key | 에러가 아님. `ignored` 에 `{"name": str(key), "reason": "unsupported_argument"}` 추가 |
| 정의된 key 인데 타입 변환 실패 / 범위 위반 / choices 위반 | `GenerationError(ErrorCode.INVALID_INPUT, message)` |
| 정상 | `values[key]` 에 변환된 값 저장 |

- **순수 함수**여야 한다: GPU/torch 사용 금지, 빠르게 끝나야 하며, 같은 입력에 같은 결과를 낸다. API 프로세스(`main.py`, 사전 검증)와 worker(`generate()` 안) 양쪽에서 호출된다.
- `ignored` 의 각 항목은 `"name"` 과 `"reason"` 두 키를 가진 JSON 직렬화 가능한 dict 여야 한다 (`main._merge_ignored` 는 `item["name"]` 으로 중복 제거, `_ignored_header` 는 `name`/`reason` 을 읽는다).
- `extra` 의 값은 **항상 문자열**이다 (form 필드이므로). 그래서 `spec.type(raw)` 로 변환한다. 주의할 점:
  - `bool("false")` 는 `True` 다. bool 옵션이 필요하면 `type=bool` 대신 문자열을 직접 해석하는 로직을 넣는다 (`choices=("true", "false")` 로 선언하고 사용 시점에 변환하는 방법도 있다).
  - `float("nan")` 은 `min`/`max` 비교를 모두 통과한다. float 옵션에는 `math.isfinite()` 검사를 추가하는 것을 권장한다.
- `main.py` 는 `k not in MODEL_OPTIONS` 로 정의되지 않은 인자를 판별하므로, 모델이 실제로 받는 옵션은 **전부** `MODEL_OPTIONS` 에 선언되어 있어야 한다. 선언하지 않은 key 를 `req.extra` 에서 몰래 읽으면 클라이언트에게는 "무시됨"으로 보고된다 (더미의 `_simulate` 가 정확히 이 경우).
- 모델 `generate()` 안에서는 `resolve_options(req.extra).values` 를 실제 추론 인자로 **사용**한다 (더미는 `ignored` 만 쓰고 있다).

### 3.3 `GenerationRequest`

```python
@dataclass
class GenerationRequest:
    image_paths: list[str]
    audio_path: str
    choreography: str
    output_path: str
    fps: int = 25
    resolution: int = 512
    seed: Optional[int] = None
    extra: dict = field(default_factory=dict)
```

- 프로세스 간 pickle 로 전달되므로 **파일 경로 등 단순 타입만** 담는다. 바이트나 텐서를 넣지 않는다. 모듈 최상위에 정의된 dataclass 여야 한다.
- `main.py` 가 키워드 인자로 위 8개 필드를 채워 생성한다. 필드를 추가/삭제하려면 `main.py` 도 같이 수정해야 한다.

| 필드 | 호출 시점에 보장되는 것 | 모델이 해야 하는 것 |
|---|---|---|
| `image_paths` | 1~10개. 업로드 순서. 파일이 존재함. 각 20MB 이하. 파일명은 `img_00<ext>` … (확장자는 업로드 파일명 기준, 없으면 `.jpg`) | 실제로 디코드 가능한지 검증 → 실패 시 `IMAGE_DECODE_FAILED`. 얼굴 검출이 필요하면 `FACE_NOT_DETECTED` |
| `audio_path` | 파일이 존재함. 50MB 이하. `audio<ext>` (확장자 없으면 `.wav`) | 디코드 검증 → 실패 시 `AUDIO_DECODE_FAILED` |
| `choreography` | `main.py` 가 `SUPPORTED_CHOREOGRAPHIES` 로 검증 완료 | 방어적으로 한 번 더 검증 → `UNSUPPORTED_CHOREOGRAPHY` |
| `output_path` | 요청 전용 임시 디렉터리 안의 `output.mp4`. 쓰기 가능 | **여기에** 결과 mp4 를 쓴다 |
| `fps` | 1~60 정수 | |
| `resolution` | **검증되지 않은 정수** (음수/0/매우 큰 값 가능) | 허용 범위·지원 값 검증 → 위반 시 `INVALID_INPUT` |
| `seed` | `None` 또는 정수 | `None` 이면 임의 시드, 값이 있으면 재현 가능하게 |
| `extra` | 선언되지 않은 form 필드. **값은 모두 `str`**. 같은 key 가 반복되면 마지막 값 | `resolve_options(req.extra)` 로 해석 |

> 업로드 파일의 *내용*은 main 에서 전혀 검증하지 않는다 (크기만 제한). 손상된 파일, 확장자가 거짓인 파일, 빈 파일이 그대로 들어온다고 가정할 것.

### 3.4 `GenerationResult`

```python
@dataclass
class GenerationResult:
    output_path: str
    ignored_args: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
```

- worker 는 이 세 필드만 `dict` 로 스케줄러에 전달한다. **필드를 추가해도 전달되지 않는다.**
- `output_path`: 반드시 존재하는, 비어있지 않은 mp4. `main.py` 는 존재 여부를 검사하지 않고 그대로 `FileResponse` 로 보낸다 (없으면 500).
- `ignored_args`: 보통 `resolve_options(...).ignored` 를 그대로 넣는다. 항목은 `{"name": str, "reason": str}`.
- `warnings`: 짧은 문자열 목록. `X-Warnings` 헤더로 나가며 최대 10개까지만 전달된다. 헤더이므로 짧게 유지한다.

---

## 4. `VideoGenerationModel` 메서드 규약

### 4.1 `__init__(self, device: str = "cuda:0")`

- `device` 를 저장하고 필드를 초기화하기만 한다. 가볍게 끝나야 하며 GPU 메모리를 할당하지 않는다 (할당은 `load()`).

### 4.2 `load(self) -> None`

- worker 프로세스당 **1회**, 블로킹으로 호출된다. 가중치를 `self.device` 로 올리고 필요하면 warm-up 까지 끝낸다.
- 어떤 예외든 던지면 스케줄러가 worker 를 재시작한다. 연속 재시작이 `max_restarts`(5) 를 넘으면 그 GPU 는 DEAD 가 된다. 예외 종류를 구분해서 처리할 필요는 없다.
- **반드시 종료되어야 한다.** `STARTING` 상태에 대한 별도 타임아웃은 없고, 서버 시작 시에는 최대 `startup_timeout`(600초) 까지만 기다린 뒤 서비스를 시작한다. 그 뒤에도 `load()` 가 끝나지 않으면 해당 GPU 는 영구히 `starting` 으로 남는다.
- 로드 직후 점유 VRAM 이 `LOAD_VRAM_MB` 이하가 되도록 설정값을 맞춘다 (§7).

### 4.3 `generate(self, req: GenerationRequest) -> GenerationResult`

- worker 안에서 **동기·블로킹**으로 호출되며, GPU(worker) 하나당 한 번에 하나만 실행된다. 동시 호출은 없다.
- 권장 흐름: `_validate(req)` → `resolve_options(req.extra)` → 입력 디코드/전처리 → 추론 → 인코딩 → `req.output_path` 에 기록 → `GenerationResult` 반환.
- **성공 시**: `req.output_path` 에 유효한 mp4 를 쓰고 `GenerationResult(output_path=req.output_path, ignored_args=opts.ignored, ...)` 를 반환한다. 다른 경로에 결과를 쓰지 않는다.
- **실패 시**: `GenerationError` 를 던진다 (§5). `GenerationError` 가 아닌 예외는 모두 `UNKNOWN(9999)` 으로 처리되어 다른 GPU 에서 1회 재시도된다.
- **재진입 안전**: 실패한 뒤에도 같은 프로세스에서 다음 `generate()` 가 정상 동작해야 한다. `MODEL_INFERENCE_FAILED`, `IO_ERROR`, `ENCODING_FAILED` 등은 worker 재시작 없이 같은 모델 인스턴스로 재시도되거나 다음 작업이 들어온다. 전역 상태(캐시, 임시 텐서, 열린 파일 핸들)가 누적/오염되지 않게 한다.
- **재시도 시 같은 `output_path`** 로 다시 호출될 수 있다. 이전 시도가 남긴 부분 파일을 덮어쓸 수 있어야 한다.
- **시간 제한**: 1회 호출은 `JOB_TIMEOUT` 안에 끝나야 한다. 초과하면 worker 프로세스가 강제 종료(kill)된다. 모델이 스스로 시간 제한을 걸 필요는 없다.
- **강제 종료 가능성**: 타임아웃/취소 시 프로세스가 `SIGKILL` 로 죽을 수 있으므로 `finally` 정리 로직에 의존하지 않는다. 임시 파일은 `req.output_path` 와 같은 디렉터리에 만든다 (main 이 응답 후 디렉터리째 삭제한다).
- `generate()` 가 반환한 뒤에 백그라운드 스레드/서브프로세스가 `req` 의 파일을 계속 읽거나 쓰면 안 된다 (응답 직후 디렉터리가 삭제된다).
- 시드는 전역 `torch.manual_seed` 대신 요청 단위 `torch.Generator` 로 적용하는 것을 권장한다.

### 4.4 `clear_cache(self) -> None`

- 스케줄러가 **매 작업 직후** (성공/실패 무관) 호출한다. 예외는 무시된다.
- 목적: PyTorch caching allocator 가 쥐고 있는 VRAM 을 반환해 NVML 의 free VRAM 판정을 정확하게 만드는 것. 보통 `gc.collect()` + `torch.cuda.empty_cache()`.
- 모델 가중치를 해제하면 안 된다 (다음 작업에서 재사용).

---

## 5. 에러 매핑 가이드

어떤 상황에서 어떤 코드를 던지느냐가 곧 스케줄러의 동작을 결정한다.

| 상황 | 던질 코드 | 스케줄러 동작 | HTTP |
|---|---|---|---|
| 이미지 목록 비어있음, 옵션/`resolution` 등 파라미터 범위 위반 | `INVALID_INPUT` 1001 | 즉시 실패 | 422 |
| 이미지 디코드 실패 | `IMAGE_DECODE_FAILED` 1002 | 즉시 실패 | 422 |
| 오디오 디코드 실패 | `AUDIO_DECODE_FAILED` 1003 | 즉시 실패 | 422 |
| 얼굴이 검출되지 않음 | `FACE_NOT_DETECTED` 1004 | 즉시 실패 | 422 |
| 지원하지 않는 안무 | `UNSUPPORTED_CHOREOGRAPHY` 1005 | 즉시 실패 | 422 |
| `torch.cuda.OutOfMemoryError` | `CUDA_OOM` 2001 | **다른 GPU** 로 최대 2회 재시도 (백오프 1.0s) | 500 |
| CUDA `RuntimeError` (device-side assert, illegal memory access 등 — CUDA context 오염 가능) | `CUDA_RUNTIME_ERROR` 2002 | **worker 재시작** + 다른 GPU 로 최대 2회 재시도 | 500 |
| NaN 등 일시적 추론 실패 | `MODEL_INFERENCE_FAILED` 2003 | 같은 GPU 에서 1회 재시도 (백오프 0.5s) | 500 |
| 파일 읽기/쓰기 실패 (`OSError`) | `IO_ERROR` 3001 | 같은 GPU 에서 최대 2회 재시도 (백오프 1.0s) | 500 |
| ffmpeg 등 인코딩 실패 | `ENCODING_FAILED` 3002 | 1회 재시도 (백오프 0.5s) | 500 |
| 위에 해당하지 않는 예기치 못한 예외 | 그냥 raise → `UNKNOWN` 9999 | 다른 GPU 에서 1회 재시도 | 500 |

원칙
- **입력 탓이라서 다시 해도 소용없는 것만 1xxx** 로 분류한다. 일시적일 수 있는 실패를 1xxx 로 던지면 재시도 기회를 잃고, 입력 문제를 2xxx/3xxx 로 던지면 불필요하게 재시도하며 GPU 를 낭비한다.
- 예외 변환 순서에 주의: `torch.cuda.OutOfMemoryError` 는 `RuntimeError` 의 하위 클래스이므로 **OOM 을 먼저** 잡고 그다음에 일반 CUDA `RuntimeError` 를 처리한다.
- 4xxx(`TIMEOUT`, `WORKER_CRASHED`, `QUEUE_FULL`, `NO_GPU_AVAILABLE`) 는 스케줄러 전용이다. 모델에서 던지지 않는다.
- `SystemExit`, `os._exit()`, 시그널 핸들러 변경 등으로 프로세스를 스스로 종료하지 않는다. 더미의 `WORKER_CRASH` 훅(`sys.exit(-1)`)은 스케줄러의 크래시 복구를 시험하기 위한 것이다.

### `message` 작성 규칙 (권장)

`GenerationError.message` 는 **HTTP 응답 `detail.message` 로 클라이언트에 그대로 노출**된다.
- 짧게 쓴다 (200자 이내 권장). 스택 트레이스, 절대 경로, 내부 모델 이름/버전, 서버 환경 정보를 넣지 않는다.
- 파일을 언급해야 하면 `os.path.basename()` 만 쓴다.
- 사용자 입력 값을 그대로 echo 해야 하면 길이를 자른다 (`resolve_options` 는 `str(raw)[:50]`).
- 자세한 원인은 서버 로그에 남기고 message 에는 요약만 둔다.

---

## 6. 검증 책임 분담

| 항목 | main.py | generation.py (모델) |
|---|---|---|
| 필수 필드 존재 / 타입 | FastAPI 가 검증 | |
| 이미지 개수 (1~10) | O | `_validate` 에서 빈 목록 방어 |
| 이미지/오디오 크기 (20MB / 50MB) | O | |
| 이미지/오디오 **내용**(디코드 가능 여부) | X | **O** (1002 / 1003) |
| 얼굴 검출 등 모델 전처리 조건 | X | **O** (1004 등) |
| `choreography` ∈ 지원 목록 | O | 방어적 재검증 (1005) |
| `fps` 1~60 | O | |
| `resolution` | **X** | **O** (`INVALID_INPUT`) |
| `seed` | 정수 파싱만 | 의미 검증 필요하면 모델에서 |
| 모델 옵션 값 | `resolve_options` 호출 | `resolve_options` 구현 + `generate()` 에서 재호출 |
| 정의되지 않은 인자 | 수집·보고 | `MODEL_OPTIONS` 에 없는 key 를 `ignored` 로 분류 |

---

## 7. VRAM 설정 가이드

`main.py` 의 두 환경변수는 모델의 실제 메모리 사용량에 맞춰야 스케줄러가 올바르게 동작한다.

| 변수 | 의미 | 산정 방법 (권장) |
|---|---|---|
| `LOAD_VRAM_MB` | worker 를 띄우기 전에 GPU 에 남아 있어야 하는 free VRAM | `load()` 직후 점유량 + 여유. 현재 더미 기준 약 2GB 점유 |
| `INFER_VRAM_MB` | 작업 배정 직전에 (모델이 이미 올라간 상태에서) 남아 있어야 하는 free VRAM | `generate()` 의 **피크 추가 점유량**(최악의 입력: 이미지 10장, 최대 `resolution`, 최대 `num_inference_steps` 등) + 여유. 현재 더미 기준 약 1GB |

- 측정: `generate()` 전후로 `torch.cuda.max_memory_allocated()` / `max_memory_reserved()` 를 확인하고, `nvidia-smi` 로 교차 확인한다.
- NVML 의 free 값은 **GPU 전체 기준**이다. 같은 GPU 를 다른 프로세스가 쓰고 있으면 그만큼 줄어든다.
- caching allocator 가 예약한 메모리도 free 에서 빠진다. 그래서 `clear_cache()` 가 중요하다.
- 기준에 못 미치면 OOM 대신 GPU 가 `busy(low_vram_load / low_vram_infer)` 로 빠져 요청을 받지 않는다. 값이 너무 크면 정상 GPU 도 계속 `busy` 로 남는다.
- `JOB_TIMEOUT` 은 **1회 시도의 최대 실행 시간**이다 (큐 대기 제외). 더미용 기본값 30초이므로, 실제 모델의 최악 소요 시간(+여유)로 반드시 바꾼다.
- `scheduler.py` 자체의 기본값(`load_vram_mb=20000`, `infer_vram_mb=40000`, `job_timeout=600`)은 `main.py` 가 환경변수 값으로 덮어쓴다.

---

## 8. 금지 / 주의 사항

1. **더미 테스트 훅 제거**: `_simulate`, `_alloc_mb`, `_sleep_sec` 를 처리하는 분기를 전부 삭제한다. 남아 있으면 외부 요청 하나로 worker 종료(`sys.exit`), 대량 VRAM 할당, 장시간 sleep 을 유발할 수 있다.
2. **모듈 최상단 `torch` import 금지** (§2.1).
3. **다른 GPU 건드리지 않기**: `cuda:0` / `self.device` 만 사용 (§2.2).
4. **전역 상태 오염 금지**: `torch.manual_seed` 같은 전역 시드, `torch.set_default_device`, 환경변수 변경 등을 요청 단위로 남기지 않는다.
5. **결과는 `req.output_path` 에만 쓴다.** 다른 위치에 쓰거나 전역 `/tmp` 에 임시 파일을 만들지 않는다.
6. **`generate()` 반환 후 남는 작업 금지** (백그라운드 스레드, 서브프로세스) (§4.3).
7. **로깅 설정 변경 금지**: `logging.basicConfig()` 호출 등. 필요하면 모듈 로거(`logging.getLogger("generation")`)만 쓴다.
8. **내부 정보 노출 금지**: `GenerationError.message` 규칙 (§5).
9. 더미의 `_validate` 처럼 지원 안무 목록을 함수 안에 따로 복사해 두지 말고, 모듈의 `SUPPORTED_CHOREOGRAPHIES` 하나만 쓴다 (두 군데가 어긋나면 main 은 통과시키고 모델은 거절한다).

---

## 9. 골격 코드

`ErrorCode` 는 실제 모델의 실패 유형에 맞게 수정이 필요할 수 있다 (§3.1, 수정 시 꼭 알려줄 것). `GenerationError` ~ `GenerationResult` 부분은 더미에서 그대로 두고, 아래 부분을 실제 구현으로 바꾼다.

> 아래 골격의 `ErrorCode.X` 이름은 현재 더미 기준이다. `ErrorCode` 를 재정의했다면 그에 맞게 바꿔서 사용한다.

```python
# ----- 실제 값으로 교체 -----
MODEL_OPTIONS: dict[str, OptionSpec] = {
    "guidance_scale":      OptionSpec(float, 3.0, min=0.0, max=20.0, description="CFG scale"),
    "num_inference_steps": OptionSpec(int, 30, min=1, max=100),
    "background":          OptionSpec(str, "original", choices=("original", "blur", "studio")),
}

SUPPORTED_CHOREOGRAPHIES = {"dance_a", "dance_b", "dance_c", "kpop_01", "kpop_02"}

MIN_RESOLUTION, MAX_RESOLUTION = 256, 1024   # 예시 — 모델 스펙에 맞게


class VideoGenerationModel:
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.pipe = None

    # worker 프로세스당 1회
    def load(self) -> None:
        import torch                                   # 무거운 import 는 함수 안에서
        # self.pipe = MyPipeline.from_pretrained(...).to(self.device)
        # warm-up 이 필요하면 여기서 수행
        ...

    def generate(self, req: GenerationRequest) -> GenerationResult:
        import torch

        self._validate(req)
        opts = resolve_options(req.extra)              # opts.values 를 추론에 사용

        try:
            images = self._load_images(req.image_paths)   # 실패 시 IMAGE_DECODE_FAILED
            audio = self._load_audio(req.audio_path)      # 실패 시 AUDIO_DECODE_FAILED
            # 얼굴 미검출 시 FACE_NOT_DETECTED

            gen = torch.Generator(device=self.device)
            if req.seed is not None:
                gen.manual_seed(req.seed)

            with torch.inference_mode():
                frames = self.pipe(...)                   # opts.values["guidance_scale"] 등 사용

            self._encode(frames, req.fps, req.output_path)  # 실패 시 ENCODING_FAILED

        except GenerationError:
            raise                                          # 이미 분류된 에러는 그대로
        except torch.cuda.OutOfMemoryError as e:           # RuntimeError 보다 먼저!
            raise GenerationError(ErrorCode.CUDA_OOM, "GPU out of memory") from e
        except OSError as e:
            raise GenerationError(ErrorCode.IO_ERROR, type(e).__name__) from e
        except RuntimeError as e:
            if "CUDA" in str(e):
                raise GenerationError(ErrorCode.CUDA_RUNTIME_ERROR, "CUDA runtime error") from e
            raise                                          # 그 외는 UNKNOWN 으로 처리됨

        return GenerationResult(output_path=req.output_path, ignored_args=opts.ignored)

    def clear_cache(self) -> None:
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ----- helpers -----
    def _validate(self, req: GenerationRequest) -> None:
        if not req.image_paths:
            raise GenerationError(ErrorCode.INVALID_INPUT, "no images")
        if req.choreography not in SUPPORTED_CHOREOGRAPHIES:
            raise GenerationError(ErrorCode.UNSUPPORTED_CHOREOGRAPHY, req.choreography)
        if not MIN_RESOLUTION <= req.resolution <= MAX_RESOLUTION:
            raise GenerationError(ErrorCode.INVALID_INPUT,
                                  f"resolution must be in [{MIN_RESOLUTION}, {MAX_RESOLUTION}]")
```

---

## 10. 통합 전 검증 (스모크 테스트)

서버 없이 `generation.py` 단독(단일 GPU)으로 규약을 지키는지 확인한다. 서버를 띄워서 하는 통합 확인(`POST /generate`, 큐, 타임아웃, VRAM 회수 등)은 서버 담당자가 수행한다.

> 아래 스크립트가 사용하는 에러 코드 이름(`INVALID_INPUT`, `UNSUPPORTED_CHOREOGRAPHY`, `IMAGE_DECODE_FAILED`)은 현재 더미 기준이다. 에러 코드를 재정의했다면 이름을 맞춰서 수정해서 사용한다.

`smoke_test_generation.py` 로 저장해서 실행한다: `python smoke_test_generation.py <GPU index> <image> <audio>`

```python
import os, pickle, sys, tempfile

# worker 와 동일한 환경을 재현
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
image, audio = sys.argv[2], sys.argv[3]

import generation as g                                   # (1) import 가 가볍고 부작용이 없는가
assert "torch" not in sys.modules, "모듈 최상단 torch import 금지"

# (2) resolve_options 규약
r = g.resolve_options({"foo": "bar"})
assert r.ignored == [{"name": "foo", "reason": "unsupported_argument"}]
assert set(r.values) == set(g.MODEL_OPTIONS)
for k, spec in g.MODEL_OPTIONS.items():                  # 잘못된 값 → INVALID_INPUT
    try:
        g.resolve_options({k: "not-a-valid-value-\x00"})
    except g.GenerationError as e:
        assert e.code == g.ErrorCode.INVALID_INPUT
    else:
        assert spec.type is str and spec.choices is None, f"{k}: 잘못된 값이 통과됨"

# (3) 모델 로드 + 정상 생성
m = g.VideoGenerationModel("cuda:0")
m.load()
d = tempfile.mkdtemp()
req = g.GenerationRequest(
    image_paths=[image], audio_path=audio,
    choreography=sorted(g.SUPPORTED_CHOREOGRAPHIES)[0],
    output_path=os.path.join(d, "output.mp4"),
)
pickle.dumps(req)                                        # 프로세스 간 전달 가능한가
res = m.generate(req)
assert res.output_path == req.output_path and os.path.getsize(res.output_path) > 0
m.clear_cache()

# (4) 입력 오류는 GenerationError(1xxx) 로 나오는가
def expect(code, **override):
    bad = g.GenerationRequest(**{**req.__dict__, **override})
    try:
        m.generate(bad)
    except g.GenerationError as e:
        assert e.code == code, (e.code, code)
    else:
        raise AssertionError(f"{code.name} 이 발생해야 함")

expect(g.ErrorCode.INVALID_INPUT, image_paths=[])
expect(g.ErrorCode.UNSUPPORTED_CHOREOGRAPHY, choreography="nope")
expect(g.ErrorCode.INVALID_INPUT, resolution=-1)
open(os.path.join(d, "broken.jpg"), "wb").write(b"not an image")
expect(g.ErrorCode.IMAGE_DECODE_FAILED, image_paths=[os.path.join(d, "broken.jpg")])

# (5) 실패 후에도 재진입 가능한가
m.generate(req)
print("OK")
```

---

## 11. 더미에서 바꿔야 할 부분 요약

`main.py` / `scheduler.py` 는 직접 수정하지 않는다. 아래 표에서 **★** 표시된 항목은 두 파일의 수정이 필요할 수 있으므로, 변경 내용을 서버 담당자에게 알려준다.

| 더미의 내용 | 실제 구현에서 | 알림 |
|---|---|---|
| 파일 최상단 `import torch` (미사용 `import os` 포함) | 제거 후 함수 안에서 import | |
| `ErrorCode` 의 1xxx~3xxx 모델 에러 코드 (더미가 가정한 목록) | 실제 모델에서 발생하는 실패 유형에 맞게 추가/삭제/이름·번호 변경 (§3.1, §5). 4xxx 와 `UNKNOWN` 은 스케줄러용이므로 유지 | ★ 변경한 코드 목록. `main.py` 는 `INVALID_INPUT`, `UNSUPPORTED_CHOREOGRAPHY` 를, `scheduler.py` 의 `RETRY_POLICIES` 는 `CUDA_OOM`, `CUDA_RUNTIME_ERROR`, `MODEL_INFERENCE_FAILED`, `IO_ERROR`, `ENCODING_FAILED` 를 이름으로 참조하므로, 이 이름을 바꾸거나 지우면 수정이 필요함. 새 코드는 재시도 정책(재시도 횟수, 다른 GPU, worker 재시작) 결정 필요 |
| `_simulate` 로 흉내 내던 에러들 (이미지/오디오 디코드, 얼굴 미검출, OOM, 추론 실패, I/O, 인코딩) | 실제 예외를 에러 코드로 변환하는 코드로 교체 (§5 표 기준) | ★ 새로 던지는 코드가 있으면 |
| `generate()` 의 `_simulate` / `_alloc_mb` / `_sleep_sec` 분기 전체 | **삭제** | |
| `load()` 의 `torch.empty(500*1024*1024 ...)` (약 2GB 더미 텐서) | 실제 가중치 로드 | |
| 정상 루틴의 1GB 더미 텐서 + `time.sleep(5)` + `b"DUMMY_MP4_CONTENT"` 기록 | 실제 추론 + mp4 인코딩 | |
| `resolve_options(req.extra)` 결과의 `values` 미사용 | 추론 인자로 사용 | |
| `MODEL_OPTIONS` (`guidance_scale`, `num_inference_steps`, `background`) | 실제 모델 옵션 | |
| `SUPPORTED_CHOREOGRAPHIES` 와 `_validate` 안의 중복 `SUPPORTED` | 목록 하나로 통일 (`main.py` 가 이 값을 import 하므로 변경은 자동 반영됨) | |
| `resolution` 검증 없음 | 검증 추가 | |
| `GenerationRequest` / `GenerationResult` 필드, export 이름 | 가급적 변경하지 않는다 | ★ 필드를 추가/삭제하거나 이름을 바꿀 때 (`main.py` 가 `GenerationRequest` 필드를 직접 채우고, worker 는 `GenerationResult` 의 3개 필드만 전달함) |
| 주석에 남은 `[cite: …]` 표기 | 제거 | |
| `main.py` 의 `JOB_TIMEOUT=30`, `LOAD_VRAM_MB=INFER_VRAM_MB=5000` (더미 기준 값) | 실측 값으로 설정 (§7) | ★ 실측한 값 |