"""
main.py
-------
동영상 생성 API 서버.

실행 (반드시 worker 1개: GPU 스케줄러가 이 프로세스 안에서 GPU worker 들을 관리한다)
    pip install fastapi uvicorn python-multipart nvidia-ml-py
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1

요청 예시 (guidance_scale 은 모델 옵션, foo 는 정의되지 않은 인자)
    curl -i -X POST http://localhost:8000/generate \
      -F "images=@a.jpg" -F "images=@b.jpg" -F "audio=@voice.wav" \
      -F "choreography=dance_a" -F "fps=25" \
      -F "guidance_scale=5.0" -F "foo=bar" \
      -o result.mp4

정의되지 않은 인자는 실패시키지 않고 무시하며, 응답 헤더로 알려준다.
    X-Ignored-Args: [{"name":"foo","reason":"unsupported_argument"}]
    X-Ignored-Args-Count: 1
(실패 응답에서는 JSON detail.ignored_args 로 동일한 정보를 제공)
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from starlette.datastructures import UploadFile as StarletteUploadFile

from generation import (MODEL_OPTIONS, SUPPORTED_CHOREOGRAPHIES, ErrorCode,
                        GenerationError, GenerationRequest, resolve_options)
from scheduler import GPUScheduler, NvmlMonitor, SchedulerError

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

NUM_GPUS = int(os.environ.get("NUM_GPUS", 4))
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", 32))
JOB_TIMEOUT = int(os.environ.get("JOB_TIMEOUT", 30))
LOAD_VRAM_MB = float(os.environ.get("LOAD_VRAM_MB", 5000))     # 모델 로드에 필요한 free VRAM (spawn 전 확인)
INFER_VRAM_MB = float(os.environ.get("INFER_VRAM_MB", 5000))   # 추론에 필요한 free VRAM (작업 배정 전 확인)
HEALTH_CHECK_INTERVAL = float(os.environ.get("HEALTH_CHECK_INTERVAL", 300))
ENABLE_NVML = os.environ.get("DISABLE_NVML", "0") != "1"   # GPU 없는 개발환경에서만 DISABLE_NVML=1
WORK_ROOT = os.environ.get("WORK_ROOT", tempfile.gettempdir())

MAX_IMAGES = 10
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_AUDIO_BYTES = 50 * 1024 * 1024
MAX_REPORTED_IGNORED = 20       # 헤더 크기 제한

# API 가 직접 정의한 필드. 이 외의 form 필드는 'extra' 로 모델에 전달되고,
# 모델이 정의하지 않은 것은 ignored 로 회신된다.
CORE_FIELDS = {"images", "audio", "choreography", "fps", "resolution", "seed"}

scheduler = GPUScheduler(
    num_gpus=NUM_GPUS,
    max_queue=MAX_QUEUE,
    job_timeout=JOB_TIMEOUT,
    load_vram_mb=LOAD_VRAM_MB,
    infer_vram_mb=INFER_VRAM_MB,
    health_check_interval=HEALTH_CHECK_INTERVAL,
    monitor=NvmlMonitor(enabled=ENABLE_NVML),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await scheduler.start()   # GPU 별 worker 기동 + 모델 로딩 (서버 시작 시 1회)
    yield
    await scheduler.stop()


app = FastAPI(title="Video Generation API", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _http_status(code: ErrorCode) -> int:
    if 1000 <= code < 2000:
        return 422                     # 입력 문제
    if code in (ErrorCode.QUEUE_FULL, ErrorCode.NO_GPU_AVAILABLE):
        return 503
    if code == ErrorCode.TIMEOUT:
        return 504
    return 500


def _error(code: ErrorCode, message: str, attempts: int = 0,
           ignored: Optional[list] = None) -> HTTPException:
    return HTTPException(
        status_code=_http_status(code),
        detail={"error_code": int(code), "error_name": code.name, "message": message,
                "attempts": attempts, "ignored_args": ignored or []},
    )


def _merge_ignored(*lists: list) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for lst in lists:
        for item in lst:
            if item["name"] not in seen:
                seen.add(item["name"])
                out.append(item)
    return out


async def _collect_extra_args(request: Request) -> tuple[dict, list[dict]]:
    """
    선언되지 않은 form 필드 / 파일 / query 파라미터를 수집한다. 절대 실패시키지 않는다.
      - 선언되지 않은 일반 form 필드 -> extra (모델로 전달, 모델이 모르면 모델 쪽에서 ignored 처리)
      - 선언되지 않은 파일 필드, query 파라미터 -> API 레벨에서 바로 ignored
    FastAPI 가 이미 파싱한 form 은 Starlette 가 캐시하므로 여기서 다시 읽어도 안전하다.
    """
    extra: dict = {}
    ignored: list[dict] = []

    form = await request.form()
    for key, value in form.multi_items():
        if key in CORE_FIELDS:
            continue
        if isinstance(value, StarletteUploadFile):
            ignored.append({"name": key, "reason": "unsupported_file_field"})
        else:
            extra[key] = value          # 같은 key 가 반복되면 마지막 값 사용

    for key in request.query_params.keys():
        ignored.append({"name": key, "reason": "unsupported_query_param"})

    return extra, ignored


def _ignored_header(ignored: list[dict]) -> dict[str, str]:
    if not ignored:
        return {}
    shown = [{"name": i["name"][:64], "reason": i["reason"]}
             for i in ignored[:MAX_REPORTED_IGNORED]]
    return {
        # ensure_ascii=True 라서 비ASCII/개행도 이스케이프되어 헤더에 안전하다
        "X-Ignored-Args": json.dumps(shown, ensure_ascii=True, separators=(",", ":")),
        "X-Ignored-Args-Count": str(len(ignored)),
    }


class _UploadTooLarge(Exception):
    pass


async def _save_upload(upload: UploadFile, dest: str, max_bytes: int) -> None:
    size = 0
    with open(dest, "wb") as f:
        while chunk := await upload.read(1 << 20):
            size += len(chunk)
            if size > max_bytes:
                raise _UploadTooLarge(f"{upload.filename} exceeds {max_bytes // (1 << 20)}MB")
            f.write(chunk)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.post("/generate")
async def generate(
    request: Request,
    images: list[UploadFile] = File(..., description="입력 이미지 여러 장"),
    audio: UploadFile = File(..., description="음성 녹음 파일"),
    choreography: str = Form(..., description=f"안무 선택: {sorted(SUPPORTED_CHOREOGRAPHIES)}"),
    fps: int = Form(25),
    resolution: int = Form(512),
    seed: Optional[int] = Form(None),
):
    # ---- 0) 정의되지 않은 인자 수집 (실패 X) + 모델 옵션 사전 검증 ----
    extra, api_ignored = await _collect_extra_args(request)
    try:
        resolved = resolve_options(extra)   # 정의된 옵션의 '값'이 잘못된 경우만 실패
    except GenerationError as e:
        # 값 오류로 실패하더라도 정의되지 않은 인자는 함께 알려준다
        unknown = [{"name": k, "reason": "unsupported_argument"}
                   for k in extra if k not in MODEL_OPTIONS]
        raise _error(e.code, e.message, ignored=_merge_ignored(api_ignored, unknown))
    ignored = _merge_ignored(api_ignored, resolved.ignored)
    if ignored:
        log.info("ignored args: %s", [i["name"] for i in ignored])

    # ---- 1) 가벼운 입력 검증 (GPU 를 쓰기 전에 걸러낸다) ----
    if not 1 <= len(images) <= MAX_IMAGES:
        raise _error(ErrorCode.INVALID_INPUT, f"images must be 1..{MAX_IMAGES}", ignored=ignored)
    if choreography not in SUPPORTED_CHOREOGRAPHIES:
        raise _error(ErrorCode.UNSUPPORTED_CHOREOGRAPHY, f"unknown choreography: {choreography}",
                     ignored=ignored)
    if not 1 <= fps <= 60:
        raise _error(ErrorCode.INVALID_INPUT, "fps must be 1..60", ignored=ignored)

    workdir = tempfile.mkdtemp(prefix="vidgen_", dir=WORK_ROOT)
    keep_workdir = False   # 성공 시 응답 전송 후 삭제하도록 BackgroundTask 에 위임
    try:
        # ---- 2) 업로드 파일을 디스크에 저장 (worker 에는 경로만 전달) ----
        try:
            image_paths: list[str] = []
            for i, img in enumerate(images):
                ext = os.path.splitext(img.filename or "")[1] or ".jpg"
                path = os.path.join(workdir, f"img_{i:02d}{ext}")
                await _save_upload(img, path, MAX_IMAGE_BYTES)
                image_paths.append(path)

            audio_ext = os.path.splitext(audio.filename or "")[1] or ".wav"
            audio_path = os.path.join(workdir, f"audio{audio_ext}")
            await _save_upload(audio, audio_path, MAX_AUDIO_BYTES)
        except _UploadTooLarge as e:
            raise _error(ErrorCode.INVALID_INPUT, str(e), ignored=ignored)

        req = GenerationRequest(
            image_paths=image_paths,
            audio_path=audio_path,
            choreography=choreography,
            output_path=os.path.join(workdir, "output.mp4"),
            fps=fps,
            resolution=resolution,
            seed=seed,
            extra=extra,
        )

        # ---- 3) 스케줄러에 위임 (재시도 / 에러 처리는 scheduler 담당) ----
        try:
            result = await scheduler.generate(req)
        except SchedulerError as e:
            log.error("generation failed: %s", e)
            raise _error(e.code, e.message, e.attempts, ignored=ignored)

        # 모델이 보고한 ignored 와 API 레벨 ignored 를 합쳐 응답 헤더로 회신
        ignored = _merge_ignored(ignored, result.ignored_args)
        headers = _ignored_header(ignored)
        if result.warnings:
            headers["X-Warnings"] = json.dumps(result.warnings[:10], ensure_ascii=True,
                                               separators=(",", ":"))

        keep_workdir = True
        return FileResponse(
            result.output_path,
            media_type="video/mp4",
            filename="result.mp4",
            headers=headers,
            background=BackgroundTask(shutil.rmtree, workdir, ignore_errors=True),
        )
    finally:
        if not keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)


@app.get("/health")
async def health():
    return scheduler.status()


@app.get("/queue")
async def queue():
    return scheduler.queue_snapshot()