"""
generation.py
-------------
동영상 생성 모델 객체 (DUMMY 구현).

실제 모델로 교체할 때는 VideoGenerationModel 의 load() / generate() 내부만 바꾸면 된다.
(주의: main.py 도 이 모듈을 import 하므로 torch 같은 무거운 import 는 load() 안에서 할 것)

scheduler / main 이 의존하는 것
  - ErrorCode / GenerationError
  - GenerationRequest / GenerationResult
  - MODEL_OPTIONS / resolve_options()   <- 모델 전용 옵션 선언 및 검증
  - VideoGenerationModel(device).load() / .generate(request) / .clear_cache()
"""
from __future__ import annotations

import os
import sys
import torch
import time
import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# Error codes
# --------------------------------------------------------------------------- #
class ErrorCode(IntEnum):
    # 1xxx: 입력 문제 (재시도 해도 소용없음 -> 즉시 클라이언트에 반환)
    INVALID_INPUT = 1001
    IMAGE_DECODE_FAILED = 1002
    AUDIO_DECODE_FAILED = 1003
    FACE_NOT_DETECTED = 1004
    UNSUPPORTED_CHOREOGRAPHY = 1005

    # 2xxx: GPU / 모델 추론 문제 (일시적일 수 있음)
    CUDA_OOM = 2001
    CUDA_RUNTIME_ERROR = 2002        # CUDA context 가 오염됐을 수 있음 -> worker 재시작 필요
    MODEL_INFERENCE_FAILED = 2003    # NaN 등 일시적 추론 실패

    # 3xxx: I/O 문제
    IO_ERROR = 3001
    ENCODING_FAILED = 3002

    # 4xxx: scheduler 가 직접 발생시키는 코드
    TIMEOUT = 4001
    WORKER_CRASHED = 4002
    QUEUE_FULL = 4003
    NO_GPU_AVAILABLE = 4004

    UNKNOWN = 9999


class GenerationError(Exception):
    """모델 쪽에서 발생시키는 에러. 항상 ErrorCode 를 달고 나온다."""

    def __init__(self, code: ErrorCode, message: str = ""):
        super().__init__(f"[{code.name}({code.value})] {message}")
        self.code = code
        self.message = message


# --------------------------------------------------------------------------- #
# Model-specific options
#   클라이언트가 보낸 '정의되지 않은 인자'는 에러가 아니라 ignored 로 분류해서 돌려준다.
#   '정의된 옵션인데 값이 잘못된 경우'만 INVALID_INPUT 으로 실패시킨다.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OptionSpec:
    type: type
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    choices: Optional[tuple] = None
    description: str = ""


MODEL_OPTIONS: dict[str, OptionSpec] = {
    "guidance_scale":      OptionSpec(float, 3.0, min=0.0, max=20.0, description="CFG scale"),
    "num_inference_steps": OptionSpec(int, 30, min=1, max=100, description="denoising steps"),
    "background":          OptionSpec(str, "original", choices=("original", "blur", "studio")),
}


@dataclass
class ResolvedOptions:
    values: dict                                            # 기본값이 채워진 최종 옵션
    ignored: list[dict] = field(default_factory=list)       # [{"name":..., "reason":...}]


def resolve_options(extra: Optional[dict]) -> ResolvedOptions:
    values = {k: spec.default for k, spec in MODEL_OPTIONS.items()}
    ignored: list[dict] = []

    for key, raw in (extra or {}).items():
        spec = MODEL_OPTIONS.get(key)
        if spec is None:
            ignored.append({"name": str(key), "reason": "unsupported_argument"})
            continue
        try:
            value = spec.type(raw)
        except (TypeError, ValueError):
            raise GenerationError(
                ErrorCode.INVALID_INPUT,
                f"option '{key}': cannot parse {str(raw)[:50]!r} as {spec.type.__name__}")
        if spec.min is not None and value < spec.min or spec.max is not None and value > spec.max:
            raise GenerationError(
                ErrorCode.INVALID_INPUT, f"option '{key}' must be in [{spec.min}, {spec.max}]")
        if spec.choices is not None and value not in spec.choices:
            raise GenerationError(
                ErrorCode.INVALID_INPUT, f"option '{key}' must be one of {list(spec.choices)}")
        values[key] = value

    return ResolvedOptions(values=values, ignored=ignored)


# --------------------------------------------------------------------------- #
# Request / Result
# --------------------------------------------------------------------------- #
SUPPORTED_CHOREOGRAPHIES = {"dance_a", "dance_b", "dance_c", "kpop_01", "kpop_02"}


@dataclass
class GenerationRequest:
    """프로세스 간 pickle 로 전달되므로 파일 '경로'만 담는다 (바이트 X)."""
    image_paths: list[str]
    audio_path: str
    choreography: str
    output_path: str
    fps: int = 25
    resolution: int = 512
    seed: Optional[int] = None
    extra: dict = field(default_factory=dict)   # API 에 정의되지 않은 추가 인자 (모델이 알아서 해석)


@dataclass
class GenerationResult:
    output_path: str
    ignored_args: list[dict] = field(default_factory=list)   # 사용되지 않은 인자
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Model (dummy)
# --------------------------------------------------------------------------- #
class VideoGenerationModel:
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.model_tensor = None

    def load(self) -> None:
        """모델 가중치 로드 시뮬레이션"""
        time.sleep(2.0)
        # 기본 모델 가중치 VRAM 점유 (약 2GB)
        self.model_tensor = torch.empty(500 * 1024 * 1024, dtype=torch.float32, device=self.device)

    def generate(self, req: GenerationRequest) -> GenerationResult:
        self._validate(req)
        opts = resolve_options(req.extra)
        
        sim_type = req.extra.get("_simulate")

        # --- 1xxx 에러 그룹 (입력/분석 실패) ---
        if sim_type == "IMAGE_DECODE_FAILED":
            raise GenerationError(ErrorCode.IMAGE_DECODE_FAILED, "실제: PIL.UnidentifiedImageError 감지됨")
        elif sim_type == "AUDIO_DECODE_FAILED":
            raise GenerationError(ErrorCode.AUDIO_DECODE_FAILED, "실제: librosa 오디오 디코드 실패")
        elif sim_type == "FACE_NOT_DETECTED":
            raise GenerationError(ErrorCode.FACE_NOT_DETECTED, "실제: cv2 얼굴 인식 결과 0건")

        # --- 2xxx 에러 그룹 (GPU 및 추론 에러) ---
        elif sim_type == "CUDA_OOM":
            # [OOM 스위칭 테스트 핵심 로직]
            # 지정된 크기만큼 실제 VRAM을 할당하여 장착된 GPU 스펙에 따라 
            # 실제로 OutOfMemoryError 가 터지거나, 메모리가 넉넉한 GPU에서는 성공하도록 만듭니다.
            alloc_mb = int(req.extra.get("_alloc_mb", 30000))
            try:
                num_elements = (alloc_mb * 1024 * 1024) // 4
                huge_tensor = torch.empty(num_elements, dtype=torch.float32, device=self.device)
                del huge_tensor  # 해당 GPU의 VRAM이 충분하여 성공한 경우
            except torch.cuda.OutOfMemoryError:
                # 메모리가 부족한 GPU에서는 실제 OOM이 발생하며, 이를 에러코드로 변환하여 스케줄러로 던짐[cite: 1]
                raise GenerationError(ErrorCode.CUDA_OOM, f"{alloc_mb}MB 실제 할당 시도 중 OOM 발생")
                
        elif sim_type == "CUDA_RUNTIME_ERROR":
            try:
                # 실제 CUDA device-side assert 발생 흉내
                raise RuntimeError("CUDA error: device-side assert triggered")
            except RuntimeError as e:
                if "CUDA" in str(e):
                    raise GenerationError(ErrorCode.CUDA_RUNTIME_ERROR, str(e))
                    
        elif sim_type == "MODEL_INFERENCE_FAILED":
            raise GenerationError(ErrorCode.MODEL_INFERENCE_FAILED, "텐서 연산 중 NaN 값 발생")

        # --- 3xxx 에러 그룹 (I/O 및 시스템) ---
        elif sim_type == "IO_ERROR":
            try:
                # 존재하지 않는 경로를 읽으려 시도하여 실제 OSError 발생 유도
                with open("/path/to/non_existent.file", "r") as f:
                    f.read()
            except OSError as e:
                raise GenerationError(ErrorCode.IO_ERROR, f"파일 열기 실패: {str(e)}")
                
        elif sim_type == "ENCODING_FAILED":
            raise GenerationError(ErrorCode.ENCODING_FAILED, "ffmpeg 프로세스 비정상 종료 (exit code 1)")

        # --- 4xxx 에러 그룹 (워커 비정상 상태 유발) ---
        elif sim_type == "WORKER_CRASH":
            # 파이썬 워커 프로세스를 C 세그폴트나 강제 종료처럼 즉시 죽임 
            # -> 스케줄러가 WORKER_CRASHED(4002)로 감지하고 워커 재기동 및 재시도 수행[cite: 3]
            sys.exit(-1)
            
        elif sim_type == "TIMEOUT":
            # 설정된 timeout 시간 이상 슬립하여 타임아웃 강제 유발
            sleep_time = int(req.extra.get("_sleep_sec", 35))
            time.sleep(sleep_time)

        # --- 정상 처리 루틴 ---
        try:
            # 추론 중 사용되는 1GB VRAM 시뮬레이션
            infer_tensor = torch.empty(250 * 1024 * 1024, dtype=torch.float32, device=self.device)
            time.sleep(5)
            
            with open(req.output_path, "wb") as f:
                f.write(b"DUMMY_MP4_CONTENT")
        except OSError as e:
            raise GenerationError(ErrorCode.IO_ERROR, str(e))
        finally:
            if 'infer_tensor' in locals():
                del infer_tensor

        return GenerationResult(output_path=req.output_path, ignored_args=opts.ignored)

    def clear_cache(self) -> None:
        torch.cuda.empty_cache()

    def _validate(self, req: GenerationRequest) -> None:
        if not req.image_paths:
            raise GenerationError(ErrorCode.INVALID_INPUT, "no images")
        # 지원되지 않는 안무는 1005 (UNSUPPORTED_CHOREOGRAPHY) 반환[cite: 1]
        SUPPORTED = {"dance_a", "dance_b", "dance_c", "kpop_01", "kpop_02"}
        if req.choreography not in SUPPORTED:
            raise GenerationError(ErrorCode.UNSUPPORTED_CHOREOGRAPHY, req.choreography)