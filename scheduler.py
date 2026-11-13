"""
scheduler.py
------------
GPU 당 1개의 worker 프로세스를 띄우고, 각 worker 가 모델을 한 번만 로드해 상주시키는 스케줄러.

구조
  FastAPI(asyncio) ──generate()──> GPUScheduler ──task_q[gpu]──> worker process (모델 상주)
                                        ^   |                          |
                                        |   └─ NVML health check(5분)   |
                                        └────── result_q (공유) <───────┘

VRAM 기준 2개
  load_vram_mb  : 모델을 '로드하기 위해' 필요한 free VRAM.  worker 를 spawn 하기 직전에 확인한다.
                  부족하면 worker 를 띄우지 않고 BUSY(low_vram_load) 로 대기 (모델 미로딩 상태).
  infer_vram_mb : '추론을 위해' 모델 로드 후에도 남아 있어야 하는 free VRAM.  작업 배정 직전에 확인한다.
                  부족하면 그 GPU 는 BUSY(low_vram_infer) 로 두고 다른 GPU 를 고른다.

GPU(worker) 상태
  IDLE  : 모델이 로드되어 있고 작업을 받을 수 있음
  BUSY  : reason = job(작업 중) | low_vram_infer(추론 여유 VRAM 부족, 모델 로드됨)
                 | low_vram_load(로드용 VRAM 부족, 모델 미로드 / worker 없음)
  DEAD  : GPU 자체 오류(NVML 조회 실패) 또는 worker 재시작 한도 초과
  (STARTING : worker 기동 / 모델 로딩 중)

  health check 는 health_check_interval(기본 300초)마다 모든 GPU 를 점검한다.
    - NVML 오류                              -> DEAD (worker kill, 진행 중 작업은 WORKER_CRASHED 로 재시도)
    - DEAD 인데 NVML 정상으로 복구            -> (로드용 VRAM 확인 후) worker 재기동 -> 모델 로딩 후 IDLE
    - IDLE + free < infer_vram_mb            -> BUSY(low_vram_infer)
    - BUSY(low_vram_infer) + 회복            -> IDLE
    - BUSY(low_vram_load) + free >= load_vram_mb -> worker spawn (모델 로딩)
  작업 중(BUSY/job)인 GPU 는 VRAM 이 낮은 게 정상이므로 VRAM 판정에서 제외한다 (NVML 오류 판정만 수행).

에러 코드별 재시도 정책은 RETRY_POLICIES 참고.
"""
from __future__ import annotations

import asyncio
import enum
import logging
import multiprocessing as mp
import os
import queue
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from generation import (ErrorCode, GenerationError, GenerationRequest,
                        GenerationResult)

log = logging.getLogger("scheduler")


# --------------------------------------------------------------------------- #
# Retry policy
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 0           # 해당 에러 코드에 대해 허용하는 재시도 횟수 (0 = 재시도 없음)
    switch_gpu: bool = False       # 재시도 시 직전 GPU 를 피할지
    restart_worker: bool = False   # 에러 후 worker 프로세스를 재시작할지 (CUDA context 오염 등)
    backoff_sec: float = 0.0       # 재시도 전 대기 시간


RETRY_POLICIES: dict[ErrorCode, RetryPolicy] = {
    # GPU / 추론 문제 -> 다른 GPU 에서 재시도
    ErrorCode.CUDA_OOM:               RetryPolicy(max_retries=2, switch_gpu=True, backoff_sec=1.0),
    ErrorCode.CUDA_RUNTIME_ERROR:     RetryPolicy(max_retries=2, switch_gpu=True, restart_worker=True),
    ErrorCode.MODEL_INFERENCE_FAILED: RetryPolicy(max_retries=1, backoff_sec=0.5),
    # I/O 문제 -> 같은 GPU 에서 재시도해도 무방
    ErrorCode.IO_ERROR:               RetryPolicy(max_retries=2, backoff_sec=1.0),
    ErrorCode.ENCODING_FAILED:        RetryPolicy(max_retries=1, backoff_sec=0.5),
    # scheduler 레벨 문제
    ErrorCode.TIMEOUT:                RetryPolicy(max_retries=1, switch_gpu=True, restart_worker=True),
    ErrorCode.WORKER_CRASHED:         RetryPolicy(max_retries=2, switch_gpu=True, restart_worker=True),
    ErrorCode.UNKNOWN:                RetryPolicy(max_retries=1, switch_gpu=True),
    # 1xxx (입력 오류) 는 등록하지 않음 -> DEFAULT_POLICY(재시도 없음) 적용
}
DEFAULT_POLICY = RetryPolicy(max_retries=0)


class SchedulerError(Exception):
    """재시도 후에도 실패했거나, 재시도 대상이 아닌 에러. main.py 에서 HTTP 응답으로 변환된다."""

    def __init__(self, code: ErrorCode, message: str = "", attempts: int = 0):
        super().__init__(f"[{code.name}({code.value})] {message}")
        self.code = code
        self.message = message
        self.attempts = attempts


# --------------------------------------------------------------------------- #
# NVML monitor
# --------------------------------------------------------------------------- #
@dataclass
class GpuProbe:
    ok: bool
    free_mb: Optional[float] = None   # NVML 비활성화 시 None
    error: str = ""


class NvmlMonitor:
    """pynvml 래퍼. (pip install nvidia-ml-py)  enabled=False 면 항상 정상으로 취급 (개발/테스트용)."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._pynvml = None
        self._inited = False
        self._lock = threading.Lock()

    def _ensure_init(self) -> None:
        with self._lock:
            if self._inited:
                return
            import pynvml  # lazy import
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._inited = True

    def _reset(self) -> None:
        with self._lock:
            if self._inited:
                try:
                    self._pynvml.nvmlShutdown()
                except Exception:  # noqa: BLE001
                    pass
            self._inited = False

    def probe(self, gpu_id: int) -> GpuProbe:
        """blocking 호출 (run_in_executor 로 실행할 것)."""
        if not self.enabled:
            return GpuProbe(ok=True)
        try:
            self._ensure_init()
            nv = self._pynvml
            handle = nv.nvmlDeviceGetHandleByIndex(gpu_id)
            mem = nv.nvmlDeviceGetMemoryInfo(handle)
            return GpuProbe(ok=True, free_mb=mem.free / (1024 ** 2))
        except Exception as e:  # noqa: BLE001  (NVMLError, ImportError 등 전부 GPU 이상으로 간주)
            self._reset()   # 다음 probe 때 NVML 재초기화 (드라이버 복구 대응)
            return GpuProbe(ok=False, error=f"{type(e).__name__}: {e}")

    def shutdown(self) -> None:
        self._reset()


# --------------------------------------------------------------------------- #
# Worker process (GPU 1장당 1개)
# --------------------------------------------------------------------------- #
def _worker_main(gpu_id: int, epoch: int, task_q, result_q) -> None:
    # torch 를 import 하기 전에 가시 GPU 를 제한 -> 프로세스 안에서는 항상 "cuda:0"
    # PCI_BUS_ID 순서로 고정해야 NVML(pynvml) 의 GPU index 와 일치한다.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from generation import VideoGenerationModel  # noqa: E402  (env 설정 이후 import)

    base = {"gpu_id": gpu_id, "epoch": epoch}
    try:
        model = VideoGenerationModel(device="cuda:0")
        model.load()
    except Exception as e:  # noqa: BLE001
        result_q.put({**base, "type": "load_failed", "message": repr(e)})
        return
    result_q.put({**base, "type": "ready"})

    while True:
        task = task_q.get()
        if task is None:  # shutdown sentinel
            break
        task_id, request = task
        try:
            res = model.generate(request)
            result_q.put({**base, "type": "result", "task_id": task_id, "ok": True,
                          "output_path": res.output_path,
                          "ignored_args": res.ignored_args, "warnings": res.warnings})
        except GenerationError as e:
            result_q.put({**base, "type": "result", "task_id": task_id,
                          "ok": False, "code": int(e.code), "message": e.message})
        except Exception as e:  # noqa: BLE001
            result_q.put({**base, "type": "result", "task_id": task_id,
                          "ok": False, "code": int(ErrorCode.UNKNOWN), "message": repr(e)})
        finally:
            # caching allocator 가 잡고 있는 VRAM 을 반환해야 NVML 의 free VRAM 판정이 정확하다.
            try:
                model.clear_cache()
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
# Scheduler
# --------------------------------------------------------------------------- #
class WorkerState(enum.Enum):
    STARTING = "starting"   # 프로세스 기동 / 모델 로딩 중
    IDLE = "idle"
    BUSY = "busy"           # busy_reason: "job" | "low_vram_infer" | "low_vram_load"
    DEAD = "dead"           # dead_reason: "gpu_error: ..." | "restart_limit_exceeded"


@dataclass
class _Worker:
    gpu_id: int
    process: Optional[mp.Process] = None
    task_q: Optional[object] = None
    state: WorkerState = WorkerState.STARTING
    busy_reason: str = ""
    dead_reason: str = ""
    free_mb: Optional[float] = None   # 마지막 NVML 조회 값
    epoch: int = 0          # 재시작될 때마다 증가 (이전 프로세스의 stale 메시지 / release 무시용)
    restarts: int = 0
    inflight: Optional[str] = None
    jobs_done: int = 0


@dataclass
class _Job:
    """GET /queue 용 요청 추적 정보. (파일 경로 등 민감 정보는 담지 않는다)"""
    job_id: str
    choreography: str
    num_images: int
    submitted_at: float                 # time.monotonic()
    waiting_since: float                # 대기 시작 시각 (재시도로 다시 대기하면 갱신)
    state: str = "waiting"              # "waiting" | "running"
    gpu_id: Optional[int] = None
    started_at: Optional[float] = None  # 현재 시도의 실행 시작 시각
    attempts: int = 0


class GPUScheduler:
    def __init__(
        self,
        num_gpus: int = 4,
        max_queue: int = 32,
        job_timeout: float = 600.0,
        max_total_attempts: int = 4,
        max_restarts: int = 5,
        startup_timeout: float = 600.0,
        load_vram_mb: float = 20000,      # 모델 로드에 필요한 free VRAM (모델 크기에 맞게 설정!)
        infer_vram_mb: float = 40000,     # 추론에 필요한 free VRAM (모델 로드 후 기준)
        health_check_interval: float = 300.0,
        monitor: Optional[NvmlMonitor] = None,
    ):
        self._num_gpus = num_gpus
        self._max_queue = max_queue
        self._job_timeout = job_timeout
        self._max_attempts = max_total_attempts
        self._max_restarts = max_restarts
        self._startup_timeout = startup_timeout
        self._load_vram_mb = load_vram_mb
        self._infer_vram_mb = infer_vram_mb
        self._health_interval = health_check_interval
        self._nvml = monitor or NvmlMonitor(enabled=True)

        self._ctx = mp.get_context("spawn")  # CUDA 는 fork 와 호환되지 않음
        self._result_q = self._ctx.Queue()
        self._workers: list[_Worker] = [_Worker(gpu_id=i) for i in range(num_gpus)]
        self._futures: dict[str, asyncio.Future] = {}
        self._waiting = 0
        self._jobs: dict[str, _Job] = {}     # 큐에 들어와 있거나 실행 중인 모든 요청
        self._stopping = False

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._cond: Optional[asyncio.Condition] = None
        self._collector: Optional[threading.Thread] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._bg_tasks: set[asyncio.Task] = set()

    # ---------------------------- lifecycle ---------------------------- #
    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._cond = asyncio.Condition()

        # 기동 전 1회 GPU 점검
        #   - NVML 오류            -> worker 를 띄우지 않고 DEAD 로 시작
        #   - free < load_vram_mb  -> worker 를 띄우지 않고 BUSY(low_vram_load) 로 시작
        #   (둘 다 health check 에서 해소되면 자동으로 worker 를 띄운다)
        for w in self._workers:
            probe = await self._loop.run_in_executor(None, self._nvml.probe, w.gpu_id)
            w.free_mb = probe.free_mb
            if probe.ok:
                self._spawn_or_park(w, probe)
            else:
                w.state = WorkerState.DEAD
                w.dead_reason = f"gpu_error: {probe.error}"
                log.error("gpu=%d unusable at startup: %s", w.gpu_id, probe.error)

        self._collector = threading.Thread(target=self._collect_loop, daemon=True,
                                           name="scheduler-collector")
        self._collector.start()
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        self._health_task = asyncio.create_task(self._health_loop())

        # 모든 worker 가 모델 로딩을 마칠 때까지 대기 (일부 실패해도 나머지로 서비스 시작)
        deadline = time.monotonic() + self._startup_timeout
        while time.monotonic() < deadline:
            if all(w.state != WorkerState.STARTING for w in self._workers):
                break
            await asyncio.sleep(0.5)
        log.info("scheduler started: %s", self.status())

    async def stop(self) -> None:
        self._stopping = True
        for t in (self._monitor_task, self._health_task):
            if t:
                t.cancel()
        for w in self._workers:
            if w.process and w.process.is_alive():
                try:
                    w.task_q.put(None)
                except Exception:  # noqa: BLE001
                    pass
        for w in self._workers:
            if w.process:
                await self._loop.run_in_executor(None, w.process.join, 5)
                if w.process.is_alive():
                    w.process.kill()
        self._nvml.shutdown()

    # ---------------------------- public API --------------------------- #
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """성공 시 GenerationResult 반환, 실패 시 SchedulerError."""
        if self._waiting >= self._max_queue:
            raise SchedulerError(ErrorCode.QUEUE_FULL, "too many pending requests")

        now = time.monotonic()
        job = _Job(job_id=uuid.uuid4().hex[:12], choreography=request.choreography,
                   num_images=len(request.image_paths), submitted_at=now, waiting_since=now)
        self._jobs[job.job_id] = job
        try:
            return await self._generate_inner(request, job)
        finally:
            self._jobs.pop(job.job_id, None)   # 성공 / 실패 / 취소 어떤 경우에도 제거

    async def _generate_inner(self, request: GenerationRequest, job: _Job) -> GenerationResult:
        counts: Counter = Counter()
        exclude: set[int] = set()
        attempts = 0

        while True:
            job.state = "waiting"
            job.gpu_id = None
            job.started_at = None
            job.waiting_since = time.monotonic()
            w = await self._acquire_gpu(exclude)
            epoch = w.epoch
            # 추론 직전 VRAM / GPU 상태 확인. 부적합하면 그 GPU 는 BUSY/DEAD 로 빠지고
            # 다른 GPU 를 다시 고른다 (이 경우는 시도 횟수(attempts)를 소모하지 않음)
            if not await self._precheck_infer(w, epoch):
                continue
            attempts += 1
            job.attempts = attempts
            job.state = "running"
            job.gpu_id = w.gpu_id
            job.started_at = time.monotonic()
            msg = await self._run_on_worker(w, epoch, request)

            if msg["ok"]:
                w.restarts = 0
                w.jobs_done += 1
                self._release(w, epoch, restart=False)
                return GenerationResult(
                    output_path=msg["output_path"],
                    ignored_args=msg.get("ignored_args", []),
                    warnings=msg.get("warnings", []),
                )

            code = self._to_code(msg["code"])
            policy = RETRY_POLICIES.get(code, DEFAULT_POLICY)
            self._release(w, epoch, restart=policy.restart_worker)
            counts[code] += 1
            log.warning("gpu=%d attempt=%d failed: %s (%s)", w.gpu_id, attempts,
                        code.name, msg.get("message"))

            if counts[code] > policy.max_retries or attempts >= self._max_attempts:
                raise SchedulerError(code, msg.get("message", ""), attempts)

            if policy.switch_gpu:
                exclude.add(w.gpu_id)
            if policy.backoff_sec:
                await asyncio.sleep(policy.backoff_sec)

    def status(self) -> dict:
        return {
            "waiting": self._waiting,
            "gpus": [
                {
                    "gpu": w.gpu_id,
                    "state": w.state.value,
                    "reason": w.busy_reason or w.dead_reason or None,
                    "free_vram_mb": None if w.free_mb is None else round(w.free_mb),
                    "jobs_done": w.jobs_done,
                    "restarts": w.restarts,
                }
                for w in self._workers
            ],
        }

    def queue_snapshot(self) -> dict:
        """현재 큐(대기 중) + 실행 중인 요청 목록. 대기열은 FIFO 순서로 position 이 붙는다."""
        now = time.monotonic()
        jobs = list(self._jobs.values())
        waiting = sorted((j for j in jobs if j.state == "waiting"),
                         key=lambda j: j.waiting_since)
        running = sorted((j for j in jobs if j.state == "running"),
                         key=lambda j: j.started_at or 0.0)
        return {
            "max_queue": self._max_queue,
            "queue_full": self._waiting >= self._max_queue,
            "waiting_count": len(waiting),
            "running_count": len(running),
            "waiting": [
                {
                    "position": i,
                    "job_id": j.job_id,
                    "choreography": j.choreography,
                    "num_images": j.num_images,
                    "attempts": j.attempts,   # 이미 실행해 본 횟수 (재시도로 다시 대기 중이면 >0)
                    "waited_sec": round(now - j.waiting_since, 1),
                    "total_sec": round(now - j.submitted_at, 1),
                }
                for i, j in enumerate(waiting, start=1)
            ],
            "running": [
                {
                    "job_id": j.job_id,
                    "gpu": j.gpu_id,
                    "choreography": j.choreography,
                    "num_images": j.num_images,
                    "attempt": j.attempts,
                    "running_sec": round(now - (j.started_at or now), 1),
                    "timeout_in_sec": round(max(0.0, self._job_timeout - (now - (j.started_at or now))), 1),
                    "total_sec": round(now - j.submitted_at, 1),
                }
                for j in running
            ],
        }

    # ------------------------- GPU acquire / release ------------------- #
    async def _acquire_gpu(self, exclude: set[int]) -> _Worker:
        self._waiting += 1
        try:
            async with self._cond:
                while True:
                    alive = [w for w in self._workers if w.state != WorkerState.DEAD]
                    if not alive:
                        raise SchedulerError(ErrorCode.NO_GPU_AVAILABLE, "all GPUs are dead")

                    # 살아있는 GPU 가 전부 제외 대상이면 제외 조건을 무시
                    eff_exclude = exclude if any(w.gpu_id not in exclude for w in alive) else set()
                    candidates = [w for w in alive
                                  if w.state == WorkerState.IDLE and w.gpu_id not in eff_exclude]
                    if candidates:
                        w = min(candidates, key=lambda x: x.jobs_done)  # 처리량이 적은 GPU 우선
                        w.state = WorkerState.BUSY
                        w.busy_reason = "job"
                        return w
                    await self._cond.wait()
        finally:
            self._waiting -= 1

    async def _precheck_infer(self, w: _Worker, epoch: int) -> bool:
        """작업 배정 직전 점검. True 면 w(BUSY/job) 로 작업을 보내도 된다."""
        try:
            probe = await self._loop.run_in_executor(None, self._nvml.probe, w.gpu_id)
        except asyncio.CancelledError:
            self._release(w, epoch, restart=False)   # 작업을 보내기 전이므로 재시작 불필요
            raise
        if w.epoch != epoch:        # 점검 중 다른 경로에서 DEAD / 재시작 처리됨
            return False
        if not probe.ok:
            self._mark_dead(w, f"gpu_error: {probe.error}")
            return False
        w.free_mb = probe.free_mb
        if probe.free_mb is not None and probe.free_mb < self._infer_vram_mb:
            log.warning("gpu=%d free VRAM %.0fMB < infer %.0fMB -> BUSY(low_vram_infer)",
                        w.gpu_id, probe.free_mb, self._infer_vram_mb)
            w.busy_reason = "low_vram_infer"        # state 는 BUSY 유지, 작업은 보내지 않음
            return False
        return True

    def _release(self, w: _Worker, epoch: int, restart: bool) -> None:
        if w.epoch != epoch:  # 이미 monitor / health check 에 의해 재시작·DEAD 처리됨
            return
        if restart:
            self._restart_worker(w)
        else:
            w.state = WorkerState.IDLE
            w.busy_reason = ""
            self._notify_nowait()

    # ------------------------------ dispatch --------------------------- #
    async def _run_on_worker(self, w: _Worker, epoch: int, request: GenerationRequest) -> dict:
        task_id = uuid.uuid4().hex
        fut = self._loop.create_future()
        self._futures[task_id] = fut
        w.inflight = task_id
        w.task_q.put((task_id, request))
        try:
            return await asyncio.wait_for(fut, timeout=self._job_timeout)
        except asyncio.TimeoutError:
            return {"ok": False, "code": int(ErrorCode.TIMEOUT),
                    "message": f"job exceeded {self._job_timeout}s"}
        except asyncio.CancelledError:
            # 클라이언트 취소: GPU 작업은 중단할 수 없으므로 worker 를 재시작해 GPU 를 회수
            self._release(w, epoch, restart=True)
            raise
        finally:
            self._futures.pop(task_id, None)
            if w.epoch == epoch:
                w.inflight = None

    # --------------------------- worker mgmt --------------------------- #
    def _spawn(self, w: _Worker) -> None:
        w.task_q = self._ctx.Queue()
        w.state = WorkerState.STARTING
        w.busy_reason = ""
        w.dead_reason = ""
        w.process = self._ctx.Process(
            target=_worker_main,
            args=(w.gpu_id, w.epoch, w.task_q, self._result_q),
            daemon=True,
            name=f"gpu-worker-{w.gpu_id}",
        )
        w.process.start()
        log.info("spawned worker gpu=%d epoch=%d pid=%s", w.gpu_id, w.epoch, w.process.pid)

    def _begin_spawn(self, w: _Worker) -> None:
        """(재)기동 경로 공통 진입점: 로드용 VRAM 을 확인한 뒤에 worker 를 띄운다."""
        w.state = WorkerState.STARTING
        w.busy_reason = ""
        w.dead_reason = ""
        self._bg(self._probe_then_spawn(w, w.epoch))

    async def _probe_then_spawn(self, w: _Worker, epoch: int) -> None:
        probe = await self._loop.run_in_executor(None, self._nvml.probe, w.gpu_id)
        if w.epoch != epoch:      # 그 사이 다른 경로에서 처리됨
            return
        if not probe.ok:
            self._mark_dead(w, f"gpu_error: {probe.error}")
            return
        w.free_mb = probe.free_mb
        self._spawn_or_park(w, probe)

    def _spawn_or_park(self, w: _Worker, probe: GpuProbe) -> None:
        if probe.free_mb is not None and probe.free_mb < self._load_vram_mb:
            # 모델을 올릴 공간이 없음 -> OOM 을 내지 않고 worker 를 띄우지 않은 채 대기
            # (재시작 횟수는 올리지 않음: worker 결함이 아니므로)
            log.warning("gpu=%d free VRAM %.0fMB < load %.0fMB -> BUSY(low_vram_load), not spawned",
                        w.gpu_id, probe.free_mb, self._load_vram_mb)
            w.state = WorkerState.BUSY
            w.busy_reason = "low_vram_load"
            w.dead_reason = ""
            return
        self._spawn(w)

    def _kill_process(self, w: _Worker) -> None:
        old = w.process
        w.process = None          # monitor 가 죽은 핸들을 보고 중복 재시작하지 않도록
        if old and old.is_alive():
            old.kill()
        if old:
            self._loop.run_in_executor(None, old.join, 5)  # zombie 방지

    def _fail_inflight(self, w: _Worker, message: str) -> None:
        if w.inflight:
            fut = self._futures.get(w.inflight)
            if fut and not fut.done():
                fut.set_result({"ok": False, "code": int(ErrorCode.WORKER_CRASHED),
                                "message": message})

    def _restart_worker(self, w: _Worker) -> None:
        self._kill_process(w)
        w.restarts += 1
        w.epoch += 1  # 이후 이전 epoch 의 메시지/release 는 무시된다
        w.inflight = None
        if w.restarts > self._max_restarts:
            log.error("worker gpu=%d exceeded restart limit -> DEAD", w.gpu_id)
            w.state = WorkerState.DEAD
            w.busy_reason = ""
            w.dead_reason = "restart_limit_exceeded"
            self._notify_nowait()
            return
        self._begin_spawn(w)

    def _mark_dead(self, w: _Worker, reason: str) -> None:
        log.error("gpu=%d -> DEAD (%s)", w.gpu_id, reason)
        self._kill_process(w)
        self._fail_inflight(w, reason)   # 진행 중 요청은 WORKER_CRASHED 로 다른 GPU 에서 재시도됨
        w.epoch += 1
        w.inflight = None
        w.state = WorkerState.DEAD
        w.busy_reason = ""
        w.dead_reason = reason
        self._notify_nowait()            # 대기 중인 요청이 NO_GPU_AVAILABLE 을 판단할 수 있도록

    def _recover_worker(self, w: _Worker) -> None:
        log.info("gpu=%d recovered -> respawning worker", w.gpu_id)
        w.restarts = 0
        w.epoch += 1
        self._begin_spawn(w)   # 로드용 VRAM 확인 -> spawn -> (ready) -> IDLE

    async def _monitor_loop(self) -> None:
        """worker 프로세스가 (OOM-kill, segfault 등으로) 죽었는지 1초마다 확인."""
        while not self._stopping:
            await asyncio.sleep(1.0)
            for w in self._workers:
                if w.state == WorkerState.DEAD or w.process is None or w.process.is_alive():
                    continue
                log.error("worker gpu=%d died (exitcode=%s)", w.gpu_id, w.process.exitcode)
                self._fail_inflight(w, f"worker exit code {w.process.exitcode}")
                self._restart_worker(w)

    # ----------------------------- GPU health -------------------------- #
    async def _health_loop(self) -> None:
        while not self._stopping:
            await asyncio.sleep(self._health_interval)
            await asyncio.gather(*(self._check_gpu(w) for w in self._workers),
                                 return_exceptions=True)

    async def _check_gpu(self, w: _Worker) -> None:
        try:
            probe = await self._loop.run_in_executor(None, self._nvml.probe, w.gpu_id)
            self._apply_probe(w, probe)
        except Exception:  # noqa: BLE001
            log.exception("health check failed for gpu=%d", w.gpu_id)

    def _apply_probe(self, w: _Worker, probe: GpuProbe) -> None:
        """NVML 결과에 따라 상태 전이. (await 이후 '현재' 상태 기준으로 판단하므로 race 에 안전)"""
        if not probe.ok:
            if w.state != WorkerState.DEAD:
                self._mark_dead(w, f"gpu_error: {probe.error}")
            else:
                w.dead_reason = f"gpu_error: {probe.error}"
            return

        w.free_mb = probe.free_mb
        free = probe.free_mb
        below_infer = free is not None and free < self._infer_vram_mb
        below_load = free is not None and free < self._load_vram_mb

        if w.state == WorkerState.DEAD:
            self._recover_worker(w)          # 에러 해소 -> (로드용 VRAM 확인 후) worker 재기동
        elif w.state == WorkerState.IDLE and below_infer:
            log.warning("gpu=%d free VRAM %.0fMB < infer %.0fMB -> BUSY(low_vram_infer)",
                        w.gpu_id, free, self._infer_vram_mb)
            w.state = WorkerState.BUSY
            w.busy_reason = "low_vram_infer"
        elif w.state == WorkerState.BUSY and w.busy_reason == "low_vram_infer" and not below_infer:
            log.info("gpu=%d VRAM recovered -> IDLE", w.gpu_id)
            w.state = WorkerState.IDLE
            w.busy_reason = ""
            self._notify_nowait()
        elif w.state == WorkerState.BUSY and w.busy_reason == "low_vram_load" and not below_load:
            log.info("gpu=%d enough VRAM to load model -> spawning worker", w.gpu_id)
            self._spawn(w)                   # STARTING -> (ready) -> IDLE
        # BUSY(job) / STARTING 은 VRAM 판정 제외

    # --------------------------- result channel ------------------------ #
    def _collect_loop(self) -> None:
        """mp.Queue 는 블로킹 API 이므로 별도 thread 에서 읽어 event loop 로 전달."""
        while not self._stopping:
            try:
                msg = self._result_q.get(timeout=1.0)
            except queue.Empty:
                continue
            except (EOFError, OSError):
                break
            self._loop.call_soon_threadsafe(self._handle_message, msg)

    def _handle_message(self, msg: dict) -> None:
        w = self._workers[msg["gpu_id"]]
        if msg["epoch"] != w.epoch:  # 이전(재시작 전) 프로세스가 보낸 stale 메시지
            return

        kind = msg["type"]
        if kind == "ready":
            w.state = WorkerState.IDLE
            log.info("worker gpu=%d ready", w.gpu_id)
            self._notify_nowait()
            self._bg(self._check_gpu(w))     # 모델 로딩 직후 VRAM 즉시 재판정
        elif kind == "load_failed":
            log.error("worker gpu=%d model load failed: %s", w.gpu_id, msg.get("message"))
            self._restart_worker(w)
        elif kind == "result":
            fut = self._futures.get(msg["task_id"])
            if fut and not fut.done():
                fut.set_result(msg)

    # ------------------------------- utils ----------------------------- #
    def _bg(self, coro) -> None:
        t = self._loop.create_task(coro)
        self._bg_tasks.add(t)
        t.add_done_callback(self._bg_tasks.discard)

    def _notify_nowait(self) -> None:
        async def _notify():
            async with self._cond:
                self._cond.notify_all()
        self._bg(_notify())

    @staticmethod
    def _to_code(value: int) -> ErrorCode:
        try:
            return ErrorCode(value)
        except ValueError:
            return ErrorCode.UNKNOWN