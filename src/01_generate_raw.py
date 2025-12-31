from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import math
import random

KST = timezone(timedelta(hours=9))

@dataclass
class Config:
    out_path: Path = Path("data/raw/events.jsonl")
    days: int = 14
    services: tuple[str, ...] = ("api-gateway", "payment", "auth")
    seed: int = 42

def seasonal_lambda(service: str, dt: datetime) -> float:
    hour = dt.hour
    dow = dt.weekday()

    # 기본 트래픽 규모는 서비스별로 다르게 설정
    base = {"api-gateway": 220, "payment": 70, "auth": 110}[service]

    # 일중 패턴 : 낮에 높고 새벽에 낮게
    diurnal = 0.55 + 0.45 * (math.sin((hour -6) / 24 * 2 * math.pi) + 1) / 2

    # 주중 주말 차이
    weekend_factor = 0.75 if dow >= 5 else 1.0

    return base * diurnal * weekend_factor

def is_anomaly_window(dt: datetime) -> bool:
    
    t = dt.time()
    return (t.hour == 10 and 30 <= t.minute < 45) or (t.hour == 21 and 10 <= t.minute < 25)

def main(cfg: Config) -> None:
    random.seed(cfg.seed)
    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)

    # 지금 기준으로 과거 cfg.days 생성 (분단위)
    end = datetime.now(tz=KST).replace(second=0, microsecond=0)
    start = end - timedelta(days=cfg.days)

    with cfg.out_path.open("w", encoding="utf-8") as f:
        dt = start
        while dt < end:
            for service in cfg.services:
                lam = seasonal_lambda(service, dt)

                # 분당 요청 수를 포아송처럼 시뮬레이션 (간단히 정규/클램프)
                req = max(0, int(random.gauss(lam, math.sqrt(lam))))

                # 정상 에러율
                base_err = {"api-gateway": 0.002, "payment": 0.006, "auth": 0.003}[service]
                err_rate = base_err

                # 이상 구간에서는 에러율 상승 + 트래픽 변동
                if is_anomaly_window(dt) and service in ("payment", "api-gateway"):
                    err_rate *= 12
                    req = int(req * 1.4) if service == "api-gateway" else int(req * 0.8)

                # 에러 카운트 생성
                err = 0
                for _ in range(req):
                    if random.random() < err_rate:
                        err += 1

                # p95 latency 시뮬레이션
                base_lat = {"api-gateway": 120, "payment": 220, "auth": 140}[service]
                p95 = base_lat + random.gauss(0, 15)
                if is_anomaly_window(dt) and service == "payment":
                    p95 += 180

                event = {
                    "ts": dt.isoformat(),
                    "service": service,
                    "req_count": req,
                    "err_count": err,
                    "p95_latency_ms": max(1, int(p95)),
                }
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

            dt += timedelta(minutes=1)

    print(f"wrote: {cfg.out_path} (days={cfg.days}, service={len(cfg.services)})")

if __name__ == "__main__":
    main(Config())