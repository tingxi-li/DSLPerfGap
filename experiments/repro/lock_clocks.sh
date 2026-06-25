#!/usr/bin/env bash
# lock_clocks.sh -- pin GPU clocks to a SUSTAINABLE value and verify under load.
#
# WHY THIS EXISTS (the gotcha): locking at the GPU's MAX clock is often NOT honored
# under sustained tensor-core load, because the board hits its power cap and silently
# throttles -- giving you a fluctuating clock (exactly the run-to-run noise you locked
# to remove). On A100-SXM4-40GB (400 W) the 1410 MHz max power-caps to a bouncing
# ~1200-1215 MHz; locking at 1215 instead holds DEAD FLAT. Always lock at a clock the
# board can SUSTAIN, then verify it holds under a real load.
#
# Also: this driver (>= 610) REJECTS a combined "-lgc X -lmc Y" ("Only one device
# modification may be done at a time") -- you MUST issue -lgc and -lmc separately.
#
# Usage:
#   bash lock_clocks.sh [GR_MHZ] [MEM_MHZ]
#     no args  -> queries this GPU's max clocks and load-tests them (discovery mode);
#                 if the clock power-caps under load it prints the SUSTAINED value to
#                 re-lock with.
#     GR_MHZ MEM_MHZ -> locks those exact values and verifies (use the sustained value
#                 discovery mode reported).
#
# Per-arch sustained values we have verified (lock at these, NOT the max):
#   RTX 4000 Ada (sm_89, 130 W):  -lgc 1400 -lmc 9001   (gr holds 1410; mem -> 8551 under load)
#   A100-SXM4-40GB (sm_80, 400 W): -lgc 1215 -lmc 1215  (1410 power-caps; 1215 holds flat)
#   H100-SXM5-80GB (sm_90, 700 W): query first -- 1980 MAY hold (more power headroom);
#                                   if it caps under load, re-lock at the sustained value.
set -u
IDX=0
VENV="${PYTHON:-/home/ubuntu/dslperf-venv/bin/python}"
read -r MAXGR MAXMEM < <(nvidia-smi --query-gpu=clocks.max.gr,clocks.max.mem --format=csv,noheader,nounits -i $IDX | tr ',' ' ')
GR="${1:-$MAXGR}"; MEM="${2:-$MAXMEM}"
echo "GPU max clocks: gr=$MAXGR MHz, mem=$MAXMEM MHz   ->  locking gr=$GR, mem=$MEM"

sudo nvidia-smi -i $IDX -pm 1 >/dev/null
sudo nvidia-smi -i $IDX -lgc "$GR"        # SEPARATE commands (combined form is rejected)
sudo nvidia-smi -i $IDX -lmc "$MEM"
echo "idle clocks after lock: $(nvidia-smi --query-gpu=clocks.gr,clocks.mem --format=csv,noheader -i $IDX)"

echo "--- load test (12 s sustained fp16 matmul; sampling clock/power/throttle) ---"
nohup "$VENV" - >/dev/null 2>&1 <<'PY' &
import torch,time
a=torch.randn(8192,8192,device='cuda',dtype=torch.float16); b=torch.randn(8192,8192,device='cuda',dtype=torch.float16)
t=time.time()
while time.time()-t<12: c=a@b
torch.cuda.synchronize()
PY
LOADPID=$!
MINGR=999999
for i in 1 2 3 4 5; do
  sleep 2
  read -r u g m thr p tC < <(nvidia-smi --query-gpu=utilization.gpu,clocks.gr,clocks.mem,clocks_event_reasons.active,power.draw,temperature.gpu --format=csv,noheader,nounits -i $IDX | tr ',' ' ')
  echo "  util=${u}% gr=${g}MHz mem=${m}MHz throttle=${thr} power=${p}W temp=${tC}C"
  [ "$g" -lt "$MINGR" ] && MINGR=$g
done
wait $LOADPID 2>/dev/null
echo "--- min graphics clock observed under load: ${MINGR} MHz (target was ${GR}) ---"
if [ "$MINGR" -lt "$GR" ]; then
  echo "WARNING: clock power-capped under load (throttle bit 0x4 = SW Power Cap)."
  echo "         Re-lock at the SUSTAINED value:  bash $0 ${MINGR} ${MEM}"
else
  echo "OK: clock HELD at ${GR} MHz under load -- this is your locked-clock config."
  echo "    Use it in significance:  python experiments/exp_significance.py --lock-gr-mhz ${GR} --lock-mem-mhz ${MEM}"
fi
echo "Reset afterward with:  sudo nvidia-smi -i $IDX -rgc ; sudo nvidia-smi -i $IDX -rmc"
