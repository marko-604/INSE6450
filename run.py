import demos
import sys

task = sys.argv[1].lower()
method = sys.argv[2].lower()
N = int(sys.argv[3])
M = int(sys.argv[4])

# Optional flags after positional args
extra_args = sys.argv[5:]  # everything after M

pure_oracle = False
if "--pure_oracle" in extra_args:
    pure_oracle = True
    extra_args.remove("--pure_oracle")

if method == "nonbatch" or method == "random":
    demos.nonbatch(task, method, N, M)

elif method in {"greedy", "medoids", "boundary_medoids", "successive_elimination", "dpp"}:
    if len(extra_args) < 1:
        raise SystemExit("Usage: python run.py <task> <method> N M b [human_topk] [--pure_oracle]")

    b = int(extra_args[0])

    # human_topk is optional; default 0
    human_topk = int(extra_args[1]) if len(extra_args) >= 2 else 0

    demos.batch(task, method, N, M, b, human_topk=human_topk, pure_oracle=pure_oracle)

else:
    print("There is no method called " + method)
