import argparse

from graphkmc.Preset_Runners import SteadyStateRunnerTTA

def main(*args, **kwargs):
    runner = SteadyStateRunnerTTA(**kwargs)
    runner.run()

def bool_string(*s):
    # print(s)
    out = []
    for val in s:
        # print(f"val = {val}")
        if "false" in val.lower():
            out.append(False)
        elif "true" in val.lower():
            out.append(True)
        else:
            raise ValueError("Not a boolean value")
    if len(out) == 1:
        out = out[0]
    elif len(out) == 0:
        raise ValueError("Could not find boolean")
    # print(f"out = {out}")
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Steady State KMC runner with TTA.")
    parser.add_argument("--Rf", type=float, default=1.)
    parser.add_argument("--n", type=int, default=11)
    parser.add_argument("--G", type=float, default=1.)
    parser.add_argument("--max_time", type=float, default=1.)
    parser.add_argument("--max_steps", type=int, default=int(1e7))
    parser.add_argument("--varKappa", type=bool_string, default=True)
    parser.add_argument("--latticetype", type=str, default="scc")
    args = parser.parse_args()
    print(f"args = {args}")
    print(f"vars(args) = \r\n", *[f"{k}: {v}\r\n" for k, v in vars(args).items()])
    out = main(**vars(args))