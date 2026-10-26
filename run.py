import argparse
import os

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Run the local ONTHEDOT question workbench.")
    parser.add_argument("--backend", choices=["mlx", "ollama"], default=os.environ.get("ONTHEDOT_BACKEND", "mlx"))
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()
    # select the local runtime before the application imports its parser
    os.environ["ONTHEDOT_BACKEND"] = args.backend
    uvicorn.run("app:app", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
