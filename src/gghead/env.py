from pathlib import Path

from environs import Env

REPO_ROOT_DIR = f"{Path(__file__).parent.resolve()}/../.."

env = Env(expand_vars=True)
env_file_path = Path(f"{Path.home()}/.config/gghead/.env")
if env_file_path.exists():
    env.read_env(str(env_file_path), recurse=False)

with env.prefixed("GGHEAD_"):
    GGH_MODELS_PATH = env("MODELS_PATH", f"<<<Define GGHEAD_MODELS_PATH in {env_file_path}>>>")
    GGH_RENDERINGS_PATH = env("RENDERINGS_PATH", f"<<<Define GGHEAD_RENDERINGS_PATH in {env_file_path}>>>")
