#!/usr/bin/env bash
set -e
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"
if (( $# > 2 )); then
  echo "Usage: bash run.sh [CONFIG_YAML] or bash run.sh OUTPUT_DIR CONFIG_YAML" >&2
  exit 2
fi
config_path="$(realpath -e -- "${2:-${1:-$script_dir/run-op-6.yaml}}")"
config_name="$(basename -- "$config_path")"
run_name="${config_name%.*}_$(date -u +%Y%m%d_%H%M%S_%N)"
logs_dir="$script_dir/logs"
mkdir -p "$logs_dir"
log_path="$logs_dir/$run_name.log"
if (( $# == 2 )); then
  output_dir="$(realpath -m -- "$1")"
else
  output_dir="$logs_dir/$run_name"
fi
echo "Log: $log_path"
echo "Output: $output_dir"
exec > "$log_path" 2>&1
echo "Config: $config_path"
echo "Output: $output_dir"
cd "$repo_dir"
source /etc/profile.d/proxy.sh
source /team/xyq/data_juicer/envs/nemo-1/bin/activate
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export RAY_num_server_call_thread=1 RAY_core_worker_num_server_call_thread=1
export RAY_num_grpc_internal_threads=1 RAY_worker_num_grpc_internal_threads=1
export RAY_gcs_server_rpc_client_thread_num=1 RAY_gcs_server_rpc_server_thread_num=1
export RAY_enable_worker_prestart=0 RAY_prestart_worker_first_driver=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
set +e
python -u tutorials/video/getting-started/data_juicer_compatible_pipeline.py \
  --pipeline-config "$config_path" \
  --output-path "$output_dir" \
  --ray-temp-dir /tmp/curator_gpu4_7_ray
pipeline_exit_code=$?
echo "Exit code: $pipeline_exit_code"
exit "$pipeline_exit_code"
