#!/bin/bash
set -e
models=${1:-"mistralai/Mistral-Small-3.2-24B-Instruct-2506"}
datasets=${2:-"medxpertqa"}
tasks=${3:-"text,mm"}
output_dir=${4:-"dev"}
method=${5:-"zero_shot"}
prompting_type=${6:-"cot"}
temperature=${7:-0}

echo "Starting SEQUENTIAL Mistral evaluation..."
echo "Models: $models"
echo "Datasets: $datasets"
echo "Tasks: $tasks"

IFS=","
for model in $models; do
    for dataset in $datasets; do
        for task in $tasks; do
            date +"%Y-%m-%d %H:%M:%S"
            echo "Model: ${model}"
            echo "Dataset: ${dataset}"
            echo "Task: ${task}"
            echo "Output: ${output_dir}"
            
            # Create log directory structure with safe file naming
            model_safe=$(echo "${model}" | sed 's/\//-/g')
            log_dir="outputs/${output_dir}/${model}/${dataset}/${method}/${prompting_type}/logs"
            if [ ! -d "${log_dir}" ]; then
                mkdir -p "${log_dir}"
                echo "Created directory: ${log_dir}"
            fi
            
            log_file="${log_dir}/run-${model_safe}-${dataset}-${task}.log"
            
            # Copy files to log directory
            echo "Copying files to log directory..."
            [ -f "${BASH_SOURCE[0]}" ] && cp "${BASH_SOURCE[0]}" "${log_dir}/run.sh"
            [ -f main.py ] && cp main.py "${log_dir}/main.py"
            [ -f utils.py ] && cp utils.py "${log_dir}/utils.py"
            [ -f model/api_agent.py ] && cp model/api_agent.py "${log_dir}/api_agent.py"
            [ -f config/prompt_templates.py ] && cp config/prompt_templates.py "${log_dir}/prompt_templates.py"
            
            echo "Starting SEQUENTIAL job with log file: ${log_file}"
            
            # Run SEQUENTIALLY (not in background with &)
            python main.py \
                --model "${model}" \
                --dataset "${dataset}" \
                --task "${task}" \
                --output-dir "${output_dir}" \
                --method "${method}" \
                --prompting-type "${prompting_type}" \
                --temperature "${temperature}" \
                --num-threads 1 \
                2>&1 | tee "${log_file}"
            
            echo "Completed task: ${task}"
            
            # Clean up GPU memory between tasks
            echo "Cleaning GPU memory..."
            python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"
            sleep 5
        done
    done
done

echo "All tasks completed sequentially."