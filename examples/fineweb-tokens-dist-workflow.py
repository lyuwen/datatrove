from datatrove.executor.base import PipelineExecutor
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers import JsonlReader, ParquetReader
from datatrove.pipeline.tokens.megatron_tokenizer import MegatronTokenizer
from datatrove.pipeline.tokens.megatron_merger import MegatronTokenizerMerger


def process(input_folder, output_folder, filename, job_name, tokenizer, njobs=100):
    merge = True
    pipeline_1 = [
        ParquetReader(
             input_folder,
             glob_pattern="*.parquet",
             text_key="text",
             ),
        MegatronTokenizer(
            output_folder=f"{output_folder}/tokens/",
            local_working_dir=f"{output_folder}/scratch/",
            save_filename=f"{filename}",
            tokenizer_name_or_path=f"{tokenizer}",
            batch_size=8,
            shuffle=not merge,
        ),
    ]

    pipeline_2 = [
        DistTokenizerMergerPlanner(
            input_folder=f"{output_folder}/tokens/",
            output_folder=f"{output_folder}/dmerge-test/",
            plan_folder=f"{output_folder}/dmerge-plan/",
            save_filename=f"{filename}",
        ),
    ]


    pipeline_3 = [
        DistTokenizerMergerExecutor(
            input_folder=f"{output_folder}/tokens/",
            output_folder=f"{output_folder}/dmerge-out/",
            plan_folder=f"{output_folder}/dmerge-plan/",
            save_filename=f"{filename}",
        ),
    ]


    executor_1: PipelineExecutor = SlurmPipelineExecutor(
        pipeline=pipeline_1,
        cpus_per_task=8,
        mem_per_cpu_gb=12,
        job_name=f"{job_name}-tokens",
        partition="debug",
        time="12:00:00",
        env_command="module load conda/23.07 && conda activate py3",
        logging_dir=f"log_dir/tokens",
        workers=njobs,
        tasks=njobs,
        )


    executor_2: PipelineExecutor = SlurmPipelineExecutor(
        pipeline=pipeline_2,
        cpus_per_task=4,
        mem_per_cpu_gb=12,
        job_name=f"{job_name}-merge-plan",
        partition="debug",
        time="12:00:00",
        env_command="module load conda/23.07 && conda activate py3",
        logging_dir=f"log_dir/plan",
        workers=1,
        tasks=1,
        depends=executor_1,
        )


    executor_3: PipelineExecutor = SlurmPipelineExecutor(
        pipeline=pipeline_3,
        cpus_per_task=4,
        mem_per_cpu_gb=12,
        job_name=f"{job_name}-merge",
        partition="debug",
        time="12:00:00",
        env_command="module load conda/23.07 && conda activate py3",
        logging_dir=f"log_dir/execute",
        workers=njobs,
        tasks=bjobs,
        depends=executor_2,
        )


    if not merge:
        print(executor_1.run())
    else:
        print(executor_3.run())


if __name__ == "__main__":
    DATA_BASE = "..."
    OUTPUT_BASE = "..."
    TOKENIZER_DIR = "..."
    PART="CC-MAIN-2024-18"
    process(
        input_folder=f"{DATA_BASE}/{PART}",
        output_folder=f"{OUTPUT_BASE}/{PART}",
        filename=PART,
        job_name=f"FW-{PART}",
        tokenizer=TOKENIZER_DIR,
        njobs=200,
        )
