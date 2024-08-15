import subprocess

from tqdm import tqdm
from loguru import logger

from datatrove.data import DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep


class ZstdCompressor(PipelineStep):
    """Compress files with Zstd.

    Args:
        input_folder (DataFolderLike): the input folder containing the tokenized documents
        output_folder (DataFolderLike): the output folder where to save the merged tokenized documents
        recursive (bool): whether to search files recursively. Ignored if paths_file is provided
        glob_pattern (bool): pattern that all files must match exactly to be included (relative to data_folder). Ignored if paths_file is provided
        progress (bool): show progress bar for documents
        remove (bool): remove original file after compression
        nthreads (int): number of threads for compression
        zstd_bin (str): path to zstd binary
    """

    name = "📦 - Zstd Compressor"
    type = "🛒 - POSTPROCESSOR"

    def __init__(
        self,
        input_folder: DataFolderLike,
        output_folder: DataFolderLike,
        recursive: bool = True,
        glob_pattern: str | None = None,
        progress: bool = True,
        remove: bool = True,
        nthreads: int = 1,
        zstd_bin: str | None = None,
    ):
        super().__init__()
        self.input_folder = get_datafolder(input_folder)
        self.output_folder = get_datafolder(output_folder)
        self.recursive = recursive
        self.glob_pattern = glob_pattern
        self.progress = progress
        self.remove = remove
        self.nthreads = nthreads
        if zstd_bin is None:
            self.zstd_bin = subprocess.check_output("which zstd", shell=True, encoding="utf-8").strip()
            if not self.zstd_bin:
                raise RuntimeError("Failed to find zstd binary.")
        else:
            self.zstd_bin = zstd_bin

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """Main method to compress files.

        Args:
            data: DocumentsPipeline
                The data to be processed as a Generator typically created by a Reader initial pipeline step
            rank: int
                The rank of the process
            world_size: int
                The total number of processes
        """
        files_shard = self.plan_folder.get_shard(rank, world_size, recursive=self.recursive, glob_pattern=self.glob_pattern)
        with tqdm(total=len(files_shard), desc="File progress", unit="file", disable=not self.file_progress) as file_pbar:
            for file in files_shard:
                rm = "--rm" if self.remove else ""
                cmd = f"{self.zstd_bin} -T{self.nthreads} {rm} {file}"
                logger.info(f"Compress file {file} with command \"{cmd}\"")
                subprocess.check_output(cmd)
                file_pbar.update()
