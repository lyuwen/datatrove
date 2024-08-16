import subprocess

from tqdm import tqdm
from loguru import logger

from datatrove.data import DocumentsPipeline
from datatrove.io import DataFileLike, DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep


class ZstdCompressor(PipelineStep):
    """Compress files with Zstd.

    Args:
        input_folder (DataFolderLike): the input folder containing the tokenized documents
        file_paths (DataFileLike): the files to process
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
        input_folder: DataFolderLike | None = None,
        *,
        file_paths: DataFileLike | None = None,
        recursive: bool = True,
        glob_pattern: str | None = None,
        progress: bool = True,
        remove: bool = True,
        nthreads: int = 1,
        zstd_bin: str | None = None,
    ):
        super().__init__()
        if file_paths is not None:
            if input_folder:
              input_folder = get_datafolder(input_folder)
              file_paths = input_folder._join(file_paths)
            self.files_to_process = file_paths
        else:
            input_folder = get_datafolder(input_folder)
            self.files_to_process = input_folder._join(input_folder.list_files(recursive=recursive, glob_pattern=glob_pattern))
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
        files_shard = self.files_to_process[rank::world_size]
        with tqdm(total=len(files_shard), desc="File progress", unit="file", disable=not self.progress) as file_pbar:
            for file in files_shard:
                cmd = f"{self.zstd_bin} -T{self.nthreads} {file}"
                logger.info(f"Compress file {file} with command \"{cmd}\"")
                subprocess.check_output(cmd.split())
                if self.remove:
                    self.input_folder.rm_file(file)
                file_pbar.update()
