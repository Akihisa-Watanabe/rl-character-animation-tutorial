import os
import subprocess
import time
from typing import Dict, List, Optional, Union

import tensorboardX

import util.logger as logger


class TBLogger(logger.Logger):
    """
    TensorBoard logger for logging variables and displaying them in TensorBoard.

    Args:
        run_tb (bool): Whether to run TensorBoard in the background.
    """

    MISC_TAG: str = "Misc"

    def __init__(self, run_tb: bool = False) -> None:
        super().__init__()

        self._writer: Optional[tensorboardX.SummaryWriter] = None
        self._step_var_key: Optional[str] = None
        self._collections: Dict[str, List[str]] = {}
        self._run_tb: bool = run_tb
        self._tb_proc: Optional[subprocess.Popen] = None

    def __del__(self) -> None:
        """
        Destructor that kills the TensorBoard process if it's running.
        """
        if self._tb_proc is not None:
            self._tb_proc.kill()

    def reset(self) -> None:
        """
        Resets the logger state.
        """
        super().reset()

    def configure_output_file(self, filename: Optional[str] = None) -> None:
        """
        Configures the output file and initializes the TensorBoard writer.

        Args:
            filename (Optional[str]): The name of the output file.
        """
        super().configure_output_file(filename)
        output_dir = os.path.dirname(filename)
        self._delete_event_files(output_dir)
        self._writer = tensorboardX.SummaryWriter(output_dir)
        if self._run_tb:
            self._run_tensorboard(output_dir)

    def set_step_key(self, var_key: str) -> None:
        """
        Sets the step variable key for TensorBoard.

        Args:
            var_key (str): The step variable key.
        """
        self._step_key = var_key

    def log(self, key: str, val: Union[int, float, str], collection: Optional[str] = None, quiet: bool = False) -> None:
        """
        Logs a value with an optional collection tag.

        Args:
            key (str): The key for the log.
            val (Union[int, float, str]): The value to log.
            collection (Optional[str]): The collection tag.
            quiet (bool): Whether the log should be quiet.
        """
        super().log(key, val, quiet)
        if collection is not None:
            self._add_collection(collection, key)

    def write_log(self) -> None:
        """
        Writes logs into the TensorBoard.
        """
        row_count = self._row_count
        super().write_log()

        if self._writer is not None:
            if row_count == 0:
                self._key_tags = self._build_key_tags()

            curr_time = time.time()
            step_val = self.log_current_row.get(self._step_key, "").val
            for i, key in enumerate(self.log_headers):
                if key != self._step_key:
                    entry = self.log_current_row.get(key, "")
                    val = entry.val
                    tag = self._key_tags[i]
                    self._writer.add_scalar(tag, val, step_val)

    def _add_collection(self, name: str, key: str) -> None:
        """
        Adds a collection of keys.

        Args:
            name (str): The name of the collection.
            key (str): The key to add to the collection.
        """
        if name not in self._collections:
            self._collections[name] = []
        self._collections[name].append(key)

    def _delete_event_files(self, dir: str) -> None:
        """
        Deletes TensorBoard event files in a directory.

        Args:
            dir (str): The directory path.
        """
        if os.path.exists(dir):
            files = os.listdir(dir)
            for file in files:
                if "events.out.tfevents." in file:
                    file_path = os.path.join(dir, file)
                    print(f"Deleting event file: {file_path}")
                    os.remove(file_path)

    def _build_key_tags(self) -> List[str]:
        """
        Builds key tags for TensorBoard.

        Returns:
            List[str]: The list of built key tags.
        """
        tags = []
        for key in self.log_headers:
            curr_tag = TBLogger.MISC_TAG
            for col_tag, col_keys in self._collections.items():
                if key in col_keys:
                    curr_tag = col_tag
            curr_tags = f"{curr_tag}/{key}"
            tags.append(curr_tags)
        return tags

    def _run_tensorboard(self, output_dir: str) -> None:
        """
        Runs TensorBoard as a background process.

        Args:
            output_dir (str): The directory where TensorBoard logs are stored.
        """
        cmd = f"tensorboard --logdir {output_dir}"
        self._tb_proc = subprocess.Popen(cmd, shell=True)
