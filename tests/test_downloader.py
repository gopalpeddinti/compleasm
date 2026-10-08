import io
import os
import tarfile
import tempfile
import threading
import unittest
from types import SimpleNamespace
from urllib.error import URLError
from unittest.mock import patch

from compleasm import AutoLineager, CompleasmRunner, Downloader, Error, ProteinRunner, download


class DownloaderLineageTests(unittest.TestCase):
    lineage = "eukaryota_odb12"

    def make_downloader(self, download_dir):
        downloader = Downloader.__new__(Downloader)
        downloader.download_dir = download_dir
        downloader.base_url = "https://example.test/data/"
        downloader.lineage_description = {self.lineage: ["20260101", "hash", "lineages"]}
        return downloader

    def create_dataset(self, download_dir):
        lineage_dir = os.path.join(download_dir, self.lineage)
        hmm_dir = os.path.join(lineage_dir, "hmms")
        os.makedirs(hmm_dir, exist_ok=True)
        for relative_path, content in (
            ("refseq_db.faa", b">protein\nM\n"),
            ("scores_cutoff", b"gene 1\n"),
            ("hmms/gene.hmm", b"HMMER3/f\n"),
        ):
            with open(os.path.join(lineage_dir, relative_path), "wb") as output:
                output.write(content)
        return lineage_dir

    def write_dataset_archive(self, archive_path):
        with tarfile.open(archive_path, "w:gz") as archive:
            for name, content, member_type in (
                (f"{self.lineage}/refseq_db.faa", b">protein\nM\n", tarfile.REGTYPE),
                (f"{self.lineage}/scores_cutoff", b"gene 1\n", tarfile.REGTYPE),
                (f"{self.lineage}/hmms", b"", tarfile.DIRTYPE),
                (f"{self.lineage}/hmms/gene.hmm", b"HMMER3/f\n", tarfile.REGTYPE),
            ):
                member = tarfile.TarInfo(name)
                member.type = member_type
                member.size = len(content)
                member.mode = 0o755 if member_type == tarfile.DIRTYPE else 0o644
                archive.addfile(member, io.BytesIO(content) if content else None)

    def test_valid_local_dataset_without_done_marker_is_reused(self):
        with tempfile.TemporaryDirectory() as download_dir:
            lineage_dir = self.create_dataset(download_dir)
            downloader = self.make_downloader(download_dir)

            with patch.object(downloader, "download_single_file") as download:
                downloader.download_lineage("eukaryota", "odb12")

            download.assert_not_called()
            self.assertTrue(os.path.isfile(lineage_dir + ".done"))
            self.assertEqual(downloader.lineage_description[self.lineage][3], lineage_dir)

    def test_download_single_file_handles_standard_url_error(self):
        with tempfile.TemporaryDirectory() as download_dir:
            downloader = self.make_downloader(download_dir)
            with patch("compleasm.urllib.request.urlretrieve", side_effect=URLError("offline")):
                self.assertFalse(downloader.download_single_file("https://example.test/file", "unused", "hash"))

    def test_invalid_dataset_with_done_marker_is_downloaded(self):
        with tempfile.TemporaryDirectory() as download_dir:
            lineage_dir = os.path.join(download_dir, self.lineage)
            os.makedirs(lineage_dir)
            open(lineage_dir + ".done", "w").close()
            downloader = self.make_downloader(download_dir)

            def download_archive(_remote_path, local_path, _expected_hash):
                self.write_dataset_archive(local_path)
                return True

            with patch.object(downloader, "download_single_file", side_effect=download_archive) as download:
                downloader.download_lineage("eukaryota", "odb12")

            download.assert_called_once()
            self.assertTrue(Downloader._is_valid_lineage(lineage_dir))
            self.assertTrue(os.path.isfile(lineage_dir + ".done"))
            self.assertFalse(os.path.exists(lineage_dir + ".tmp"))

    def test_concurrent_call_waits_and_reuses_downloaded_dataset(self):
        with tempfile.TemporaryDirectory() as download_dir:
            first = self.make_downloader(download_dir)
            second = self.make_downloader(download_dir)
            download_started = threading.Event()
            allow_download = threading.Event()
            call_count = 0
            call_count_lock = threading.Lock()
            errors = []

            def download_archive(_downloader, _remote_path, local_path, _expected_hash):
                nonlocal call_count
                with call_count_lock:
                    call_count += 1
                self.write_dataset_archive(local_path)
                self.create_dataset(download_dir)
                download_started.set()
                if not allow_download.wait(5):
                    raise TimeoutError("test download was not released")
                return True

            def run_download(downloader):
                try:
                    downloader.download_lineage("eukaryota", "odb12")
                except Exception as error:
                    errors.append(error)

            with patch.object(Downloader, "download_single_file", download_archive):
                first_thread = threading.Thread(target=run_download, args=(first,))
                second_thread = threading.Thread(target=run_download, args=(second,))
                first_thread.start()
                self.assertTrue(download_started.wait(2))
                second_thread.start()
                self.assertTrue(second_thread.is_alive())
                allow_download.set()
                first_thread.join(5)
                second_thread.join(5)

            self.assertFalse(first_thread.is_alive())
            self.assertFalse(second_thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(call_count, 1)
            self.assertTrue(Downloader._is_valid_lineage(os.path.join(download_dir, self.lineage)))

    def test_failed_download_cleans_marker_and_releases_lock_for_retry(self):
        with tempfile.TemporaryDirectory() as download_dir:
            downloader = self.make_downloader(download_dir)
            with patch.object(downloader, "download_single_file", return_value=False):
                with self.assertRaises(Error):
                    downloader.download_lineage("eukaryota", "odb12")

            lineage_dir = os.path.join(download_dir, self.lineage)
            self.assertFalse(os.path.exists(lineage_dir + ".tmp"))
            with patch.object(
                downloader,
                "download_single_file",
                side_effect=lambda _remote, path, _hash: (self.write_dataset_archive(path) or True),
            ):
                downloader.download_lineage("eukaryota", "odb12")

            self.assertTrue(Downloader._is_valid_lineage(lineage_dir))


class LineageSelectionTests(unittest.TestCase):
    def test_download_command_does_not_download_default_lineage(self):
        args = SimpleNamespace(odb="odb12", library_path="library", lineages=["primates"])
        with patch("compleasm.Downloader") as downloader_type:
            download(args)

        downloader_type.assert_called_once_with(
            odb="odb12", download_dir="library", download_lineage=False
        )
        downloader_type.return_value.download_lineage.assert_called_once_with("primates", "odb12")

    def test_protein_runner_defers_download_until_selected_lineage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("compleasm.Downloader") as downloader_type:
                ProteinRunner(
                    protein_path="proteins.faa",
                    output_folder=os.path.join(temp_dir, "out"),
                    library_path=os.path.join(temp_dir, "library"),
                    lineage="primates",
                    odb="odb12",
                    nthreads=1,
                    hmmsearch_execute_command="hmmsearch",
                )

        downloader_type.assert_called_once_with(
            odb="odb12", download_dir=os.path.join(temp_dir, "library"), download_lineage=False
        )

    def test_genome_runner_downloads_default_only_for_autolineage(self):
        for autolineage in (False, True):
            with self.subTest(autolineage=autolineage):
                with patch("compleasm.MiniprotRunner"), patch("compleasm.Downloader") as downloader_type, patch(
                    "compleasm.AutoLineager"
                ):
                    CompleasmRunner(
                        assembly_path="assembly.fna",
                        output_folder="output",
                        library_path="library",
                        lineage="primates",
                        odb="odb12",
                        autolineage=autolineage,
                        retrocopy=False,
                        nthreads=1,
                        outs=0.95,
                        miniprot_execute_command="miniprot",
                        hmmsearch_execute_command="hmmsearch",
                        sepp_execute_command="run_sepp.py",
                        min_diff=0.2,
                        min_length_percent=0.6,
                        min_identity=0.4,
                        min_complete=None,
                        min_rise=0.5,
                        specified_contigs=None,
                    )

                downloader_type.assert_called_once_with(
                    odb="odb12", download_dir="library", download_lineage=autolineage
                )

    def test_autolineager_explicitly_requests_eukaryota_bootstrap(self):
        with patch("compleasm.Downloader") as downloader_type:
            AutoLineager("sepp_output", "sepp_tmp", "library", "odb12", 1)

        downloader_type.assert_called_once_with(
            odb="odb12", download_dir="library", download_lineage=True, autolineage=True
        )


if __name__ == "__main__":
    unittest.main()