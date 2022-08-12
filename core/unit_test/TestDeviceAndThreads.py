import unittest
import subprocess

PREFIX = "$<TARGET_FILE_DIR:KokkosCore_UnitTest_DeviceAndThreads>"
EXECUTABLE = "$<TARGET_FILE_NAME:KokkosCore_UnitTest_DeviceAndThreads>"
COMMAND = "/".join([PREFIX, EXECUTABLE])

def GetFlag(flag, *extra_args):
    p = subprocess.run([COMMAND, flag, *extra_args], capture_output=True)
    if p.returncode != 0:
        raise Exception(p.stderr.decode("utf-8"))
    return int(p.stdout)


class KokkosInitializationTestCase(unittest.TestCase):
    def test_num_threads(self):
        max_threads = GetFlag("max_threads")
        if max_threads == 1:
            self.skipTest("no host parallel backend enabled")
        for num_threads in range(1, max_threads + 1):
            self.assertEqual(num_threads,
                             GetFlag("num_threads", "--kokkos-num-threads={}".format(num_threads)))

    def test_device_id(self):
        device_count = GetFlag("device_count")
        if device_count == 0:
            self.skipTest("no device detected")
        self.assertEqual(0, GetFlag("device_id"))  # by default use the first GPU available for execution
        for device_id in range(device_count):
            self.assertEqual(device_id, GetFlag("device_id", "--kokkos-device-id={}".format(device_id)))


if __name__ == '__main__':
    unittest.main()
