# SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
import subprocess

COMMAND = "$<TARGET_FILE:Kokkos_CoreUnitTest_DeathTestStyle>"

def GetFlag(*extra_args, env={}):
    p = subprocess.run([COMMAND, *extra_args], env=env, capture_output=True)
    if p.returncode != 0:
        raise Exception(p.stderr.decode("utf-8"))
    return str(p.stdout.decode("utf-8")).rstrip()

class GoogleTestInitializationTestCase(unittest.TestCase):
    def test_death_style(self):
        self.assertEqual(GetFlag(), "threadsafe")

        self.assertEqual(GetFlag("--gtest_death_test_style=fast"), "fast")
        self.assertEqual(GetFlag("--gtest_death_test_style=threadsafe"), "threadsafe")

        self.assertEqual(GetFlag(env={"GTEST_DEATH_TEST_STYLE": "threadsafe"}), "threadsafe")
        self.assertEqual(GetFlag(env={"GTEST_DEATH_TEST_STYLE": "fast"}), "fast")

        self.assertEqual(GetFlag("--gtest_death_test_style=fast", env={"GTEST_DEATH_TEST_STYLE": "threadsafe"}), "fast")
        self.assertEqual(GetFlag("--gtest_death_test_style=threadsafe", env={"GTEST_DEATH_TEST_STYLE": "fast"}), "threadsafe")


if __name__ == '__main__':
    unittest.main()
