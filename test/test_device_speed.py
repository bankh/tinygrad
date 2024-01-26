import unittest
from tinygrad import Device
from tinygrad.helpers import Timing, Profiling
from tinygrad.device import Compiled

@unittest.skipIf(not isinstance(Device[Device.DEFAULT], Compiled), "only for compiled backend")
class TestDeviceSpeed(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.dev = Device[Device.DEFAULT]
    cls.empty = Device[Device.DEFAULT].renderer("test", [])

  def test_empty_compile(self):
    with Timing("compiler "):
      self.dev.compiler(self.empty)

  def test_launch_speed(self):
    prg_bin = self.dev.compiler(self.empty)
    prg = self.dev.runtime("test", prg_bin)
    for _ in range(10): prg() # ignore first launches
    with Timing("launch 1000x "):
      for _ in range(1000): prg()
    with Timing("launch 1000x with wait "):
      for _ in range(1000): prg(wait=True)

  def test_profile_launch_speed(self):
    prg_bin = self.dev.compiler(self.empty)
    prg = self.dev.runtime("test", prg_bin)
    for _ in range(10): prg() # ignore first launches
    with Profiling():
      for _ in range(1000): prg()

if __name__ == '__main__':
  unittest.main()