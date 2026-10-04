import importlib.util, sys, unittest
name = 'torchtitan.models.kimi_k3.pipeline_parallel.vision_dep.plan'
spec = importlib.util.spec_from_file_location(name, sys.argv[1])
mod = importlib.util.module_from_spec(spec)
sys.modules[name] = mod
spec.loader.exec_module(mod)
sys.path.insert(0, 'tests/unit_tests/cpu')
import test_kimi_k3_vision_dep_plan as t
assert t.VisionDepPlan is mod.VisionDepPlan
suite = unittest.TestSuite()
for n in ('test_three_ranks_and_six_microbatches_lay_out_as_in_the_k3_report',
          'test_the_first_and_last_pipeline_degree_microbatches_run_outside_the_schedule',
          'test_a_backward_uses_the_rest_of_an_idle_run_that_began_before_its_gradient'):
    suite.addTest(t.TestVisionDepPlan(n))
r = unittest.TextTestRunner(verbosity=0).run(suite)
print(sys.argv[1].split('/')[-1], 'failures', len(r.failures), 'errors', len(r.errors), 'subtests failed', sum(1 for f in r.failures))
