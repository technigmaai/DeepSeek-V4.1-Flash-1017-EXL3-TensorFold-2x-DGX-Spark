"""Run the actual scheduler completion method with CPU tensors and stubbed GPU work."""
import ast
import importlib.util
import os
import sys
import time
import unittest
from types import SimpleNamespace
from dataclasses import dataclass
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('TF_DISPLAY_SOURCE', '/tmp/opencode/tfv41-display-source'))
FAMILY = SOURCE / 'src/tensorfold/families/deepseek_v41/cuda'

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value

apply = module('local_test_display_apply', HERE / 'apply.py')
fixes = module('local_test_engine_fixes', HERE / 'overlay/local_engine_fixes.py')
spec = importlib.util.spec_from_file_location('actual_streams', SOURCE / 'src/tensorfold/cuda/streams.py')
streams = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = streams
spec.loader.exec_module(streams)
Stream = streams.Stream


@dataclass
class ImageState:
    patches: torch.Tensor


def load_fill(source):
    filename = str(FAMILY / "multi.py")
    tree = ast.parse(source)
    decoder = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'MultiDecoder')
    method = next(node for node in decoder.body if isinstance(node, ast.FunctionDef) and node.name == '_fill')
    decoder.body = [method]
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), decoder], type_ignores=[])
    scope = {'Stream': Stream, 'time': time}
    exec(compile(ast.fix_missing_locations(module), filename, 'exec'), scope)
    return scope['MultiDecoder']._fill


def complete():
    if False:
        yield
    return object()


def request(sid, images=True):
    stream = Stream([1, 2, 3], 10, sid=sid,
                    vision=ImageState(torch.tensor([1, 2])) if images else None)
    stream.steps = complete()
    stream.filled = 0
    return stream


class FakeDecoder:
    def __init__(self, queue):
        self.filling = queue
        self.streams = {}
        self.errors = []
        self.steps = []
    def _step(self, active): self.steps.append(active)
    def _agree(self, *args): pass
    def _shape(self): return []
    def _first(self, *args): return 7
    def _ends(self, *args): return ()
    def _grammar_end(self, *args): pass
    def _broken(self, error): self.errors.append(error)
    def _stream_failed(self, stream, error): raise error


display_source = apply.transform('multi.py', (FAMILY / 'multi.py').read_text())
fixed = load_fill(fixes.transform('multi.py', display_source))
original = load_fill(display_source)


class SchedulerIdentityTests(unittest.TestCase):
    def test_original_reproduces_the_reported_crash(self):
        queue = [request(0), request(1)]
        decoder = FakeDecoder(queue)
        with self.assertRaisesRegex(RuntimeError, 'Boolean value of Tensor with more than one value is ambiguous'):
            original(decoder, queue[1])
        self.assertEqual(len(decoder.errors), 1)
        self.assertEqual(decoder.steps, [True, False])

    def test_every_completion_position_with_one_to_four_image_requests(self):
        for length in range(1, 5):
            for index in range(length):
                with self.subTest(length=length, index=index):
                    queue = [request(i) for i in range(length)]
                    target = queue[index]
                    remaining = [id(s) for s in queue if s is not target]
                    decoder = FakeDecoder(queue)
                    self.assertEqual(fixed(decoder, target), [])
                    self.assertEqual([id(s) for s in queue], remaining)
                    self.assertIs(decoder.streams[target.sid], target)
                    self.assertEqual(target.out, [7])
                    self.assertIsNone(target.steps)
                    self.assertEqual(decoder.errors, [])
                    self.assertEqual(decoder.steps, [True, False])

    def test_text_only_completion_preserves_other_requests(self):
        queue = [request(0, False), request(1, False)]
        first, target = queue
        decoder = FakeDecoder(queue)
        fixed(decoder, target)
        self.assertEqual(len(queue), 1)
        self.assertIs(queue[0], first)

    def test_incomplete_prefill_keeps_request_queued(self):
        target = request(0)
        target.steps = iter([1])
        decoder = FakeDecoder([target])
        self.assertEqual(fixed(decoder, target), [])
        self.assertIs(decoder.filling[0], target)
        self.assertEqual(target.filled, 1)
        self.assertEqual(decoder.streams, {})

    def test_missing_request_still_fails_the_invariant(self):
        decoder = FakeDecoder([request(0)])
        with self.assertRaisesRegex(ValueError, 'not in the filling queue'):
            fixed(decoder, request(1))
        self.assertEqual(len(decoder.errors), 1)
        self.assertEqual(decoder.steps, [True, False])





class BaseApp:
    def __init__(self, *args, **kwargs):
        self.vision = args[0].vision
        self.template = SimpleNamespace(render=lambda *args, **kwargs: '')



class VisionImageLimitTests(unittest.TestCase):
    def test_configured_limit_reaches_real_decoder(self):
        source = ast.parse(fixes.transform('app.py', apply.transform('app.py', (FAMILY / 'app.py').read_text())))
        nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
        nodes += [node for node in source.body if isinstance(node, ast.ClassDef) and node.name == 'DsApp']
        namespace = {'App': BaseApp, 'EFFORT_VALUES': {}, 'Any': object}
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), 'app.py', 'exec'), namespace)

        vision_source = ast.parse((FAMILY / 'vision.py').read_text())
        vision_class = next(node for node in vision_source.body if isinstance(node, ast.ClassDef) and node.name == 'DsVision')
        vision_class.body = [node for node in vision_class.body if isinstance(node, ast.FunctionDef) and node.name in ['__init__', 'split']]
        vision_nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), vision_class]
        vision_namespace = {'PLACEHOLDER': '<image>', 'decode': lambda data, cfg: data, 'load_image_bytes': lambda *args: b'image'}
        exec(compile(ast.fix_missing_locations(ast.Module(body=vision_nodes, type_ignores=[])), 'vision.py', 'exec'), vision_namespace)

        decoder = vision_namespace['DsVision'](None)
        engine = SimpleNamespace(vision=decoder)
        namespace['DsApp'](engine)
        assert decoder.max_images == 8, 'Absent option must preserve the original default'
        namespace['DsApp'](engine, vision_max_images=32)
        assert decoder.max_images == 32, 'CLI setting must reach the actual DeepSeek decoder'

        def messages(count):
            return [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,eA=='}}] * count}]

        for count in [8, 9, 32]:
            _, pictures = decoder.split(messages(count))
            assert len(pictures) == count
        try:
            decoder.split(messages(33))
        except ValueError as error:
            assert str(error) == 'a request may carry at most 32 images'
        else:
            raise AssertionError('33 images must be rejected')


class LocalPatchDriftTests(unittest.TestCase):
    def test_reinstallation_and_missing_anchors_are_rejected(self):
        for name in ('app.py', 'multi.py'):
            original = apply.transform(name, (FAMILY / name).read_text())
            patched = fixes.transform(name, original)
            with self.assertRaises(ValueError):
                fixes.transform(name, patched)
            with self.assertRaises(ValueError):
                fixes.transform(name, '')


if __name__ == '__main__':
    unittest.main(verbosity=2)
