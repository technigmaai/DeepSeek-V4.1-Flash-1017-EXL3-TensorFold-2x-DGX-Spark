#!/usr/bin/env python3
"""Apply the local DeepSeek fixes after the pinned display-KV transformer.

Exact anchors and syntax checks make source drift fail the image build. Validate
both edits before writing either file. This runs offline and needs no GPU.
"""
from __future__ import annotations

import ast
from pathlib import Path
import sys

FAMILY = Path("src/tensorfold/families/deepseek_v41/cuda")


def transform(name: str, source: str) -> str:
    if name == "app.py":
        old = "        super().__init__(*args, **kwargs)\n        template = self.template\n"
        new = (
            "        super().__init__(*args, **kwargs)\n"
            "        # DeepSeek uses DsVision rather than App's generic image decoder.\n"
            "        if self.vision is not None and kwargs.get(\"vision_max_images\") is not None:\n"
            "            self.vision.max_images = int(kwargs[\"vision_max_images\"])\n"
            "        template = self.template\n"
        )
    elif name == "multi.py":
        old = "            self.filling.remove(s)\n"
        new = (
            "            # Stream value equality can compare image tensors. Use identity.\n"
            "            for index, queued in enumerate(self.filling):\n"
            "                if queued is s:\n"
            "                    del self.filling[index]\n"
            "                    break\n"
            "            else:\n"
            "                raise ValueError(\"completed stream is not in the filling queue\")\n"
        )
    else:
        raise ValueError(f"unsupported local fix: {name}")
    if source.count(old) != 1:
        raise ValueError(f"{name}: expected one unmodified local-fix anchor")
    result = source.replace(old, new)
    ast.parse(result, filename=name)
    return result


def install(checkout: Path) -> None:
    edits = {checkout / FAMILY / name: transform(name, (checkout / FAMILY / name).read_text())
             for name in ("app.py", "multi.py")}
    for path, source in edits.items():
        path.write_text(source)


if __name__ == "__main__":
    install(Path(sys.argv[1]))
