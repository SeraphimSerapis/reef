// Copies LiteRT.js (@litertjs/core + @litertjs/wasm-utils) from node_modules into space/vendor/,
// with one patch: an `int8` entry in LiteRT.js's dtype table.
//
// Why: the Reef encoder is a fully int8 TFLite model (int8 input and output, Edge TPU style). The
// LiteRT.js wasm runtime runs it fine, but the JS wrapper only maps float32/int32/uint8 tensors and
// throws "Element type INT8 is not supported." while reading the model's signature. Adding the
// mapping (Int8Array <-> ElementType.INT8, byte width 1) is all that is needed; the C++ side already
// checks buffer/model element types itself.
//
//   cd space/tools && npm install && node vendor_litert.mjs
//
// The .wasm binaries are NOT vendored (about 9 MB each); the app loads them from jsDelivr at the
// same pinned version (see LITERT_VERSION in ../reef/backend.js).

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const nm = join(here, "node_modules", "@litertjs");
const out = join(here, "..", "vendor");
mkdirSync(out, { recursive: true });

const corePkg = JSON.parse(readFileSync(join(nm, "core", "package.json"), "utf8"));
const utilsPkg = JSON.parse(readFileSync(join(nm, "wasm-utils", "package.json"), "utf8"));

function replaceOnce(src, from, to, what) {
  const n = src.split(from).length - 1;
  if (n !== 1) throw new Error(`patch anchor for ${what} found ${n} times (expected 1); LiteRT.js changed?`);
  return src.replace(from, to);
}

const header = (name, version, note) =>
  `// ${name}@${version} (Apache-2.0, https://github.com/google-ai-edge/LiteRT), vendored by\n` +
  `// space/tools/vendor_litert.mjs. ${note}\n`;

let core = readFileSync(join(nm, "core", "dist", "index.js"), "utf8");
core = core.replace(/\n\/\/# sourceMappingURL=.*\s*$/, "\n");
core = replaceOnce(
  core,
  `    dtype: "uint8",
    typedArrayConstructor: Uint8Array,
    elementType: ElementType.UINT8
  }
]);`,
  `    dtype: "uint8",
    typedArrayConstructor: Uint8Array,
    elementType: ElementType.UINT8
  },
  // Reef patch: int8 tensors (fully quantised models with int8 inputs/outputs).
  {
    dtype: "int8",
    typedArrayConstructor: Int8Array,
    elementType: ElementType.INT8
  }
]);`,
  "dtype table",
);
core = replaceOnce(
  core,
  `import { createWasmLib } from "@litertjs/wasm-utils";`,
  `import { createWasmLib } from "./litert-wasm-utils.js";`,
  "wasm-utils import",
);
writeFileSync(
  join(out, "litert-core.js"),
  header("@litertjs/core", corePkg.version, "Patched: int8 dtype entry, relative wasm-utils import.") + core,
);

let utils = readFileSync(join(nm, "wasm-utils", "dist", "index.js"), "utf8");
utils = utils.replace(/\n\/\/# sourceMappingURL=.*\s*$/, "\n");
writeFileSync(join(out, "litert-wasm-utils.js"), header("@litertjs/wasm-utils", utilsPkg.version, "Unmodified.") + utils);

console.log(`vendored @litertjs/core@${corePkg.version} and @litertjs/wasm-utils@${utilsPkg.version} into ${out}`);
