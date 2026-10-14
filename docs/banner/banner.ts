// The README banner: coral "REEF" letters over a rolling ascii sea, made with ascii.rest
// (MIT, https://ascii.rest). Rebuild with: cd docs/banner && npm install && npm run build
// (Node 22.18+ runs this .ts file directly). Writes ../banner.svg and ../banner.dark.svg.
import { writeFileSync } from "node:fs";
import { banner } from "ascii.rest/banner";
import { column, sea } from "ascii.rest/kit";
import { svg } from "ascii.rest/svg";

const coral = { light: ["#e8505b", "#f2777a"], dark: ["#ff7f6e", "#ff9e8a"] };
const letters = banner("REEF", { font: "round", shadow: "rounded", color: coral, glint: { every: 6 } });
const water = sea({ palette: "ocean", cols: 88, rows: 13, horizon: "high", period: 8 });
const reef = column([water.behind(letters), "tiny typed-decision models  ~  grown on a Coral Edge TPU"], { gap: 0 })
  .named("reef", "REEF in coral letters over a rolling ascii sea");

const label = "Reef: tiny typed-decision models, grown on a Coral Edge TPU";
writeFileSync(new URL("../banner.svg", import.meta.url), svg(reef, { label }));
writeFileSync(new URL("../banner.dark.svg", import.meta.url), svg(reef, { dark: true, label }));
