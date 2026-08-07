import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const ROOT = process.cwd();
const INPUT_DIR = path.join(ROOT, "outputs", "data_splits");
const OUTPUT_XLSX = path.join(INPUT_DIR, "data_splits_review.xlsx");
const PREVIEW_DIR = path.join(INPUT_DIR, ".workbook_previews");

const COLORS = {
  navy: "#183A5A",
  blue: "#2F75B5",
  paleBlue: "#DCEAF5",
  green: "#548235",
  paleGreen: "#E2F0D9",
  amber: "#BF9000",
  paleAmber: "#FFF2CC",
  red: "#C65954",
  paleRed: "#FCE4D6",
  grey: "#E7E6E6",
  darkGrey: "#595959",
  white: "#FFFFFF",
};

const DATA_SHEETS = [
  ["Well Assignment", "well_fold_matrix.csv", "WellAssignmentTable"],
  ["Split Summary", "split_summary.csv", "SplitSummaryTable"],
  ["Concentration Balance", "concentration_balance.csv", "ConcentrationBalanceTable"],
  ["Patch Outer Fold", "outer_fold_assignments.csv", "PatchOuterFoldTable"],
  ["Nested Roles", "nested_split_assignments.csv", "NestedRolesTable"],
];


function toCellValue(value) {
  const trimmed = value.trim();
  if (trimmed === "") return null;
  if (trimmed === "True" || trimmed === "true") return true;
  if (trimmed === "False" || trimmed === "false") return false;
  if (/^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(trimmed)) {
    const numeric = Number(trimmed);
    if (Number.isFinite(numeric)) return numeric;
  }
  return trimmed;
}


function parseCsv(text) {
  const rows = [];
  let row = [];
  let value = "";
  let quoted = false;
  const normalized = text.replace(/^\uFEFF/, "");
  for (let index = 0; index < normalized.length; index += 1) {
    const char = normalized[index];
    const next = normalized[index + 1];
    if (quoted) {
      if (char === '"' && next === '"') {
        value += '"';
        index += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        value += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ",") {
      row.push(toCellValue(value));
      value = "";
    } else if (char === "\n") {
      row.push(toCellValue(value.replace(/\r$/, "")));
      rows.push(row);
      row = [];
      value = "";
    } else {
      value += char;
    }
  }
  if (value.length || row.length) {
    row.push(toCellValue(value.replace(/\r$/, "")));
    rows.push(row);
  }
  return rows;
}


function columnName(index) {
  let value = index + 1;
  let output = "";
  while (value > 0) {
    const remainder = (value - 1) % 26;
    output = String.fromCharCode(65 + remainder) + output;
    value = Math.floor((value - 1) / 26);
  }
  return output;
}


function styleHeader(range) {
  range.format = {
    fill: COLORS.blue,
    font: { name: "Malgun Gothic", size: 9, bold: true, color: COLORS.white },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "inside", style: "thin", color: "#B4C7E7" },
  };
  range.format.rowHeight = 30;
}


function setColumnWidths(sheet, headers) {
  const wideText = new Set(["patch_id", "image_id", "crop_file"]);
  const mediumText = new Set(["well_id", "analyte", "pipeline", "role", "ml_role", "dl_role"]);
  headers.forEach((header, index) => {
    const column = sheet.getRange(`${columnName(index)}:${columnName(index)}`);
    if (wideText.has(header)) {
      column.format.columnWidth = header === "crop_file" ? 42 : 27;
    } else if (mediumText.has(header)) {
      column.format.columnWidth = 15;
    } else if (header.includes("percentage")) {
      column.format.columnWidth = 18;
    } else if (header.includes("concentration")) {
      column.format.columnWidth = 19;
    } else {
      column.format.columnWidth = 14;
    }
  });
}


function applySemanticFormats(sheet, headers, rowCount) {
  headers.forEach((header, index) => {
    const letter = columnName(index);
    const dataRange = sheet.getRange(`${letter}2:${letter}${rowCount}`);
    if (header === "percentage_of_analyte") {
      dataRange.format.numberFormat = "0.0%";
    } else if (header === "concentration_mg_ml") {
      dataRange.format.numberFormat = "0.00###";
    } else if (header.includes("fraction")) {
      dataRange.format.numberFormat = "0.000";
    } else if (
      header.includes("count") ||
      header.includes("fold") ||
      header.includes("order") ||
      header === "grid_row" ||
      header === "grid_col"
    ) {
      dataRange.format.numberFormat = "0";
    }
    if (header === "ml_role" || header === "dl_role" || header === "role") {
      dataRange.conditionalFormats.add("containsText", {
        text: "test",
        format: { fill: COLORS.paleRed, font: { color: COLORS.red, bold: true } },
      });
      dataRange.conditionalFormats.add("containsText", {
        text: "validation",
        format: { fill: COLORS.paleAmber, font: { color: COLORS.amber, bold: true } },
      });
      dataRange.conditionalFormats.add("containsText", {
        text: "train",
        format: { fill: COLORS.paleGreen, font: { color: COLORS.green } },
      });
    }
  });
}


async function addDataSheet(workbook, sheetName, csvName, tableName) {
  const csvText = await fs.readFile(path.join(INPUT_DIR, csvName), "utf8");
  const rows = parseCsv(csvText);
  const headers = rows[0];
  const rowCount = rows.length;
  const columnCount = headers.length;
  const lastColumn = columnName(columnCount - 1);
  const sheet = workbook.worksheets.add(sheetName);
  sheet.showGridLines = false;
  sheet.getRange(`A1:${lastColumn}${rowCount}`).values = rows;
  styleHeader(sheet.getRange(`A1:${lastColumn}1`));
  sheet.getRange(`A2:${lastColumn}${rowCount}`).format = {
    font: { name: "Aptos", size: 9, color: "#262626" },
    verticalAlignment: "center",
  };
  sheet.getRange(`A2:${lastColumn}${rowCount}`).format.rowHeight = 18;
  const table = sheet.tables.add(`A1:${lastColumn}${rowCount}`, true, tableName);
  table.style = "TableStyleMedium2";
  table.showBandedRows = true;
  table.showFilterButton = true;
  setColumnWidths(sheet, headers);
  applySemanticFormats(sheet, headers, rowCount);
  sheet.freezePanes.freezeRows(1);
  if (sheetName === "Well Assignment") sheet.freezePanes.freezeColumns(1);
  return { sheet, rows, headers, lastColumn, rowCount };
}


function buildOverview(workbook) {
  const sheet = workbook.worksheets.getItem("Overview");
  sheet.showGridLines = false;
  sheet.getRange("A1:H1").merge();
  sheet.getRange("A1").values = [["Group-aware Cross-validation Split Review"]];
  sheet.getRange("A1:H1").format = {
    fill: COLORS.navy,
    font: { name: "Malgun Gothic", size: 18, bold: true, color: COLORS.white },
    verticalAlignment: "center",
  };
  sheet.getRange("A1:H1").format.rowHeight = 36;

  sheet.getRange("A3:B3").values = [["검증 항목", "값"]];
  styleHeader(sheet.getRange("A3:B3"));
  sheet.getRange("A4:A9").values = [
    ["전체 패치 수"],
    ["포도당 패치 수"],
    ["케톤 패치 수"],
    ["고유 well_id 수"],
    ["외부 fold 수"],
    ["고정 seed"],
  ];
  sheet.getRange("B4:B9").formulas = [
    ["=COUNTA('Patch Outer Fold'!$A$2:$A$1825)"],
    ["=COUNTIF('Patch Outer Fold'!$C$2:$C$1825,\"glucose\")"],
    ["=COUNTIF('Patch Outer Fold'!$C$2:$C$1825,\"ketone\")"],
    ["=COUNTA('Well Assignment'!$A$2:$A$97)"],
    ["=MAX('Well Assignment'!$E$2:$E$97)"],
    ["='Protocol'!$B$2"],
  ];
  sheet.getRange("A4:A9").format = { fill: COLORS.paleBlue, font: { name: "Malgun Gothic", bold: true } };
  sheet.getRange("B4:B9").format = { font: { name: "Aptos", bold: true }, numberFormat: "#,##0" };
  sheet.getRange("A3:B9").format.borders = { preset: "outside", style: "thin", color: "#A6A6A6" };

  sheet.getRange("D3:H3").merge();
  sheet.getRange("D3").values = [["확정한 분할 원칙"]];
  styleHeader(sheet.getRange("D3:H3"));
  sheet.getRange("D4:H8").merge(true);
  sheet.getRange("D4:D8").values = [
    ["외부 분할: 농도 순번과 edge/interior를 층화한 5-fold"],
    ["그룹 변수: well_id, 동일 위치의 19개 농도 패치에 같은 역할 배정"],
    ["ML: 외부 train/test, 외부 train 내부의 5-fold로 하이퍼파라미터 선택"],
    ["DL: inner_fold 1을 validation, inner_fold 2–5를 train으로 사용"],
    ["해석 범위: 위치 그룹을 분리한 패치 수준 내부 교차검증"],
  ];
  sheet.getRange("D4:H8").format = {
    fill: "#F7FAFC",
    font: { name: "Malgun Gothic", size: 10, color: "#333333" },
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "outside", style: "thin", color: "#A6A6A6" },
  };
  sheet.getRange("D4:H8").format.rowHeight = 28;

  sheet.getRange("A12:F12").values = [[
    "외부 fold",
    "DL Train well",
    "DL Validation well",
    "Outer Test well",
    "Test edge",
    "Test interior",
  ]];
  styleHeader(sheet.getRange("A12:F12"));
  sheet.getRange("A13:A17").values = [[1], [2], [3], [4], [5]];
  for (let fold = 1; fold <= 5; fold += 1) {
    const row = 12 + fold;
    const roleColumn = columnName(4 + fold);
    sheet.getRange(`B${row}:F${row}`).formulas = [[
      `=COUNTIF('Well Assignment'!$${roleColumn}$2:$${roleColumn}$97,\"train\")`,
      `=COUNTIF('Well Assignment'!$${roleColumn}$2:$${roleColumn}$97,\"validation\")`,
      `=COUNTIF('Well Assignment'!$${roleColumn}$2:$${roleColumn}$97,\"test\")`,
      `=COUNTIFS('Well Assignment'!$E$2:$E$97,$A${row},'Well Assignment'!$D$2:$D$97,1)`,
      `=D${row}-E${row}`,
    ]];
  }
  sheet.getRange("A13:F17").format = {
    font: { name: "Aptos", size: 10 },
    horizontalAlignment: "center",
    numberFormat: "0",
    borders: { preset: "inside", style: "thin", color: "#D9E2F3" },
  };
  sheet.getRange("A13:F17").conditionalFormats.add("Custom", {
    formula: "=MOD(ROW(),2)=1",
    format: { fill: "#F3F6FA" },
  });

  sheet.getRange("A20:H20").merge();
  sheet.getRange("A20").values = [["해석상 주의"]];
  sheet.getRange("A20:H20").format = {
    fill: COLORS.amber,
    font: { name: "Malgun Gothic", bold: true, color: COLORS.white },
  };
  sheet.getRange("A21:H22").merge();
  sheet.getRange("A21").values = [[
    "농도별 원본 이미지가 1장뿐이므로 서로 다른 well_id라도 같은 원본 이미지에서 추출됨. 새로운 플레이트·촬영 조건에 대한 외부 일반화 성능이 아닌 내부 검증 결과임.",
  ]];
  sheet.getRange("A21:H22").format = {
    fill: COLORS.paleAmber,
    font: { name: "Malgun Gothic", size: 10, color: "#7F6000" },
    wrapText: true,
    verticalAlignment: "center",
    borders: { preset: "outside", style: "thin", color: "#D6B656" },
  };

  ["A", "B", "C", "D", "E", "F", "G", "H"].forEach((column) => {
    sheet.getRange(`${column}:${column}`).format.columnWidth = column === "A" ? 22 : 16;
  });
  sheet.getRange("D:H").format.columnWidth = 17;
  sheet.freezePanes.freezeRows(1);
  return sheet;
}


function buildProtocol(workbook, config) {
  const sheet = workbook.worksheets.add("Protocol");
  sheet.showGridLines = false;
  const entries = [
    ["random_seed", config.random_seed],
    ["n_splits", config.n_splits],
    ["outer_splitter", config.outer_splitter],
    ["outer_stratification", config.outer_stratification],
    ["outer_group", config.outer_group],
    ["inner_splitter", config.inner_splitter],
    ["inner_random_seed_rule", config.inner_random_seed_rule],
    ["inner_stratification", config.inner_stratification],
    ["inner_group", config.inner_group],
    ["ml_usage", config.ml_usage],
    ["dl_usage", config.dl_usage],
    ["interpretation", config.interpretation],
    ["known_limitation", config.known_limitation],
  ];
  sheet.getRange(`A1:B${entries.length + 1}`).values = [["parameter", "value"], ...entries];
  styleHeader(sheet.getRange("A1:B1"));
  sheet.getRange(`A2:A${entries.length + 1}`).format = {
    fill: COLORS.paleBlue,
    font: { name: "Aptos", size: 9, bold: true },
  };
  sheet.getRange(`B2:B${entries.length + 1}`).format = {
    font: { name: "Aptos", size: 9 },
    wrapText: true,
  };
  sheet.getRange("A:A").format.columnWidth = 26;
  sheet.getRange("B:B").format.columnWidth = 78;
  sheet.getRange(`A2:B${entries.length + 1}`).format.rowHeight = 26;
  sheet.getRange(`A${entries.length + 1}:B${entries.length + 1}`).format.rowHeight = 42;
  const table = sheet.tables.add(`A1:B${entries.length + 1}`, true, "ProtocolTable");
  table.style = "TableStyleMedium2";
  sheet.freezePanes.freezeRows(1);
  return sheet;
}


const workbook = Workbook.create();
workbook.worksheets.add("Overview");
const dataSheets = {};
for (const [sheetName, csvName, tableName] of DATA_SHEETS) {
  dataSheets[sheetName] = await addDataSheet(workbook, sheetName, csvName, tableName);
}
const config = JSON.parse(await fs.readFile(path.join(INPUT_DIR, "run_config.json"), "utf8"));
buildProtocol(workbook, config);
buildOverview(workbook);

await fs.mkdir(PREVIEW_DIR, { recursive: true });
const renderTargets = [
  ["Overview", "A1:H22"],
  ["Well Assignment", "A1:J18"],
  ["Split Summary", "A1:J18"],
  ["Concentration Balance", "A1:J18"],
  ["Patch Outer Fold", "A1:J16"],
  ["Nested Roles", "A1:M16"],
  ["Protocol", "A1:B14"],
];
for (const [sheetName, range] of renderTargets) {
  const preview = await workbook.render({ sheetName, range, scale: 1.25, format: "png" });
  const bytes = new Uint8Array(await preview.arrayBuffer());
  await fs.writeFile(path.join(PREVIEW_DIR, `${sheetName.replaceAll(" ", "_")}.png`), bytes);
}

const overviewInspect = await workbook.inspect({
  kind: "table",
  range: "Overview!A1:H22",
  include: "values,formulas",
  tableMaxRows: 24,
  tableMaxCols: 10,
});
console.log(overviewInspect.ndjson);
const formulaErrors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(formulaErrors.ndjson);

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(OUTPUT_XLSX);
console.log(`Saved ${OUTPUT_XLSX}`);
