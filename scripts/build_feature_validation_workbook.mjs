import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const ROOT = process.cwd();
const INPUT_DIR = path.join(ROOT, "outputs", "feature_validation");
const OUTPUT_XLSX = path.join(INPUT_DIR, "feature_validation_review.xlsx");
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
  white: "#FFFFFF",
};

const TABLES = [
  ["Concentration Summary", "concentration_summary.csv", "ConcentrationSummaryTable"],
  ["Trends", "concentration_trends.csv", "TrendSummaryTable"],
  ["Adjacent Overlap", "adjacent_concentration_overlap.csv", "AdjacentOverlapTable"],
  ["Variability", "within_image_variability.csv", "VariabilityTable"],
  ["Position Bias", "position_bias_summary.csv", "PositionBiasTable"],
  ["Position Well", "position_well_summary.csv", "PositionWellTable"],
  ["Background Control", "background_control_summary.csv", "BackgroundControlTable"],
  ["Feature Correlation", "feature_correlation_summary.csv", "FeatureCorrelationTable"],
  ["Model Feature Sets", "model_feature_sets.csv", "ModelFeatureSetsTable"],
];


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


function toCellValue(value) {
  const trimmed = value.trim();
  if (trimmed === "") return null;
  if (/^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(trimmed)) {
    const numeric = Number(trimmed);
    if (Number.isFinite(numeric)) return numeric;
  }
  return trimmed;
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


function styleTitle(sheet, address, title) {
  const range = sheet.getRange(address);
  range.merge();
  range.values = [[title]];
  range.format = {
    fill: COLORS.navy,
    font: { name: "Malgun Gothic", size: 18, bold: true, color: COLORS.white },
    verticalAlignment: "center",
  };
  range.format.rowHeight = 34;
}


function styleHeader(range) {
  range.format = {
    fill: COLORS.blue,
    font: { name: "Malgun Gothic", size: 9, bold: true, color: COLORS.white },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: "#9EADBA" },
  };
  range.format.rowHeight = 36;
}


function styleDataSheet(sheet, rows, tableName) {
  const rowCount = rows.length;
  const colCount = rows[0].length;
  const lastColumn = columnName(colCount - 1);
  const fullRange = sheet.getRange(`A1:${lastColumn}${rowCount}`);
  fullRange.values = rows;
  fullRange.format.font = { name: "Aptos", size: 9 };
  styleHeader(sheet.getRange(`A1:${lastColumn}1`));
  sheet.tables.add(`A1:${lastColumn}${rowCount}`, true, tableName);
  sheet.freezePanes.freezeRows(1);
  sheet.showGridLines = false;
  for (let index = 0; index < colCount; index += 1) {
    const letter = columnName(index);
    const header = String(rows[0][index] ?? "");
    let width = 14;
    if (/id$|analyte|feature$|feature_|scope|rationale|role|label/.test(header)) width = 22;
    if (/interpretation|rationale/.test(header)) width = 45;
    if (/concentration/.test(header)) width = 18;
    sheet.getRange(`${letter}:${letter}`).format.columnWidth = width;
    if (/interpretation|rationale/.test(header) && rowCount > 1) {
      sheet.getRange(`${letter}2:${letter}${rowCount}`).format.wrapText = true;
      sheet.getRange(`${letter}2:${letter}${rowCount}`).format.rowHeight = 30;
    }
  }
  if (rowCount > 1) {
    const numericStart = sheet.getRange(`D2:${lastColumn}${rowCount}`);
    numericStart.format.numberFormat = "0.0000";
  }
  return { rowCount, colCount, lastColumn };
}


async function savePreview(workbook, sheetName, range, fileName, scale = 1) {
  const preview = await workbook.render({
    sheetName,
    range,
    scale,
    format: "png",
  });
  await fs.writeFile(
    path.join(PREVIEW_DIR, fileName),
    new Uint8Array(await preview.arrayBuffer()),
  );
}


async function main() {
  await fs.mkdir(PREVIEW_DIR, { recursive: true });
  const workbook = Workbook.create();
  const overview = workbook.worksheets.add("Overview");
  const sheetMetadata = new Map();

  for (const [sheetName, fileName, tableName] of TABLES) {
    const csvText = await fs.readFile(path.join(INPUT_DIR, fileName), "utf8");
    const rows = parseCsv(csvText);
    const sheet = workbook.worksheets.add(sheetName);
    const metadata = styleDataSheet(sheet, rows, tableName);
    sheetMetadata.set(sheetName, metadata);
  }

  overview.showGridLines = false;
  styleTitle(overview, "A1:F1", "IFMBE 색상 특징 검증 결과");
  overview.getRange("A3:B3").values = [["검증 항목", "결과"]];
  styleHeader(overview.getRange("A3:B3"));
  overview.getRange("A4:A10").values = [
    ["원본 이미지"],
    ["포도당 이미지"],
    ["케톤 이미지"],
    ["RGB 주 특징"],
    ["HSV 주 특징"],
    ["최대 위치 잔차"],
    ["자동 제외 패치"],
  ];
  overview.getRange("B4:B10").formulas = [
    ["=COUNTA('Concentration Summary'!$B$2:$B$286)/15"],
    ["=COUNTIF('Concentration Summary'!$A$2:$A$286,\"glucose\")/15"],
    ["=COUNTIF('Concentration Summary'!$A$2:$A$286,\"ketone\")/15"],
    ["=COUNTIF('Model Feature Sets'!$A$2:$A$28,\"RGB_primary\")"],
    ["=COUNTIF('Model Feature Sets'!$A$2:$A$28,\"HSV_primary\")"],
    ["=MAX('Position Bias'!$H$2:$H$13)"],
    ["=0"],
  ];
  overview.getRange("A4:B10").format = {
    borders: { preset: "all", style: "thin", color: "#B8C4CE" },
    font: { name: "Malgun Gothic", size: 10 },
  };
  overview.getRange("A4:A10").format = { fill: COLORS.paleBlue, font: { name: "Malgun Gothic", size: 10, bold: true } };
  overview.getRange("B4:B10").format.horizontalAlignment = "right";
  overview.getRange("B4:B8").format.numberFormat = "0";
  overview.getRange("B9:B9").format.numberFormat = "0.000";
  overview.getRange("B10:B10").format.numberFormat = "0";

  overview.getRange("D3:F3").values = [["검증 원칙", "적용 방법", "해석 범위"]];
  styleHeader(overview.getRange("D3:F3"));
  overview.getRange("D4:F8").values = [
    ["농도 추세", "농도별 원본 이미지 대표값 사용", "11개·8개 이미지의 기술적 추세"],
    ["패치 변동", "이미지별 96개 패치 IQR/median", "이미지 내부 기술적 변동"],
    ["위치 편향", "이미지별 median 제거 후 IQR 단위화", "농도와 분리된 배열 위치 효과"],
    ["배경 대조", "흰 기판 RGB만 별도 평가", "센서 화학과 무관한 음성 대조"],
    ["데이터 제외", "사전 고정 QC 기준만 적용", "결과 기반 사후 제외 없음"],
  ];
  overview.getRange("D4:F8").format = {
    borders: { preset: "all", style: "thin", color: "#B8C4CE" },
    wrapText: true,
    verticalAlignment: "top",
    font: { name: "Malgun Gothic", size: 9 },
  };
  overview.getRange("D4:D8").format = { fill: COLORS.paleBlue, font: { name: "Malgun Gothic", size: 9, bold: true } };

  overview.getRange("A12:F12").merge();
  overview.getRange("A12:F12").values = [["주요 결과"]];
  overview.getRange("A12:F12").format = {
    fill: COLORS.green,
    font: { name: "Malgun Gothic", size: 11, bold: true, color: COLORS.white },
  };
  overview.getRange("A13:F13").values = [["분석물", "주요 농도 추세", "인접 농도 중첩", "배경 대조", "위치 효과", "해석"]];
  styleHeader(overview.getRange("A13:F13"));
  overview.getRange("A14:F16").values = [
    ["포도당", "G median ρ=0.900; Hue ρ=-0.900", "10-15 mg/mL의 RGB·S·V 중첩", "최대 |ρ|=0.287", "가장자리 intensity +0.598 IQR", "전반적 추세와 국소 비단조성 공존"],
    ["케톤", "Hue ρ=-1.000; G·R·V ρ=-0.976", "0.05-0.1 mg/mL의 R·G·V 중첩", "G 배경 ρ=-0.554", "가장자리 S -0.743 IQR", "Hue와 G 중심의 강한 단조 추세"],
    ["공통", "센서 특징이 배경보다 강한 농도 연관성", "일부 인접 농도 분포 중첩", "주 모델 입력에서 제외", "행·가장자리 방향 편향 확인", "위치별 그룹 교차검증 필요"],
  ];
  overview.getRange("A14:F16").format = {
    borders: { preset: "all", style: "thin", color: "#B8C4CE" },
    wrapText: true,
    verticalAlignment: "top",
    font: { name: "Malgun Gothic", size: 9 },
  };
  overview.getRange("A14:A16").format = { fill: COLORS.paleGreen, font: { name: "Malgun Gothic", size: 9, bold: true } };
  overview.getRange("A14:F16").format.rowHeight = 44;

  overview.getRange("A18:F18").merge();
  overview.getRange("A18:F18").values = [["모델링 결정"]];
  overview.getRange("A18:F18").format = {
    fill: COLORS.amber,
    font: { name: "Malgun Gothic", size: 11, bold: true, color: COLORS.white },
  };
  overview.getRange("A19:F23").merge(true);
  overview.getRange("A19:A23").values = [
    ["• 주 분석: RGB median 3개와 HSV의 sin(H), cos(H), S median, V median을 별도 모델로 비교함"],
    ["• RGB+HSV, chromaticity, 배경 보정 및 texture 특징은 보조 분석으로 제한함"],
    ["• 흰 배경 RGB는 농도 예측 모델이 아닌 음성 대조 모델에만 사용함"],
    ["• 위치 편향이 확인되어 well_id가 train과 test에 겹치지 않는 그룹 교차검증을 권장함"],
    ["• 현재 자료에서 고정 QC 기준으로 제외되는 패치는 없음"],
  ];
  overview.getRange("A19:F23").format = {
    wrapText: true,
    verticalAlignment: "top",
    font: { name: "Malgun Gothic", size: 9 },
  };
  overview.getRange("A19:F23").format.rowHeight = 26;
  overview.getRange("A:A").format.columnWidth = 17;
  overview.getRange("B:B").format.columnWidth = 25;
  overview.getRange("C:C").format.columnWidth = 24;
  overview.getRange("D:D").format.columnWidth = 23;
  overview.getRange("E:E").format.columnWidth = 25;
  overview.getRange("F:F").format.columnWidth = 30;
  overview.freezePanes.freezeRows(1);

  const overviewInspect = await workbook.inspect({
    kind: "region",
    sheetId: "Overview",
    range: "A1:F23",
    maxChars: 5000,
  });
  await fs.writeFile(
    path.join(PREVIEW_DIR, "overview_inspection.txt"),
    overviewInspect.ndjson ?? String(overviewInspect),
    "utf8",
  );
  const formulaErrors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 100 },
    summary: "feature-validation workbook formula error scan",
  });
  await fs.writeFile(
    path.join(PREVIEW_DIR, "formula_error_scan.txt"),
    formulaErrors.ndjson ?? String(formulaErrors),
    "utf8",
  );

  await savePreview(workbook, "Overview", "A1:F23", "00_overview.png", 1.1);
  let previewIndex = 1;
  for (const [sheetName] of TABLES) {
    const metadata = sheetMetadata.get(sheetName);
    const lastPreviewRow = Math.min(metadata.rowCount, 18);
    await savePreview(
      workbook,
      sheetName,
      `A1:${metadata.lastColumn}${lastPreviewRow}`,
      `${String(previewIndex).padStart(2, "0")}_${sheetName.toLowerCase().replaceAll(" ", "_")}.png`,
      0.8,
    );
    previewIndex += 1;
  }

  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(OUTPUT_XLSX);
  await fs.rm(`${OUTPUT_XLSX}.inspect.ndjson`, { force: true });
  console.log(`Saved ${OUTPUT_XLSX}`);
}


await main();
