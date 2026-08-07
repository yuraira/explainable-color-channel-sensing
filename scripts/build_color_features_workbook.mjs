import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const ROOT = process.cwd();
const OUTPUT_DIR = path.join(ROOT, "outputs", "color_features");
const FEATURES_CSV = path.join(OUTPUT_DIR, "features.csv");
const SUMMARY_CSV = path.join(OUTPUT_DIR, "feature_extraction_summary.csv");
const OUTPUT_XLSX = path.join(OUTPUT_DIR, "color_features_review.xlsx");
const PREVIEW_DIR = path.join(OUTPUT_DIR, "workbook_previews");

const COLORS = {
  navy: "#17365D",
  blue: "#2F75B5",
  lightBlue: "#D9EAF7",
  paleBlue: "#EEF5FB",
  green: "#548235",
  paleGreen: "#E2F0D9",
  amber: "#BF9000",
  paleAmber: "#FFF2CC",
  grey: "#E7E6E6",
  darkGrey: "#595959",
  white: "#FFFFFF",
};


function parseSimpleCsv(text) {
  return text
    .replace(/^\uFEFF/, "")
    .trimEnd()
    .split(/\r?\n/)
    .map((line) => line.split(",").map((value) => {
      const trimmed = value.trim();
      if (trimmed === "") return "";
      const numeric = Number(trimmed);
      return Number.isFinite(numeric) ? numeric : trimmed;
    }));
}


function styleTitle(sheet, range, title) {
  range.merge();
  range.values = [[title]];
  range.format = {
    fill: COLORS.navy,
    font: { bold: true, color: COLORS.white, size: 18 },
    verticalAlignment: "center",
  };
  range.format.rowHeight = 34;
}


function styleHeader(range) {
  range.format = {
    fill: COLORS.blue,
    font: { bold: true, color: COLORS.white },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: "#A6A6A6" },
  };
}


function featureDictionary() {
  const rows = [];
  const add = (field, category, definition, unit, recommendedUse) => {
    rows.push([field, category, definition, unit, recommendedUse]);
  };

  add("patch_id", "ID", "패치별 고유 ID (image_id + well_id)", "-", "식별자; 모델 입력 제외");
  add("image_id", "ID", "농도별 원본 이미지 ID", "-", "그룹/출처 추적; 모델 입력 제외");
  add("analyte", "Label", "분석물 종류", "glucose/ketone", "분석물별 모델 분리 또는 층화");
  add("concentration_order", "Label", "분석물 내 농도 순서", "ordinal", "층화·표시용");
  add("concentration_mg_ml", "Target", "예측 대상 농도", "mg/mL", "회귀 종속변수");
  add("well_id", "Position", "8×12 배열의 위치 ID", "A01-H12", "위치 편향 검증; 기본 모델 입력 제외");
  add("grid_row", "Position", "배열 행 번호", "1-8", "위치 편향 검증; 기본 모델 입력 제외");
  add("grid_col", "Position", "배열 열 번호", "1-12", "위치 편향 검증; 기본 모델 입력 제외");
  add("crop_file", "Path", "손실 없는 패치 PNG 상대 경로", "-", "재현 및 영상 모델 입력");
  add("circle_radius_px", "Geometry", "검출된 원 반지름", "pixel", "QC; 기본 모델 입력 제외");

  for (const channel of ["r", "g", "b"]) {
    const upper = channel.toUpperCase();
    for (const stat of ["mean", "median", "std", "iqr"]) {
      const korean = { mean: "평균", median: "중앙값", std: "표준편차", iqr: "사분위 범위" }[stat];
      add(`${channel}_${stat}`, "Raw RGB", `중심 ROI의 ${upper} 채널 ${korean}`, "0-255", "RGB-only 후보 특징");
    }
  }
  for (const channel of ["s", "v"]) {
    const label = channel === "s" ? "채도(S)" : "명도(V)";
    for (const stat of ["mean", "median", "std", "iqr"]) {
      const korean = { mean: "평균", median: "중앙값", std: "표준편차", iqr: "사분위 범위" }[stat];
      add(`${channel}_${stat}`, "Raw HSV", `중심 ROI의 ${label} ${korean}`, "0-1", "HSV-only 후보 특징");
    }
  }
  add("h_circular_mean_deg", "Raw HSV", "채도 가중 Hue 원형 평균", "degree", "HSV-only 후보 특징");
  add("h_circular_std_deg", "Raw HSV", "채도 가중 Hue 원형 표준편차", "degree", "색상 균질성 후보 특징");
  add("h_resultant_length", "Raw HSV", "Hue 집중도; 1에 가까울수록 방향이 일관됨", "0-1", "색상 균질성 후보 특징");
  add("h_sin_weighted", "Raw HSV", "채도 가중 Hue 사인 성분", "-1 to 1", "Hue 경계 문제를 피하는 모델 입력");
  add("h_cos_weighted", "Raw HSV", "채도 가중 Hue 코사인 성분", "-1 to 1", "Hue 경계 문제를 피하는 모델 입력");

  for (const channel of ["r", "g", "b"]) {
    const upper = channel.toUpperCase();
    add(`${channel}_chromaticity_mean`, "Normalized color", `픽셀별 ${upper}/(R+G+B)의 평균`, "0-1", "밝기 영향 완화 후보 특징");
    add(`${channel}_chromaticity_median`, "Normalized color", `픽셀별 ${upper}/(R+G+B)의 중앙값`, "0-1", "밝기 영향 완화 후보 특징");
  }
  add("intensity_mean", "Intensity", "픽셀별 (R+G+B)/3의 평균", "0-255", "밝기 의존성 평가");
  add("intensity_median", "Intensity", "픽셀별 (R+G+B)/3의 중앙값", "0-255", "밝기 의존성 평가");

  for (const channel of ["r", "g", "b"]) {
    const upper = channel.toUpperCase();
    add(`${channel}_bg_median`, "Local background", `패치 밖 고리의 ${upper} 중앙값`, "0-255", "촬영/조명 QC");
  }
  for (const channel of ["r", "g", "b"]) {
    const upper = channel.toUpperCase();
    add(`${channel}_delta_bg`, "Background adjusted", `${upper} ROI 중앙값 - 국소 배경 중앙값`, "intensity difference", "보조 보정 특징");
  }
  for (const channel of ["r", "g", "b"]) {
    const upper = channel.toUpperCase();
    add(`${channel}_ratio_bg`, "Background adjusted", `${upper} ROI 중앙값 / 국소 배경 중앙값`, "ratio", "보조 보정 특징");
  }

  add("roi_pixel_count", "QC", "중심 ROI 전체 픽셀 수", "pixel", "QC; 모델 입력 제외");
  add("valid_pixel_count", "QC", "반사광 제외 후 유효 픽셀 수", "pixel", "QC; 모델 입력 제외");
  add("highlight_pixel_count", "QC", "고정 규칙으로 식별한 흰색 반사광 픽셀 수", "pixel", "QC; 모델 입력 제외");
  add("highlight_fraction", "QC", "ROI 중 반사광 픽셀 비율", "0-1", "QC; 민감도 분석 가능");
  add("valid_fraction", "QC", "ROI 중 유효 픽셀 비율", "0-1", "QC; 모델 입력 제외");
  add("dark_fraction", "QC", "ROI 중 V≤0.10인 픽셀 비율", "0-1", "QC; 모델 입력 제외");
  add("background_pixel_count", "QC", "국소 배경 고리 픽셀 수", "pixel", "QC; 모델 입력 제외");
  add("qc_status", "QC", "고정 임계값에 따른 pass/review", "-", "검토 플래그; 자동 제외하지 않음");
  add("qc_reason", "QC", "review 사유", "-", "검토 기록");
  return rows;
}


async function savePreview(workbook, sheetName, fileName, scale = 1, range = undefined) {
  const preview = await workbook.render({
    sheetName,
    range,
    autoCrop: "all",
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
  const featuresText = (await fs.readFile(FEATURES_CSV, "utf8")).replace(/^\uFEFF/, "");
  const summaryText = await fs.readFile(SUMMARY_CSV, "utf8");

  const workbook = await Workbook.fromCSV(featuresText, { sheetName: "Features" });
  const overview = workbook.worksheets.add("Overview");
  const qcSummary = workbook.worksheets.add("QC Summary");
  const dictionary = workbook.worksheets.add("Data Dictionary");

  overview.showGridLines = false;
  styleTitle(overview, overview.getRange("A1:F1"), "IFMBE 색상 특징 추출 결과");
  overview.getRange("A3:B3").values = [["데이터 항목", "값"]];
  styleHeader(overview.getRange("A3:B3"));
  overview.getRange("A4:A11").values = [
    ["원본 이미지"],
    ["전체 패치"],
    ["포도당 패치"],
    ["케톤 패치"],
    ["색상 후보 특징"],
    ["QC review"],
    ["최대 반사광 비율"],
    ["최소 유효 픽셀 비율"],
  ];
  overview.getRange("B4:B11").formulas = [
    ["=COUNTA('QC Summary'!$A$2:$A$20)"],
    ["=COUNTA(Features!$A$2:$A$1825)"],
    ["=COUNTIF(Features!$C$2:$C$1825,\"glucose\")"],
    ["=COUNTIF(Features!$C$2:$C$1825,\"ketone\")"],
    ["=42"],
    ["=COUNTIF(Features!$BH$2:$BH$1825,\"review\")"],
    ["=MAX(Features!$BD$2:$BD$1825)"],
    ["=MIN(Features!$BE$2:$BE$1825)"],
  ];
  overview.getRange("A4:B11").format = {
    borders: { preset: "all", style: "thin", color: "#BFBFBF" },
  };
  overview.getRange("A4:A11").format = { fill: COLORS.paleBlue, font: { bold: true } };
  overview.getRange("B4:B11").format = { horizontalAlignment: "right" };
  overview.getRange("B10:B11").format.numberFormat = "0.0%";

  overview.getRange("D3:F3").values = [["고정 추출 규칙", "설정", "목적"]];
  styleHeader(overview.getRange("D3:F3"));
  overview.getRange("D4:F9").values = [
    ["중심 ROI", "검출 반지름의 70%", "테두리와 인접 배경 제외"],
    ["반사광", "V≥0.96 및 S≤0.30", "흰색 specular pixel 제외"],
    ["국소 배경", "반지름 1.05-1.16배 고리", "보조 조명 참고값"],
    ["Hue", "채도 가중 원형 통계", "0/360도 경계 문제 방지"],
    ["색상 스케일", "RGB 0-255; HSV S/V 0-1", "해석 단위 고정"],
    ["사전 보정", "리사이즈·화이트밸런스 없음", "원본 픽셀값 보존"],
  ];
  overview.getRange("D4:F9").format = {
    borders: { preset: "all", style: "thin", color: "#BFBFBF" },
    wrapText: true,
    verticalAlignment: "top",
  };
  overview.getRange("D4:D9").format = { fill: COLORS.paleBlue, font: { bold: true } };

  overview.getRange("A13:F13").merge();
  overview.getRange("A13:F13").values = [["해석 시 주의사항"]];
  overview.getRange("A13:F13").format = {
    fill: COLORS.paleAmber,
    font: { bold: true, color: "#7F6000" },
  };
  overview.getRange("A14:F17").merge(true);
  overview.getRange("A14:F17").values = [
    ["• 모든 1,824개 패치는 QC 표에 유지되며 자동 제외하지 않았습니다."],
    ["• RGB-only와 HSV-only를 주 분석으로 비교하고, 상관된 RGB+HSV 결합은 보조 분석으로 다룹니다."],
    ["• 국소 배경 보정 특징은 원본 특징을 대체하지 않고 별도 모델 조합에서만 평가합니다."],
    ["• 현재 자료는 농도당 원본 이미지가 1장이므로 패치 CV는 동일 이미지 내부 일반화만 평가합니다."],
  ];
  overview.getRange("A14:F17").format = { wrapText: true, verticalAlignment: "top" };
  overview.getRange("A1:F17").format.font = { name: "Malgun Gothic", size: 10 };
  overview.getRange("A1:F1").format.font = { name: "Malgun Gothic", size: 18, bold: true, color: COLORS.white };
  overview.getRange("A:A").format.columnWidth = 22;
  overview.getRange("B:B").format.columnWidth = 13;
  overview.getRange("C:C").format.columnWidth = 3;
  overview.getRange("D:D").format.columnWidth = 18;
  overview.getRange("E:E").format.columnWidth = 25;
  overview.getRange("F:F").format.columnWidth = 30;
  overview.getRange("A14:F17").format.rowHeight = 26;
  overview.freezePanes.freezeRows(1);

  const features = workbook.worksheets.getItem("Features");
  features.showGridLines = false;
  for (const address of ["D2:E1825", "G2:H1825", "J2:BG1825"]) {
    const numericRange = features.getRange(address);
    numericRange.values = numericRange.values.map((row) =>
      row.map((value) => {
        const numeric = Number(value);
        return Number.isFinite(numeric) ? numeric : value;
      }),
    );
  }
  const featuresUsed = features.getUsedRange();
  featuresUsed.format.font = { name: "Aptos", size: 9 };
  features.getRange("A1:BI1").format = {
    fill: COLORS.navy,
    font: { bold: true, color: COLORS.white, name: "Aptos", size: 9 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: "#7F8FA4" },
  };
  features.getRange("A1:BI1").format.rowHeight = 42;
  features.freezePanes.freezeRows(1);
  features.freezePanes.freezeColumns(5);
  features.getRange("A:A").format.columnWidth = 24;
  features.getRange("B:B").format.columnWidth = 20;
  features.getRange("C:C").format.columnWidth = 11;
  features.getRange("D:E").format.columnWidth = 15;
  features.getRange("F:F").format.columnWidth = 10;
  features.getRange("G:H").format.columnWidth = 13;
  features.getRange("I:I").format.columnWidth = 55;
  features.getRange("J:J").format.columnWidth = 15;
  features.getRange("K:BG").format.columnWidth = 13;
  features.getRange("BH:BI").format.columnWidth = 18;
  features.getRange("E2:E1825").format.numberFormat = "0.00####";
  features.getRange("K2:BG1825").format.numberFormat = "0.000000";
  features.tables.add("A1:BI1825", true, "ColorFeatureTable");

  const summaryRows = parseSimpleCsv(summaryText);
  qcSummary.showGridLines = false;
  qcSummary.getRangeByIndexes(0, 0, summaryRows.length, summaryRows[0].length).values = summaryRows;
  styleHeader(qcSummary.getRange("A1:K1"));
  qcSummary.getRange("A1:K20").format.font = { name: "Malgun Gothic", size: 9 };
  qcSummary.getRange("A:A").format.columnWidth = 22;
  qcSummary.getRange("B:B").format.columnWidth = 11;
  qcSummary.getRange("C:K").format.columnWidth = 15;
  qcSummary.getRange("H2:K20").format.numberFormat = "0.0%";
  qcSummary.getRange("A1:K20").format.borders = { preset: "all", style: "thin", color: "#D9D9D9" };
  qcSummary.tables.add("A1:K20", true, "QCSummaryTable");
  qcSummary.freezePanes.freezeRows(1);

  const dictionaryRows = featureDictionary();
  dictionary.showGridLines = false;
  styleTitle(dictionary, dictionary.getRange("A1:E1"), "색상 특징 데이터 사전");
  dictionary.getRange("A3:E3").values = [["필드명", "구분", "정의", "단위/범위", "권장 사용"]];
  styleHeader(dictionary.getRange("A3:E3"));
  dictionary.getRangeByIndexes(3, 0, dictionaryRows.length, 5).values = dictionaryRows;
  const dictionaryLastRow = 3 + dictionaryRows.length;
  dictionary.getRange(`A3:E${dictionaryLastRow}`).format = {
    borders: { preset: "all", style: "thin", color: "#D9D9D9" },
    wrapText: true,
    verticalAlignment: "top",
    font: { name: "Malgun Gothic", size: 9 },
  };
  dictionary.getRange(`B4:B${dictionaryLastRow}`).format.fill = COLORS.paleBlue;
  dictionary.getRange("A:A").format.columnWidth = 27;
  dictionary.getRange("B:B").format.columnWidth = 20;
  dictionary.getRange("C:C").format.columnWidth = 48;
  dictionary.getRange("D:D").format.columnWidth = 18;
  dictionary.getRange("E:E").format.columnWidth = 35;
  dictionary.freezePanes.freezeRows(3);
  dictionary.tables.add(`A3:E${dictionaryLastRow}`, true, "FeatureDictionaryTable");

  const inspection = await workbook.inspect({
    kind: "workbook,sheet,formula",
    maxChars: 9000,
    tableMaxRows: 5,
    tableMaxCols: 8,
  });
  await fs.writeFile(
    path.join(PREVIEW_DIR, "workbook_inspection.txt"),
    inspection.ndjson ?? String(inspection),
    "utf8",
  );

  await savePreview(workbook, "Overview", "overview.png", 1.2);
  await savePreview(workbook, "Features", "features_sample.png", 0.9, "A1:H12");
  await savePreview(workbook, "QC Summary", "qc_summary.png", 1.0);
  await savePreview(workbook, "Data Dictionary", "data_dictionary.png", 0.9);

  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(OUTPUT_XLSX);
  await fs.rm(`${OUTPUT_XLSX}.inspect.ndjson`, { force: true });
  console.log(`Saved ${OUTPUT_XLSX}`);
}


await main();
