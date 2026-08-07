import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";


const workbookPath = path.join(
  process.cwd(),
  "outputs",
  "color_features",
  "color_features_review.xlsx",
);

const input = await FileBlob.load(workbookPath);
const workbook = await SpreadsheetFile.importXlsx(input);
const sheetNames = Array.from(workbook.worksheets).map((sheet) => sheet.name);
const features = workbook.worksheets.getItem("Features");
const overview = workbook.worksheets.getItem("Overview");

const errors = [];
const expectedSheets = ["Features", "Overview", "QC Summary", "Data Dictionary"];
for (const name of expectedSheets) {
  if (!sheetNames.includes(name)) errors.push(`Missing sheet: ${name}`);
}

const featureHeader = features.getRange("A1:BI1").values[0];
const patchIds = features.getRange("A2:A1825").values.flat();
const numericSample = features.getRange("D2:E3").values.flat();
const overviewValues = overview.getRange("B4:B11").values.flat();

if (featureHeader[0] !== "patch_id") errors.push("The first feature header is not patch_id");
if (featureHeader.length !== 61) errors.push(`Expected 61 columns, found ${featureHeader.length}`);
if (patchIds.length !== 1824) errors.push(`Expected 1824 data rows, found ${patchIds.length}`);
if (new Set(patchIds).size !== 1824) errors.push("Duplicate patch IDs in workbook");
if (!numericSample.every((value) => typeof value === "number")) {
  errors.push("Numeric feature cells were imported as text");
}
if (overviewValues.some((value) => typeof value === "string" && value.startsWith("#"))) {
  errors.push("Formula error displayed on Overview sheet");
}

const report = {
  workbook: workbookPath,
  sheets: sheetNames,
  feature_rows: patchIds.length,
  feature_columns: featureHeader.length,
  unique_patch_ids: new Set(patchIds).size,
  numeric_cells_verified: numericSample,
  overview_values: overviewValues,
  errors,
};
console.log(JSON.stringify(report, null, 2));
if (errors.length) process.exitCode = 1;
